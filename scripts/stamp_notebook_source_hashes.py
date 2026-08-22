#!/usr/bin/env python3
"""Stamp output-bearing notebook code cells with source SHA-256 digests."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any


class NotebookStampError(ValueError):
    """Raised when a notebook cannot be safely validated or stamped."""


def logical_source(source: object) -> str:
    """Return the notebook-format logical source string without normalization."""
    if isinstance(source, str):
        return source
    if isinstance(source, list) and all(isinstance(part, str) for part in source):
        return "".join(source)
    raise NotebookStampError("notebook cell source must be a JSON string or a list of strings")


def compute_source_hash(source: object) -> str:
    """Hash the exact UTF-8 logical source and return a bare lowercase digest."""
    return hashlib.sha256(logical_source(source).encode("utf-8")).hexdigest()


def _validate_document(document: object) -> dict[str, Any]:
    if not isinstance(document, dict):
        raise NotebookStampError("notebook JSON root must be an object")
    cells = document.get("cells")
    if not isinstance(cells, list):
        raise NotebookStampError("notebook cells must be a list")
    metadata = document.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        raise NotebookStampError("notebook metadata must be an object")

    for index, cell in enumerate(cells):
        if not isinstance(cell, dict):
            raise NotebookStampError(f"notebook cell {index} must be an object")
        cell_type = cell.get("cell_type")
        if cell_type not in {"code", "markdown", "raw"}:
            raise NotebookStampError(f"notebook cell {index} has invalid cell_type")
        if "source" not in cell:
            raise NotebookStampError(f"notebook cell {index} is missing source")
        logical_source(cell["source"])
        cell_metadata = cell.get("metadata")
        if cell_metadata is not None and not isinstance(cell_metadata, dict):
            raise NotebookStampError(f"notebook cell {index} metadata must be an object")
        if cell_type == "code":
            outputs = cell.get("outputs")
            if not isinstance(outputs, list):
                raise NotebookStampError(f"notebook code cell {index} outputs must be a list")
            if not all(isinstance(output, dict) for output in outputs):
                raise NotebookStampError(f"notebook code cell {index} outputs must contain objects")
            if "execution_count" in cell and cell["execution_count"] is not None and not isinstance(
                cell["execution_count"], int
            ):
                raise NotebookStampError(f"notebook code cell {index} execution_count must be an integer or null")
        if "id" in cell and not isinstance(cell["id"], str):
            raise NotebookStampError(f"notebook cell {index} id must be a string")
    return document


def stamp_document(document: dict[str, object]) -> int:
    """Update source markers in a validated notebook mapping in place."""
    validated = _validate_document(document)
    changed = 0
    for cell in validated["cells"]:
        if cell["cell_type"] != "code":
            continue
        if cell["outputs"]:
            metadata = cell.setdefault("metadata", {})
            if not isinstance(metadata, dict):  # guarded by _validate_document
                raise NotebookStampError("code cell metadata must be an object")
            digest = compute_source_hash(cell["source"])
            if metadata.get("source_hash") != digest:
                metadata["source_hash"] = digest
                changed += 1
        elif "source_hash" in cell.get("metadata", {}):
            metadata = cell["metadata"]
            del metadata["source_hash"]
            changed += 1
    return changed


def _parse_notebook(raw: bytes, notebook: Path) -> dict[str, object]:
    try:
        text = raw.decode("utf-8")
        document = json.loads(text, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise NotebookStampError(f"invalid notebook JSON in {notebook}: {exc}") from exc
    try:
        return _validate_document(document)
    except NotebookStampError as exc:
        raise NotebookStampError(f"invalid notebook structure in {notebook}: {exc}") from exc


def _serialize(document: dict[str, object]) -> bytes:
    try:
        return (json.dumps(document, indent=1, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise NotebookStampError(f"cannot serialize notebook: {exc}") from exc


def stamp_path(notebook: Path) -> int:
    """Atomically stamp one notebook, returning the number of marker changes."""
    notebook = Path(notebook)
    try:
        original = notebook.read_bytes()
        mode = stat.S_IMODE(notebook.stat().st_mode)
    except OSError as exc:
        raise NotebookStampError(f"cannot read notebook {notebook}: {exc}") from exc

    document = _parse_notebook(original, notebook)
    changed = stamp_document(document)
    serialized = _serialize(document)
    if serialized == original:
        return changed

    temporary: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{notebook.name}.", suffix=".tmp", dir=notebook.parent
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, notebook)
        temporary = None
    except OSError as exc:
        if temporary is not None:
            try:
                temporary.unlink()
            except OSError:
                pass
        raise NotebookStampError(f"atomic notebook replacement failed for {notebook}: {exc}") from exc
    return changed


def active_notebook_paths(repo: Path) -> tuple[Path, ...]:
    """Discover active notebooks from the authoritative YAML task inventory."""
    config_path = Path(repo) / "scripts" / "verify_repo_config.yaml"
    try:
        import yaml

        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (ImportError, OSError, ValueError) as exc:
        raise NotebookStampError(f"cannot load notebook inventory {config_path}: {exc}") from exc
    if not isinstance(config, dict) or not isinstance(config.get("active_task_dirs"), list):
        raise NotebookStampError("notebook inventory must define active_task_dirs as a list")

    notebooks_root = Path(repo) / "notebooks"
    discovered: set[Path] = set()
    for task_dir in config["active_task_dirs"]:
        if not isinstance(task_dir, str):
            raise NotebookStampError("active_task_dirs entries must be strings")
        directory = notebooks_root / task_dir
        if not directory.is_dir():
            continue
        for path in directory.rglob("*.ipynb"):
            if path.is_file() and "archive" not in path.relative_to(notebooks_root).parts:
                discovered.add(path)
    return tuple(sorted(discovered, key=lambda path: path.relative_to(repo).as_posix()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="notebook paths to stamp")
    parser.add_argument("--all-active", action="store_true", help="stamp notebooks in active_task_dirs")
    args = parser.parse_args(argv)
    if bool(args.paths) == args.all_active:
        parser.error("provide one or more paths or --all-active (but not both)")

    notebooks = active_notebook_paths(Path(__file__).resolve().parent.parent) if args.all_active else args.paths
    try:
        for notebook in notebooks:
            stamp_path(notebook)
    except NotebookStampError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
