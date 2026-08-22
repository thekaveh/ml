#!/usr/bin/env python3
"""Stamp output-bearing notebook code cells with source SHA-256 digests."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable


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


def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_output(
    output: object, index: int, output_index: int, *, allow_failed_execution: bool = False
) -> None:
    if not isinstance(output, dict):
        raise NotebookStampError(f"notebook code cell {index} output {output_index} must be an object")
    output_type = output.get("output_type")
    if output_type == "error":
        if not isinstance(output.get("ename"), str) or not isinstance(output.get("evalue"), str):
            raise NotebookStampError(f"notebook code cell {index} error output names must be strings")
        traceback = output.get("traceback")
        if not isinstance(traceback, list) or not all(isinstance(line, str) for line in traceback):
            raise NotebookStampError(f"notebook code cell {index} error output traceback must be a list of strings")
        if not allow_failed_execution:
            raise NotebookStampError(f"notebook code cell {index} contains an error output")
        return
    if output_type == "stream":
        if output.get("name") not in {"stdout", "stderr"}:
            raise NotebookStampError(f"notebook code cell {index} stream output has invalid name")
        text = output.get("text")
        if not isinstance(text, str) and not (
            isinstance(text, list) and all(isinstance(part, str) for part in text)
        ):
            raise NotebookStampError(f"notebook code cell {index} stream output text must be a string or list")
        return
    if output_type in {"display_data", "execute_result"}:
        if not isinstance(output.get("data"), dict):
            raise NotebookStampError(f"notebook code cell {index} output data must be an object")
        if not isinstance(output.get("metadata"), dict):
            raise NotebookStampError(f"notebook code cell {index} output metadata must be an object")
        if output_type == "execute_result" and not (
            _is_integer(output.get("execution_count"))
            or (allow_failed_execution and output.get("execution_count") is None)
        ):
            raise NotebookStampError(
                f"notebook code cell {index} execute_result execution_count must be an integer"
            )
        return
    raise NotebookStampError(f"notebook code cell {index} has invalid output type {output_type!r}")


def _is_json_value(value: object) -> bool:
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_is_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _is_json_value(item) for key, item in value.items())
    return False


def _validate_attachments(attachments: object, index: int) -> None:
    if not isinstance(attachments, dict):
        raise NotebookStampError(f"notebook markdown cell {index} attachments must be an object")
    for filename, bundle in attachments.items():
        if not isinstance(filename, str) or not filename:
            raise NotebookStampError(f"notebook markdown cell {index} attachment filename must be non-empty")
        if not isinstance(bundle, dict) or not bundle:
            raise NotebookStampError(f"notebook markdown cell {index} attachment bundle must be a non-empty object")
        for mime_type, value in bundle.items():
            if not isinstance(mime_type, str) or not mime_type:
                raise NotebookStampError(f"notebook markdown cell {index} attachment MIME key must be non-empty")
            if re.fullmatch(r"application/(?:.*\+)?json", mime_type):
                if not _is_json_value(value):
                    raise NotebookStampError(f"notebook markdown cell {index} attachment JSON value is invalid")
            elif not isinstance(value, str) and not (
                isinstance(value, list) and all(isinstance(part, str) for part in value)
            ):
                raise NotebookStampError(f"notebook markdown cell {index} attachment value must be a string or list")


def _validate_document(
    document: object, *, allow_failed_execution: bool = False
) -> dict[str, Any]:
    if not isinstance(document, dict):
        raise NotebookStampError("notebook JSON root must be an object")
    if "nbformat" not in document or not _is_integer(document["nbformat"]) or document["nbformat"] != 4:
        raise NotebookStampError("notebook nbformat must be integer version 4")
    if "nbformat_minor" not in document or not _is_integer(document["nbformat_minor"]):
        raise NotebookStampError("notebook nbformat_minor must be a non-negative integer")
    if document["nbformat_minor"] < 0:
        raise NotebookStampError("notebook nbformat_minor must be a non-negative integer")
    if "metadata" not in document or not isinstance(document["metadata"], dict):
        raise NotebookStampError("notebook metadata must be an object")
    cells = document.get("cells")
    if not isinstance(cells, list):
        raise NotebookStampError("notebook cells must be a list")

    for index, cell in enumerate(cells):
        if not isinstance(cell, dict):
            raise NotebookStampError(f"notebook cell {index} must be an object")
        cell_type = cell.get("cell_type")
        if cell_type not in {"code", "markdown", "raw"}:
            raise NotebookStampError(f"notebook cell {index} has invalid cell_type")
        if "source" not in cell:
            raise NotebookStampError(f"notebook cell {index} is missing source")
        logical_source(cell["source"])
        if "metadata" not in cell or not isinstance(cell["metadata"], dict):
            raise NotebookStampError(f"notebook cell {index} metadata must be an object")
        if "attachments" in cell:
            if cell_type != "markdown":
                raise NotebookStampError(f"notebook cell {index} attachments are only valid on markdown cells")
            _validate_attachments(cell["attachments"], index)
        if cell_type == "code":
            if "execution_count" not in cell or (
                cell["execution_count"] is not None and not _is_integer(cell["execution_count"])
            ):
                raise NotebookStampError(f"notebook code cell {index} execution_count must be an integer or null")
            outputs = cell.get("outputs")
            if not isinstance(outputs, list):
                raise NotebookStampError(f"notebook code cell {index} outputs must be a list")
            if outputs and cell["execution_count"] is None and not allow_failed_execution:
                raise NotebookStampError(f"notebook code cell {index} output-bearing execution_count must be an integer")
            for output_index, output in enumerate(outputs):
                _validate_output(
                    output,
                    index,
                    output_index,
                    allow_failed_execution=allow_failed_execution,
                )
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


def clear_document(document: dict[str, object]) -> int:
    """Remove source markers from every code cell in a validated notebook mapping."""
    validated = _validate_document(document, allow_failed_execution=True)
    changed = 0
    for cell in validated["cells"]:
        if cell["cell_type"] == "code" and "source_hash" in cell["metadata"]:
            del cell["metadata"]["source_hash"]
            changed += 1
    return changed


def _parse_notebook(
    raw: bytes, notebook: Path, *, allow_failed_execution: bool = False
) -> dict[str, object]:
    try:
        text = raw.decode("utf-8")
        document = json.loads(text, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise NotebookStampError(f"invalid notebook JSON in {notebook}: {exc}") from exc
    try:
        return _validate_document(document, allow_failed_execution=allow_failed_execution)
    except NotebookStampError as exc:
        raise NotebookStampError(f"invalid notebook structure in {notebook}: {exc}") from exc


def _serialize(document: dict[str, object]) -> bytes:
    try:
        return (json.dumps(document, indent=1, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise NotebookStampError(f"cannot serialize notebook: {exc}") from exc


def _update_path(
    notebook: Path,
    update_document: Callable[[dict[str, object]], int],
    *,
    allow_failed_execution: bool = False,
) -> int:
    notebook = Path(notebook)
    try:
        original = notebook.read_bytes()
        mode = stat.S_IMODE(notebook.stat().st_mode)
    except OSError as exc:
        raise NotebookStampError(f"cannot read notebook {notebook}: {exc}") from exc

    document = _parse_notebook(
        original, notebook, allow_failed_execution=allow_failed_execution
    )
    changed = update_document(document)
    if changed == 0:
        return 0
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


def stamp_path(notebook: Path) -> int:
    """Atomically stamp one notebook, returning the number of marker changes."""
    return _update_path(notebook, stamp_document)


def clear_path(notebook: Path) -> int:
    """Atomically clear one notebook, returning the number of markers removed."""
    return _update_path(notebook, clear_document, allow_failed_execution=True)


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
    parser.add_argument("--clear", action="store_true", help="clear source hashes from explicit notebook paths")
    args = parser.parse_args(argv)
    if args.clear and (args.all_active or not args.paths):
        parser.error("--clear requires one or more explicit notebook paths and cannot use --all-active")
    if not args.clear and bool(args.paths) == args.all_active:
        parser.error("provide one or more paths or --all-active (but not both)")

    notebooks = active_notebook_paths(Path(__file__).resolve().parent.parent) if args.all_active else args.paths
    update_path = clear_path if args.clear else stamp_path
    try:
        for notebook in notebooks:
            update_path(notebook)
    except NotebookStampError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
