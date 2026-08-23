"""Shared repository traversal and subprocess primitives."""
from __future__ import annotations

import subprocess
from collections.abc import Callable, Iterator
from pathlib import Path

import nbformat

from .models import VerifierConfig


DEFAULT_SUBPROCESS_TIMEOUT = 120


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, FileNotFoundError):
        return ""


def _subprocess_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def run_command(
    cmd: list[str], cwd: Path, timeout: int | None = DEFAULT_SUBPROCESS_TIMEOUT
) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        # rc=124 mirrors GNU `timeout(1)`: callers already branch on rc != 0
        # to surface a Finding, so a hung make target produces a clean
        # error rather than crashing the verifier.
        stdout = _subprocess_text(e.stdout)
        stderr = _subprocess_text(e.stderr)
        return 124, stdout, stderr + f"\n[verify_repo] timed out after {timeout}s"
    return proc.returncode, proc.stdout, proc.stderr


def iter_notebooks(
    repo: Path, config: VerifierConfig, notebook_root: Path
) -> Iterator[Path]:
    for task in config.active_task_dirs:
        for notebook_path in (repo / notebook_root / task).glob("*.ipynb"):
            yield notebook_path


def iter_in_scope_text_files(
    repo: Path,
    config: VerifierConfig,
    notebook_root: Path,
    read_text: Callable[[Path], str],
) -> Iterator[Path]:
    candidates = [
        repo / "README.md",
        repo / "CONTRIBUTING.md",
        repo / "CHANGELOG.md",
    ]
    manifest_path = repo / "docs" / "manifest.yaml"
    if manifest_path.exists():
        from scripts.docs.check_docs import manifest_markdown_sources
        from scripts.docs.manifest import load_manifest

        try:
            manifest = load_manifest(manifest_path, repo)
        except (OSError, ValueError):
            manifest = None
        if manifest is not None:
            candidates.extend(
                repo / source
                for source in sorted(manifest_markdown_sources(manifest))
                if Path(source).parent == Path(".")
                and Path(source).suffix.lower() == ".md"
            )
    candidates.extend(
        path
        for path in sorted((repo / "docs").rglob("*.md"))
        if not path.relative_to(repo).as_posix().startswith("docs/superpowers/")
    )
    candidates.extend(
        path
        for task in config.active_task_dirs
        for path in (repo / notebook_root / task).glob("*.md")
    )
    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        yield path


def iter_in_scope_markdown_documents(
    repo: Path,
    config: VerifierConfig,
    notebook_root: Path,
    read_text: Callable[[Path], str],
) -> Iterator[tuple[Path, Path, str]]:
    for markdown_path in iter_in_scope_text_files(
        repo, config, notebook_root, read_text
    ):
        yield markdown_path, markdown_path.parent, read_text(markdown_path)
    for notebook_path in iter_notebooks(repo, config, notebook_root):
        try:
            document = nbformat.read(notebook_path, as_version=4)
        except Exception:
            continue
        text = "\n\n".join(
            cell.source for cell in document.cells if cell.cell_type == "markdown"
        )
        yield notebook_path, notebook_path.parent, text
