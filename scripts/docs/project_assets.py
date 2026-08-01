from __future__ import annotations

import os
import shutil
from pathlib import Path


class ProjectAssetError(ValueError):
    """Raised when project assets cannot be projected safely."""


def _entries(root: Path):
    for current, directories, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in directories:
            yield current_path / name
        for name in files:
            yield current_path / name


def validate_generated_output(out_dir: Path) -> None:
    if out_dir.is_symlink():
        raise ProjectAssetError(f"generated output contains symlink: {out_dir}")
    if not out_dir.exists():
        return
    for path in _entries(out_dir):
        if path.is_symlink():
            relative = path.relative_to(out_dir)
            raise ProjectAssetError(
                f"generated output contains symlink: {relative}"
            )


def cleanup_generated_output(out_dir: Path, expected: set[Path]) -> None:
    validate_generated_output(out_dir)
    entries = sorted(
        out_dir.rglob("*"),
        key=lambda path: len(path.relative_to(out_dir).parts),
        reverse=True,
    )
    for path in entries:
        if path.is_file() and path not in expected:
            path.unlink()
        elif path.is_dir() and not any(path.iterdir()):
            path.rmdir()


def copy_project_assets(repo_root: Path, out_dir: Path, expected: set[Path]) -> list[Path]:
    source_root = repo_root / "docs/assets"
    if source_root.is_symlink():
        raise ProjectAssetError(
            f"canonical project assets contain symlink: {source_root}"
        )
    if not source_root.exists():
        return []
    for source in _entries(source_root):
        if source.is_symlink():
            relative = source.relative_to(source_root)
            raise ProjectAssetError(
                f"canonical project assets contain symlink: {relative}"
            )
    validate_generated_output(out_dir)
    written: list[Path] = []
    for source in sorted(path for path in source_root.rglob("*") if path.is_file()):
        destination = out_dir / "assets" / source.relative_to(source_root)
        if destination in expected:
            relative = destination.relative_to(out_dir)
            raise ProjectAssetError(
                f"project asset destination collision with generated output: {relative}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        expected.add(destination)
        written.append(destination)
    return written
