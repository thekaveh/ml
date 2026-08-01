from __future__ import annotations

import shutil
from pathlib import Path


def copy_project_assets(repo_root: Path, out_dir: Path, expected: set[Path]) -> list[Path]:
    source_root = repo_root / "docs/assets"
    if not source_root.exists():
        return []
    written: list[Path] = []
    for source in sorted(path for path in source_root.rglob("*") if path.is_file()):
        destination = out_dir / "assets" / source.relative_to(source_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        expected.add(destination)
        written.append(destination)
    return written
