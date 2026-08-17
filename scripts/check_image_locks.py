"""Networked validation for pinned multi-platform base-image identities."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from scripts.dependency_locks import DependencyLockError, load_image_ledger


Inspector = Callable[[str], Mapping[str, object]]


class ImageLockError(RuntimeError):
    """A stable, redacted image-lock validation failure."""


def _inspect_with_buildx(source: str) -> Mapping[str, object]:
    try:
        result = subprocess.run(
            ("docker", "buildx", "imagetools", "inspect", source, "--raw"),
            check=False,
            capture_output=True,
            text=True,
        )
    except (OSError, UnicodeError) as exc:
        raise ImageLockError("image lock check failed: registry: execution") from exc
    if result.returncode != 0:
        raise ImageLockError("image lock check failed: registry: inspect")
    try:
        raw = json.loads(result.stdout)
        manifests = raw["manifests"]
        children = {
            f"{item['platform']['os']}/{item['platform']['architecture']}": item["digest"]
            for item in manifests
            if item.get("platform", {}).get("os") == "linux"
            and item.get("platform", {}).get("architecture") in {"amd64", "arm64"}
        }
        digest = subprocess.run(
            ("docker", "buildx", "imagetools", "inspect", source, "--format", "{{json .Manifest.Digest}}"),
            check=False,
            capture_output=True,
            text=True,
        )
        if digest.returncode != 0:
            raise ImageLockError("image lock check failed: registry: digest")
        return {
            "source": source,
            "index_digest": json.loads(digest.stdout),
            "media_type": raw["mediaType"],
            "children": children,
        }
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ImageLockError("image lock check failed: registry: schema") from exc


def check_image_locks(repo: Path, inspect: Inspector = _inspect_with_buildx) -> None:
    try:
        images = load_image_ledger(repo)
    except DependencyLockError as exc:
        raise ImageLockError("image lock check failed: ledger") from exc
    for image in images:
        try:
            actual = inspect(image.source)
        except ImageLockError:
            raise
        except Exception as exc:
            raise ImageLockError("image lock check failed: registry: execution") from exc
        if (
            actual.get("source") != image.source
            or actual.get("index_digest") != image.index_digest
            or actual.get("media_type")
            not in {
                "application/vnd.oci.image.index.v1+json",
                "application/vnd.docker.distribution.manifest.list.v2+json",
            }
            or actual.get("children") != dict(image.children)
        ):
            raise ImageLockError("image lock check failed: registry: identity")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    arguments = parser.parse_args(argv)
    try:
        check_image_locks(arguments.repo_root.resolve())
    except ImageLockError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
