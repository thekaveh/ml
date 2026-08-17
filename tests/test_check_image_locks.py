from __future__ import annotations

from pathlib import Path

import pytest

from scripts.check_image_locks import ImageLockError, check_image_locks
from scripts.dependency_locks import load_image_ledger


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_injected_registry_identities_match_the_exact_ledger() -> None:
    identities = {image.source: image for image in load_image_ledger(REPO_ROOT)}

    def inspect(source: str) -> dict[str, object]:
        image = identities[source]
        return {
            "source": source,
            "index_digest": image.index_digest,
            "media_type": "application/vnd.oci.image.index.v1+json",
            "children": dict(image.children),
        }

    check_image_locks(REPO_ROOT, inspect=inspect)


@pytest.mark.parametrize("mutation", ("index", "child", "missing", "extra", "media"))
def test_registry_identity_drift_fails_closed(mutation: str) -> None:
    identities = {image.source: image for image in load_image_ledger(REPO_ROOT)}

    def inspect(source: str) -> dict[str, object]:
        image = identities[source]
        children = dict(image.children)
        document: dict[str, object] = {
            "source": source,
            "index_digest": image.index_digest,
            "media_type": "application/vnd.oci.image.index.v1+json",
            "children": children,
        }
        if mutation == "index":
            document["index_digest"] = "sha256:" + "0" * 64
        elif mutation == "child":
            children["linux/amd64"] = "sha256:" + "0" * 64
        elif mutation == "missing":
            children.pop("linux/arm64")
        elif mutation == "extra":
            children["linux/s390x"] = "sha256:" + "0" * 64
        elif mutation == "media":
            document["media_type"] = "application/vnd.oci.image.manifest.v1+json"
        return document

    with pytest.raises(ImageLockError, match="registry"):
        check_image_locks(REPO_ROOT, inspect=inspect)
