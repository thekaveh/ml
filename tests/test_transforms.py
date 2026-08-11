# tests/test_transforms.py
from __future__ import annotations

from pathlib import Path

from scripts.docs.transforms import build_source_map, rewrite_for_surface
from scripts.docs.manifest import load_manifest, parse_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]

MANIFEST = parse_manifest(
    """
surfaces: [repo, site, wiki]
numbering: baked
sections:
  - id: overview
    number: "1"
    title: Overview
    source: docs/index.md
  - id: architecture
    number: "2"
    title: Architecture
    source: docs/architecture.md
    children:
      - id: system
        number: "2.1"
        title: System view
notebooks:
  - task: tabular_classification-iris-mlp-pytorch
    number: "8.1"
    family: tabular
    depth: full
    doc: docs/notebooks/tabular_classification-iris-mlp-pytorch.md
    spec: notebooks/tabular_classification-iris-mlp-pytorch/docs/spec.yaml
diagrams: []
"""
)


def test_build_source_map_site():
    sm = build_source_map(MANIFEST, surface="site")
    assert sm["docs/index.md"] == "index.md"
    assert sm["docs/architecture.md"] == "architecture.md"
    assert sm["docs/notebooks/tabular_classification-iris-mlp-pytorch.md"] == "notebooks/tabular_classification-iris-mlp-pytorch.md"


def test_build_source_map_includes_parent_and_children_sources():
    manifest = parse_manifest(
        """surfaces: [repo, site, wiki]
numbering: baked
sections:
  - id: environment
    number: "4"
    title: Environment
    source: docs/env-setup.md
    children:
      - id: jupyterhub
        number: "4.1"
        title: JupyterHub
        source: docs/jupyterhub-integration.md
notebooks: []
diagrams: []
"""
    )

    assert build_source_map(manifest, "site") == {
        "docs/env-setup.md": "env-setup.md",
        "docs/jupyterhub-integration.md": "jupyterhub-integration.md",
    }


def test_build_source_map_wiki():
    sm = build_source_map(MANIFEST, surface="wiki")
    assert sm["docs/index.md"] == "Home.md"
    assert sm["docs/architecture.md"] == "2-Architecture.md"
    assert sm["docs/notebooks/tabular_classification-iris-mlp-pytorch.md"] == "8-1-tabular-classification-iris-mlp-pytorch.md"


def test_rewrite_strips_forbidden_links_to_bare_text():
    md = "see [the README](https://github.com/thekaveh/ml-eng-lab/blob/main/README.md) for more."
    out = rewrite_for_surface(md, surface="site", source_map={})
    assert out == "see the README for more."


def test_rewrite_rewrites_md_links_via_source_map():
    sm = {"docs/architecture.md": "architecture.md"}
    md = "see [arch](docs/architecture.md)."
    assert rewrite_for_surface(md, "site", sm) == "see [arch](architecture.md)."


def test_rewrite_drops_ipynb_links_to_bare_text():
    md = "open [the notebook](notebook.ipynb)."
    assert rewrite_for_surface(md, "site", {}) == "open the notebook."


def test_rewrite_strips_non_manifest_md_links_to_bare_text():
    # Relative .md link to a doc NOT in the surface's source_map (e.g. env-setup, a README) is
    # valid in-repo but absent from the generated site/wiki → bare text (self-contained surfaces).
    md = "see [env setup](../env-setup.md) and [the README](../../notebooks/t/README.md)."
    assert rewrite_for_surface(md, "site", {}) == "see env setup and the README."


def test_real_manifest_maps_root_security_and_rewrites_policy_links():
    manifest = load_manifest(REPO_ROOT / "docs/manifest.yaml", REPO_ROOT)
    policy = (REPO_ROOT / "SECURITY.md").read_text(encoding="utf-8")
    site_map = build_source_map(manifest, "site")
    wiki_map = build_source_map(manifest, "wiki")

    assert site_map["SECURITY.md"] == "SECURITY.md"
    assert wiki_map["SECURITY.md"] == "13-Security-policy.md"

    for source in (
        "docs/conventions.md",
        "docs/dependency-contracts.md",
        "docs/atlas-pin-bump-runbook.md",
        "docs/jupyterhub-integration.md",
    ):
        assert f"]({source})" in policy

    site_policy = rewrite_for_surface(policy, "site", site_map)
    assert "](conventions.md)" in site_policy
    assert "](dependency-contracts.md)" in site_policy
    assert "](atlas-pin-bump-runbook.md)" in site_policy
    assert "](jupyterhub-integration.md)" in site_policy

    wiki_policy = rewrite_for_surface(policy, "wiki", wiki_map)
    assert "](5-Repository-conventions.md)" in wiki_policy
    assert "](6-1-Dependency-ledger.md)" in wiki_policy
    assert "](6-2-Atlas-pin-bump-and-service-admission-runbook.md)" in wiki_policy
    assert "](4-2-JupyterHub-integration.md)" in wiki_policy
