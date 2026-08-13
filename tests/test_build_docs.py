# tests/test_build_docs.py
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.docs.build_docs import build, render_mkdocs_yml, render_site
from scripts.docs.manifest import load_manifest, parse_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_YAML = """
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
    children:
      - id: system
        number: "2.1"
        title: System view
        source: docs/architecture.md
        diagrams: [system]
notebooks:
  - task: tabular_classification-iris-mlp-pytorch
    number: "8.1"
    family: tabular
    depth: full
    doc: docs/notebooks/tabular_classification-iris-mlp-pytorch.md
    spec: notebooks/tabular_classification-iris-mlp-pytorch/docs/spec.yaml
diagrams:
  - id: system
    master: docs/diagrams/ml-eng-lab-system.html
"""


def _seed(repo: Path) -> None:
    files = {
        "docs/index.md": "# 1. Overview\n",
        "docs/architecture.md": "## 2.1 System view\n\n![d](diagrams/img/system.png)\n",
        "docs/notebooks/tabular_classification-iris-mlp-pytorch.md": "## 8.1 Iris MLP\n",
        "notebooks/tabular_classification-iris-mlp-pytorch/docs/spec.yaml": "title: x\n",
        "docs/diagrams/ml-eng-lab-system.html": "<html><svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 10 10'></svg></html>",
        "docs/stylesheets/extra.css": "/* obsidian */\n",
        "docs/manifest.yaml": MANIFEST_YAML,
    }
    for rel, text in files.items():
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    poster = repo / "docs/assets/ml-eng-lab-poster.png"
    poster.parent.mkdir(parents=True, exist_ok=True)
    poster.write_bytes(b"poster")
    badge = repo / "docs/assets/badges/python.svg"
    badge.parent.mkdir(parents=True, exist_ok=True)
    badge.write_text("<svg xmlns='http://www.w3.org/2000/svg'/>", encoding="utf-8")


def test_render_site_writes_pages_and_assets(tmp_path):
    _seed(tmp_path)
    m = parse_manifest(MANIFEST_YAML)
    out = tmp_path / "generated/site"
    written = render_site(m, tmp_path, out, trusted_output_root=tmp_path)
    assert (out / "index.md").read_text().startswith("# 1. Overview")
    assert (out / "architecture.md").exists()
    assert (out / "notebooks/tabular_classification-iris-mlp-pytorch.md").exists()
    # PNG image ref rewritten to site SVG asset
    arch = (out / "architecture.md").read_text()
    assert "assets/img/system.svg" in arch and "diagrams/img/system.png" not in arch
    # stylesheet copied
    assert (out / "stylesheets/extra.css").exists()
    assert (out / "assets/ml-eng-lab-poster.png").read_bytes() == b"poster"
    assert (out / "assets/badges/python.svg").exists()
    assert any(p.name == "index.md" for p in written)


def test_render_mkdocs_yml_has_generated_nav_and_no_repo_url(tmp_path):
    _seed(tmp_path)
    m = parse_manifest(MANIFEST_YAML)
    text = render_mkdocs_yml(m, tmp_path, tmp_path / "generated/site")
    assert "docs_dir: generated/site" in text
    assert "site_dir: site" in text
    assert "exclude_docs" not in text
    assert "repo_url" not in text and "edit_uri" not in text
    parsed = yaml.safe_load(text)
    titles = [item if isinstance(item, str) else list(item)[0] for item in parsed["nav"]]
    assert any("1. Overview" in t for t in titles)
    assert any("2. Architecture" in t for t in titles)


def test_real_manifest_renders_root_security_page_and_site_navigation(tmp_path):
    manifest = load_manifest(REPO_ROOT / "docs/manifest.yaml", REPO_ROOT)
    out = tmp_path / "generated/site"

    render_site(manifest, REPO_ROOT, out, trusted_output_root=tmp_path)
    policy = (out / "SECURITY.md").read_text(encoding="utf-8")
    nav = yaml.safe_load(render_mkdocs_yml(manifest, REPO_ROOT, out))["nav"]

    assert "](dependency-contracts.md)" in policy
    assert {"13. Security policy": "SECURITY.md"} in nav


def test_real_manifest_projects_current_vulnerability_snapshot_to_site(tmp_path):
    manifest = load_manifest(REPO_ROOT / "docs/manifest.yaml", REPO_ROOT)
    out = tmp_path / "generated/site"
    render_site(manifest, REPO_ROOT, out, trusted_output_root=tmp_path)
    ledger = (out / "dependency-contracts.md").read_text(encoding="utf-8")
    assert "### 6.1.1.2 Current accepted advisories" in ledger
    assert "2026-08-12" in ledger
    assert "archived audit provenance only" in ledger
    assert "../SECURITY.md" not in ledger
    assert "](#6112-current-accepted-advisories)" in ledger


def test_nav_preserves_a_section_source_and_children():
    manifest = parse_manifest(
        MANIFEST_YAML.replace(
            "  - id: architecture\n    number: \"2\"\n    title: Architecture\n    children:",
            "  - id: architecture\n    number: \"2\"\n    title: Architecture\n    source: docs/architecture-parent.md\n    children:",
        )
    )

    nav = yaml.safe_load(render_mkdocs_yml(manifest, Path("."), Path(".")))["nav"]

    architecture = next(item["2. Architecture"] for item in nav if "2. Architecture" in item)
    assert architecture == [
        {"2. Architecture": "architecture-parent.md"},
        {"2.1. System view": "architecture.md"},
    ]


def test_nav_places_notebooks_before_later_numbered_sections():
    manifest = parse_manifest(
        MANIFEST_YAML.replace(
            "notebooks:",
            "  - id: findings\n    number: \"9\"\n    title: Findings\n"
            "    source: docs/findings.md\nnotebooks:",
        )
    )

    nav = yaml.safe_load(render_mkdocs_yml(manifest, Path("."), Path(".")))["nav"]
    labels = [next(iter(item)) for item in nav]

    assert labels.index("8. Notebooks") < labels.index("9. Findings")


def test_build_check_is_deterministic(tmp_path):
    _seed(tmp_path)
    rc1 = build(tmp_path / "docs/manifest.yaml", tmp_path, site=True, check=True)
    rc2 = build(tmp_path / "docs/manifest.yaml", tmp_path, site=True, check=True)
    assert rc1 == 0 and rc2 == 0


def test_render_site_removes_stale_generated_files(tmp_path):
    _seed(tmp_path)
    out = tmp_path / "generated/site"
    out.mkdir(parents=True)
    (out / "obsolete.md").write_text("stale", encoding="utf-8")

    render_site(
        parse_manifest(MANIFEST_YAML),
        tmp_path,
        out,
        trusted_output_root=tmp_path,
    )

    assert not (out / "obsolete.md").exists()


def test_render_site_removes_stale_empty_directories_recursively(tmp_path):
    _seed(tmp_path)
    out = tmp_path / "generated/site"
    stale = out / "old/nested"
    stale.mkdir(parents=True)
    (stale / "obsolete.md").write_text("stale", encoding="utf-8")

    render_site(
        parse_manifest(MANIFEST_YAML),
        tmp_path,
        out,
        trusted_output_root=tmp_path,
    )

    assert not (out / "old").exists()


@pytest.mark.parametrize("kind", ["root", "file", "directory"])
def test_render_site_rejects_symlinks_in_canonical_project_assets(tmp_path, kind):
    _seed(tmp_path)
    assets = tmp_path / "docs/assets"
    if kind == "root":
        target = tmp_path / "canonical-assets"
        assets.rename(target)
        assets.symlink_to(target, target_is_directory=True)
    elif kind == "file":
        source = assets / "ml-eng-lab-poster.png"
        target = tmp_path / "poster-target.png"
        source.rename(target)
        source.symlink_to(target)
    else:
        source = assets / "badges"
        target = tmp_path / "badge-targets"
        source.rename(target)
        source.symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="canonical project assets.*symlink"):
        render_site(
            parse_manifest(MANIFEST_YAML),
            tmp_path,
            tmp_path / "generated/site",
            trusted_output_root=tmp_path,
        )


@pytest.mark.parametrize("kind", ["root", "ancestor", "entry"])
def test_render_site_rejects_symlinks_in_generated_output(tmp_path, kind):
    _seed(tmp_path)
    out = tmp_path / "generated/site"
    target_dir = tmp_path / "generated-target"
    target_dir.mkdir()
    if kind == "root":
        out.parent.mkdir(parents=True)
        out.symlink_to(target_dir, target_is_directory=True)
    elif kind == "ancestor":
        out.mkdir(parents=True)
        (out / "assets").symlink_to(target_dir, target_is_directory=True)
    else:
        entry = out / "assets/ml-eng-lab-poster.png"
        entry.parent.mkdir(parents=True)
        target = tmp_path / "poster-target.png"
        target.write_bytes(b"original")
        entry.symlink_to(target)

    with pytest.raises(ValueError, match="generated output.*symlink"):
        render_site(
            parse_manifest(MANIFEST_YAML),
            tmp_path,
            out,
            trusted_output_root=tmp_path,
        )


def test_render_site_rejects_symlinked_generated_parent(tmp_path):
    _seed(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    generated = tmp_path / "generated"
    generated.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="generated output.*symlink"):
        render_site(
            parse_manifest(MANIFEST_YAML),
            tmp_path,
            generated / "site",
            trusted_output_root=tmp_path,
        )

    assert list(outside.iterdir()) == []


def test_render_site_rejects_project_asset_collision_with_diagram(tmp_path):
    _seed(tmp_path)
    collision = tmp_path / "docs/assets/img/system.svg"
    collision.parent.mkdir(parents=True)
    collision.write_text("project asset", encoding="utf-8")

    with pytest.raises(ValueError, match="destination collision.*assets/img/system.svg"):
        render_site(
            parse_manifest(MANIFEST_YAML),
            tmp_path,
            tmp_path / "generated/site",
            trusted_output_root=tmp_path,
        )


def test_rewrite_images_site_preserves_subdir_prefix():
    from scripts.docs.build_docs import _rewrite_images_site
    # deep-dive in notebooks/ uses ../diagrams/img/... → must keep ../ for the generated site
    assert _rewrite_images_site("![MLP](../diagrams/img/mlp.png)") == "![MLP](../assets/img/mlp.svg)"
    # root doc (no prefix) still resolves at the site root
    assert _rewrite_images_site("![x](diagrams/img/system.png)") == "![x](assets/img/system.svg)"


def test_assert_dirs_equal_catches_content_drift(tmp_path):
    from scripts.docs.build_docs import _assert_dirs_equal

    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "x.md").write_text("one")
    (b / "x.md").write_text("two")  # same path, different content
    with pytest.raises(AssertionError, match="content-diff"):
        _assert_dirs_equal(
            a,
            b,
            a_trusted_output_root=tmp_path,
            b_trusted_output_root=tmp_path,
        )
    (b / "x.md").write_text("one")  # now byte-identical
    _assert_dirs_equal(
        a,
        b,
        a_trusted_output_root=tmp_path,
        b_trusted_output_root=tmp_path,
    )  # no raise


def test_assert_dirs_equal_catches_entry_type_and_empty_directory_drift(tmp_path):
    from scripts.docs.build_docs import _assert_dirs_equal

    a, b = tmp_path / "a", tmp_path / "b"
    (a / "same-path").mkdir(parents=True)
    b.mkdir()
    (b / "same-path").write_text("file", encoding="utf-8")

    with pytest.raises(AssertionError, match="type-diff.*same-path"):
        _assert_dirs_equal(
            a,
            b,
            a_trusted_output_root=tmp_path,
            b_trusted_output_root=tmp_path,
        )

    (b / "same-path").unlink()
    (b / "same-path").mkdir()
    (a / "only-empty-directory").mkdir()
    with pytest.raises(AssertionError, match="only-in-temp.*only-empty-directory"):
        _assert_dirs_equal(
            a,
            b,
            a_trusted_output_root=tmp_path,
            b_trusted_output_root=tmp_path,
        )
