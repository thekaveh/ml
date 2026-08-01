# tests/test_check_docs.py
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.docs.check_docs import (
    check_notebook_infrastructure,
    check_completeness,
    check_numbering,
    check_placeholders,
    check_project_opening,
    check_repo_self_containment,
    check_self_containment,
    manifest_markdown_sources,
)
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
notebooks:
  - task: t-iris-mlp-pytorch
    number: "8.1"
    family: tabular
    depth: full
    doc: docs/notebooks/t.md
    spec: notebooks/t/docs/spec.yaml
diagrams: []
"""

BADGE_GROUPS = (
    (
        "Core ML",
        (
            ("Python", "python.svg"),
            ("Jupyter", "jupyter.svg"),
            ("NumPy", "numpy.svg"),
            ("pandas", "pandas.svg"),
            ("PyTorch", "pytorch.svg"),
            ("PyTorch Geometric", "pytorch-geometric.svg"),
            ("scikit-learn", "scikit-learn.svg"),
        ),
    ),
    (
        "NLP and graphs",
        (
            ("spaCy", "spacy.svg"),
            ("NLTK", "nltk.svg"),
            ("NetworkX", "networkx.svg"),
        ),
    ),
    (
        "Runtime",
        (
            ("Atlas", "atlas.svg"),
            ("Docker", "docker.svg"),
            ("VS Code", "vscode.svg"),
            ("GitHub Codespaces", "github-codespaces.svg"),
        ),
    ),
    (
        "Engineering",
        (
            ("NNx", "nnx.svg"),
            ("Papermill", "papermill.svg"),
            ("pytest", "pytest.svg"),
            ("Ruff", "ruff.svg"),
            ("GitHub Actions", "github-actions.svg"),
        ),
    ),
)

PROJECT_SUMMARY = (
    "ml-eng-lab is a portfolio of self-contained machine-learning notebook experiments built for "
    "local editing in VS Code and recommended remote execution through JupyterHub on Atlas's ML "
    "Engineering track. Unlike a loose notebook collection, each task declares its runtime needs in a "
    "checked infrastructure contract, keeping notebook dependencies explicit as the lab expands beyond "
    "JupyterHub. Narrative experiments, reproducible execution tiers, exact dependency pins, validation "
    "gates, and the reusable thekaveh-nnx toolkit evolve together.\n\n"
    "Contributors can use Browser JupyterLab for mounted-workspace tasks or choose a local virtual "
    "environment, Docker, or GitHub Codespaces when Atlas is not the right fit. Host-native Ollama is "
    "the only approved Ollama source whenever a future task needs it; containerized Ollama is "
    "intentionally excluded. This makes the lab both a practical portfolio and a controlled environment "
    "for growing machine-learning systems without hiding operational assumptions inside notebooks."
)


def test_self_contamination_flags_cross_surface_link(tmp_path):
    site = tmp_path / "site/page.md"
    site.parent.mkdir(parents=True)
    site.write_text("see [repo](https://github.com/thekaveh/ml-eng-lab/blob/main/README.md)", encoding="utf-8")
    findings = check_self_containment(tmp_path)
    assert any(f.severity == "error" and "site" in f.message.lower() for f in findings)


def test_self_contamination_clean(tmp_path):
    (tmp_path / "site").mkdir()
    (tmp_path / "site/page.md").write_text("see [next](next.md)", encoding="utf-8")
    assert check_self_containment(tmp_path) == []


def test_self_containment_rejects_missing_local_raw_html_image(tmp_path):
    page = tmp_path / "site/index.md"
    page.parent.mkdir(parents=True)
    page.write_text(
        '<img src="assets/ml-eng-lab-poster.png">\n'
        '<img src="https://example.com/remote-badge.svg">\n',
        encoding="utf-8",
    )

    findings = check_self_containment(tmp_path)

    assert len(findings) == 1
    assert "local image target missing" in findings[0].message
    assert "assets/ml-eng-lab-poster.png" in findings[0].message


def test_self_containment_ignores_raw_html_image_examples_in_code(tmp_path):
    page = tmp_path / "site/index.md"
    page.parent.mkdir(parents=True)
    page.write_text(
        "```html\n"
        '<img src="examples/fenced.png">\n'
        "```\n\n"
        'Use `<img src="examples/inline.png">` as an example.\n'
        '<img src="assets/rendered.png">\n',
        encoding="utf-8",
    )

    findings = check_self_containment(tmp_path)

    assert len(findings) == 1
    assert "assets/rendered.png" in findings[0].message


@pytest.mark.parametrize(
    "source",
    ["../outside.png", "/../outside.png", "%2e%2e/outside.png"],
)
def test_self_containment_rejects_raw_html_image_root_escapes(tmp_path, source):
    page = tmp_path / "site/index.md"
    page.parent.mkdir(parents=True)
    (tmp_path / "outside.png").write_bytes(b"outside")
    page.write_text(f'<img src="{source}">\n', encoding="utf-8")

    findings = check_self_containment(tmp_path)

    assert len(findings) == 1
    assert "escapes generated surface" in findings[0].message


def test_self_containment_rejects_raw_html_image_symlink_escape(tmp_path):
    page = tmp_path / "site/index.md"
    assets = tmp_path / "site/assets"
    assets.mkdir(parents=True)
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    (assets / "redirect.png").symlink_to(outside)
    page.write_text('<img src="assets/redirect.png">\n', encoding="utf-8")

    findings = check_self_containment(tmp_path)

    assert len(findings) == 1
    assert "escapes generated surface" in findings[0].message


@pytest.mark.parametrize("source", ["file:assets/existing.png", "ftp://example.com/image.png"])
def test_self_containment_rejects_non_web_raw_html_image_schemes(tmp_path, source):
    page = tmp_path / "site/index.md"
    page.parent.mkdir(parents=True)
    (page.parent / "assets").mkdir()
    (page.parent / "assets/existing.png").write_bytes(b"image")
    page.write_text(f'<img src="{source}">\n', encoding="utf-8")

    findings = check_self_containment(tmp_path)

    assert len(findings) == 1
    assert "unsupported raw HTML image source" in findings[0].message


def test_self_containment_allows_web_data_and_contained_root_images(tmp_path):
    page = tmp_path / "site/index.md"
    asset = tmp_path / "site/assets/existing.png"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"image")
    page.write_text(
        '<img src="https://example.com/image.png">\n'
        '<img src="//example.com/image.png">\n'
        '<img src="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw==">\n'
        '<img src="/assets/existing.png">\n',
        encoding="utf-8",
    )

    assert check_self_containment(tmp_path) == []


def test_self_containment_parses_unquoted_raw_html_image_src(tmp_path):
    page = tmp_path / "site/index.md"
    page.parent.mkdir(parents=True)
    page.write_text("<img src=assets/missing.png>\n", encoding="utf-8")

    findings = check_self_containment(tmp_path)

    assert len(findings) == 1
    assert "assets/missing.png" in findings[0].message


def test_self_containment_does_not_mistake_data_src_for_src(tmp_path):
    page = tmp_path / "site/index.md"
    asset = tmp_path / "site/assets/existing.png"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"image")
    page.write_text(
        '<img data-src="assets/existing.png">\n'
        '<img data-src="assets/existing.png" src=assets/missing.png>\n',
        encoding="utf-8",
    )

    messages = [finding.message for finding in check_self_containment(tmp_path)]

    assert len(messages) == 2
    assert any("missing src" in message for message in messages)
    assert any("assets/missing.png" in message for message in messages)


def test_self_containment_ignores_raw_html_images_in_comments_and_indented_code(tmp_path):
    page = tmp_path / "site/index.md"
    page.parent.mkdir(parents=True)
    page.write_text(
        '<!-- <img src="examples/comment.png"> -->\n'
        '    <img src="examples/spaces.png">\n'
        '\t<img src="examples/tab.png">\n'
        '<img src="assets/rendered.png">\n',
        encoding="utf-8",
    )

    findings = check_self_containment(tmp_path)

    assert len(findings) == 1
    assert "assets/rendered.png" in findings[0].message


def test_self_containment_flags_indented_raw_html_image_inside_html_block(tmp_path):
    page = tmp_path / "site/index.md"
    page.parent.mkdir(parents=True)
    page.write_text(
        "<div>\n"
        "    <img src=assets/missing.png>\n"
        "</div>\n",
        encoding="utf-8",
    )

    findings = check_self_containment(tmp_path)

    assert len(findings) == 1
    assert "assets/missing.png" in findings[0].message


def test_repo_self_containment_rejects_site_and_wiki_links(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "README.md").write_text(
        "https://thekaveh.github.io/ml-eng-lab/\n", encoding="utf-8"
    )
    (tmp_path / "docs/page.md").write_text(
        "[wiki](https://github.com/thekaveh/ml-eng-lab/wiki/Page)\n",
        encoding="utf-8",
    )

    findings = check_repo_self_containment(tmp_path)

    assert len(findings) == 2
    assert all(finding.severity == "error" for finding in findings)


def test_repo_self_containment_allows_relative_links(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "README.md").write_text("[docs](docs/page.md)\n", encoding="utf-8")
    (tmp_path / "docs/page.md").write_text("[next](next.md)\n", encoding="utf-8")

    assert check_repo_self_containment(tmp_path) == []


def test_completeness_flags_missing_spec(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/index.md").write_text("x", encoding="utf-8")
    m = parse_manifest(MANIFEST_YAML)
    findings = check_completeness(m, tmp_path)
    assert any("spec" in f.message.lower() or "doc" in f.message.lower() for f in findings)


def test_completeness_flags_unmanifested_markdown(tmp_path):
    manifest = _write_valid_notebook_infrastructure_fixture(tmp_path)
    (tmp_path / "docs/extra.md").write_text("# 9 Extra\n", encoding="utf-8")

    findings = check_completeness(manifest, tmp_path)

    assert any("docs/extra.md" in finding.message and "not declared" in finding.message for finding in findings)


def test_manifest_markdown_sources_contains_sections_and_notebooks():
    manifest = parse_manifest(MANIFEST_YAML)

    assert manifest_markdown_sources(manifest) == {
        "docs/index.md",
        "docs/notebooks/t.md",
    }


def test_real_manifest_declares_every_canonical_markdown_file():
    manifest = load_manifest(REPO_ROOT / "docs/manifest.yaml", REPO_ROOT)
    actual = {
        str(path.relative_to(REPO_ROOT))
        for path in (REPO_ROOT / "docs").rglob("*.md")
    }

    assert manifest_markdown_sources(manifest) == actual


def test_real_manifest_sections_are_source_leaves_or_children_groups():
    manifest = load_manifest(REPO_ROOT / "docs/manifest.yaml", REPO_ROOT)

    assert all(bool(section.source) ^ bool(section.children) for section in manifest.sections)


def test_real_user_facing_docs_match_the_atlas_runtime_contract():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    runtime_docs = "\n".join(
        (REPO_ROOT / path).read_text(encoding="utf-8")
        for path in (
            "docs/env-setup.md",
            "docs/jupyterhub-integration.md",
            "docs/vscode-remote-access.md",
            "docs/atlas-pin-bump-runbook.md",
        )
    )
    infrastructure = (REPO_ROOT / "docs/notebook-infrastructure.md").read_text(
        encoding="utf-8"
    )

    assert "MkDocs" not in readme
    assert "mkdocs.yml" not in readme
    assert "short-lived" not in readme + runtime_docs
    assert "token-bearing" in readme + runtime_docs
    assert "track defaults are not notebook authorization" in infrastructure
    assert "Additional Atlas services stay inactive" not in infrastructure


def _write_project_opening(
    repo_root: Path,
    *,
    readme_tagline: str = "Local notebooks. Remote Atlas execution. Explicit infrastructure contracts.",
    landing_tagline: str = "Local notebooks. Remote Atlas execution. Explicit infrastructure contracts.",
    readme_summary: str = PROJECT_SUMMARY,
    landing_summary: str = PROJECT_SUMMARY,
) -> None:
    badge_assets = repo_root / "docs/assets/badges"
    badge_assets.mkdir(parents=True)
    for _, badges in BADGE_GROUPS:
        for _, filename in badges:
            (badge_assets / filename).write_text("badge", encoding="utf-8")
    (repo_root / "docs/assets/ml-eng-lab-poster.png").write_text(
        "poster", encoding="utf-8"
    )

    def badge_html(prefix: str) -> str:
        rows = []
        for label, badges in BADGE_GROUPS:
            images = " ".join(
                f'<img alt="{alt}" src="{prefix}badges/{filename}">'
                for alt, filename in badges
            )
            rows.append(
                '<p align="center">\n'
                f"  <sub><strong>{label}</strong></sub><br>\n"
                f"  {images}\n"
                "</p>"
            )
        return "\n\n".join(rows)

    def opener(
        *, asset_prefix: str, title: str, tagline: str, summary: str
    ) -> str:
        return (
            '<p align="center">\n'
            f'  <img src="{asset_prefix}ml-eng-lab-poster.png" '
            'alt="ML Eng Lab — notebooks, systems, and reproducibility" width="100%">\n'
            "</p>\n\n"
            f'<h1 align="center">{title}</h1>\n\n'
            f'<p align="center"><strong>{tagline}</strong></p>\n\n'
            f"{badge_html(asset_prefix)}\n\n"
            "<!-- project-summary:start -->\n"
            f"{summary}\n"
            "<!-- project-summary:end -->\n"
            "\n## 1.1 Repository map\n"
        )

    (repo_root / "README.md").write_text(
        opener(
            asset_prefix="docs/assets/",
            title="ML ENG LAB",
            tagline=readme_tagline,
            summary=readme_summary,
        ),
        encoding="utf-8",
    )
    (repo_root / "docs/index.md").write_text(
        opener(
            asset_prefix="assets/",
            title="1 · ML ENG LAB",
            tagline=landing_tagline,
            summary=landing_summary,
        ),
        encoding="utf-8",
    )


def test_project_opening_rejects_tagline_and_summary_drift(tmp_path):
    _write_project_opening(
        tmp_path,
        landing_tagline="A different tagline.",
        landing_summary=f"{PROJECT_SUMMARY} changed",
    )

    messages = [finding.message for finding in check_project_opening(tmp_path)]

    assert any("tagline" in message for message in messages)
    assert any("summary" in message and "differ" in message for message in messages)


def test_project_opening_rejects_missing_poster_and_summary_outside_word_range(tmp_path):
    short_summary = f"ml-eng-lab {' '.join(['short'] * 20)}"
    _write_project_opening(
        tmp_path,
        readme_summary=short_summary,
        landing_summary=short_summary,
    )
    readme = tmp_path / "README.md"
    readme.write_text(
        readme.read_text(encoding="utf-8").replace(
            '  <img src="docs/assets/ml-eng-lab-poster.png" '
            'alt="ML Eng Lab — notebooks, systems, and reproducibility" width="100%">\n',
            "",
        ),
        encoding="utf-8",
    )

    messages = [finding.message for finding in check_project_opening(tmp_path)]

    assert any("poster" in message for message in messages)
    assert any("100-150 words" in message for message in messages)


def test_project_opening_rejects_missing_project_title(tmp_path):
    _write_project_opening(tmp_path)
    landing = tmp_path / "docs/index.md"
    landing.write_text(
        landing.read_text(encoding="utf-8").replace(
            '<h1 align="center">1 · ML ENG LAB</h1>',
            '<h1 align="center">1 · Overview</h1>',
        ),
        encoding="utf-8",
    )

    assert any("title" in finding.message for finding in check_project_opening(tmp_path))


def test_project_opening_rejects_content_inserted_before_poster(tmp_path):
    _write_project_opening(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text(
        readme.read_text(encoding="utf-8").replace(
            '<p align="center">\n  <img src="docs/assets/ml-eng-lab-poster.png"',
            'Unexpected prose.\n\n<p align="center">\n  <img src="docs/assets/ml-eng-lab-poster.png"',
        ),
        encoding="utf-8",
    )

    assert any("order" in finding.message for finding in check_project_opening(tmp_path))


def test_project_opening_rejects_runtime_diagram_as_poster(tmp_path):
    _write_project_opening(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text(
        readme.read_text(encoding="utf-8").replace(
            "docs/assets/ml-eng-lab-poster.png",
            "docs/diagrams/img/runtime-flow.png",
            1,
        ),
        encoding="utf-8",
    )

    assert any(
        "runtime-flow" in finding.message
        for finding in check_project_opening(tmp_path)
    )


def test_project_opening_rejects_left_aligned_markdown_title(tmp_path):
    _write_project_opening(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text(
        readme.read_text(encoding="utf-8").replace(
            '<h1 align="center">ML ENG LAB</h1>', "# ML ENG LAB", 1
        ),
        encoding="utf-8",
    )

    assert any(
        "centered HTML title" in finding.message
        for finding in check_project_opening(tmp_path)
    )


def test_project_opening_rejects_missing_badge_and_plain_text_stack(tmp_path):
    _write_project_opening(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text(
        readme.read_text(encoding="utf-8").replace(
            '<img alt="Python" src="docs/assets/badges/python.svg">',
            "Python",
            1,
        ),
        encoding="utf-8",
    )

    assert any("badge" in finding.message for finding in check_project_opening(tmp_path))


def test_project_opening_rejects_missing_local_asset(tmp_path):
    _write_project_opening(tmp_path)
    (tmp_path / "docs/assets/badges/python.svg").unlink()

    assert any(
        "asset missing" in finding.message
        for finding in check_project_opening(tmp_path)
    )


def test_project_opening_rejects_missing_poster_asset(tmp_path):
    _write_project_opening(tmp_path)
    (tmp_path / "docs/assets/ml-eng-lab-poster.png").unlink()

    assert any(
        "poster asset missing" in finding.message
        for finding in check_project_opening(tmp_path)
    )


def test_project_opening_rejects_single_paragraph_summary(tmp_path):
    _write_project_opening(tmp_path)
    landing = tmp_path / "docs/index.md"
    landing.write_text(
        landing.read_text(encoding="utf-8").replace("together.\n\nContributors", "together. Contributors", 1),
        encoding="utf-8",
    )

    assert any(
        "two paragraphs" in finding.message
        for finding in check_project_opening(tmp_path)
    )


def test_project_opening_rejects_runtime_flow_in_summary_tail(tmp_path):
    _write_project_opening(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text(
        readme.read_text(encoding="utf-8").replace(
            "<!-- project-summary:end -->\n\n## 1.1 Repository map",
            "<!-- project-summary:end -->\n\n"
            "![Runtime flow](docs/diagrams/img/runtime-flow.png)\n\n"
            "## 1.1 Repository map",
        ),
        encoding="utf-8",
    )

    assert any(
        "runtime-flow" in finding.message
        for finding in check_project_opening(tmp_path)
    )


def test_project_opening_rejects_nonwhitespace_summary_tail(tmp_path):
    _write_project_opening(tmp_path)
    landing = tmp_path / "docs/index.md"
    landing.write_text(
        landing.read_text(encoding="utf-8").replace(
            "<!-- project-summary:end -->\n\n## 1.1 Repository map",
            "<!-- project-summary:end -->\n\nOne-surface tail drift.\n\n"
            "## 1.1 Repository map",
        ),
        encoding="utf-8",
    )

    assert any(
        "only whitespace" in finding.message
        for finding in check_project_opening(tmp_path)
    )


def test_project_opening_allows_summary_line_reflow(tmp_path):
    _write_project_opening(
        tmp_path,
        landing_summary=PROJECT_SUMMARY.replace(
            "recommended remote execution through JupyterHub",
            "recommended remote execution\nthrough JupyterHub",
        ),
    )

    assert check_project_opening(tmp_path) == []


def test_project_opening_rejects_moved_summary_paragraph_boundary(tmp_path):
    _write_project_opening(
        tmp_path,
        landing_summary=PROJECT_SUMMARY.replace(
            "together.\n\nContributors can use Browser JupyterLab",
            "together. Contributors can use Browser\n\nJupyterLab",
        ),
    )

    messages = [finding.message for finding in check_project_opening(tmp_path)]

    assert any("summary differs" in message for message in messages)
    assert not any("exactly two paragraphs" in message for message in messages)


def test_project_opening_ignores_summary_tail_whitespace_drift(tmp_path):
    _write_project_opening(tmp_path)
    landing = tmp_path / "docs/index.md"
    landing.write_text(
        landing.read_text(encoding="utf-8").replace(
            "<!-- project-summary:end -->\n\n## 1.1 Repository map",
            "<!-- project-summary:end -->\n \n\n   \n## 1.1 Repository map",
        ),
        encoding="utf-8",
    )

    assert check_project_opening(tmp_path) == []


def test_real_project_opening_is_canonical():
    assert check_project_opening(REPO_ROOT) == []


def test_atlas_migration_records_are_marked_complete():
    design = (
        REPO_ROOT
        / "docs/superpowers/specs/2026-07-30-atlas-infrastructure-migration-design.md"
    ).read_text(encoding="utf-8")
    plan = (
        REPO_ROOT
        / "docs/superpowers/plans/2026-07-30-atlas-infrastructure-migration-implementation-plan.md"
    ).read_text(encoding="utf-8")

    assert "**Status:** Implemented" in design
    assert "**Status:** Completed" in plan
    assert "- [ ]" not in plan


def test_numbering_requires_manifest_h1_and_hierarchical_children(tmp_path):
    manifest = _write_valid_notebook_infrastructure_fixture(tmp_path)
    (tmp_path / "docs/index.md").write_text(
        "# Overview\n\n## 1. Wrong depth\n\n### Child without a number\n",
        encoding="utf-8",
    )

    messages = [finding.message for finding in check_numbering(manifest, tmp_path)]

    assert any("H1" in message for message in messages)
    assert any("H2" in message for message in messages)
    assert any("H3" in message for message in messages)


def test_numbering_accepts_hierarchical_headings_and_ignores_fences(tmp_path):
    manifest = _write_valid_notebook_infrastructure_fixture(tmp_path)
    (tmp_path / "docs/index.md").write_text(
        "# 1 Overview\n\n## 1.1 First\n\n### 1.1.1 Child\n\n"
        "```bash\n# shell comment\n## not a heading\n```\n",
        encoding="utf-8",
    )

    assert check_numbering(manifest, tmp_path) == []


def test_numbering_accepts_centered_html_h1(tmp_path):
    manifest = _write_valid_notebook_infrastructure_fixture(tmp_path)
    (tmp_path / "docs/index.md").write_text(
        '<h1 align="center">1 · ML ENG LAB</h1>\n\n## 1.1 Repository map\n',
        encoding="utf-8",
    )

    assert check_numbering(manifest, tmp_path) == []


def test_numbering_ignores_markdown_and_html_headings_in_html_comments(tmp_path):
    manifest = _write_valid_notebook_infrastructure_fixture(tmp_path)
    (tmp_path / "docs/index.md").write_text(
        "# 1 Overview\n\n"
        "<!-- ## 1.1 Single-line Markdown comment -->\n"
        '<!-- <h1 align="center">1 Commented title</h1> -->\n'
        "<!--\n"
        "## 1.1 Multiline Markdown comment\n"
        "-->\n"
        "<!--\n"
        '<h1 align="center">1 Multiline commented title</h1>\n'
        "-->\n"
        "## 1.1 Repository map\n",
        encoding="utf-8",
    )

    assert check_numbering(manifest, tmp_path) == []


@pytest.mark.parametrize(
    ("opener", "shorter_closer", "trailing_closer", "closer"),
    [
        ("````", "```", "```` trailing", "````"),
        ("~~~~", "~~~", "~~~~ trailing", "~~~~"),
    ],
)
def test_numbering_keeps_fences_open_for_shorter_or_trailing_closers(
    tmp_path, opener, shorter_closer, trailing_closer, closer
):
    manifest = _write_valid_notebook_infrastructure_fixture(tmp_path)
    (tmp_path / "docs/index.md").write_text(
        "# 1 Overview\n\n"
        f"{opener}text\n"
        "## 1.1 Still in the fence\n"
        f"{shorter_closer}\n"
        "## 1.1 Still in the fence\n"
        f"{trailing_closer}\n"
        "## 1.1 Still in the fence\n"
        f"{closer}\n\n"
        "## 1.1 Repository map\n",
        encoding="utf-8",
    )

    assert check_numbering(manifest, tmp_path) == []


@pytest.mark.parametrize("marker", ["```", "~~~"])
def test_numbering_keeps_fence_open_for_overindented_closer(tmp_path, marker):
    manifest = _write_valid_notebook_infrastructure_fixture(tmp_path)
    (tmp_path / "docs/index.md").write_text(
        "# 1 Overview\n\n"
        f"{marker}\n"
        f"    {marker}\n"
        '<h1 align="center">Second title still inside fence</h1>\n'
        f"{marker}\n\n"
        "## 1.1 Repository map\n",
        encoding="utf-8",
    )

    assert check_numbering(manifest, tmp_path) == []


def test_placeholders_flag_tbd(tmp_path):
    (tmp_path / "site").mkdir()
    (tmp_path / "site/p.md").write_text("## TODO fill this in\nTBD\n", encoding="utf-8")
    findings = check_placeholders(tmp_path)
    assert findings and findings[0].severity == "error"


def _write_valid_notebook_infrastructure_fixture(tmp_path):
    (tmp_path / "docs/notebooks").mkdir(parents=True)
    (tmp_path / "docs/index.md").write_text("# 1. Overview\n", encoding="utf-8")
    (tmp_path / "docs/notebooks/t.md").write_text("# 8.1 Task\n", encoding="utf-8")
    (tmp_path / "notebooks/t/docs").mkdir(parents=True)
    (tmp_path / "notebooks/t/docs/spec.yaml").write_text(
        "title: Task\ntier: A\natlas:\n  executor: jupyterhub\n  default_mode: vscode-remote\n"
        "  required_services: [jupyterhub]\n  workspace_access: remote\n"
        "  artifact_policy: atlas-jupyter-volume\n  constraints: []\n",
        encoding="utf-8",
    )
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/verify_repo_config.yaml").write_text(
        "active_task_dirs: [t-iris-mlp-pytorch]\n", encoding="utf-8"
    )
    return parse_manifest(MANIFEST_YAML)


def test_notebook_infrastructure_flags_only_a_drifted_canonical_table(tmp_path):
    manifest = _write_valid_notebook_infrastructure_fixture(tmp_path)
    (tmp_path / "docs/notebook-infrastructure.md").write_text(
        "<!-- atlas-task-contracts:start -->\n| stale |\n<!-- atlas-task-contracts:end -->\n",
        encoding="utf-8",
    )

    findings = check_notebook_infrastructure(manifest, tmp_path)

    assert len(findings) == 1
    assert findings[0].severity == "error" and "drifted" in findings[0].message


def test_contract_failure_stops_before_generated_surface_build(tmp_path, monkeypatch):
    from scripts.docs import build_docs, check_docs

    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/manifest.yaml").write_text(MANIFEST_YAML, encoding="utf-8")
    (tmp_path / "docs/index.md").write_text("# 1. Overview\n", encoding="utf-8")
    (tmp_path / "docs/notebooks").mkdir()
    (tmp_path / "docs/notebooks/t.md").write_text("# 8.1 Task\n", encoding="utf-8")
    (tmp_path / "notebooks/t/docs").mkdir(parents=True)
    (tmp_path / "notebooks/t/docs/spec.yaml").write_text("title: Task\n", encoding="utf-8")
    monkeypatch.setattr(check_docs, "check_notebook_infrastructure", lambda *_: [check_docs.Finding("error", "bad contract")])
    monkeypatch.setattr(build_docs, "build", lambda *args, **kwargs: pytest.fail("generated build ran"))

    assert check_docs.check(tmp_path, tmp_path / "generated") == 1
