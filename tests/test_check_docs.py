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
    readme_summary: str,
    landing_summary: str,
) -> None:
    (repo_root / "docs").mkdir()
    (repo_root / "README.md").write_text(
        "# ml-eng-lab — personal ML lab\n\n"
        "![ml-eng-lab runtime paths](docs/diagrams/img/runtime-flow.png)\n\n"
        f"*{readme_tagline}*\n\n"
        "<!-- project-summary:start -->\n"
        f"{readme_summary}\n"
        "<!-- project-summary:end -->\n",
        encoding="utf-8",
    )
    (repo_root / "docs/index.md").write_text(
        "# 1 ml-eng-lab — personal ML lab\n\n"
        "![ml-eng-lab runtime paths](diagrams/img/runtime-flow.png)\n\n"
        f"*{landing_tagline}*\n\n"
        "<!-- project-summary:start -->\n"
        f"{landing_summary}\n"
        "<!-- project-summary:end -->\n",
        encoding="utf-8",
    )


def test_project_opening_rejects_tagline_and_summary_drift(tmp_path):
    summary = " ".join(["grounded"] * 100)
    _write_project_opening(
        tmp_path,
        landing_tagline="A different tagline.",
        readme_summary=summary,
        landing_summary=f"{summary} changed",
    )

    messages = [finding.message for finding in check_project_opening(tmp_path)]

    assert any("tagline" in message for message in messages)
    assert any("summary" in message and "differ" in message for message in messages)


def test_project_opening_rejects_missing_poster_and_summary_outside_word_range(tmp_path):
    short_summary = " ".join(["short"] * 20)
    _write_project_opening(
        tmp_path,
        readme_summary=short_summary,
        landing_summary=short_summary,
    )
    readme = tmp_path / "README.md"
    readme.write_text(
        readme.read_text(encoding="utf-8").replace(
            "![ml-eng-lab runtime paths](docs/diagrams/img/runtime-flow.png)\n\n", ""
        ),
        encoding="utf-8",
    )

    messages = [finding.message for finding in check_project_opening(tmp_path)]

    assert any("poster" in message for message in messages)
    assert any("100-150 words" in message for message in messages)


def test_project_opening_rejects_missing_project_title(tmp_path):
    summary = " ".join(["grounded"] * 100)
    _write_project_opening(
        tmp_path,
        readme_summary=summary,
        landing_summary=summary,
    )
    landing = tmp_path / "docs/index.md"
    landing.write_text(
        landing.read_text(encoding="utf-8").replace(
            "# 1 ml-eng-lab — personal ML lab", "# 1 Overview"
        ),
        encoding="utf-8",
    )

    assert any("title" in finding.message for finding in check_project_opening(tmp_path))


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
