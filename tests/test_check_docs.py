# tests/test_check_docs.py
from __future__ import annotations

import pytest

from scripts.docs.check_docs import (
    check_notebook_infrastructure,
    check_completeness,
    check_placeholders,
    check_self_containment,
)
from scripts.docs.manifest import parse_manifest

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


def test_completeness_flags_missing_spec(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/index.md").write_text("x", encoding="utf-8")
    m = parse_manifest(MANIFEST_YAML)
    findings = check_completeness(m, tmp_path)
    assert any("spec" in f.message.lower() or "doc" in f.message.lower() for f in findings)


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
