"""Tests for scripts/verify_repo.py — the four-check oracle."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from scripts import verify_repo

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "verify_repo.py"
ACTIVE_FIXTURE_DIR = "notebooks/image_classification-mnist-ffnn-numpy"
TEST_SUBPROCESS_TIMEOUT = 30


def run_verify(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=TEST_SUBPROCESS_TIMEOUT,
    )


def _temp_repo(tmp_path: Path) -> Path:
    (tmp_path / ACTIVE_FIXTURE_DIR).mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
        timeout=TEST_SUBPROCESS_TIMEOUT,
    )
    return tmp_path


def test_docs_adapter_skips_synthetic_fixture_without_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(verify_repo, "REQUIRED_SECTIONS", {})

    result = verify_repo.check_docs(tmp_path)

    assert not [finding for finding in result.findings if finding.id == "D10.notebook_infrastructure"]


def test_docs_adapter_reports_drift_for_a_real_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(verify_repo, "REQUIRED_SECTIONS", {})
    (tmp_path / "docs/notebooks").mkdir(parents=True)
    (tmp_path / "docs/manifest.yaml").write_text(
        "surfaces: [repo, site, wiki]\nnumbering: baked\nsections:\n  - id: overview\n"
        "    number: '1'\n    title: Overview\n    source: docs/index.md\nnotebooks:\n"
        "  - task: task\n    number: '8.1'\n    family: test\n    depth: full\n"
        "    doc: docs/notebooks/task.md\n    spec: notebooks/task/docs/spec.yaml\ndiagrams: []\n",
        encoding="utf-8",
    )
    (tmp_path / "docs/index.md").write_text("# 1. Overview\n", encoding="utf-8")
    (tmp_path / "docs/notebooks/task.md").write_text("# 8.1 Task\n", encoding="utf-8")
    (tmp_path / "docs/notebook-infrastructure.md").write_text(
        "<!-- atlas-task-contracts:start -->\n| stale |\n<!-- atlas-task-contracts:end -->\n",
        encoding="utf-8",
    )
    (tmp_path / "notebooks/task/docs").mkdir(parents=True)
    (tmp_path / "notebooks/task/docs/spec.yaml").write_text(
        "title: Task\ntier: A\natlas:\n  executor: jupyterhub\n  default_mode: vscode-remote\n"
        "  required_services: [jupyterhub]\n  workspace_access: remote\n"
        "  artifact_policy: atlas-jupyter-volume\n  constraints: []\n",
        encoding="utf-8",
    )
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/verify_repo_config.yaml").write_text("active_task_dirs: [task]\n", encoding="utf-8")

    result = verify_repo.check_docs(tmp_path)

    assert [finding.id for finding in result.findings if finding.id == "D10.notebook_infrastructure"] == ["D10.notebook_infrastructure"]


def test_help_lists_all_checks():
    r = run_verify("--help")
    assert r.returncode == 0
    for ch in ("structure", "execution", "docs", "comments", "all"):
        assert ch in r.stdout


def test_help_does_not_require_adjacent_config(tmp_path):
    script_copy = tmp_path / "scripts" / "verify_repo.py"
    script_copy.parent.mkdir()
    script_copy.write_text(SCRIPT.read_text())
    r = subprocess.run(
        [sys.executable, str(script_copy), "--help"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=TEST_SUBPROCESS_TIMEOUT,
    )
    assert r.returncode == 0, r.stderr
    assert "--check" in r.stdout


def test_unknown_check_errors():
    r = run_verify("--check", "garbage")
    assert r.returncode != 0


def test_missing_check_errors_without_phase_b_out():
    r = run_verify()
    assert r.returncode != 0
    assert "--check is required unless --phase-b-out is used" in r.stderr


def test_emits_valid_json_schema(tmp_path):
    out = tmp_path / "findings.json"
    r = run_verify("--check", "structure", "--out", str(out), "--fast")
    assert out.exists(), f"no output file; stderr={r.stderr}"
    data = json.loads(out.read_text())
    assert isinstance(data, dict)
    assert "schema_version" in data
    assert data["schema_version"] == 1
    assert "findings" in data
    assert isinstance(data["findings"], list)
    assert "summary" in data
    assert "checks_run" in data["summary"]
    assert "structure" in data["summary"]["checks_run"]


def test_finding_shape():
    """Every finding must have id, check, severity, location, message."""
    r = run_verify("--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    for f in data.get("findings", []):
        assert {"id", "check", "severity", "location", "message"} <= set(f.keys())
        assert f["severity"] in ("error", "warning")


def test_structure_s1_notebooks_parse():
    r = run_verify("--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    s1 = [f for f in data["findings"] if f["id"].startswith("S1")]
    assert data["summary"]["by_check"]["structure"] == len(s1) + sum(
        1 for f in data["findings"] if not f["id"].startswith("S1") and f["check"] == "structure"
    )


def test_structure_s3_current_markdown_links_resolve():
    r = run_verify("--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    s3 = [f for f in data["findings"] if f["id"].startswith("S3")]
    assert s3 == [], f"S3 found broken current-repository links: {s3}"


def test_structure_s1_flags_missing_notebook_cell_id(tmp_path):
    """nbformat currently auto-fills missing cell ids, so check raw JSON too."""
    repo = _temp_repo(tmp_path)
    name = "missing-cell-id.ipynb"
    fake = repo / ACTIVE_FIXTURE_DIR / name
    fake.write_text(json.dumps({
        "cells": [{
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "x = 1\n",
        }],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }))
    r = run_verify("--repo-root", str(repo), "--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [
        f for f in data["findings"]
        if f["id"] == "S1.cell_id" and name in f["location"]
    ]
    assert hits, f"expected S1.cell_id for {name}; got {data.get('findings')}"


def test_structure_s1_flags_missing_archive_notebook_cell_id(tmp_path):
    """Archive notebooks must satisfy the same raw nbformat cell-id policy."""
    repo = _temp_repo(tmp_path)
    name = "archive-missing-cell-id.ipynb"
    archive = repo / "notebooks" / "archive" / "old-task" / name
    archive.parent.mkdir(parents=True)
    archive.write_text(json.dumps({
        "cells": [{
            "cell_type": "markdown",
            "metadata": {},
            "source": "# Archived\n",
        }],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }))
    r = run_verify("--repo-root", str(repo), "--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [
        f for f in data["findings"]
        if f["id"] == "S1.cell_id" and name in f["location"]
    ]
    assert hits, f"expected S1.cell_id for archive {name}; got {data.get('findings')}"


def test_structure_s1_flags_invalid_notebook_schema(tmp_path):
    """Raw notebook schema validation should catch id/minor-version mismatches."""
    repo = _temp_repo(tmp_path)
    name = "invalid-schema.ipynb"
    fake = repo / ACTIVE_FIXTURE_DIR / name
    fake.write_text(json.dumps({
        "cells": [{
            "cell_type": "code",
            "execution_count": None,
            "id": "abc123",
            "metadata": {},
            "outputs": [],
            "source": "x = 1\n",
        }],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 2,
    }))

    r = run_verify("--repo-root", str(repo), "--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [
        f for f in data["findings"]
        if f["id"] == "S1.schema" and name in f["location"]
    ]
    assert hits, f"expected S1.schema for invalid notebook schema; got {data.get('findings')}"


def test_repo_root_uses_target_repo_config_for_active_dirs(tmp_path):
    """`--repo-root` should verify notebooks listed by that repo's own config."""
    repo = tmp_path
    task_dir = repo / "notebooks" / "custom-task"
    task_dir.mkdir(parents=True)
    (repo / "scripts").mkdir()
    (repo / "scripts" / "verify_repo_config.yaml").write_text(
        "\n".join([
            "active_task_dirs:",
            "  - custom-task",
            "tier_a_notebooks:",
            "  - notebooks/custom-task/notebook.ipynb",
            "",
        ]),
        encoding="utf-8",
    )
    nb_path = task_dir / "notebook.ipynb"
    nb_path.write_text(json.dumps({
        "cells": [{
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "x = 1\n",
        }],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }))
    subprocess.run(
        ["git", "init"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
        timeout=TEST_SUBPROCESS_TIMEOUT,
    )

    r = run_verify("--repo-root", str(repo), "--check", "structure", "--fast")

    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [
        f for f in data["findings"]
        if f["id"] == "S1.cell_id"
        and f["location"] == "notebooks/custom-task/notebook.ipynb:cell[0]"
    ]
    assert hits, f"expected target repo config to include custom notebook; got {data.get('findings')}"


def test_structure_s5_no_common_imports():
    """No `from common.` import anywhere in active task notebooks or scripts."""
    r = run_verify("--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    s5 = [f for f in data["findings"] if f["id"].startswith("S5")]
    assert s5 == [], f"S5 found stray common.* imports: {s5}"


def test_structure_s5_flags_common_alias_inside_python_multi_import(tmp_path):
    """A valid first import must not hide a forbidden common import alias."""
    repo = _temp_repo(tmp_path)
    module = repo / "stray_common_multi_import.py"
    module.write_text("import os, common.utils\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "-f", str(module)],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
        timeout=TEST_SUBPROCESS_TIMEOUT,
    )

    r = run_verify("--repo-root", str(repo), "--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [
        f for f in data["findings"]
        if f["id"] == "S5.common_import"
        and f["location"] == "stray_common_multi_import.py:1"
    ]
    assert hits, f"expected S5.common_import for multi-import alias; got {data.get('findings')}"


def test_structure_s5_flags_common_alias_inside_notebook_multi_import(tmp_path):
    """Notebook cells should get the same multi-import common scan as scripts."""
    import nbformat

    repo = _temp_repo(tmp_path)
    name = "multi-import-common.ipynb"
    fake = repo / ACTIVE_FIXTURE_DIR / name
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_code_cell("import os, common.utils\n")
    ]
    nbformat.write(nb, str(fake))
    subprocess.run(
        ["git", "add", "-f", str(fake)],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
        timeout=TEST_SUBPROCESS_TIMEOUT,
    )

    r = run_verify("--repo-root", str(repo), "--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [
        f for f in data["findings"]
        if f["id"] == "S5.common_import"
        and f["location"] == f"{ACTIVE_FIXTURE_DIR}/{name}:cell[0]:line[1]"
    ]
    assert hits, f"expected S5.common_import for notebook multi-import alias; got {data.get('findings')}"


def test_structure_s2_checks_every_module_in_multi_import(tmp_path):
    """A valid first import must not hide a missing second import on the same line."""
    import nbformat

    repo = _temp_repo(tmp_path)
    name = "multi-import-missing.ipynb"
    fake = repo / ACTIVE_FIXTURE_DIR / name
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_code_cell("import json, definitely_missing_module_for_s2\n")
    ]
    nbformat.write(nb, str(fake))

    r = run_verify("--repo-root", str(repo), "--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [
        f for f in data["findings"]
        if f["id"] == "S2.unresolved_import"
        and name in f["location"]
        and "definitely_missing_module_for_s2" in f["message"]
    ]
    assert hits, f"expected S2.unresolved_import for second import; got {data.get('findings')}"


def test_structure_s2_reports_one_based_line_numbers(tmp_path):
    """S2 locations should use the same one-based line convention as other findings."""
    import nbformat

    repo = _temp_repo(tmp_path)
    name = "one-based-import-line.ipynb"
    fake = repo / ACTIVE_FIXTURE_DIR / name
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_code_cell(
            "x = 1\nimport definitely_missing_module_for_line_number\n"
        )
    ]
    nbformat.write(nb, str(fake))

    r = run_verify("--repo-root", str(repo), "--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [
        f for f in data["findings"]
        if f["id"] == "S2.unresolved_import"
        and name in f["location"]
        and "definitely_missing_module_for_line_number" in f["message"]
    ]
    assert hits, f"expected S2.unresolved_import; got {data.get('findings')}"
    assert hits[0]["location"].endswith(":cell[0]:line[2]")


def test_structure_s2_checks_multi_import_after_notebook_magic(tmp_path):
    """Notebook magics must not push S2 back to a first-module-only regex fallback."""
    import nbformat

    repo = _temp_repo(tmp_path)
    name = "magic-multi-import-missing.ipynb"
    fake = repo / ACTIVE_FIXTURE_DIR / name
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_code_cell(
            "%matplotlib inline\nimport json, definitely_missing_module_after_magic\n"
        )
    ]
    nbformat.write(nb, str(fake))

    r = run_verify("--repo-root", str(repo), "--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [
        f for f in data["findings"]
        if f["id"] == "S2.unresolved_import"
        and name in f["location"]
        and "definitely_missing_module_after_magic" in f["message"]
    ]
    assert hits, f"expected S2.unresolved_import after notebook magic; got {data.get('findings')}"


def test_structure_s2_ignores_acknowledged_runtime_only_imports(tmp_path):
    """Tier-C runtime-container modules should not create recurring local warnings."""
    import nbformat

    repo = _temp_repo(tmp_path)
    name = "runtime-only-import.ipynb"
    fake = repo / ACTIVE_FIXTURE_DIR / name
    nb = nbformat.v4.new_notebook()
    nb.cells = [nbformat.v4.new_code_cell("import torch_sparse\n")]
    nbformat.write(nb, str(fake))

    r = run_verify("--repo-root", str(repo), "--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [
        f for f in data["findings"]
        if f["id"] == "S2.unresolved_import"
        and name in f["location"]
        and "torch_sparse" in f["message"]
    ]
    assert not hits, f"runtime-only import should be acknowledged, got {hits}"


def test_structure_s2_checks_literal_dynamic_imports(tmp_path):
    """Literal importlib/__import__ calls should not bypass unresolved-import checks."""
    import nbformat

    repo = _temp_repo(tmp_path)
    name = "literal-dynamic-imports.ipynb"
    fake = repo / ACTIVE_FIXTURE_DIR / name
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_code_cell(
            "import importlib\n"
            "importlib.import_module('definitely_missing_dynamic_import')\n"
            "__import__('also_missing_dynamic_import')\n"
        )
    ]
    nbformat.write(nb, str(fake))

    r = run_verify("--repo-root", str(repo), "--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    messages = [
        f["message"] for f in data["findings"]
        if f["id"] == "S2.unresolved_import" and name in f["location"]
    ]
    assert any("definitely_missing_dynamic_import" in m for m in messages), data.get("findings")
    assert any("also_missing_dynamic_import" in m for m in messages), data.get("findings")


def test_structure_s2_checks_literal_dynamic_import_aliases(tmp_path):
    """Literal dynamic imports should be checked through common importlib aliases."""
    import nbformat

    repo = _temp_repo(tmp_path)
    name = "literal-dynamic-import-aliases.ipynb"
    fake = repo / ACTIVE_FIXTURE_DIR / name
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_code_cell(
            "from importlib import import_module\n"
            "import importlib as il\n"
            "import_module('definitely_missing_from_import_alias')\n"
            "il.import_module('definitely_missing_module_alias')\n"
        )
    ]
    nbformat.write(nb, str(fake))

    r = run_verify("--repo-root", str(repo), "--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    messages = [
        f["message"] for f in data["findings"]
        if f["id"] == "S2.unresolved_import" and name in f["location"]
    ]
    assert any("definitely_missing_from_import_alias" in m for m in messages), data.get("findings")
    assert any("definitely_missing_module_alias" in m for m in messages), data.get("findings")


def test_structure_s2_checks_missing_dotted_submodules(tmp_path):
    """An importable top-level package must not hide a missing dotted submodule."""
    import nbformat

    repo = _temp_repo(tmp_path)
    name = "missing-dotted-submodules.ipynb"
    fake = repo / ACTIVE_FIXTURE_DIR / name
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_code_cell(
            "import json.definitely_missing_submodule_for_s2\n"
            "from json.definitely_missing_from_submodule import VALUE\n"
            "importlib.import_module('json.definitely_missing_dynamic_submodule')\n"
        )
    ]
    nbformat.write(nb, str(fake))

    r = run_verify("--repo-root", str(repo), "--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    messages = [
        f["message"] for f in data["findings"]
        if f["id"] == "S2.unresolved_import" and name in f["location"]
    ]
    assert any("json.definitely_missing_submodule_for_s2" in m for m in messages), data.get("findings")
    assert any("json.definitely_missing_from_submodule" in m for m in messages), data.get("findings")
    assert any("json.definitely_missing_dynamic_submodule" in m for m in messages), data.get("findings")


def test_structure_s2_fallback_checks_literal_dynamic_import_aliases(tmp_path):
    """Syntax-error fallback should still check importlib alias calls."""
    import nbformat

    repo = _temp_repo(tmp_path)
    name = "literal-dynamic-import-alias-fallback.ipynb"
    fake = repo / ACTIVE_FIXTURE_DIR / name
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_code_cell(
            "import importlib as il\n"
            "il.import_module('definitely_missing_alias_fallback')\n"
            "x = [\n"
        )
    ]
    nbformat.write(nb, str(fake))

    r = run_verify("--repo-root", str(repo), "--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [
        f for f in data["findings"]
        if f["id"] == "S2.unresolved_import"
        and name in f["location"]
        and "definitely_missing_alias_fallback" in f["message"]
    ]
    assert hits, f"expected S2.unresolved_import for fallback dynamic alias; got {data.get('findings')}"


def test_structure_s2_fallback_checks_multiline_literal_dynamic_import_aliases(tmp_path):
    """Syntax-error fallback should still track parenthesized importlib aliases."""
    import nbformat

    repo = _temp_repo(tmp_path)
    name = "literal-dynamic-import-multiline-alias-fallback.ipynb"
    fake = repo / ACTIVE_FIXTURE_DIR / name
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_code_cell(
            "from importlib import (\n"
            "    import_module as im,\n"
            ")\n"
            "im('definitely_missing_multiline_alias_fallback')\n"
            "if True print('force fallback')\n"
        )
    ]
    nbformat.write(nb, str(fake))

    r = run_verify("--repo-root", str(repo), "--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [
        f for f in data["findings"]
        if f["id"] == "S2.unresolved_import"
        and name in f["location"]
        and "definitely_missing_multiline_alias_fallback" in f["message"]
    ]
    assert hits, f"expected S2.unresolved_import for multiline fallback alias; got {data.get('findings')}"


def test_structure_s2_fallback_ignores_parentheses_in_import_comments(tmp_path):
    """Comment text must not break fallback reconstruction of multiline imports."""
    import nbformat

    repo = _temp_repo(tmp_path)
    name = "fallback-comment-parentheses.ipynb"
    fake = repo / ACTIVE_FIXTURE_DIR / name
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_code_cell(
            "from importlib import (  # keep ) in comment\n"
            "    import_module,\n"
            ")\n"
            "not valid python ???\n"
            "import_module('definitely_missing_comment_paren_dynamic')\n"
        )
    ]
    nbformat.write(nb, str(fake))

    r = run_verify("--repo-root", str(repo), "--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [
        f for f in data["findings"]
        if f["id"] == "S2.unresolved_import"
        and name in f["location"]
        and "definitely_missing_comment_paren_dynamic" in f["message"]
    ]
    assert hits, f"expected S2.unresolved_import with comment paren import alias; got {data.get('findings')}"


def test_structure_s2_fallback_checks_multiline_literal_dynamic_import_calls(tmp_path):
    """Syntax-error fallback should still check multiline import_module calls."""
    import nbformat

    repo = _temp_repo(tmp_path)
    name = "literal-dynamic-import-multiline-call-fallback.ipynb"
    fake = repo / ACTIVE_FIXTURE_DIR / name
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_code_cell(
            "import importlib\n"
            "importlib.import_module(\n"
            "    'definitely_missing_multiline_call_fallback'\n"
            ")\n"
            "if True print('force fallback')\n"
        )
    ]
    nbformat.write(nb, str(fake))

    r = run_verify("--repo-root", str(repo), "--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [
        f for f in data["findings"]
        if f["id"] == "S2.unresolved_import"
        and name in f["location"]
        and "definitely_missing_multiline_call_fallback" in f["message"]
    ]
    assert hits, f"expected S2.unresolved_import for multiline fallback call; got {data.get('findings')}"


def test_structure_s2_fallback_checks_backslash_continued_multi_imports(tmp_path):
    """Syntax-error fallback should still check every backslash-continued import."""
    import nbformat

    repo = _temp_repo(tmp_path)
    name = "backslash-continued-multi-import-fallback.ipynb"
    fake = repo / ACTIVE_FIXTURE_DIR / name
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_code_cell(
            "import json, \\\n"
            "    definitely_missing_backslash_import\n"
            "if True print('force fallback')\n"
        )
    ]
    nbformat.write(nb, str(fake))

    r = run_verify("--repo-root", str(repo), "--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [
        f for f in data["findings"]
        if f["id"] == "S2.unresolved_import"
        and name in f["location"]
        and "definitely_missing_backslash_import" in f["message"]
    ]
    assert hits, f"expected S2.unresolved_import for backslash import; got {data.get('findings')}"


def test_structure_s2_ignores_non_python_cell_magic_body(tmp_path):
    """Shell cell magics must not make S2 scan shell text as Python imports."""
    import nbformat

    repo = _temp_repo(tmp_path)
    name = "bash-cell-magic-import-text.ipynb"
    fake = repo / ACTIVE_FIXTURE_DIR / name
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_code_cell(
            "%%bash\n"
            "echo import definitely_missing_module_inside_shell_magic\n"
            "import definitely_missing_module_inside_shell_magic\n"
        )
    ]
    nbformat.write(nb, str(fake))

    r = run_verify("--repo-root", str(repo), "--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [
        f for f in data["findings"]
        if name in f["location"]
        and "definitely_missing_module_inside_shell_magic" in f["message"]
    ]
    assert hits == [], f"shell cell magic body should not be scanned as Python; got {hits}"


def test_structure_s2_flags_notebook_relative_imports(tmp_path):
    """Relative imports in notebooks are runtime-broken and should be explicit findings."""
    import nbformat

    repo = _temp_repo(tmp_path)
    name = "relative-import.ipynb"
    fake = repo / ACTIVE_FIXTURE_DIR / name
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_code_cell("from . import definitely_missing_relative_helper\n")
    ]
    nbformat.write(nb, str(fake))

    r = run_verify("--repo-root", str(repo), "--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [
        f for f in data["findings"]
        if f["id"] == "S2.relative_import"
        and name in f["location"]
        and "definitely_missing_relative_helper" in f["message"]
    ]
    assert hits, f"expected S2.relative_import for notebook relative import; got {data.get('findings')}"


def test_structure_s2_flags_dotted_notebook_relative_imports(tmp_path):
    """Sibling helper files must not hide dotted relative imports in notebooks."""
    import nbformat

    repo = _temp_repo(tmp_path)
    name = "dotted-relative-import.ipynb"
    task_dir = repo / ACTIVE_FIXTURE_DIR
    (task_dir / "helpers.py").write_text("VALUE = 1\n", encoding="utf-8")
    fake = task_dir / name
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_code_cell("from .helpers import VALUE\n")
    ]
    nbformat.write(nb, str(fake))

    r = run_verify("--repo-root", str(repo), "--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [
        f for f in data["findings"]
        if f["id"] == "S2.relative_import"
        and name in f["location"]
        and "helpers" in f["message"]
    ]
    assert hits, f"expected S2.relative_import for dotted relative import; got {data.get('findings')}"


def test_structure_s2_dedupe_does_not_hide_relative_imports(tmp_path):
    """A normal sibling import must not suppress a later relative-import finding."""
    import nbformat

    repo = _temp_repo(tmp_path)
    name = "deduped-relative-import.ipynb"
    task_dir = repo / ACTIVE_FIXTURE_DIR
    (task_dir / "helpers.py").write_text("VALUE = 1\n", encoding="utf-8")
    fake = task_dir / name
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_code_cell(
            "import helpers\nfrom .helpers import VALUE\n"
        )
    ]
    nbformat.write(nb, str(fake))

    r = run_verify("--repo-root", str(repo), "--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [
        f for f in data["findings"]
        if f["id"] == "S2.relative_import"
        and name in f["location"]
        and "helpers" in f["message"]
    ]
    assert hits, f"expected relative import despite earlier normal import; got {data.get('findings')}"


def test_structure_s2_fallback_ignores_multiline_string_import_text(tmp_path):
    """Syntax fallback must not scan import-looking text inside multiline strings."""
    import nbformat

    repo = _temp_repo(tmp_path)
    name = "fallback-string-import-text.ipynb"
    fake = repo / ACTIVE_FIXTURE_DIR / name
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_code_cell(
            "note = '''\n"
            "import definitely_missing_inside_multiline_string\n"
            "'''\n"
            "if True print('force fallback')\n"
        )
    ]
    nbformat.write(nb, str(fake))

    r = run_verify("--repo-root", str(repo), "--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [
        f for f in data["findings"]
        if name in f["location"]
        and "definitely_missing_inside_multiline_string" in f["message"]
    ]
    assert hits == [], f"fallback should ignore multiline string import text; got {hits}"


def test_structure_s7_no_pycache_tracked():
    """No __pycache__, .ipynb_checkpoints, .DS_Store should be tracked."""
    r = run_verify("--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    s7 = [f for f in data["findings"] if f["id"].startswith("S7")]
    assert s7 == [], f"S7 found tracked bloat: {s7}"


def test_structure_s6_allows_committed_superpowers_specs_and_plans(tmp_path):
    """Committed Superpowers spec/plan docs are intentional planning records."""
    repo = _temp_repo(tmp_path)
    (repo / ".gitignore").write_text("docs/superpowers/\n", encoding="utf-8")
    spec = repo / "docs" / "superpowers" / "specs" / "design.md"
    plan = repo / "docs" / "superpowers" / "plans" / "plan.md"
    spec.parent.mkdir(parents=True, exist_ok=True)
    plan.parent.mkdir(parents=True, exist_ok=True)
    spec.write_text("# Design\n", encoding="utf-8")
    plan.write_text("# Plan\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "-f", str(spec), str(plan)],
        cwd=repo,
        check=True,
        timeout=TEST_SUBPROCESS_TIMEOUT,
    )

    r = run_verify("--repo-root", str(repo), "--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}

    forbidden = {str(spec.relative_to(repo)), str(plan.relative_to(repo))}
    hits = [
        f for f in data["findings"]
        if f["id"] == "S6.tracked_bloat" and f["location"] in forbidden
    ]
    assert not hits, f"intentional planning docs were flagged as bloat: {hits}"


def test_structure_s6_flags_other_tracked_superpowers_files(tmp_path):
    """Only committed spec/plan records are exempt from docs/superpowers bloat."""
    repo = _temp_repo(tmp_path)
    (repo / ".gitignore").write_text("docs/superpowers/\n", encoding="utf-8")
    scratch = repo / "docs" / "superpowers" / "scratch.md"
    scratch.parent.mkdir(parents=True, exist_ok=True)
    scratch.write_text("# Scratch\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "-f", str(scratch)],
        cwd=repo,
        check=True,
        timeout=TEST_SUBPROCESS_TIMEOUT,
    )

    r = run_verify("--repo-root", str(repo), "--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}

    hits = [
        f for f in data["findings"]
        if f["id"] == "S6.tracked_bloat" and f["location"] == str(scratch.relative_to(repo))
    ]
    assert hits, f"expected non-plan docs/superpowers file to be flagged; got {data.get('findings')}"


def test_structure_s8_script_shebang_executable_parity():
    """Direct CLI scripts should keep shebang and executable bit in sync."""
    r = run_verify("--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    s8 = [f for f in data["findings"] if f["id"].startswith("S8")]
    assert s8 == [], f"S8 found script mode drift: {s8}"


def test_structure_s3_flags_missing_markdown_fragment(tmp_path):
    """Internal Markdown links must validate `#fragment` anchors, not just files."""
    repo = _temp_repo(tmp_path)
    name = "bad_anchor.md"
    fake = repo / ACTIVE_FIXTURE_DIR / name
    fake.write_text("# 1. Existing Heading\n\n[bad](#2-missing-heading)\n")
    r = run_verify("--repo-root", str(repo), "--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [
        f for f in data["findings"]
        if f["id"] == "S3.broken_anchor" and name in f["location"]
    ]
    assert hits, f"expected S3.broken_anchor for {name}; got {data.get('findings')}"


def test_structure_s3_ignores_markdown_link_examples_in_code_spans(tmp_path):
    """Historical examples like ``[§4](#old-heading)`` should not be live links."""
    repo = _temp_repo(tmp_path)
    name = "code_span_anchor.md"
    fake = repo / ACTIVE_FIXTURE_DIR / name
    fake.write_text("# 1. Existing Heading\n\nLiteral example: `[bad](#missing-heading)`.\n")
    r = run_verify("--repo-root", str(repo), "--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [
        f for f in data["findings"]
        if f["id"].startswith("S3.") and name in f["location"]
    ]
    assert not hits, f"code-span Markdown link example was treated as live: {hits}"


def test_structure_s3_ignores_markdown_link_examples_in_fenced_code(tmp_path):
    """Fenced snippets often contain example Markdown links that are not live."""
    repo = _temp_repo(tmp_path)
    name = "fenced_anchor.md"
    fake = repo / ACTIVE_FIXTURE_DIR / name
    fake.write_text(
        "# 1. Existing Heading\n\n"
        "```md\n"
        "[bad](#missing-heading)\n"
        "```\n"
    )
    r = run_verify("--repo-root", str(repo), "--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [
        f for f in data["findings"]
        if f["id"].startswith("S3.") and name in f["location"]
    ]
    assert not hits, f"fenced Markdown link example was treated as live: {hits}"


def test_structure_s3_checks_notebook_markdown_links(tmp_path):
    """Notebook markdown links should be covered by the same S3 hygiene."""
    import nbformat

    repo = _temp_repo(tmp_path)
    name = "bad_link.ipynb"
    fake = repo / ACTIVE_FIXTURE_DIR / name
    nb = nbformat.v4.new_notebook()
    nb.cells = [nbformat.v4.new_markdown_cell("[bad](missing-local-doc.md)\n")]
    nbformat.write(nb, str(fake))
    r = run_verify("--repo-root", str(repo), "--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [
        f for f in data["findings"]
        if f["id"] == "S3.broken_link" and name in f["location"]
    ]
    assert hits, f"expected S3.broken_link for notebook markdown; got {data.get('findings')}"


def test_structure_s3_checks_nested_docs_markdown_links(tmp_path):
    """Nested docs should be covered by the same S3 link hygiene as shallow docs."""
    repo = _temp_repo(tmp_path)
    nested = repo / "docs" / "maintenance" / "history.md"
    nested.parent.mkdir(parents=True, exist_ok=True)
    nested.write_text("# History\n\n[missing](missing-local-doc.md)\n", encoding="utf-8")

    r = run_verify("--repo-root", str(repo), "--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [
        f for f in data["findings"]
        if f["id"] == "S3.broken_link" and "docs/maintenance/history.md" in f["location"]
    ]
    assert hits, f"expected S3.broken_link for nested docs markdown; got {data.get('findings')}"


def test_structure_s3_flags_relative_links_that_escape_repo(tmp_path):
    """Repo docs should not silently validate sibling-directory links."""
    repo = _temp_repo(tmp_path / "repo")
    sibling = tmp_path / "nnx"
    sibling.mkdir()
    changelog = repo / "CHANGELOG.md"
    changelog.write_text("Historical example: (via [`nnx`](../nnx))\n", encoding="utf-8")

    r = run_verify("--repo-root", str(repo), "--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [
        f for f in data["findings"]
        if f["id"] == "S3.repo_escape_link" and f["location"] == "CHANGELOG.md"
    ]
    assert hits, f"expected repo-escaping link to be flagged; got {data.get('findings')}"


def test_structure_s3_ignores_notebook_markdown_code_span_links(tmp_path):
    """Notebook prose can show Markdown link syntax as a literal example."""
    import nbformat

    repo = _temp_repo(tmp_path)
    name = "code_span_link.ipynb"
    fake = repo / ACTIVE_FIXTURE_DIR / name
    nb = nbformat.v4.new_notebook()
    nb.cells = [nbformat.v4.new_markdown_cell("Literal: `[bad](missing-local-doc.md)`\n")]
    nbformat.write(nb, str(fake))
    r = run_verify("--repo-root", str(repo), "--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [
        f for f in data["findings"]
        if f["id"].startswith("S3.") and name in f["location"]
    ]
    assert not hits, f"notebook code-span Markdown link was treated as live: {hits}"


def test_docs_d1_known_notebooks_have_required_sections():
    """All tracked notebooks must have their REQUIRED_SECTIONS H1s present.

    Regression guard: if a future edit deletes / reorders an H1 in a tracked
    notebook listed in REQUIRED_SECTIONS, D1.missing_sections fires here.
    Also catches D1.missing_notebook if a listed file gets renamed without
    updating the config.
    """
    r = run_verify("--check", "docs", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    d1 = [f for f in data["findings"] if f["id"].startswith("D1.")]
    assert d1 == [], f"D1 reported issues: {d1}"


def test_docs_d1_unconfigured_active_notebook_is_error(tmp_path):
    """A new active notebook must not bypass docs/E7 checks by being omitted from YAML."""
    import nbformat

    repo = _temp_repo(tmp_path)
    name = "unconfigured.ipynb"
    fake = repo / ACTIVE_FIXTURE_DIR / name
    nb = nbformat.v4.new_notebook()
    nb.cells = [nbformat.v4.new_markdown_cell("# 1. Overview\n")]
    nbformat.write(nb, str(fake))
    r = run_verify("--repo-root", str(repo), "--check", "docs", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [
        f for f in data["findings"]
        if f["id"] == "D1.unconfigured_notebook" and name in f["location"]
    ]
    assert hits, f"expected D1.unconfigured_notebook for {name}; got {data.get('findings')}"
    assert all(f["severity"] == "error" for f in hits)


def test_docs_d8_terminology_consistency_known_canonicals():
    """The check should mention canonical spellings in its allow-list logic."""
    SCRIPT_TEXT = SCRIPT.read_text()
    for token in ("JupyterHub", "NumPy", "PyTorch"):
        assert token in SCRIPT_TEXT, f"D8 missing canonical {token!r}"


def test_docs_d9_current_numbered_docs_are_consistent():
    """Active numbered docs should use dotted numeric headings consistently."""
    r = run_verify("--check", "docs", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    d9 = [f for f in data["findings"] if f["id"] == "D9.numbered_heading"]
    assert d9 == [], f"D9 reported numbered-heading issues: {d9}"


def test_docs_d9_flags_malformed_numbered_headings(tmp_path):
    """H3 headings need a dotted number plus trailing period, e.g. `3.1.`."""
    repo = _temp_repo(tmp_path)
    readme = repo / ACTIVE_FIXTURE_DIR / "README.md"
    readme.write_text(
        "# Fixture\n\n"
        "## 1. Task summary\n\n"
        "## 2. Why this exists\n\n"
        "## 3. What's in the notebook\n\n"
        "### 3.1 Phase without dotted-number terminator\n\n"
        "## 4. How to run\n\n"
        "## 5. Dependencies\n\n"
        "## 6. Known issues\n",
        encoding="utf-8",
    )
    r = run_verify("--repo-root", str(repo), "--check", "docs", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [
        f for f in data["findings"]
        if f["id"] == "D9.numbered_heading" and "README.md:9" in f["location"]
    ]
    assert hits, f"expected D9.numbered_heading for malformed H3; got {data.get('findings')}"


def test_docs_d9_flags_malformed_published_docs_page_heading(tmp_path):
    """Published MkDocs pages should be included in numbered-heading checks."""
    repo = _temp_repo(tmp_path)
    page = repo / "docs" / "index.md"
    page.parent.mkdir()
    page.write_text(
        "# 1. Overview\n\n"
        "## 1.1. Nested heading depth on an H2\n",
        encoding="utf-8",
    )
    r = run_verify("--repo-root", str(repo), "--check", "docs", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [
        f for f in data["findings"]
        if f["id"] == "D9.numbered_heading" and f["location"] == "docs/index.md:3"
    ]
    assert hits, f"expected D9.numbered_heading for docs/index.md; got {data.get('findings')}"


def test_docs_d9_flags_malformed_published_diagram_provenance_heading(tmp_path):
    """Published diagram provenance docs should be included in numbered-heading checks."""
    repo = _temp_repo(tmp_path)
    page = repo / "docs" / "diagrams" / "README.md"
    page.parent.mkdir(parents=True)
    page.write_text(
        "# Diagram Provenance\n\n"
        "## 1.1. Nested heading depth on an H2\n",
        encoding="utf-8",
    )
    r = run_verify("--repo-root", str(repo), "--check", "docs", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [
        f for f in data["findings"]
        if f["id"] == "D9.numbered_heading" and f["location"] == "docs/diagrams/README.md:3"
    ]
    assert hits, f"expected D9.numbered_heading for docs/diagrams/README.md; got {data.get('findings')}"


def test_docs_d10_dependency_ledger_counts_match_current_doc():
    """Package counts and advisory feed-record counts should reconcile."""
    r = run_verify("--check", "docs", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    d10 = [f for f in data["findings"] if f["id"] == "D10.dependency_ledger_count"]
    assert d10 == [], f"D10 reported dependency-ledger issues: {d10}"


def test_docs_d10_current_atlas_infra_gitlink_matches_ledger():
    r = run_verify("--check", "docs", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [
        f for f in data["findings"]
        if f["id"] == "D10.dependency_ledger_submodule_sha"
    ]
    assert hits == [], f"D10 reported Atlas infra ledger issues: {hits}"


def test_docs_d10_flags_dependency_ledger_count_drift(tmp_path):
    """The dependency ledger should not collapse duplicated advisory feed records."""
    repo = _temp_repo(tmp_path)
    docs = repo / "docs"
    docs.mkdir()
    (docs / "dependency-contracts.md").write_text(
        "# Dependency Contracts\n\n"
        "## 1. Audit Snapshot\n\n"
        "Result: 2 known vulnerabilities across one resolved package:\n\n"
        "| Package | Manifest Constraint | Audited Resolved Version | Finding Count | Current Disposition |\n"
        "| --- | --- | ---: | ---: | --- |\n"
        "| `torch` | `torch==2.4.1` | `2.4.1` | 2 | Accepted temporarily. |\n\n"
        "| Package | Advisory ID | Feed Records | Fix Versions |\n"
        "| --- | --- | ---: | --- |\n"
        "| `torch` | `PYSEC-2025-41` | 1 | `2.6.0` |\n",
        encoding="utf-8",
    )
    r = run_verify("--repo-root", str(repo), "--check", "docs", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [f for f in data["findings"] if f["id"] == "D10.dependency_ledger_count"]
    assert hits, f"expected D10.dependency_ledger_count; got {data.get('findings')}"


def test_docs_d10_flags_dependency_ledger_submodule_sha_drift(tmp_path, monkeypatch):
    """The Atlas ledger SHA should match the superproject infra gitlink."""
    repo = _temp_repo(tmp_path)
    (repo / "infra").mkdir()
    docs = repo / "docs"
    docs.mkdir()
    ledger_sha = "b96a2924b5d30aa30eddb2fa43f9b7a47fc81bcb"
    gitlink_sha = "163134451a19d024e0e1c0df51139fd8c0a2ca52"
    (docs / "dependency-contracts.md").write_text(
        "# Dependency Contracts\n\n"
        "## 7. Atlas Infra Submodule Contract\n\n"
        f"Current Atlas `infra` gitlink SHA: `{ledger_sha}`.\n",
        encoding="utf-8",
    )
    verify_repo = _load_verify_module()

    def fake_run(cmd, cwd, timeout=None):
        if cmd == ["git", "ls-files", "--stage", "--", "infra"]:
            return 0, f"160000 {gitlink_sha} 0\tinfra\n", ""
        return 0, "", ""

    monkeypatch.setattr(verify_repo, "_run", fake_run)

    findings = verify_repo._dependency_ledger_findings(repo)

    hits = [f for f in findings if f.id == "D10.dependency_ledger_submodule_sha"]
    assert hits
    assert hits[0].detail == {"ledger_sha": ledger_sha, "gitlink_sha": gitlink_sha}


def test_docs_d10_flags_missing_dependency_ledger_submodule_sha(tmp_path, monkeypatch):
    """The Atlas ledger must keep a parseable pinned tree-entry SHA."""
    repo = _temp_repo(tmp_path)
    (repo / "infra").mkdir()
    docs = repo / "docs"
    docs.mkdir()
    (docs / "dependency-contracts.md").write_text(
        "# Dependency Contracts\n\n"
        "## 7. Atlas Infra Submodule Contract\n\n"
        "Current Atlas `infra` gitlink SHA: `not-a-sha`.\n",
        encoding="utf-8",
    )
    verify_repo = _load_verify_module()

    def fake_run(cmd, cwd, timeout=None):
        if cmd == ["git", "ls-files", "--stage", "--", "infra"]:
            return 0, "160000 ba21661e8a63b3727b9c4a14eaf5e61262d4b48e 0\tinfra\n", ""
        return 0, "", ""

    monkeypatch.setattr(verify_repo, "_run", fake_run)

    findings = verify_repo._dependency_ledger_findings(repo)

    hits = [f for f in findings if f.id == "D10.dependency_ledger_submodule_sha"]
    assert hits
    assert "parseable" in hits[0].message


def test_docs_d10_flags_missing_dependency_ledger_gitlink(tmp_path, monkeypatch):
    """The Atlas ledger SHA must be checked against a parseable gitlink."""
    repo = _temp_repo(tmp_path)
    (repo / "infra").mkdir()
    docs = repo / "docs"
    docs.mkdir()
    ledger_sha = "ba21661e8a63b3727b9c4a14eaf5e61262d4b48e"
    (docs / "dependency-contracts.md").write_text(
        "# Dependency Contracts\n\n"
        "## 7. Atlas Infra Submodule Contract\n\n"
        f"Current Atlas `infra` gitlink SHA: `{ledger_sha}`.\n",
        encoding="utf-8",
    )
    verify_repo = _load_verify_module()

    def fake_run(cmd, cwd, timeout=None):
        if cmd == ["git", "ls-files", "--stage", "--", "infra"]:
            return 0, "", ""
        return 0, "", ""

    monkeypatch.setattr(verify_repo, "_run", fake_run)

    findings = verify_repo._dependency_ledger_findings(repo)

    hits = [f for f in findings if f.id == "D10.dependency_ledger_submodule_sha"]
    assert hits
    assert "gitlink" in hits[0].message
    assert hits[0].detail == {"ledger_sha": ledger_sha, "gitlink_sha": None}


def test_docs_d10_requires_atlas_ledger_entry_when_infra_exists(tmp_path):
    repo = _temp_repo(tmp_path)
    (repo / "infra").mkdir()
    docs = repo / "docs"
    docs.mkdir()
    (docs / "dependency-contracts.md").write_text(
        "# Dependency Contracts\n\n"
        "## 8. Other dependency record\n\n"
        "The unrelated tree entry is `10f840252404eb5399550f96fbb560153f1a47c7`.\n",
        encoding="utf-8",
    )

    findings = _load_verify_module()._dependency_ledger_findings(repo)

    hits = [f for f in findings if f.id == "D10.dependency_ledger_submodule_sha"]
    assert len(hits) == 1
    assert "Atlas Infra Submodule Contract" in hits[0].message


def test_docs_d10_does_not_use_legacy_sha_for_malformed_atlas_entry(tmp_path, monkeypatch):
    repo = _temp_repo(tmp_path)
    (repo / "infra").mkdir()
    docs = repo / "docs"
    docs.mkdir()
    legacy_sha = "61c7c5103660e2226bf107c115dae42bf46f8374"
    (docs / "dependency-contracts.md").write_text(
        "# Dependency Contracts\n\n"
        "## 7. Atlas Infra Submodule Contract\n\n"
        "Current Atlas `infra` gitlink SHA: `not-a-sha`.\n\n"
        "## 8. Other dependency record\n\n"
        f"The repository currently pins tree entry `{legacy_sha}`.\n",
        encoding="utf-8",
    )
    module = _load_verify_module()

    def fail_if_gitlink_checked(cmd, cwd, timeout=None):
        assert cmd != ["git", "ls-files", "--stage", "--", "infra"]
        return 0, "", ""

    monkeypatch.setattr(module, "_run", fail_if_gitlink_checked)

    findings = module._dependency_ledger_findings(repo)

    hits = [f for f in findings if f.id == "D10.dependency_ledger_submodule_sha"]
    assert len(hits) == 1
    assert "parseable" in hits[0].message


def test_docs_d10_reads_atlas_sha_without_capturing_legacy_rollback_sha(tmp_path, monkeypatch):
    repo = _temp_repo(tmp_path)
    (repo / "infra").mkdir()
    docs = repo / "docs"
    docs.mkdir()
    atlas_sha = "61c7c5103660e2226bf107c115dae42bf46f8374"
    legacy_sha = "10f840252404eb5399550f96fbb560153f1a47c7"
    (docs / "dependency-contracts.md").write_text(
        "# Dependency Contracts\n\n"
        "## 7. Atlas Infra Submodule Contract\n\n"
        f"Current Atlas `infra` gitlink SHA: `{atlas_sha}`.\n\n"
        "## 8. Other dependency record\n\n"
        f"The repository currently pins tree entry `{legacy_sha}`.\n",
        encoding="utf-8",
    )
    module = _load_verify_module()

    def fake_run(cmd, cwd, timeout=None):
        if cmd == ["git", "ls-files", "--stage", "--", "infra"]:
            return 0, f"160000 {atlas_sha} 0\tinfra\n", ""
        return 0, "", ""

    monkeypatch.setattr(module, "_run", fake_run)

    findings = module._dependency_ledger_findings(repo)

    assert not [f for f in findings if f.id == "D10.dependency_ledger_submodule_sha"]


def test_docs_d10_flags_workflow_action_refs_that_are_not_sha_pinned(tmp_path):
    repo = _temp_repo(tmp_path)
    workflow = repo / ".github" / "workflows" / "ci.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text(
        "jobs:\n"
        "  test:\n"
        "    steps:\n"
        "      - uses: actions/checkout@v7\n",
        encoding="utf-8",
    )
    docs = repo / "docs"
    docs.mkdir()
    (docs / "dependency-contracts.md").write_text(
        "# Dependency Contracts\n\n"
        "## 8. GitHub Actions Pins\n\n"
        "| Action | Reviewed Tag | Pinned SHA |\n"
        "| --- | --- | --- |\n"
        "| `actions/checkout` | `v7` | `9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0` |\n",
        encoding="utf-8",
    )
    r = run_verify("--repo-root", str(repo), "--check", "docs", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [f for f in data["findings"] if f["id"] == "D10.workflow_action_pin"]
    assert hits, f"expected D10.workflow_action_pin; got {data.get('findings')}"
    assert "actions/checkout@v7" in hits[0]["message"]


def test_docs_d10_flags_yaml_workflow_action_refs_that_are_not_sha_pinned(tmp_path):
    repo = _temp_repo(tmp_path)
    workflow = repo / ".github" / "workflows" / "ci.yaml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text(
        "jobs:\n"
        "  test:\n"
        "    steps:\n"
        "      - uses: actions/checkout@v7\n",
        encoding="utf-8",
    )
    docs = repo / "docs"
    docs.mkdir()
    (docs / "dependency-contracts.md").write_text(
        "# Dependency Contracts\n\n"
        "## 8. GitHub Actions Pins\n\n"
        "| Action | Reviewed Tag | Pinned SHA |\n"
        "| --- | --- | --- |\n"
        "| `actions/checkout` | `v7` | `9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0` |\n",
        encoding="utf-8",
    )
    r = run_verify("--repo-root", str(repo), "--check", "docs", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [f for f in data["findings"] if f["id"] == "D10.workflow_action_pin"]
    assert hits, f"expected D10.workflow_action_pin for .yaml workflow; got {data.get('findings')}"
    assert hits[0]["location"] == ".github/workflows/ci.yaml:4"


def test_docs_d10_flags_workflow_action_refs_missing_from_ledger(tmp_path):
    repo = _temp_repo(tmp_path)
    workflow = repo / ".github" / "workflows" / "ci.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text(
        "jobs:\n"
        "  test:\n"
        "    steps:\n"
        "      - uses: actions/cache@aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa # v5\n",
        encoding="utf-8",
    )
    docs = repo / "docs"
    docs.mkdir()
    (docs / "dependency-contracts.md").write_text(
        "# Dependency Contracts\n\n"
        "## 8. GitHub Actions Pins\n\n"
        "| Action | Reviewed Tag | Pinned SHA |\n"
        "| --- | --- | --- |\n",
        encoding="utf-8",
    )
    r = run_verify("--repo-root", str(repo), "--check", "docs", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [f for f in data["findings"] if f["id"] == "D10.workflow_action_pin"]
    assert hits, f"expected D10.workflow_action_pin; got {data.get('findings')}"
    assert "ledger" in hits[0]["message"]


def test_docs_d10_current_workflow_action_pins_match_ledger():
    r = run_verify("--check", "docs", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [f for f in data["findings"] if f["id"] == "D10.workflow_action_pin"]
    assert hits == []


def test_docs_d11_current_layout_guidance_is_not_stale():
    """Contributor-facing docs should point new tasks at notebooks/<task>/."""
    r = run_verify("--check", "docs", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    d11 = [f for f in data["findings"] if f["id"] == "D11.stale_notebook_layout"]
    assert d11 == [], f"D11 reported stale layout guidance: {d11}"


def test_docs_d11_flags_old_flat_layout_guidance(tmp_path):
    """The verifier should catch the pre-migration top-level task convention."""
    repo = _temp_repo(tmp_path)
    (repo / "README.md").write_text(
        "# Fixture\n\n"
        "## 1. Overview\n\n"
        "Each top-level folder is a self-contained task.\n\n"
        "See archive/README.md for preserved work.\n",
        encoding="utf-8",
    )
    (repo / "CONTRIBUTING.md").write_text(
        "# Contributing\n\n"
        "Use https://nbviewer.org/github/thekaveh/ml-eng-lab/blob/main/<folder>/<notebook>.ipynb.\n",
        encoding="utf-8",
    )
    r = run_verify("--repo-root", str(repo), "--check", "docs", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [f for f in data["findings"] if f["id"] == "D11.stale_notebook_layout"]
    assert len(hits) >= 3, f"expected stale-layout findings; got {data.get('findings')}"


def test_comments_phase_a_flags_obvious_state_the_what(tmp_path):
    """Synthetic .py file with a known bad comment should produce a finding.

    The synthetic file lives in an isolated repo root so this test never mutates
    the real checkout.
    """
    repo = _temp_repo(tmp_path)
    name = "state_the_what.py"
    fake = repo / ACTIVE_FIXTURE_DIR / name
    fake.write_text("# import numpy as np\nimport numpy as np\n")
    r = run_verify("--repo-root", str(repo), "--check", "comments", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [
        f for f in data["findings"]
        if f["check"] == "comments" and name in f["location"]
    ]
    assert hits, f"expected at least one state-the-what flag; got summary={data.get('summary')}"


def test_comments_phase_a_skips_explanatory_comments(tmp_path):
    """A WHY-style comment should NOT be flagged."""
    repo = _temp_repo(tmp_path)
    name = "why.py"
    fake = repo / ACTIVE_FIXTURE_DIR / name
    fake.write_text(
        "# Xavier init keeps variance stable across depths; default torch init blows up here.\n"
        "weight = xavier_init(shape)\n"
    )
    r = run_verify("--repo-root", str(repo), "--check", "comments", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [
        f for f in data["findings"]
        if f["check"] == "comments" and name in f["location"]
    ]
    assert not hits, f"WHY-style comment falsely flagged: {hits}"


def test_comments_phase_a_skips_parameters_tagged_cells(tmp_path):
    """C.state_the_what must skip papermill `parameters`-tagged cells.

    Their boilerplate (per scripts/inject_smoke_test_cell.py) carries lines
    like `# Set via: papermill -p SMOKE_TEST 1 in.ipynb out.ipynb` that
    document the papermill invocation contract — not state-the-what hits
    on the next code line. Same self-exclusion principle as the
    verify_repo.py-as-scanner skip.
    """
    import nbformat
    repo = _temp_repo(tmp_path)
    name = "params.ipynb"
    fake = repo / ACTIVE_FIXTURE_DIR / name
    nb = nbformat.v4.new_notebook()
    cell = nbformat.v4.new_code_cell(
        # Comment matches the `^# (initialize|init|set|assign)` rule; without
        # the parameters tag the C check would flag this. The tag must
        # suppress that.
        "# Set via: papermill -p X 1 in.ipynb out.ipynb\nX = 0\n"
    )
    cell.metadata["tags"] = ["parameters"]
    nb.cells = [cell]
    nbformat.write(nb, str(fake))
    r = run_verify("--repo-root", str(repo), "--check", "comments", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [
        f for f in data["findings"]
        if f["check"] == "comments" and name in f["location"]
    ]
    assert not hits, f"parameters-tagged cell falsely flagged: {hits}"


def test_execution_fast_mode_skips_e1_e2_e3():
    """In --fast mode, slow targets (E1-E3) must not be invoked."""
    r = run_verify("--check", "execution", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    assert "execution" in data["summary"]["checks_run"]
    forbidden_ids = ("E1.tier_a_failed", "E2.tier_b_smoke_failed", "E3.tier_c_smoke_failed")
    for f in data.get("findings", []):
        assert f["id"] not in forbidden_ids, f"slow check ran in --fast mode: {f}"


def test_execution_e5_baseline_missing_warns_not_errors():
    """Before pre-cleanup-baseline tag exists, E5 should warn (not error)."""
    r = run_verify("--check", "execution", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    e5 = [f for f in data["findings"] if f["id"] == "E5.no_baseline"]
    if e5:
        for f in e5:
            assert f["severity"] == "warning", f"E5.no_baseline must be warning, got {f}"


def test_runtime_available_requires_pyg_extension_stack(monkeypatch):
    """Full notebook execution needs the PyG binary extension stack, not just torch_geometric."""
    verify_repo = _load_verify_module()
    present = {"torch", "torch_geometric"}

    def fake_find_spec(name):
        return object() if name in present else None

    monkeypatch.setattr(verify_repo.importlib.util, "find_spec", fake_find_spec)

    assert verify_repo._runtime_available() is False


def test_full_execution_uses_temporary_tier_a_outputs(tmp_path, monkeypatch):
    verify_repo = _load_verify_module()
    repo = _temp_repo(tmp_path)
    calls: list[list[str]] = []

    monkeypatch.setattr(verify_repo, "ACTIVE_TASK_DIRS", ())
    monkeypatch.setattr(verify_repo, "REQUIRED_SECTIONS", {})
    monkeypatch.setattr(verify_repo, "TIER_A_NOTEBOOKS", ())
    monkeypatch.setattr(verify_repo, "_phase3_code_cells_unchanged", lambda _repo: [])
    monkeypatch.setattr(verify_repo, "_runtime_available", lambda: True)

    def fake_run(cmd, cwd, timeout=None):
        del cwd, timeout
        calls.append(cmd)
        return 0, "", ""

    monkeypatch.setattr(verify_repo, "_run", fake_run)

    verify_repo.check_execution(repo, fast=False)

    assert ["make", "smoke-tier-a"] in calls
    assert ["make", "check-tier-a-artifacts"] in calls
    assert ["make", "check-tier-a-clean"] in calls
    assert ["make", "run-tier-a"] not in calls


def test_required_sections_loaded_from_yaml_config():
    """The verify_repo_config.yaml should be the source of truth for the
    REQUIRED_SECTIONS table."""
    import importlib

    scripts_dir = str(REPO / "scripts")
    sys_path_snapshot = list(sys.path)
    sys.path.insert(0, scripts_dir)
    try:
        if "verify_repo" in sys.modules:
            importlib.reload(sys.modules["verify_repo"])
        import verify_repo
        assert isinstance(verify_repo.REQUIRED_SECTIONS, dict)
        for d in verify_repo.ACTIVE_TASK_DIRS:
            assert any(k.startswith(f"notebooks/{d}/") for k in verify_repo.REQUIRED_SECTIONS), (
                f"no entries for {d}"
            )
        phase1 = verify_repo.REQUIRED_SECTIONS.get(
            "notebooks/node_classification-reddit-gnn-pyg/phase1-dataset-exploration-notebook.ipynb"
        )
        assert phase1 is not None
        assert "4. Model" not in phase1

        # YAML is the source of truth — compare TIER_A_NOTEBOOKS to what the
        # config file actually declares, not a hardcoded literal.
        import yaml  # PyYAML is a verify_repo runtime dep, so import is safe here
        config_path = REPO / "scripts" / "verify_repo_config.yaml"
        config = yaml.safe_load(config_path.read_text()) or {}
        expected_tier_a = tuple(config.get("tier_a_notebooks", ()))
        assert tuple(verify_repo.TIER_A_NOTEBOOKS) == expected_tier_a
    finally:
        sys.path[:] = sys_path_snapshot


def test_phase_b_export_runs_and_produces_json(tmp_path):
    """--phase-b-out exports candidate comments as JSON; doesn't run full check."""
    out = tmp_path / "candidates.json"
    r = run_verify("--check", "comments", "--phase-b-out", str(out))
    assert r.returncode == 0
    assert out.exists()
    data = json.loads(out.read_text())
    assert "schema_version" in data
    assert "candidate_count" in data
    assert "candidates" in data
    assert isinstance(data["candidates"], list)
    for cand in data["candidates"]:
        assert {"location", "comment", "snippet"} <= set(cand.keys())


def test_phase_b_export_does_not_require_check(tmp_path):
    out = tmp_path / "candidates.json"
    r = run_verify("--phase-b-out", str(out))
    assert r.returncode == 0, r.stderr
    assert out.exists()


def test_e7_papermill_params_tag_check():
    """Notebooks meant to be papermilled with -p should declare a parameters tag."""
    r = run_verify("--check", "execution", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    # E7 is a warning, never an error.
    e7 = [f for f in data["findings"] if f["id"] == "E7.no_papermill_params_tag"]
    for f in e7:
        assert f["severity"] == "warning"


def test_e13_current_active_notebooks_have_no_stale_repo_paths():
    """Active notebook metadata and outputs should not retain pre-rename paths."""
    r = run_verify("--check", "execution", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    e13 = [f for f in data["findings"] if f["id"] == "E13.stale_active_notebook_path"]
    assert e13 == [], f"E13 reported stale active-notebook paths: {e13}"


def test_e13_flags_stale_paths_in_active_notebooks(tmp_path, monkeypatch):
    """The stale-path guard applies to active notebooks, not the archive."""
    verify_repo = _load_verify_module()
    repo = _temp_repo(tmp_path)
    active_dir = repo / "notebooks" / "active-task"
    archive_dir = repo / "notebooks" / "archive" / "old-task"
    active_dir.mkdir(parents=True)
    archive_dir.mkdir(parents=True)
    (active_dir / "notebook.ipynb").write_text(
        '{"outputs":[{"text":["/home/jovyan/work/ml/nnx/src/file.py"]}]}',
        encoding="utf-8",
    )
    (archive_dir / "notebook.ipynb").write_text(
        '{"outputs":[{"text":["/home/jovyan/work/ml/legacy.py"]}]}',
        encoding="utf-8",
    )
    monkeypatch.setattr(verify_repo, "ACTIVE_TASK_DIRS", ("active-task",))

    result = verify_repo.check_execution(repo, fast=True)

    hits = [f for f in result.findings if f.id == "E13.stale_active_notebook_path"]
    assert len(hits) == 1
    assert hits[0].location.startswith("notebooks/active-task/notebook.ipynb")


def test_e13_flags_removed_nnx_source_tree_and_host_python_paths(tmp_path, monkeypatch):
    verify_repo = _load_verify_module()
    repo = _temp_repo(tmp_path)
    active_dir = repo / "notebooks" / "active-task"
    active_dir.mkdir(parents=True)
    (active_dir / "notebook.ipynb").write_text(
        "\n".join([
            '{"outputs":[',
            '  {"text":["/home/jovyan/work/ml-eng-lab/nnx/src/nnx/nn/params/file.py"]},',
            '  {"text":["/Users/alice/.pyenv/versions/3.11/site-packages/pkg/file.py"]}',
            ']}',
        ]),
        encoding="utf-8",
    )
    monkeypatch.setattr(verify_repo, "ACTIVE_TASK_DIRS", ("active-task",))

    result = verify_repo.check_execution(repo, fast=True)

    hits = [f for f in result.findings if f.id == "E13.stale_active_notebook_path"]
    assert [f.message for f in hits] == [
        "stale active-notebook path artifact: removed in-repo nnx source tree",
        "stale active-notebook path artifact: host-local Python environment path",
    ]


def test_e14_flags_tmp_papermill_output_path(tmp_path, monkeypatch):
    verify_repo = _load_verify_module()
    import nbformat

    repo = _temp_repo(tmp_path)
    task = "tmp-papermill-task"
    active_dir = repo / "notebooks" / task
    active_dir.mkdir(parents=True)
    nb_path = active_dir / "notebook.ipynb"

    nb = nbformat.v4.new_notebook()
    nb.metadata["papermill"] = {
        "input_path": "notebook.ipynb",
        "output_path": "/tmp/smoke-output.ipynb",
    }
    cell = nbformat.v4.new_code_cell("# parser-friendly comment\nSMOKE_TEST = 0\n")
    cell.metadata["tags"] = ["parameters"]
    nb.cells = [cell]
    nbformat.write(nb, str(nb_path))

    monkeypatch.setattr(verify_repo, "ACTIVE_TASK_DIRS", (task,))
    monkeypatch.setattr(verify_repo, "REQUIRED_SECTIONS", {str(nb_path.relative_to(repo)): ()})
    monkeypatch.setattr(verify_repo, "TIER_A_NOTEBOOKS", ())

    result = verify_repo.check_execution(repo, fast=True)

    hits = [f for f in result.findings if f.id == "E14.tmp_papermill_output_path"]
    assert hits
    assert hits[0].location == str(nb_path.relative_to(repo))


def test_e14_flags_source_notebook_papermill_metadata(tmp_path, monkeypatch):
    verify_repo = _load_verify_module()
    import nbformat

    repo = _temp_repo(tmp_path)
    task = "source-papermill-task"
    active_dir = repo / "notebooks" / task
    active_dir.mkdir(parents=True)
    nb_path = active_dir / "notebook.ipynb"

    nb = nbformat.v4.new_notebook()
    nb.metadata["papermill"] = {
        "input_path": "notebook.ipynb",
        "output_path": str(nb_path.relative_to(repo)),
    }
    cell = nbformat.v4.new_code_cell("# parser-friendly comment\nSMOKE_TEST = 0\n")
    cell.metadata["tags"] = ["parameters"]
    nb.cells = [cell]
    nbformat.write(nb, str(nb_path))

    monkeypatch.setattr(verify_repo, "ACTIVE_TASK_DIRS", (task,))
    monkeypatch.setattr(verify_repo, "REQUIRED_SECTIONS", {str(nb_path.relative_to(repo)): ()})
    monkeypatch.setattr(verify_repo, "TIER_A_NOTEBOOKS", ())

    result = verify_repo.check_execution(repo, fast=True)

    hits = [f for f in result.findings if f.id == "E14.source_papermill_metadata"]
    assert hits
    assert hits[0].location == str(nb_path.relative_to(repo))


def _write_valid_atlas_verifier_fixture(repo: Path) -> None:
    (repo / "compose").mkdir(exist_ok=True)
    (repo / "infra").mkdir(exist_ok=True)
    scripts = repo / "scripts"
    scripts.mkdir(exist_ok=True)
    (repo / "atlas.consumer.yml").write_text(
        "name: ml-eng-lab\n"
        "project_name: ml-eng-lab\n"
        "profile: dev\n"
        "brand:\n"
        "  name: ML Eng Lab\n"
        "env:\n"
        "  file: ./atlas.env.user\n"
        "  values:\n"
        "    BASE_PORT: auto\n"
        "    JUPYTERHUB_SOURCE: container\n"
        "    LLM_PROVIDER_SOURCE: ollama-localhost\n"
        "compose_overlays:\n"
        "  - ./compose/ml-eng-lab-atlas.yml\n",
        encoding="utf-8",
    )
    (repo / "atlas.env.user.example").write_text(
        "ML_ENG_LAB_REPO_PATH=/absolute/path/to/ml-eng-lab\n",
        encoding="utf-8",
    )
    (repo / "compose/ml-eng-lab-atlas.yml").write_text(
        "services:\n  jupyterhub: {}\n",
        encoding="utf-8",
    )
    for name in ("atlas-up.sh", "atlas-down.sh", "atlas-connect.sh"):
        script = scripts / name
        script.write_text("#!/usr/bin/env bash\ntrue\n", encoding="utf-8")
        script.chmod(0o755)


def _prepare_atlas_execution_check(monkeypatch, module, repo: Path, active_tasks=()):
    monkeypatch.setattr(module, "ACTIVE_TASK_DIRS", tuple(active_tasks))
    monkeypatch.setattr(module, "REQUIRED_SECTIONS", {})
    monkeypatch.setattr(module, "TIER_A_NOTEBOOKS", ())
    monkeypatch.setattr(module, "_phase3_code_cells_unchanged", lambda _repo: [])

    def fake_run(cmd, cwd, timeout=None):
        if cmd == ["git", "submodule", "status", "--", "infra"]:
            return 0, " 61c7c5103660e2226bf107c115dae42bf46f8374 infra\n", ""
        if cmd == ["git", "status", "--porcelain", "--", "."]:
            assert cwd == repo / "infra"
            return 0, "", ""
        if cmd == ["which", "shellcheck"]:
            return 1, "", ""
        return 0, "", ""

    monkeypatch.setattr(module, "_run", fake_run)


@pytest.mark.parametrize(
    ("manifest_text", "message"),
    [
        (None, "missing"),
        ("name: [unterminated\n", "valid YAML"),
    ],
)
def test_e15_flags_missing_or_malformed_atlas_manifest(
    tmp_path, monkeypatch, manifest_text, message
):
    module = _load_verify_module()
    repo = _temp_repo(tmp_path)
    _write_valid_atlas_verifier_fixture(repo)
    if manifest_text is None:
        (repo / "atlas.consumer.yml").unlink()
    else:
        (repo / "atlas.consumer.yml").write_text(manifest_text, encoding="utf-8")
    _prepare_atlas_execution_check(monkeypatch, module, repo)

    result = module.check_execution(repo, fast=True)

    hits = [finding for finding in result.findings if finding.id == "E15.atlas_manifest"]
    assert any(finding.location == "atlas.consumer.yml" and message in finding.message for finding in hits)


def test_e15_flags_illegal_manifest_track(tmp_path, monkeypatch):
    module = _load_verify_module()
    repo = _temp_repo(tmp_path)
    _write_valid_atlas_verifier_fixture(repo)
    manifest = repo / "atlas.consumer.yml"
    manifest.write_text(manifest.read_text(encoding="utf-8") + "track: ml-eng\n", encoding="utf-8")
    _prepare_atlas_execution_check(monkeypatch, module, repo)

    result = module.check_execution(repo, fast=True)

    hits = [finding for finding in result.findings if finding.id == "E15.atlas_manifest"]
    assert any(finding.location == "atlas.consumer.yml" and "track" in finding.message for finding in hits)


@pytest.mark.parametrize(
    "mutation",
    [
        "",
        "    LLM_PROVIDER_SOURCE: auto\n",
        "    LLM_PROVIDER_SOURCE: ollama-container-cpu\n",
        "    LLM_PROVIDER_SOURCE: ollama-container-gpu\n",
        "    COMFYUI_SOURCE: container-cpu\n",
        "    COMFYUI_SOURCE: container-gpu\n",
    ],
)
def test_e15_rejects_non_native_or_containerized_ai_sources(
    tmp_path, monkeypatch, mutation
):
    module = _load_verify_module()
    repo = _temp_repo(tmp_path)
    _write_valid_atlas_verifier_fixture(repo)
    manifest = repo / "atlas.consumer.yml"
    text = manifest.read_text(encoding="utf-8")
    if mutation:
        text = text.replace("    LLM_PROVIDER_SOURCE: ollama-localhost\n", mutation)
    else:
        text = text.replace("    LLM_PROVIDER_SOURCE: ollama-localhost\n", "")
    manifest.write_text(text, encoding="utf-8")
    _prepare_atlas_execution_check(monkeypatch, module, repo)

    result = module.check_execution(repo, fast=True)

    hits = [finding for finding in result.findings if finding.id == "E15.atlas_manifest"]
    assert any(finding.location == "atlas.consumer.yml" for finding in hits)


@pytest.mark.parametrize(
    "missing_path",
    [
        "atlas.env.user.example",
        "compose/ml-eng-lab-atlas.yml",
        "scripts/atlas-up.sh",
        "scripts/atlas-down.sh",
        "scripts/atlas-connect.sh",
    ],
)
def test_e15_flags_missing_atlas_contract_files(tmp_path, monkeypatch, missing_path):
    module = _load_verify_module()
    repo = _temp_repo(tmp_path)
    _write_valid_atlas_verifier_fixture(repo)
    (repo / missing_path).unlink()
    _prepare_atlas_execution_check(monkeypatch, module, repo)

    result = module.check_execution(repo, fast=True)

    hits = [finding for finding in result.findings if finding.id == "E15.atlas_manifest"]
    assert any(finding.location == missing_path and "missing" in finding.message for finding in hits)


def test_e15_flags_non_executable_atlas_lifecycle_script(tmp_path, monkeypatch):
    module = _load_verify_module()
    repo = _temp_repo(tmp_path)
    _write_valid_atlas_verifier_fixture(repo)
    script = repo / "scripts/atlas-connect.sh"
    script.chmod(0o644)
    _prepare_atlas_execution_check(monkeypatch, module, repo)

    result = module.check_execution(repo, fast=True)

    hits = [finding for finding in result.findings if finding.id == "E15.atlas_manifest"]
    assert any(finding.location == "scripts/atlas-connect.sh" and "executable" in finding.message for finding in hits)


def test_e16_uses_shared_parser_for_invalid_active_task_metadata(tmp_path, monkeypatch):
    module = _load_verify_module()
    repo = _temp_repo(tmp_path)
    _write_valid_atlas_verifier_fixture(repo)
    task = "atlas-task"
    (repo / "scripts/verify_repo_config.yaml").write_text(
        f"active_task_dirs: [{task}]\n",
        encoding="utf-8",
    )
    (repo / "docs/notebooks").mkdir(parents=True)
    (repo / "docs/notebooks/atlas-task.md").write_text("# Task\n", encoding="utf-8")
    (repo / "docs/manifest.yaml").write_text(
        "surfaces: [repo, site, wiki]\n"
        "numbering: baked\n"
        "sections: []\n"
        "notebooks:\n"
        f"  - task: {task}\n"
        '    number: "1"\n'
        "    family: test\n"
        "    depth: full\n"
        "    doc: docs/notebooks/atlas-task.md\n"
        "    spec: notebooks/atlas-task/docs/spec.yaml\n"
        "diagrams: []\n",
        encoding="utf-8",
    )
    spec = repo / "notebooks/atlas-task/docs/spec.yaml"
    spec.parent.mkdir(parents=True)
    spec.write_text(
        "title: Atlas task\n"
        "tier: A\n"
        "atlas:\n"
        "  executor: jupyterhub\n"
        "  default_mode: vscode-remote\n"
        "  required_services: [jupyterhub]\n"
        "  workspace_access: local\n"
        "  artifact_policy: atlas-jupyter-volume\n"
        "  constraints: []\n",
        encoding="utf-8",
    )
    _prepare_atlas_execution_check(monkeypatch, module, repo, active_tasks=(task,))

    result = module.check_execution(repo, fast=True)

    hits = [finding for finding in result.findings if finding.id == "E16.atlas_task_metadata"]
    assert len(hits) == 1
    assert hits[0].location == "notebooks/**/docs/spec.yaml"
    assert "workspace_access" in hits[0].message


def test_e17_flags_port_literal_in_integration_code_and_notebook_code_cell(
    tmp_path, monkeypatch
):
    import nbformat

    module = _load_verify_module()
    repo = _temp_repo(tmp_path)
    _write_valid_atlas_verifier_fixture(repo)
    task = "atlas-task"
    (repo / "scripts/atlas-up.sh").write_text(
        'MLFLOW_HOST="127.0.0.1:63040"\n',
        encoding="utf-8",
    )
    notebook_path = repo / f"notebooks/{task}/notebook.ipynb"
    notebook_path.parent.mkdir(parents=True)
    notebook = nbformat.v4.new_notebook()
    notebook.cells = [
        nbformat.v4.new_code_cell('spark = "http://localhost:63030"\n'),
    ]
    nbformat.write(notebook, notebook_path)
    _prepare_atlas_execution_check(monkeypatch, module, repo, active_tasks=(task,))

    result = module.check_execution(repo, fast=True)

    hits = [finding for finding in result.findings if finding.id == "E17.atlas_hardcoded_endpoint"]
    assert {finding.location for finding in hits} == {
        "scripts/atlas-up.sh:1",
        "notebooks/atlas-task/notebook.ipynb:cell[0]:line[1]",
    }
    assert {finding.detail["endpoint"] for finding in hits} == {
        "127.0.0.1:63040",
        "http://localhost:63030",
    }


def test_e17_excludes_docs_tests_history_notebook_prose_and_harmless_examples(
    tmp_path, monkeypatch
):
    import nbformat

    module = _load_verify_module()
    repo = _temp_repo(tmp_path)
    _write_valid_atlas_verifier_fixture(repo)
    task = "atlas-task"
    (repo / "docs").mkdir(exist_ok=True)
    (repo / "docs/example.md").write_text("Try http://localhost:63094\n", encoding="utf-8")
    (repo / "tests").mkdir()
    (repo / "tests/test_example.py").write_text(
        'EXAMPLE = "http://127.0.0.1:63040"\n',
        encoding="utf-8",
    )
    (repo / "scripts/docs").mkdir()
    (repo / "scripts/docs/example.py").write_text(
        '"""Documentation example: http://localhost:63094."""\n',
        encoding="utf-8",
    )
    (repo / "scripts/atlas-integration.py").write_text(
        '"""Prose example: http://127.0.0.1:63040."""\n'
        "# Historical example: http://localhost:63094\n"
        'TEMPLATE = "http://localhost:<port>"\n',
        encoding="utf-8",
    )
    notebook_path = repo / f"notebooks/{task}/notebook.ipynb"
    notebook_path.parent.mkdir(parents=True)
    notebook = nbformat.v4.new_notebook()
    notebook.cells = [
        nbformat.v4.new_markdown_cell("Try http://localhost:63094"),
        nbformat.v4.new_code_cell(
            "# Historical example: http://127.0.0.1:63040\n"
            'template = "http://localhost:<port>"\n'
        ),
    ]
    notebook.cells[1].outputs = [
        nbformat.v4.new_output("stream", name="stdout", text="http://localhost:63094\n")
    ]
    nbformat.write(notebook, notebook_path)
    archive_path = repo / "notebooks/archive/old/notebook.ipynb"
    archive_path.parent.mkdir(parents=True)
    archived = nbformat.v4.new_notebook()
    archived.cells = [nbformat.v4.new_code_cell('url = "http://localhost:63094"\n')]
    nbformat.write(archived, archive_path)
    _prepare_atlas_execution_check(monkeypatch, module, repo, active_tasks=(task,))

    result = module.check_execution(repo, fast=True)

    assert not [
        finding for finding in result.findings if finding.id == "E17.atlas_hardcoded_endpoint"
    ]


def test_e17_ignores_docstring_after_ipython_magic_in_active_notebook(
    tmp_path, monkeypatch
):
    import nbformat

    module = _load_verify_module()
    repo = _temp_repo(tmp_path)
    _write_valid_atlas_verifier_fixture(repo)
    task = "atlas-task"
    notebook_path = repo / f"notebooks/{task}/notebook.ipynb"
    notebook_path.parent.mkdir(parents=True)
    notebook = nbformat.v4.new_notebook()
    notebook.cells = [
        nbformat.v4.new_code_cell(
            "%matplotlib inline\n"
            '"""Prose example: http://localhost:63094."""\n'
        ),
    ]
    nbformat.write(notebook, notebook_path)
    _prepare_atlas_execution_check(monkeypatch, module, repo, active_tasks=(task,))

    result = module.check_execution(repo, fast=True)

    assert not [
        finding for finding in result.findings if finding.id == "E17.atlas_hardcoded_endpoint"
    ]


def test_e17_checks_endpoint_after_same_line_docstring(tmp_path, monkeypatch):
    module = _load_verify_module()
    repo = _temp_repo(tmp_path)
    _write_valid_atlas_verifier_fixture(repo)
    (repo / "scripts/atlas-integration.py").write_text(
        '"""Prose."""; endpoint = "http://127.0.0.1:63040"\n',
        encoding="utf-8",
    )
    _prepare_atlas_execution_check(monkeypatch, module, repo, active_tasks=())

    result = module.check_execution(repo, fast=True)

    hits = [
        finding for finding in result.findings
        if finding.id == "E17.atlas_hardcoded_endpoint"
    ]
    assert [(finding.location, finding.detail["endpoint"]) for finding in hits] == [
        ("scripts/atlas-integration.py:1", "http://127.0.0.1:63040"),
    ]


def _load_workflow(path: Path) -> dict:
    return yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def test_atlas_contract_workflow_has_recursive_checkout_and_narrow_paths():
    workflow = _load_workflow(REPO / ".github/workflows/atlas-contract.yml")
    paths = set(workflow["on"]["pull_request"]["paths"])
    assert paths == {
        ".gitmodules",
        "infra",
        "atlas.consumer.yml",
        "atlas.env.user.example",
        "compose/ml-eng-lab-atlas.yml",
        "scripts/atlas-*.sh",
        "scripts/docs/notebook_infrastructure.py",
        "docs/notebook-infrastructure.md",
        "docs/atlas-pin-bump-runbook.md",
        "docs/dependency-contracts.md",
        "notebooks/**/docs/spec.yaml",
        "scripts/verify_repo.py",
        "scripts/verify_repo_config.yaml",
        "tests/test_verify_repo.py",
        "Makefile",
        ".github/workflows/atlas-contract.yml",
        ".github/workflows/ci.yml",
        ".github/workflows/docs.yml",
    }
    steps = workflow["jobs"]["atlas-contract"]["steps"]
    checkout = next(step for step in steps if step.get("name") == "Checkout")
    assert checkout["with"]["persist-credentials"] == "false"
    assert checkout["with"]["submodules"] == "recursive"


def test_ci_runs_atlas_workflow_contract_tests():
    workflow = _load_workflow(REPO / ".github/workflows/ci.yml")
    steps = workflow["jobs"]["verify-repo"]["steps"]
    contract_tests = next(step for step in steps if step.get("name") == "Test Atlas workflow contracts")
    assert contract_tests["run"] == (
        "pytest tests/test_verify_repo.py -q -k "
        "'atlas_contract_workflow or "
        "atlas_docs_preserve_mounted_workspace_and_track_ownership or "
        "ci_covers_gitflow_pr_targets or "
        "ci_tier_a_uses_temporary_outputs_and_preserves_sources or "
        "documentation_workflows_install_cairo_and_gate_pages_inputs or "
        "documentation_direct_dependencies_are_exactly_pinned or "
        "docs_workflow_covers_atlas_metadata_inputs_and_parser_tests or "
        "ci_runs_atlas_workflow_contract_tests'"
    )


def test_ci_covers_gitflow_pr_targets():
    workflow = _load_workflow(REPO / ".github/workflows/ci.yml")

    assert set(workflow["on"]["push"]["branches"]) == {"develop", "main"}
    assert set(workflow["on"]["pull_request"]["branches"]) == {"develop", "main"}


def test_documentation_workflows_install_cairo_and_gate_pages_inputs():
    ci = _load_workflow(REPO / ".github/workflows/ci.yml")
    docs = _load_workflow(REPO / ".github/workflows/docs.yml")
    pages = _load_workflow(REPO / ".github/workflows/pages.yml")

    for steps in (
        ci["jobs"]["docs-build"]["steps"],
        docs["jobs"]["check"]["steps"],
        pages["jobs"]["build"]["steps"],
        pages["jobs"]["wiki"]["steps"],
    ):
        assert any("libcairo2" in step.get("run", "") for step in steps)

    required_paths = {
        "README.md",
        ".gitmodules",
        "infra",
        "atlas.consumer.yml",
        "atlas.env.user.example",
        "compose/**",
        "scripts/atlas-*.sh",
        "scripts/lib/atlas-dotenv.sh",
        "docs-requirements.in",
        ".github/workflows/pages.yml",
    }
    assert required_paths <= set(docs["on"]["pull_request"]["paths"])
    assert any(
        step.get("run") == "make docs-check"
        for step in pages["jobs"]["build"]["steps"]
    )


def test_documentation_direct_dependencies_are_exactly_pinned():
    requirements = {
        line
        for line in (REPO / "docs-requirements.in").read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    }

    assert requirements == {
        "mkdocs-material==9.7.7",
        "pyyaml==6.0.3",
        "cairosvg==2.9.0",
        "ruff==0.9.10",
        "pytest==9.0.3",
    }


def test_ci_tier_a_uses_temporary_outputs_and_preserves_sources():
    verify_repo = _load_verify_module()
    workflow = _load_workflow(REPO / ".github/workflows/ci.yml")
    steps = workflow["jobs"]["tier-a-papermill"]["steps"]

    execute = next(step for step in steps if step.get("name") == "Run Tier-A notebooks (papermill)")
    artifacts = next(
        step for step in steps if step.get("name") == "Check Tier-A temporary notebook outputs"
    )
    clean = next(step for step in steps if step.get("name") == "Check Tier-A source notebooks are unchanged")
    artifact = next(step for step in steps if step.get("name") == "Upload refreshed notebook outputs as artifact")
    artifact_paths = tuple(
        line.strip()
        for line in artifact["with"]["path"].splitlines()
        if line.strip()
    )

    assert execute["run"] == "make smoke-tier-a"
    assert artifacts["run"] == "make check-tier-a-artifacts"
    assert clean["run"] == "make check-tier-a-clean"
    assert artifact["with"]["if-no-files-found"] == "error"
    assert artifact_paths == tuple(
        f"/tmp/ml-tier-a/{notebook}" for notebook in verify_repo.TIER_A_NOTEBOOKS
    )
    assert "TIER_A_OUT ?= /tmp/ml-tier-a" in (REPO / "Makefile").read_text(encoding="utf-8")


def test_atlas_docs_preserve_mounted_workspace_and_track_ownership():
    numpy_spec = yaml.safe_load(
        (REPO / "notebooks/image_classification-mnist-ffnn-numpy/docs/spec.yaml").read_text(
            encoding="utf-8"
        )
    )
    constraint = " ".join(numpy_spec["atlas"]["constraints"])
    numpy_readme = (REPO / "notebooks/image_classification-mnist-ffnn-numpy/README.md").read_text(
        encoding="utf-8"
    )
    jupyterhub = (REPO / "docs/jupyterhub-integration.md").read_text(encoding="utf-8")
    vscode = (REPO / "docs/vscode-remote-access.md").read_text(encoding="utf-8")
    environment = (REPO / "docs/env-setup.md").read_text(encoding="utf-8")
    contributing = (REPO / "CONTRIBUTING.md").read_text(encoding="utf-8")

    mounted_editor = "Browser JupyterLab or VS Code attached to the JupyterHub container"
    assert numpy_spec["atlas"]["default_mode"] == "mounted-workspace"
    assert numpy_spec["atlas"]["workspace_access"] == "mounted-required"
    assert mounted_editor in constraint
    for document in (jupyterhub, vscode, contributing, numpy_readme):
        assert mounted_editor in " ".join(document.split())

    track_owner = "`scripts/atlas-up.sh` supplies `--track ml-eng`"
    assert track_owner in jupyterhub
    assert track_owner in environment
    assert "`atlas:` mapping" in contributing
    assert "`make docs-sync-notebook-infrastructure`" in contributing
    assert "future-service admission" in contributing


def test_atlas_contract_workflow_runs_exact_non_live_preflight_and_dirty_gate():
    workflow = _load_workflow(REPO / ".github/workflows/atlas-contract.yml")
    run_bodies = [
        step["run"]
        for step in workflow["jobs"]["atlas-contract"]["steps"]
        if "run" in step
    ]
    command_body = "\n".join(run_bodies)
    exact_preflight = """cp infra/.env.example infra/.env
printf 'ML_ENG_LAB_REPO_PATH=%s\\n' "$GITHUB_WORKSPACE" > atlas.env.user
(
  cd infra
  ./start.sh env backfill
  ./start.sh --consumer ../atlas.consumer.yml compose validate
  ./start.sh --consumer ../atlas.consumer.yml doctor --format json
)"""
    assert exact_preflight in command_body
    assert "git -C infra status --porcelain --untracked-files=all --ignored=no" in command_body
    assert "exit 1" in command_body
    for forbidden in (
        "make atlas-contract",
        "./scripts/atlas-up.sh",
        "--detach",
        "--track",
        "endpoints ",
        "atlas-connect",
        "docker ",
        "curl ",
        "localhost:",
        "127.0.0.1:",
    ):
        assert forbidden not in command_body


def test_docs_workflow_covers_atlas_metadata_inputs_and_parser_tests():
    workflow = _load_workflow(REPO / ".github/workflows/docs.yml")
    paths = set(workflow["on"]["pull_request"]["paths"])
    assert {
        "docs/manifest.yaml",
        "notebooks/**/docs/spec.yaml",
        "scripts/docs/notebook_infrastructure.py",
        "tests/test_notebook_infrastructure.py",
    } <= paths
    steps = workflow["jobs"]["check"]["steps"]
    unit_tests = next(step for step in steps if step.get("name") == "Unit tests (docs scripts)")
    assert "tests/test_notebook_infrastructure.py" in unit_tests["run"].split()


def test_e6_shellcheck_targets_include_only_parent_owned_scripts():
    verify_repo = _load_verify_module()

    targets = {
        str(path.relative_to(REPO))
        for path in verify_repo._shellcheck_targets(REPO)
    }

    assert "scripts/atlas-up.sh" in targets
    assert "scripts/atlas-down.sh" in targets
    assert "scripts/atlas-connect.sh" in targets
    assert not any(target.startswith(("infra/", "vendor/")) for target in targets)


def test_e6_flags_required_parent_shellcheck_target_without_executable_bit(
    tmp_path, monkeypatch
):
    module = _load_verify_module()
    repo = _temp_repo(tmp_path)
    _write_valid_atlas_verifier_fixture(repo)
    script = repo / "scripts/atlas-down.sh"
    script.chmod(0o644)
    _prepare_atlas_execution_check(monkeypatch, module, repo)

    result = module.check_execution(repo, fast=True)

    hits = [
        finding
        for finding in result.findings
        if finding.id == "E6.shellcheck_target_not_executable"
    ]
    assert [finding.location for finding in hits] == ["scripts/atlas-down.sh"]


def test_e6_flags_missing_required_parent_shellcheck_targets(tmp_path, monkeypatch):
    verify_repo = _load_verify_module()
    repo = _temp_repo(tmp_path)
    scripts = repo / "scripts"
    scripts.mkdir()

    monkeypatch.setattr(verify_repo, "ACTIVE_TASK_DIRS", ())
    monkeypatch.setattr(verify_repo, "TIER_A_NOTEBOOKS", ())
    monkeypatch.setattr(verify_repo, "_phase3_code_cells_unchanged", lambda _repo: [])

    def fake_run(cmd, cwd, timeout=None):
        if cmd == ["which", "shellcheck"]:
            return 0, "/usr/bin/shellcheck\n", ""
        if cmd and cmd[0] == "shellcheck":
            return 0, "", ""
        return 0, "", ""

    monkeypatch.setattr(verify_repo, "_run", fake_run)

    result = verify_repo.check_execution(repo, fast=True)

    hits = [f for f in result.findings if f.id == "E6.shellcheck_target_missing"]
    assert hits
    assert {
        "scripts/atlas-up.sh",
        "scripts/atlas-down.sh",
        "scripts/atlas-connect.sh",
    } == {f.location for f in hits}


def test_e6_flags_missing_required_parent_shellcheck_targets_without_shellcheck(
    tmp_path, monkeypatch
):
    verify_repo = _load_verify_module()
    repo = _temp_repo(tmp_path)
    scripts = repo / "scripts"
    scripts.mkdir()

    monkeypatch.setattr(verify_repo, "ACTIVE_TASK_DIRS", ())
    monkeypatch.setattr(verify_repo, "TIER_A_NOTEBOOKS", ())
    monkeypatch.setattr(verify_repo, "_phase3_code_cells_unchanged", lambda _repo: [])

    def fake_run(cmd, cwd, timeout=None):
        if cmd == ["which", "shellcheck"]:
            return 1, "", ""
        assert not (cmd and cmd[0] == "shellcheck")
        return 0, "", ""

    monkeypatch.setattr(verify_repo, "_run", fake_run)

    result = verify_repo.check_execution(repo, fast=True)

    missing_binary = [f for f in result.findings if f.id == "E6.shellcheck_missing"]
    assert len(missing_binary) == 1

    missing_targets = [
        f for f in result.findings if f.id == "E6.shellcheck_target_missing"
    ]
    assert {
        "scripts/atlas-up.sh",
        "scripts/atlas-down.sh",
        "scripts/atlas-connect.sh",
    } == {f.location for f in missing_targets}


def test_e6_flags_dirty_required_submodule(monkeypatch):
    verify_repo = _load_verify_module()

    def fake_run(cmd, cwd, timeout=None):
        if cmd == ["git", "submodule", "status", "--", "infra"]:
            return 0, "+163134451a19d024e0e1c0df51139fd8c0a2ca52 infra\n", ""
        if cmd == ["which", "shellcheck"]:
            return 1, "", ""
        return 0, "", ""

    monkeypatch.setattr(verify_repo, "_run", fake_run)
    monkeypatch.setattr(verify_repo, "ACTIVE_TASK_DIRS", ())
    monkeypatch.setattr(verify_repo, "TIER_A_NOTEBOOKS", ())
    monkeypatch.setattr(verify_repo, "_phase3_code_cells_unchanged", lambda _repo: [])

    result = verify_repo.check_execution(REPO, fast=True)

    hits = [f for f in result.findings if f.id == "E6.submodule_dirty"]
    assert hits
    assert hits[0].location == "infra"


def test_e6_flags_required_submodule_with_modified_worktree(monkeypatch):
    verify_repo = _load_verify_module()
    submodule_cwd = REPO / "infra"

    def fake_run(cmd, cwd, timeout=None):
        if cmd == ["git", "submodule", "status", "--", "infra"]:
            return 0, " 163134451a19d024e0e1c0df51139fd8c0a2ca52 infra\n", ""
        if cmd == ["git", "status", "--porcelain", "--", "."]:
            assert cwd == submodule_cwd
            return 0, " M services/jupyterhub/build/requirements.txt\n", ""
        if cmd == ["which", "shellcheck"]:
            return 1, "", ""
        return 0, "", ""

    monkeypatch.setattr(verify_repo, "_run", fake_run)
    monkeypatch.setattr(verify_repo, "ACTIVE_TASK_DIRS", ())
    monkeypatch.setattr(verify_repo, "TIER_A_NOTEBOOKS", ())
    monkeypatch.setattr(verify_repo, "_phase3_code_cells_unchanged", lambda _repo: [])

    result = verify_repo.check_execution(REPO, fast=True)

    hits = [f for f in result.findings if f.id == "E6.submodule_dirty"]
    assert hits
    assert hits[0].location == "infra"
    assert "local modifications" in hits[0].message


def _load_verify_module():
    import importlib.util
    if "verify_repo" in sys.modules:
        return sys.modules["verify_repo"]
    spec = importlib.util.spec_from_file_location("verify_repo", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    # @dataclass field resolution needs the module findable in sys.modules,
    # otherwise field-class lookup raises AttributeError on a NoneType.
    sys.modules["verify_repo"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_iter_notebooks_reads_active_tasks_under_notebooks(tmp_path, monkeypatch):
    verify_repo = _load_verify_module()
    active = tmp_path / "notebooks" / "task-a"
    archive = tmp_path / "notebooks" / "archive" / "old-task"
    old_root = tmp_path / "task-a"
    active.mkdir(parents=True)
    archive.mkdir(parents=True)
    old_root.mkdir()

    (active / "notebook.ipynb").write_text("{}", encoding="utf-8")
    (archive / "notebook.ipynb").write_text("{}", encoding="utf-8")
    (old_root / "notebook.ipynb").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(verify_repo, "ACTIVE_TASK_DIRS", ("task-a",))

    found = [str(p.relative_to(tmp_path)) for p in verify_repo._iter_notebooks(tmp_path)]

    assert found == ["notebooks/task-a/notebook.ipynb"]


def test_baseline_notebook_rel_removes_notebooks_prefix():
    verify_repo = _load_verify_module()
    baseline_rel = "/".join([
        "node_classification-reddit-gnn-pyg",
        "phase3-main-model-training-and-eval-notebook.ipynb",
    ])

    assert (
        verify_repo._baseline_notebook_rel(
            "notebooks/node_classification-reddit-gnn-pyg/phase3-main-model-training-and-eval-notebook.ipynb"
        )
        == baseline_rel
    )
    assert verify_repo._baseline_notebook_rel("legacy/notebook.ipynb") == "legacy/notebook.ipynb"


def test_assignment_names_ignore_comments_and_strings():
    verify_repo = _load_verify_module()
    names = verify_repo._assignment_names(
        "# COMMENT_ONLY = 1\n"
        "example = 'STRING_ONLY = 1'\n"
        "SMOKE_TEST = 0\n"
        "SMOKE_TEST_EPOCHS: int = 1\n"
        "SMOKE_TEST_SUBSET += 1\n"
        "LEFT, RIGHT = 1, 2\n"
    )

    assert {"SMOKE_TEST", "SMOKE_TEST_EPOCHS", "SMOKE_TEST_SUBSET", "LEFT", "RIGHT"} <= names
    assert "COMMENT_ONLY" not in names
    assert "STRING_ONLY" not in names


def test_e10_flags_parameters_tag_without_smoke_test_assignment(tmp_path, monkeypatch):
    verify_repo = _load_verify_module()
    import nbformat

    rel = Path("task") / "missing-smoke.ipynb"
    nb_path = tmp_path / rel
    nb_path.parent.mkdir()
    nb = nbformat.v4.new_notebook()
    cell = nbformat.v4.new_code_cell("OTHER_PARAMETER = 1\n")
    cell.metadata["tags"] = ["parameters"]
    nb.cells = [cell]
    nbformat.write(nb, str(nb_path))

    monkeypatch.setattr(verify_repo, "REQUIRED_SECTIONS", {str(rel): ("1. Any",)})
    monkeypatch.setattr(verify_repo, "TIER_A_NOTEBOOKS", ())
    result = verify_repo.check_execution(tmp_path, fast=True)

    hits = [f for f in result.findings if f.id == "E10.missing_smoke_test_parameter"]
    assert len(hits) == 1
    assert hits[0].severity == "error"
    assert hits[0].location == str(rel)


def test_e10_smoke_test_parameter_check_clean_current_repo():
    r = run_verify("--check", "execution", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [f for f in data["findings"] if f["id"] == "E10.missing_smoke_test_parameter"]
    assert hits == []


def test_makefile_variable_items_parse_continuation_list(tmp_path):
    verify_repo = _load_verify_module()
    makefile = tmp_path / "Makefile"
    makefile.write_text(
        "OTHER := ignored\n"
        "TIER_A := \\\n"
        "    first/notebook.ipynb \\\n"
        "    second/notebook.ipynb\n"
        "TIER_B := third/notebook.ipynb\n"
    )
    assert verify_repo._makefile_variable_items(tmp_path, "TIER_A") == (
        "first/notebook.ipynb",
        "second/notebook.ipynb",
    )


def test_e11_tier_a_config_matches_makefile():
    verify_repo = _load_verify_module()
    assert verify_repo._makefile_variable_items(REPO, "TIER_A") == tuple(
        verify_repo.TIER_A_NOTEBOOKS
    )
    r = run_verify("--check", "execution", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [f for f in data["findings"] if f["id"].startswith("E11.")]
    assert hits == []


def test_tier_a_excludes_external_reddit2_download_notebook():
    verify_repo = _load_verify_module()
    reddit_phase1 = "notebooks/node_classification-reddit-gnn-pyg/phase1-dataset-exploration-notebook.ipynb"

    assert reddit_phase1 not in verify_repo.TIER_A_NOTEBOOKS
    assert reddit_phase1 not in verify_repo._makefile_variable_items(REPO, "TIER_A")


def test_e11_flags_missing_makefile_tier_a(tmp_path, monkeypatch):
    verify_repo = _load_verify_module()
    monkeypatch.setattr(verify_repo, "TIER_A_NOTEBOOKS", ("task/notebook.ipynb",))
    result = verify_repo.check_execution(tmp_path, fast=True)
    hits = [f for f in result.findings if f.id == "E11.tier_a_makefile_missing"]
    assert len(hits) == 1
    assert hits[0].severity == "error"


def test_ci_tier_a_artifact_paths_parse_workflow(tmp_path):
    verify_repo = _load_verify_module()
    workflow = tmp_path / ".github" / "workflows" / "ci.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text(
        "jobs:\n"
        "  tier-a-papermill:\n"
        "    steps:\n"
        "      - name: Upload refreshed notebook outputs as artifact\n"
        "        with:\n"
        "          path: |\n"
        "            first/notebook.ipynb\n"
        "            second/notebook.ipynb\n"
    )
    assert verify_repo._ci_tier_a_artifact_paths(tmp_path) == (
        "first/notebook.ipynb",
        "second/notebook.ipynb",
    )


def test_e12_tier_a_artifact_paths_match_config():
    verify_repo = _load_verify_module()
    assert verify_repo._ci_tier_a_artifact_paths(REPO) == tuple(
        f"/tmp/ml-tier-a/{notebook}" for notebook in verify_repo.TIER_A_NOTEBOOKS
    )
    r = run_verify("--check", "execution", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    hits = [f for f in data["findings"] if f["id"].startswith("E12.")]
    assert hits == []


def test_run_helper_timeout_returns_rc_124():
    """_run must catch subprocess.TimeoutExpired and surface rc=124 + a
    diagnostic stderr suffix, so a hung make target produces a clean Finding
    instead of an uncaught traceback."""
    verify_repo = _load_verify_module()
    rc, stdout, stderr = verify_repo._run(["sleep", "5"], REPO, timeout=1)
    assert rc == 124, f"expected rc=124 on timeout, got {rc} (stdout={stdout!r}, stderr={stderr!r})"
    assert "timed out after 1s" in stderr


def test_run_helper_timeout_normalizes_byte_streams(monkeypatch):
    """TimeoutExpired can carry byte stdout/stderr even when subprocess.run used text=True."""
    verify_repo = _load_verify_module()

    def raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd=args[0],
            timeout=kwargs.get("timeout"),
            output=b"partial stdout",
            stderr=b"partial stderr",
        )

    monkeypatch.setattr(verify_repo.subprocess, "run", raise_timeout)
    rc, stdout, stderr = verify_repo._run(["fake"], REPO, timeout=1)
    assert rc == 124
    assert stdout == "partial stdout"
    assert "partial stderr" in stderr
    assert "timed out after 1s" in stderr


def test_run_helper_supplies_default_timeout(monkeypatch):
    """Callers should not have to remember a timeout for short external commands."""
    verify_repo = _load_verify_module()
    seen: dict[str, int | None] = {}

    def fake_run(cmd, cwd, capture_output, text, timeout):
        del cwd, capture_output, text
        seen["timeout"] = timeout
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(verify_repo.subprocess, "run", fake_run)
    rc, _, _ = verify_repo._run(["fake"], REPO)

    assert rc == 0
    assert seen["timeout"] == verify_repo.DEFAULT_SUBPROCESS_TIMEOUT


def test_tier_c_baseline_sources_ignore_parameter_cells():
    verify_repo = _load_verify_module()
    import nbformat

    baseline = nbformat.v4.new_notebook()
    baseline.cells = [
        nbformat.v4.new_code_cell("SMOKE_TEST = 0  # old parser-hostile comment\n"),
        nbformat.v4.new_code_cell("model.train()\n"),
    ]
    baseline.cells[0].metadata["tags"] = ["parameters"]

    head = nbformat.v4.new_notebook()
    head.cells = [
        nbformat.v4.new_code_cell("# parser-friendly comment\nSMOKE_TEST = 0\n"),
        nbformat.v4.new_code_cell("model.train()\n"),
    ]
    head.cells[0].metadata["tags"] = ["parameters"]

    assert verify_repo._code_cell_sources_for_baseline(head) == verify_repo._code_cell_sources_for_baseline(baseline)

    head.cells[1].source = "model.train(n_epochs=1)\n"
    assert verify_repo._code_cell_sources_for_baseline(head) != verify_repo._code_cell_sources_for_baseline(baseline)


def test_parameter_trailing_comment_check_flags_papermill_uninspectable_assignment():
    verify_repo = _load_verify_module()
    import nbformat

    nb = nbformat.v4.new_notebook()
    bad = nbformat.v4.new_code_cell("SMOKE_TEST = 0  # 1 = smoke mode\n")
    bad.metadata["tags"] = ["parameters"]
    good = nbformat.v4.new_code_cell("# 1 = smoke mode\nSMOKE_TEST = 0\n")
    good.metadata["tags"] = ["parameters"]

    nb.cells = [bad]
    findings = verify_repo._parameter_trailing_comment_findings(nb, "fake.ipynb")
    assert [f.id for f in findings] == ["E9.parameter_trailing_comment"]

    nb.cells = [good]
    assert verify_repo._parameter_trailing_comment_findings(nb, "fake.ipynb") == []


def test_s7_forbidden_toplevel_detects_resurrected_common(tmp_path):
    """S7.forbidden_toplevel fires if common/ ever comes back."""
    repo = _temp_repo(tmp_path)
    fake_dir = repo / "common"
    fake_dir.mkdir()
    r = run_verify("--repo-root", str(repo), "--check", "structure", "--fast")
    data = json.loads(r.stdout) if r.stdout else {"findings": []}
    s7 = [
        f for f in data["findings"]
        if f["id"] == "S7.forbidden_toplevel" and "common" in f["location"]
    ]
    assert s7, "expected S7.forbidden_toplevel to flag resurrected common/"
    for f in s7:
        assert f["severity"] == "error"
