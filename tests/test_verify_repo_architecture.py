"""Architecture contracts for the repository verifier decomposition."""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "verify_repo.py"
REPO = SCRIPT.parent.parent


def load_facade():
    name = "verify_repo_architecture_contract"
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_verifier_config_is_an_immutable_value_snapshot():
    from scripts.repo_verifier.models import VerifierConfig

    config = VerifierConfig(
        active_task_dirs=("task",),
        required_sections={"notebooks/task/main.ipynb": ("1. Overview",)},
        tier_a_notebooks=("notebooks/task/main.ipynb",),
    )

    assert config.active_task_dirs == ("task",)
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.active_task_dirs = ()
    with pytest.raises(TypeError):
        config.required_sections["new"] = ()


def test_structure_facade_passes_current_config(monkeypatch, tmp_path):
    facade = load_facade()
    monkeypatch.setattr(facade, "ACTIVE_TASK_DIRS", ("changed",))
    seen = {}
    monkeypatch.setattr(
        facade._structure_validator,
        "check_structure",
        lambda repo, config: seen.update(repo=repo, config=config) or facade.CheckResult("structure"),
    )

    facade.check_structure(tmp_path)

    assert seen["repo"] == tmp_path
    assert seen["config"].active_task_dirs == ("changed",)


def test_assets_facade_delegates_to_asset_validator(monkeypatch, tmp_path):
    facade = load_facade()
    seen = {}
    monkeypatch.setattr(
        facade._assets_validator,
        "check_assets",
        lambda repo, config: seen.update(repo=repo, config=config) or facade.CheckResult("assets"),
    )

    facade.check_assets(tmp_path)

    assert seen["repo"] == tmp_path
    assert seen["config"] == facade._config_snapshot()


def test_assets_internal_helper_is_preserved_as_a_facade_alias():
    facade = load_facade()

    assert (
        facade._nlp_asset_contract_findings
        is facade._assets_validator._nlp_asset_contract_findings
    )


def test_docs_facade_injects_current_run_hook(monkeypatch, tmp_path):
    facade = load_facade()
    sentinel = lambda cmd, cwd, timeout=None: (0, "sentinel", "")
    monkeypatch.setattr(facade, "_run", sentinel)
    seen = {}
    monkeypatch.setattr(
        facade._docs_validator,
        "check_docs",
        lambda repo, config, run: seen.update(run=run) or facade.CheckResult("docs"),
    )

    facade.check_docs(tmp_path)

    assert seen["run"] is sentinel


def test_docs_helpers_are_preserved_as_facade_aliases():
    facade = load_facade()

    assert facade._mask_dependency_raw_html is facade._docs_validator._mask_dependency_raw_html
    assert facade._strip_markdown_code is facade._docs_validator._strip_markdown_code
    assert facade._DEPENDENCY_HASH_INPUTS is facade._docs_validator._DEPENDENCY_HASH_INPUTS


def test_docs_validator_owns_dependency_helpers_through_internal_module():
    from scripts.repo_verifier import docs as docs_validator
    from scripts.repo_verifier import docs_dependencies

    assert (
        docs_validator.dependency_ledger_findings
        is docs_dependencies.dependency_ledger_findings
    )
    assert docs_validator._strip_markdown_code is docs_dependencies._strip_markdown_code


def test_comments_facade_delegates_with_current_config(monkeypatch, tmp_path):
    facade = load_facade()
    monkeypatch.setattr(facade, "ACTIVE_TASK_DIRS", ("changed",))
    seen = {}
    monkeypatch.setattr(
        facade._comments_validator,
        "check_comments",
        lambda repo, config: seen.update(repo=repo, config=config)
        or facade.CheckResult("comments"),
    )
    monkeypatch.setattr(
        facade._comments_validator,
        "export_phase_b_candidates",
        lambda repo, out_path, config: seen.update(
            export_repo=repo, out_path=out_path, config=config
        )
        or 7,
    )

    facade.check_comments(tmp_path)

    assert seen["repo"] == tmp_path
    assert seen["config"].active_task_dirs == ("changed",)
    assert facade.export_phase_b_candidates(tmp_path, tmp_path / "out.json") == 7
    assert seen["config"].active_task_dirs == ("changed",)


def test_comments_facade_preserves_comment_helper_aliases():
    facade = load_facade()

    assert facade._STATE_THE_WHAT_PATTERNS is facade._comments_validator._STATE_THE_WHAT_PATTERNS
    assert facade._scan_source_for_comments is facade._comments_validator._scan_source_for_comments


def test_comments_facade_iter_wrapper_uses_current_config(monkeypatch, tmp_path):
    facade = load_facade()
    monkeypatch.setattr(facade, "ACTIVE_TASK_DIRS", ("changed",))
    seen = {}
    monkeypatch.setattr(
        facade._comments_validator,
        "_iter_in_scope_code",
        lambda repo, config: seen.update(repo=repo, config=config) or iter(()),
    )

    assert list(facade._iter_in_scope_code(tmp_path)) == []
    assert seen["repo"] == tmp_path
    assert seen["config"].active_task_dirs == ("changed",)


def test_execution_facade_injects_current_runtime_hooks(monkeypatch, tmp_path):
    facade = load_facade()
    run = lambda cmd, cwd, timeout=None: (0, "", "")
    runtime = lambda: True
    phase3 = lambda repo: []
    monkeypatch.setattr(facade, "_run", run)
    monkeypatch.setattr(facade, "_runtime_available", runtime)
    monkeypatch.setattr(facade, "_phase3_code_cells_unchanged", phase3)
    seen = {}
    monkeypatch.setattr(
        facade._execution_validator,
        "check_execution",
        lambda repo, fast, config, run, runtime_available, phase3_check: (
            seen.update(run=run, runtime=runtime_available, phase3=phase3_check)
            or facade.CheckResult("execution")
        ),
    )

    facade.check_execution(tmp_path, fast=True)

    assert (seen["run"], seen["runtime"], seen["phase3"]) == (run, runtime, phase3)


def test_execution_facade_preserves_execution_helper_aliases():
    facade = load_facade()

    assert facade._cell_tags is facade._execution_validator._cell_tags
    assert facade._assignment_names is facade._execution_validator._assignment_names
    assert facade._atlas_manifest_findings is facade._execution_validator._atlas_manifest_findings


def test_validator_modules_own_the_five_check_implementations():
    facade = load_facade()

    assert facade._structure_validator.check_structure.__module__ == "scripts.repo_verifier.structure"
    assert facade._assets_validator.check_assets.__module__ == "scripts.repo_verifier.assets"
    assert facade._docs_validator.check_docs.__module__ == "scripts.repo_verifier.docs"
    assert facade._comments_validator.check_comments.__module__ == "scripts.repo_verifier.comments"
    assert facade._execution_validator.check_execution.__module__ == "scripts.repo_verifier.execution"


def test_common_module_owns_shared_text_reading():
    from scripts.repo_verifier import comments, common, docs_dependencies, structure

    assert structure._read_text is common.read_text
    assert docs_dependencies._read_text is common.read_text
    assert comments._read_text is common.read_text


def test_verifier_modules_stay_within_decomposition_boundaries():
    facade = REPO / "scripts" / "verify_repo.py"
    modules = sorted((REPO / "scripts" / "repo_verifier").glob("*.py"))

    assert len(facade.read_text(encoding="utf-8").splitlines()) <= 450
    assert max(len(path.read_text(encoding="utf-8").splitlines()) for path in modules) <= 1200
    assert tuple(load_facade().CHECKS) == (
        "structure", "assets", "docs", "comments", "execution"
    )
