"""Architecture contracts for the repository verifier decomposition."""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "verify_repo.py"


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
