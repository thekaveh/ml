"""Architecture contracts for the repository verifier decomposition."""
from __future__ import annotations

import dataclasses

import pytest


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
