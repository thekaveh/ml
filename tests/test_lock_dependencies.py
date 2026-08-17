from __future__ import annotations

from pathlib import Path

import pytest

from scripts.dependency_locks import load_policy
from scripts.lock_dependencies import (
    LockGenerationError,
    _validate_compiler_update_transition,
    build_compile_plan,
    compile_locks,
)


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_compile_plan_has_exact_target_and_surface_order() -> None:
    policy = load_policy(REPO_ROOT)

    plan = build_compile_plan(REPO_ROOT, policy, "check")

    assert len(plan) == 24
    expected = [
        (platform.key, surface)
        for platform in policy.platforms
        for surface in (
            "bootstrap",
            "compiler",
            "docs",
            "audit",
            "atlas-contract",
            "core",
            "runtime",
            "root",
        )
    ]
    assert [(item.target, item.surface) for item in plan] == expected
    for item in plan:
        assert item.argv[:3] == ("-m", "uv", "pip")
        assert item.argv[3] == "compile"
        assert "--generate-hashes" in item.argv
        assert "--no-annotate" in item.argv
        assert "--no-header" in item.argv
        assert "--only-binary" in item.argv
        assert ":all:" in item.argv
        assert "--python-version" in item.argv
        assert item.argv.count("--exclude-newer") == 1
        cutoff_index = item.argv.index("--exclude-newer")
        assert item.argv[cutoff_index + 1] == "2026-08-17T02:21:18Z"
        assert item.env["UV_NO_CONFIG"] == "1"
        assert item.env["UV_PYTHON_DOWNLOADS"] == "never"
        assert item.env["PIP_CONFIG_FILE"] == "/dev/null"


def test_compile_plan_pins_cutoff_for_every_target_and_surface() -> None:
    policy = load_policy(REPO_ROOT)
    plan = build_compile_plan(REPO_ROOT, policy, "check")

    assert policy.exclude_newer == "2026-08-17T02:21:18Z"
    assert len(plan) == 24
    assert {
        (item.target, item.surface, item.argv[item.argv.index("--exclude-newer") + 1])
        for item in plan
    } == {
        (platform.key, surface, policy.exclude_newer)
        for platform in policy.platforms
        for surface in (
            "bootstrap",
            "compiler",
            "docs",
            "audit",
            "atlas-contract",
            "core",
            "runtime",
            "root",
        )
    }


def test_darwin_compile_commands_pin_deployment_target() -> None:
    plan = build_compile_plan(REPO_ROOT, load_policy(REPO_ROOT), "check")
    darwin = [item for item in plan if item.target == "darwin-arm64"]

    assert darwin
    assert all(item.env["MACOSX_DEPLOYMENT_TARGET"] == "13.0" for item in darwin)
    assert all("MACOSX_DEPLOYMENT_TARGET" not in item.env for item in plan if item not in darwin)


def test_dedicated_role_compiles_include_bootstrap_as_one_resolution() -> None:
    plan = build_compile_plan(REPO_ROOT, load_policy(REPO_ROOT), "check")
    expected_inputs = {
        "docs": ("bootstrap-requirements.txt", "docs-requirements.in"),
        "audit": ("bootstrap-requirements.txt", "vulnerability-audit-requirements.txt"),
        "atlas-contract": ("bootstrap-requirements.txt", "atlas-contract-requirements.txt"),
        "core": ("bootstrap-requirements.txt", "torch-core-requirements.txt"),
        "runtime": (
            "bootstrap-requirements.txt",
            "torch-core-requirements.txt",
            "torch-requirements.txt",
        ),
    }

    for command in plan:
        if command.surface not in expected_inputs:
            continue
        compile_index = command.argv.index("compile")
        output_index = command.argv.index("--output-file")
        assert command.argv[compile_index + 1 : output_index] == expected_inputs[command.surface]


def test_check_mode_never_writes_checkout_when_runner_fails() -> None:
    before = {path: path.read_bytes() for path in REPO_ROOT.glob("requirements/locks/**/*.txt")}

    def runner(command: object) -> object:
        raise OSError("/private/tmp/secret")

    with pytest.raises(LockGenerationError, match="compile") as caught:
        compile_locks(REPO_ROOT, "check", runner=runner)

    after = {path: path.read_bytes() for path in REPO_ROOT.glob("requirements/locks/**/*.txt")}
    assert after == before
    assert "secret" not in str(caught.value)


def test_generator_rejects_noncanonical_uv_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.lock_dependencies as module

    monkeypatch.setattr(module, "_installed_uv_version", lambda: "0.11.18")
    called = False

    def runner(command: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("runner must not execute")

    with pytest.raises(LockGenerationError, match="compiler"):
        compile_locks(REPO_ROOT, "check", runner=runner)

    assert called is False


def test_compiler_update_transition_requires_all_five_exact_states() -> None:
    common = {
        "old": "0.11.18",
        "manifest": "0.11.19",
        "installed": "0.11.19",
        "atlas": "0.11.19",
        "first": {Path("compiler.txt"): b"candidate"},
        "second": {Path("compiler.txt"): b"candidate"},
    }
    _validate_compiler_update_transition(**common)

    for key, wrong in (
        ("old", ""),
        ("installed", "0.11.18"),
        ("atlas", "0.11.18"),
        ("second", {Path("compiler.txt"): b"drift"}),
    ):
        mutated = dict(common)
        mutated[key] = wrong
        with pytest.raises(LockGenerationError, match="compiler-update"):
            _validate_compiler_update_transition(**mutated)


def test_makefile_exposes_lock_generation_install_and_verification_targets() -> None:
    source = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")

    assert "lock-write:\n\t$(PYTHON) -m scripts.lock_dependencies write" in source
    assert "lock-check:\n\t$(PYTHON) -m scripts.lock_dependencies check" in source
    assert "image-lock-check:\n\t$(PYTHON) -m scripts.check_image_locks" in source
    assert "verify-dependency-locks:\n\t$(PYTHON) -m scripts.verify_dependency_locks" in source
    assert "install-bootstrap:\n\t$(PYTHON) -m scripts.install_locked_requirements bootstrap" in source
    assert "install-compiler-lock:\n\t$(PYTHON) -m scripts.install_locked_requirements compiler" in source
    assert "verify: verify-dependency-locks\n" in source
