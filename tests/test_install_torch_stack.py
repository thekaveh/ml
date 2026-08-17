from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.install_torch_stack import (
    InstallCommand,
    InstallStage,
    TorchStackInstallError,
    build_install_commands,
    install_torch_stack,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
PYPI = "https://pypi.org/simple"
TORCH_CPU = "https://download.pytorch.org/whl/cpu"
PYG = "https://data.pyg.org/whl/torch-2.11.0+cpu.html"
SUPPORTED_HOSTS = (
    ("Darwin", "arm64", "darwin-arm64", PYPI),
    ("Linux", "x86_64", "linux-x86_64", TORCH_CPU),
    ("Linux", "aarch64", "linux-aarch64", TORCH_CPU),
)


def _pip_prefix(python: str = sys.executable) -> tuple[str, ...]:
    return (
        python,
        "-m",
        "pip",
        "install",
        "--isolated",
        "--disable-pip-version-check",
        "--no-input",
        "--require-hashes",
        "--no-deps",
    )


def _expected_commands(key: str, core_index: str) -> tuple[tuple[str, ...], ...]:
    pip = _pip_prefix()
    root = f"requirements/locks/{key}"
    return (
        pip + ("--only-binary=:all:", "-r", "requirements/locks/bootstrap.txt"),
        pip + ("--only-binary=:all:", "--index-url", core_index, "-r", f"{root}/core.txt"),
        pip
        + (
            "--only-binary=:all:",
            "--index-url",
            PYPI,
            "--find-links",
            PYG,
            "-r",
            f"{root}/runtime.txt",
        ),
        pip
        + (
            "--only-binary=:all:",
            "--no-binary=python-louvain",
            "--no-build-isolation",
            "--index-url",
            PYPI,
            "--find-links",
            PYG,
            "-r",
            f"{root}/root.txt",
        ),
    )


def test_exact_four_stage_plan_for_every_supported_host() -> None:
    for system, machine, key, core_index in SUPPORTED_HOSTS:
        commands = build_install_commands(sys.executable, system, machine)
        assert tuple(command.stage for command in commands) == tuple(InstallStage)
        assert tuple(command.argv for command in commands) == _expected_commands(key, core_index)


@pytest.mark.parametrize(
    ("system", "machine"),
    (("Linux", "arm64"), ("Darwin", "x86_64"), ("Windows", "AMD64")),
)
def test_unsupported_platforms_fail_closed(system: str, machine: str) -> None:
    with pytest.raises(TorchStackInstallError, match="unsupported"):
        build_install_commands(sys.executable, system, machine)


def test_every_stage_has_exact_hash_binary_and_lock_contract() -> None:
    commands = build_install_commands(sys.executable, "Linux", "aarch64")
    for command in commands:
        assert command.argv.count("--require-hashes") == 1
        assert command.argv.count("--only-binary=:all:") == 1
        assert command.argv.count("-r") == 1
        assert not any("requirements.txt" == argument for argument in command.argv)
    assert "--no-binary=python-louvain" not in commands[2].argv
    assert "--no-binary=python-louvain" in commands[3].argv
    assert "--no-build-isolation" in commands[3].argv


@pytest.mark.parametrize("failed_index", range(4))
def test_nonzero_stage_stops_before_every_later_stage(failed_index: int) -> None:
    commands = build_install_commands(sys.executable, "Darwin", "arm64")
    seen: list[tuple[str, ...]] = []

    def runner(argv: tuple[str, ...], **kwargs: object) -> SimpleNamespace:
        seen.append(tuple(argv))
        assert kwargs["check"] is False
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        assert kwargs["cwd"] == REPO_ROOT
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        assert environment["PIP_CONFIG_FILE"] == os.devnull
        assert "PIP_INDEX_URL" not in environment
        return SimpleNamespace(returncode=int(len(seen) - 1 == failed_index), stdout="secret", stderr="secret")

    with pytest.raises(TorchStackInstallError, match=str(commands[failed_index].stage)) as caught:
        install_torch_stack(commands, repo=REPO_ROOT, runner=runner)

    assert seen == [command.argv for command in commands[: failed_index + 1]]
    assert "secret" not in str(caught.value)


def test_success_executes_each_immutable_command_once() -> None:
    commands = build_install_commands(sys.executable, "Linux", "x86_64")
    seen: list[tuple[str, ...]] = []

    def runner(argv: tuple[str, ...], **kwargs: object) -> SimpleNamespace:
        seen.append(tuple(argv))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    install_torch_stack(commands, repo=REPO_ROOT, runner=runner)

    assert seen == [command.argv for command in commands]


def test_cli_uses_supported_platform_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import install_torch_stack as module

    seen: list[tuple[InstallCommand, ...]] = []
    monkeypatch.setattr(module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(module.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(module, "install_torch_stack", lambda commands: seen.append(tuple(commands)))

    assert module.main() == 0
    assert tuple(item.argv for item in seen[0]) == _expected_commands("darwin-arm64", PYPI)
