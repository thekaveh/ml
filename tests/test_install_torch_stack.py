from __future__ import annotations

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
CORE = "torch==2.11.0\ntorchvision==0.26.0\ntorchaudio==2.11.0\n"
ECOSYSTEM = "pytorch-lightning==2.6.1\ntorchmetrics==1.9.0\ntorchao==0.18.0\n"
RUNTIME_MANIFEST = (
    "-r torch-ecosystem-requirements.txt\n"
    "--find-links https://data.pyg.org/whl/torch-2.11.0+cpu.html\n"
    "pyg-lib==0.8.0\ntorch-scatter==2.1.2\ntorch-sparse==0.6.18\n"
    "torch-cluster==1.6.3\ntorch-spline-conv==1.2.2\ntorch_geometric==2.8.0.post1\n"
)
AUDIT = (
    "-r torch-core-requirements.txt\n-r torch-ecosystem-requirements.txt\n"
    "torch_geometric==2.8.0.post1\n"
)
PYG_EXTENSIONS = (
    "torch-scatter==2.1.2\ntorch-sparse==0.6.18\n"
    "torch-cluster==1.6.3\ntorch-spline-conv==1.2.2\n"
)
LINUX_CORE = (
    sys.executable,
    "-m",
    "pip",
    "install",
    "--index-url",
    "https://download.pytorch.org/whl/cpu",
    "-r",
    "torch-core-requirements.txt",
)
RUNTIME = (
    sys.executable,
    "-m",
    "pip",
    "install",
    "--only-binary=pyg-lib,torch-scatter,torch-sparse,torch-cluster",
    "--no-binary=torch-spline-conv",
    "--no-build-isolation",
    "-r",
    "torch-requirements.txt",
)
ROOT = (
    sys.executable,
    "-m",
    "pip",
    "install",
    "--only-binary=thekaveh-nnx",
    "-r",
    "requirements.txt",
)


def test_canonical_manifest_bytes_are_exact() -> None:
    assert (REPO_ROOT / "torch-core-requirements.txt").read_text(encoding="utf-8") == CORE
    assert (REPO_ROOT / "torch-ecosystem-requirements.txt").read_text(encoding="utf-8") == ECOSYSTEM
    assert (REPO_ROOT / "torch-requirements.txt").read_text(encoding="utf-8") == RUNTIME_MANIFEST
    assert (REPO_ROOT / "torch-audit-requirements.txt").read_text(encoding="utf-8") == AUDIT
    extension_lines = (REPO_ROOT / "pyg-extension-audit-requirements.txt").read_text(encoding="utf-8").splitlines()
    assert "\n".join(line for line in extension_lines if not line.startswith("#")) + "\n" == PYG_EXTENSIONS


def test_linux_and_darwin_command_plans_are_exact() -> None:
    linux = build_install_commands(sys.executable, "Linux", "x86_64")
    darwin = build_install_commands(sys.executable, "Darwin", "arm64")

    assert tuple(command.stage for command in linux) == tuple(InstallStage)
    assert linux[1].argv == LINUX_CORE
    assert darwin[1].argv == tuple(
        token
        for token in LINUX_CORE
        if token not in ("--index-url", "https://download.pytorch.org/whl/cpu")
    )
    assert linux[2].argv == darwin[2].argv == RUNTIME
    assert linux[3].stage is InstallStage.ROOT
    assert linux[3].argv == darwin[3].argv == ROOT


@pytest.mark.parametrize(
    ("system", "machine"),
    (("Linux", "arm64"), ("Darwin", "x86_64"), ("Windows", "AMD64")),
)
def test_unsupported_platforms_fail_closed(system: str, machine: str) -> None:
    with pytest.raises(TorchStackInstallError, match="unsupported Torch stack platform"):
        build_install_commands(sys.executable, system, machine)


def test_linux_arm64_uses_the_cpu_core_index() -> None:
    commands = build_install_commands(sys.executable, "Linux", "aarch64")

    assert commands[1].argv == LINUX_CORE


def test_installer_runs_every_immutable_argv_plan_without_checking_runner() -> None:
    commands = build_install_commands(sys.executable, "Linux", "x86_64")
    seen: list[tuple[tuple[str, ...], bool]] = []

    def runner(argv, *, check):
        seen.append((tuple(argv), check))
        return SimpleNamespace(returncode=0)

    install_torch_stack(commands, runner=runner)

    assert seen == [(command.argv, False) for command in commands]
    assert all(isinstance(command.argv, tuple) for command in commands)


def test_installer_stops_on_nonzero_with_a_redacted_stable_error() -> None:
    commands = build_install_commands(sys.executable, "Linux", "x86_64")
    secret = "https://token@example.test/private-output"

    def runner(argv, *, check):
        return SimpleNamespace(returncode=1, stdout=secret, stderr=secret)

    with pytest.raises(TorchStackInstallError) as error:
        install_torch_stack(commands[2:], runner=runner)

    assert str(error.value) == "torch stack installation failed: runtime"
    assert secret not in str(error.value)


@pytest.mark.parametrize(
    "replacement",
    (
        "--only-binary=:all:",
        "--only-binary=pyg-lib,torch-scatter,torch-sparse,torch-cluster,torch-spline-conv",
        "--only-binary=pyg-lib,torch-scatter,torch-sparse",
    ),
)
def test_runtime_wheel_policy_is_not_broadened_or_weakened(replacement: str) -> None:
    command = build_install_commands(sys.executable, "Linux", "x86_64")[2]

    assert command.argv[4] == "--only-binary=pyg-lib,torch-scatter,torch-sparse,torch-cluster"
    assert replacement != command.argv[4]
    assert "--no-binary=torch-spline-conv" in command.argv
    assert "--no-build-isolation" in command.argv


def test_root_stage_is_last_and_binary_only_for_nnx() -> None:
    commands = build_install_commands(sys.executable, "Linux", "x86_64")

    assert len(commands) == 4
    assert commands[-1] == InstallCommand(InstallStage.ROOT, ROOT)
    assert "--only-binary=thekaveh-nnx" in commands[-1].argv


def test_cli_uses_platform_and_runner_seams(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import install_torch_stack as module

    seen: list[tuple[tuple[str, ...], bool]] = []
    monkeypatch.setattr(module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(module.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(module, "install_torch_stack", lambda commands: seen.extend((command.argv, False) for command in commands))

    assert module.main() == 0
    assert seen == [(command.argv, False) for command in build_install_commands(sys.executable, "Darwin", "arm64")]
