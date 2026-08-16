"""Install the repository's exact, platform-qualified Torch stack."""

from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, Sequence


CPU_INDEX = "https://download.pytorch.org/whl/cpu"


class InstallStage(StrEnum):
    UPGRADE_PIP = "upgrade-pip"
    CORE = "core"
    RUNTIME = "runtime"
    ROOT = "root"


@dataclass(frozen=True)
class InstallCommand:
    stage: InstallStage
    argv: tuple[str, ...]


class CommandResult(Protocol):
    returncode: int


class CommandRunner(Protocol):
    def __call__(self, argv: Sequence[str], *, check: bool) -> CommandResult: ...


class TorchStackInstallError(RuntimeError):
    """The fixed installer could not complete one safe stage."""


def build_install_commands(python: str, system: str, machine: str) -> tuple[InstallCommand, ...]:
    """Return the immutable four-stage install plan for one supported host."""
    if (system, machine) not in (("Linux", "x86_64"), ("Linux", "aarch64"), ("Darwin", "arm64")):
        raise TorchStackInstallError("unsupported Torch stack platform")

    pip = (python, "-m", "pip", "install")
    core = pip + (("--index-url", CPU_INDEX) if system == "Linux" else ()) + (
        "-r",
        "torch-core-requirements.txt",
    )
    return (
        InstallCommand(InstallStage.UPGRADE_PIP, pip + ("--upgrade", "pip")),
        InstallCommand(InstallStage.CORE, core),
        InstallCommand(
            InstallStage.RUNTIME,
            pip
            + (
                "--only-binary=pyg-lib,torch-scatter,torch-sparse",
                "-r",
                "torch-requirements.txt",
            ),
        ),
        InstallCommand(
            InstallStage.ROOT,
            pip + ("--only-binary=thekaveh-nnx", "-r", "requirements.txt"),
        ),
    )


def install_torch_stack(
    commands: Sequence[InstallCommand], runner: CommandRunner = subprocess.run
) -> None:
    """Execute an already-planned Torch installation without leaking subprocess output."""
    for command in commands:
        if runner(command.argv, check=False).returncode != 0:
            raise TorchStackInstallError(f"torch stack installation failed: {command.stage}")


def main() -> int:
    """Run the supported host's Torch installation plan."""
    try:
        install_torch_stack(build_install_commands(sys.executable, platform.system(), platform.machine()))
    except TorchStackInstallError as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
