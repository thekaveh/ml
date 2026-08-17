"""Install the repository's exact platform-qualified stack from reviewed locks."""

from __future__ import annotations

import platform
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from scripts.install_locked_requirements import LockedInstallError, run_install_argv


PYPI_INDEX = "https://pypi.org/simple"
CPU_INDEX = "https://download.pytorch.org/whl/cpu"
PYG_FIND_LINKS = "https://data.pyg.org/whl/torch-2.11.0+cpu.html"


class InstallStage(StrEnum):
    BOOTSTRAP = "bootstrap"
    CORE = "core"
    RUNTIME = "runtime"
    ROOT = "root"


@dataclass(frozen=True)
class InstallCommand:
    stage: InstallStage
    argv: tuple[str, ...]


Runner = Callable[..., Any]


class TorchStackInstallError(RuntimeError):
    """The fixed installer could not complete one safe stage."""


def build_install_commands(python: str, system: str, machine: str) -> tuple[InstallCommand, ...]:
    """Return the immutable four-stage lock install plan for one supported host."""
    hosts = {
        ("Darwin", "arm64"): ("darwin-arm64", PYPI_INDEX),
        ("Linux", "x86_64"): ("linux-x86_64", CPU_INDEX),
        ("Linux", "aarch64"): ("linux-aarch64", CPU_INDEX),
    }
    selected = hosts.get((system, machine))
    if selected is None:
        raise TorchStackInstallError("unsupported Torch stack platform")
    target, core_index = selected
    pip = (
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
    lock_root = f"requirements/locks/{target}"
    return (
        InstallCommand(
            InstallStage.BOOTSTRAP,
            pip + ("--only-binary=:all:", "-r", "requirements/locks/bootstrap.txt"),
        ),
        InstallCommand(
            InstallStage.CORE,
            pip
            + (
                "--only-binary=:all:",
                "--index-url",
                core_index,
                "-r",
                f"{lock_root}/core.txt",
            ),
        ),
        InstallCommand(
            InstallStage.RUNTIME,
            pip
            + (
                "--only-binary=:all:",
                "--index-url",
                PYPI_INDEX,
                "--find-links",
                PYG_FIND_LINKS,
                "-r",
                f"{lock_root}/runtime.txt",
            ),
        ),
        InstallCommand(
            InstallStage.ROOT,
            pip
            + (
                "--only-binary=:all:",
                "--no-binary=python-louvain",
                "--no-build-isolation",
                "--index-url",
                PYPI_INDEX,
                "--find-links",
                PYG_FIND_LINKS,
                "-r",
                f"{lock_root}/root.txt",
            ),
        ),
    )


def install_torch_stack(
    commands: Sequence[InstallCommand],
    *,
    repo: Path = Path.cwd(),
    runner: Runner = subprocess.run,
) -> None:
    """Execute the four-stage plan through the shared sanitized runner."""
    for command in commands:
        try:
            run_install_argv(repo, str(command.stage), command.argv, runner)
        except LockedInstallError as exc:
            raise TorchStackInstallError(
                f"torch stack installation failed: {command.stage}"
            ) from exc


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
