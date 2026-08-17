"""Install generated dependency locks through one sanitized pip boundary."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


Role = Literal["bootstrap", "compiler", "docs", "audit", "atlas-contract"]
Runner = Callable[..., Any]


class LockedInstallError(RuntimeError):
    """A stable, redacted locked-install failure."""


@dataclass(frozen=True)
class InstallCommand:
    role: str
    argv: tuple[str, ...]


_ROLE_LOCKS = {
    "bootstrap": "requirements/locks/bootstrap.txt",
    "compiler": "requirements/locks/compiler.txt",
    "docs": "docs-requirements.txt",
    "audit": "requirements/locks/audit.txt",
    "atlas-contract": "requirements/locks/atlas-contract.txt",
}


def command_for(repo: Path, role: str) -> InstallCommand:
    relative = _ROLE_LOCKS.get(role)
    if relative is None:
        raise LockedInstallError("locked requirements installation failed: role")
    repo_root = repo.resolve()
    candidate = repo / relative
    if not candidate.is_file():
        raise LockedInstallError(f"locked requirements installation failed: {role}: missing")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(repo_root)
    except (OSError, ValueError) as exc:
        raise LockedInstallError(
            f"locked requirements installation failed: {role}: unsafe escape"
        ) from exc
    if candidate.is_symlink():
        raise LockedInstallError(f"locked requirements installation failed: {role}: unsafe escape")
    return InstallCommand(
        role=role,
        argv=(
            sys.executable,
            "-m",
            "pip",
            "install",
            "--isolated",
            "--disable-pip-version-check",
            "--no-input",
            "--require-hashes",
            "--only-binary=:all:",
            "--no-deps",
            "-r",
            relative,
        ),
    )


def sanitized_environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("PIP_", "UV_")) and key not in {"PYTHONPATH", "PYTHONHOME"}
    }
    environment.update(
        {
            "PIP_CONFIG_FILE": os.devnull,
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INPUT": "1",
        }
    )
    return environment


def run_install_argv(
    repo: Path, stage: str, argv: tuple[str, ...], runner: Runner = subprocess.run
) -> None:
    try:
        result = runner(
            argv,
            cwd=repo,
            env=sanitized_environment(),
            check=False,
            capture_output=True,
            text=True,
        )
    except (OSError, UnicodeError) as exc:
        raise LockedInstallError(
            f"locked requirements installation failed: {stage}: execution"
        ) from exc
    if result.returncode != 0:
        raise LockedInstallError(f"locked requirements installation failed: {stage}")


def install_role(repo: Path, role: str, runner: Runner = subprocess.run) -> None:
    command = command_for(repo, role)
    run_install_argv(repo, command.role, command.argv, runner)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=tuple(_ROLE_LOCKS))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    arguments = parser.parse_args(argv)
    try:
        install_role(arguments.repo_root.resolve(), arguments.role)
    except LockedInstallError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
