from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.install_locked_requirements import (
    LockedInstallError,
    command_for,
    install_role,
)


REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize(
    ("role", "relative"),
    (
        ("bootstrap", "requirements/locks/bootstrap.txt"),
        ("compiler", "requirements/locks/compiler.txt"),
        ("docs", "docs-requirements.txt"),
        ("audit", "requirements/locks/audit.txt"),
        ("atlas-contract", "requirements/locks/atlas-contract.txt"),
    ),
)
def test_dedicated_install_commands_are_exact_and_binary_only(
    tmp_path: Path, role: str, relative: str
) -> None:
    lock = tmp_path / relative
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text("demo==1.0 --hash=sha256:" + "a" * 64 + "\n", encoding="utf-8")

    command = command_for(tmp_path, role)

    assert command.role == role
    assert command.argv[1:4] == ("-m", "pip", "install")
    assert command.argv[4:] == (
        "--isolated",
        "--disable-pip-version-check",
        "--no-input",
        "--require-hashes",
        "--only-binary=:all:",
        "--no-deps",
        "-r",
        relative,
    )


@pytest.mark.parametrize("role", ("root", "rogue"))
def test_unavailable_or_unknown_roles_fail_closed(tmp_path: Path, role: str) -> None:
    with pytest.raises(LockedInstallError, match="role|missing"):
        command_for(tmp_path, role)


def test_installer_stops_on_nonzero_and_redacts_runner_output(tmp_path: Path) -> None:
    lock = tmp_path / "requirements/locks/bootstrap.txt"
    lock.parent.mkdir(parents=True)
    lock.write_text("demo==1.0 --hash=sha256:" + "a" * 64 + "\n", encoding="utf-8")
    seen: list[tuple[str, ...]] = []

    def runner(argv: tuple[str, ...], **kwargs: object) -> SimpleNamespace:
        seen.append(argv)
        assert kwargs["check"] is False
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        assert environment["PIP_CONFIG_FILE"] == "/dev/null"
        assert environment["PIP_DISABLE_PIP_VERSION_CHECK"] == "1"
        return SimpleNamespace(returncode=1, stdout="secret", stderr="/private/tmp/secret")

    with pytest.raises(LockedInstallError, match="bootstrap") as caught:
        install_role(tmp_path, "bootstrap", runner=runner)

    assert seen == [command_for(tmp_path, "bootstrap").argv]
    assert "secret" not in str(caught.value)


def test_role_lock_must_remain_inside_repo_and_cannot_escape_by_symlink(tmp_path: Path) -> None:
    lock = tmp_path / "requirements/locks/bootstrap.txt"
    lock.parent.mkdir(parents=True)
    outside = tmp_path.parent / "outside-lock.txt"
    outside.write_text("demo==1.0 --hash=sha256:" + "a" * 64 + "\n", encoding="utf-8")
    lock.symlink_to(outside)

    with pytest.raises(LockedInstallError, match="escape|unsafe"):
        command_for(tmp_path, "bootstrap")
