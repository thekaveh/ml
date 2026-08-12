from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_SUBPROCESS_TIMEOUT = 30
ATLAS_TEST_FILES = (
    "tests/test_atlas_consumer_contract.py",
    "tests/test_atlas_lifecycle.py",
    "tests/test_atlas_runtime_probe.py",
    "tests/test_atlas_makefile_contract.py",
)


def _parse_phony_targets(makefile: str) -> set[str]:
    return {
        target
        for line in makefile.splitlines()
        if line.startswith(".PHONY:")
        for target in line.removeprefix(".PHONY:").split()
    }


def _target_is_phony(makefile: str, target: str) -> bool:
    return target in _parse_phony_targets(makefile)


def _help_lists_target(help_output: str, target: str) -> bool:
    prefix = f"  {target}"
    return any(
        line.startswith(prefix)
        and len(line) > len(prefix)
        and line[len(prefix)].isspace()
        for line in help_output.splitlines()
    )


def test_atlas_targets_expose_exact_lifecycle_commands():
    result = subprocess.run(
        [
            "make",
            "--no-print-directory",
            "-n",
            "atlas-setup",
            "atlas-up",
            "atlas-down",
            "atlas-connect",
            "atlas-contract",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=TEST_SUBPROCESS_TIMEOUT,
    )

    assert result.stdout.splitlines() == [
        "git submodule update --init --recursive infra",
        "./scripts/atlas-up.sh --prepare",
        "./scripts/atlas-up.sh",
        "./scripts/atlas-down.sh ",
        "./scripts/atlas-connect.sh",
        "./scripts/atlas-up.sh --validate",
    ]


def test_atlas_down_only_requests_cold_shutdown_when_explicit():
    warm = subprocess.run(
        ["make", "--no-print-directory", "-n", "atlas-down"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=TEST_SUBPROCESS_TIMEOUT,
    )
    cold = subprocess.run(
        ["make", "--no-print-directory", "-n", "COLD=1", "atlas-down"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=TEST_SUBPROCESS_TIMEOUT,
    )

    assert "--cold" not in warm.stdout
    assert cold.stdout.splitlines() == ["./scripts/atlas-down.sh --cold"]


def test_atlas_targets_are_documented_and_phony():
    help_result = subprocess.run(
        ["make", "help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=TEST_SUBPROCESS_TIMEOUT,
    )
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")

    for target in (
        "atlas-setup",
        "atlas-up",
        "atlas-down",
        "atlas-connect",
        "atlas-contract",
    ):
        assert _help_lists_target(help_result.stdout, target)
        assert _target_is_phony(makefile, target)


def test_atlas_phony_contract_rejects_prefix_only_target_matches():
    makefile = ".PHONY: atlas-setup-extra atlas-upgrade atlas-downstream\n"

    assert not _target_is_phony(makefile, "atlas-setup")
    assert not _target_is_phony(makefile, "atlas-up")
    assert not _target_is_phony(makefile, "atlas-down")


def test_atlas_help_contract_rejects_unanchored_or_prefix_only_target_matches():
    help_output = (
        "prefix atlas-up Run Atlas.\n"
        "  atlas-upgrade Upgrade Atlas.\n"
        "   atlas-down Stop Atlas.\n"
    )

    assert not _help_lists_target(help_output, "atlas-up")
    assert not _help_lists_target(help_output, "atlas-down")


def test_atlas_consumer_make_target_runs_exact_focused_test_files():
    result = subprocess.run(
        ["make", "--no-print-directory", "-n", "test-atlas-consumer"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=TEST_SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [f"pytest {' '.join(ATLAS_TEST_FILES)} -v"]


def test_atlas_consumer_make_target_is_documented_and_phony():
    help_result = subprocess.run(
        ["make", "help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=TEST_SUBPROCESS_TIMEOUT,
    )
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")

    assert _help_lists_target(help_result.stdout, "test-atlas-consumer")
    assert _target_is_phony(makefile, "test-atlas-consumer")
