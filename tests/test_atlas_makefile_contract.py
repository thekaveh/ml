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
        assert target in help_result.stdout
        assert target in makefile.split(".PHONY:", 1)[1].splitlines()[0]


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

    assert "test-atlas-consumer" in help_result.stdout
    assert "test-atlas-consumer" in makefile.split(".PHONY:", 1)[1].splitlines()[0]
