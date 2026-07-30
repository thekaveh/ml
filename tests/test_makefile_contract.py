from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_SUBPROCESS_TIMEOUT = 30


def test_setup_targets_use_selected_python_interpreter():
    custom_python = "/opt/custom/bin/python"
    result = subprocess.run(
        [
            "make",
            "-n",
            f"PYTHON={custom_python}",
            "install-torch-stack",
            "codespace-setup",
            "nlp-assets",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=TEST_SUBPROCESS_TIMEOUT,
    )
    lines = result.stdout.splitlines()

    assert f"{custom_python} -m pip install --upgrade pip" in lines
    assert f"{custom_python} -m pip install -r torch-core-requirements.txt" in lines
    assert f"{custom_python} -m pip install --no-build-isolation -r torch-requirements.txt" in lines
    assert f"{custom_python} -m pip install -r requirements.txt" in lines
    assert f"{custom_python} -m spacy download en_core_web_sm" in lines
    assert any(line.startswith(f"{custom_python} -c ") for line in lines)
    assert not any(line.startswith("pip install") or line.startswith("python ") for line in lines)


def test_atlas_targets_expose_exact_lifecycle_commands():
    result = subprocess.run(
        [
            "make",
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
        ["make", "-n", "atlas-down"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=TEST_SUBPROCESS_TIMEOUT,
    )
    cold = subprocess.run(
        ["make", "-n", "COLD=1", "atlas-down"],
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
