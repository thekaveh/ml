from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_SUBPROCESS_TIMEOUT = 30


def test_mkdocs_commands_suppress_only_the_upstream_material_banner():
    result = subprocess.run(
        ["make", "--no-print-directory", "-n", "docs-build", "docs-serve", "docs-check"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=TEST_SUBPROCESS_TIMEOUT,
    )
    mkdocs_lines = [line for line in result.stdout.splitlines() if "mkdocs " in line]

    assert mkdocs_lines
    assert all(line.startswith("NO_MKDOCS_2_WARNING=1 mkdocs ") for line in mkdocs_lines)


def test_setup_targets_use_selected_python_interpreter():
    custom_python = "/opt/custom/bin/python"
    result = subprocess.run(
        [
            "make",
            "--no-print-directory",
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


def test_smoke_tier_a_writes_to_temporary_outputs_without_mutating_sources(
    tmp_path: Path,
) -> None:
    sources = (
        tmp_path / "notebooks" / "first" / "notebook.ipynb",
        tmp_path / "notebooks" / "second" / "notebook.ipynb",
    )
    for index, source in enumerate(sources, start=1):
        source.parent.mkdir(parents=True)
        source.write_text(f"source notebook {index}\n", encoding="utf-8")
    output_root = tmp_path / "tier-a-output"
    fake_papermill = tmp_path / "papermill"
    fake_papermill.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'input="${@: -2:1}"\n'
        'output="${@: -1}"\n'
        'mkdir -p "$(dirname "$output")"\n'
        'printf "rendered:%s\\n" "$input" > "$output"\n',
        encoding="utf-8",
    )
    fake_papermill.chmod(0o755)

    result = subprocess.run(
        [
            "make",
            "-f",
            str(REPO_ROOT / "Makefile"),
            "smoke-tier-a",
            "TIER_A=notebooks/first/notebook.ipynb notebooks/second/notebook.ipynb",
            f"TIER_A_OUT={output_root}",
            f"PAPERMILL={fake_papermill}",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=TEST_SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 0, result.stderr
    assert tuple(source.read_text(encoding="utf-8") for source in sources) == (
        "source notebook 1\n",
        "source notebook 2\n",
    )
    assert tuple(
        (output_root / "notebooks" / task / "notebook.ipynb").read_text(encoding="utf-8")
        for task in ("first", "second")
    ) == ("rendered:notebook.ipynb\n", "rendered:notebook.ipynb\n")


def test_check_tier_a_artifacts_accepts_every_nonempty_mirrored_output(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "tier-a-output"
    tier_a = "notebooks/first/notebook.ipynb notebooks/second/notebook.ipynb"
    for task in ("first", "second"):
        output = output_root / "notebooks" / task / "notebook.ipynb"
        output.parent.mkdir(parents=True)
        output.write_text(f"{task} output\n", encoding="utf-8")

    result = subprocess.run(
        [
            "make",
            "-f",
            str(REPO_ROOT / "Makefile"),
            "check-tier-a-artifacts",
            f"TIER_A={tier_a}",
            f"TIER_A_OUT={output_root}",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=TEST_SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("empty_output", (False, True))
def test_check_tier_a_artifacts_reports_a_missing_or_empty_mirrored_output(
    tmp_path: Path, empty_output: bool
) -> None:
    output_root = tmp_path / "tier-a-output"
    existing = output_root / "notebooks" / "first" / "notebook.ipynb"
    existing.parent.mkdir(parents=True)
    existing.write_text("first output\n", encoding="utf-8")
    if empty_output:
        empty = output_root / "notebooks" / "second" / "notebook.ipynb"
        empty.parent.mkdir(parents=True)
        empty.write_text("", encoding="utf-8")
    tier_a = "notebooks/first/notebook.ipynb notebooks/second/notebook.ipynb"

    result = subprocess.run(
        [
            "make",
            "-f",
            str(REPO_ROOT / "Makefile"),
            "check-tier-a-artifacts",
            f"TIER_A={tier_a}",
            f"TIER_A_OUT={output_root}",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=TEST_SUBPROCESS_TIMEOUT,
    )

    assert result.returncode != 0
    assert "missing expected Tier-A notebook output" in result.stderr


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
