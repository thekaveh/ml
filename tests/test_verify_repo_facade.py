"""Characterization coverage for the verify_repo command-line facade."""
from __future__ import annotations

import builtins
import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "verify_repo.py"


def _load_facade():
    name = "verify_repo_facade_contract"
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_help_defers_config_and_validator_imports(monkeypatch):
    """A copied facade must render help without its runtime dependencies."""
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "yaml" or name.startswith("scripts.repo_verifier"):
            raise AssertionError(f"help path imported runtime dependency: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--help"])
    monkeypatch.setattr(builtins, "__import__", guarded_import)

    facade = _load_facade()
    with pytest.raises(SystemExit, match="0"):
        facade.main(["--help"])


def test_all_dispatch_preserves_order_and_forwards_fast_only_to_execution(
    tmp_path, monkeypatch, capsys
):
    facade = _load_facade()
    calls = []

    def ordinary(name):
        def check(repo):
            calls.append((name, repo, None))
            return facade.CheckResult(name)

        return check

    def execution(repo, fast):
        calls.append(("execution", repo, fast))
        return facade.CheckResult("execution")

    monkeypatch.setattr(facade, "CHECKS", {
        "structure": ordinary("structure"),
        "assets": ordinary("assets"),
        "docs": ordinary("docs"),
        "comments": ordinary("comments"),
        "execution": execution,
    })
    assert facade.main(["--check", "all", "--fast", "--repo-root", str(tmp_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [name for name, _, _ in calls] == [
        "structure", "assets", "docs", "comments", "execution"
    ]
    assert calls[-1][2] is True
    assert payload["summary"]["checks_run"] == [
        "structure", "assets", "docs", "comments", "execution"
    ]


@pytest.mark.parametrize(
    ("severity", "expected_exit", "expected_diagnostic"),
    (
        ("warning", 0, "verify_repo: 0 errors, 1 warnings"),
        ("error", 1, "verify_repo: 1 errors, 0 warnings"),
    ),
)
def test_findings_aggregate_to_exit_code_and_diagnostic(
    tmp_path, monkeypatch, capsys, severity, expected_exit, expected_diagnostic
):
    facade = _load_facade()
    finding = facade.Finding(
        id="X1.test",
        check="structure",
        severity=severity,
        location="fixture",
        message="characterized",
    )
    monkeypatch.setattr(
        facade,
        "CHECKS",
        {"structure": lambda _repo: facade.CheckResult("structure", [finding])},
    )

    assert facade.main(["--check", "structure", "--repo-root", str(tmp_path)]) == expected_exit
    captured = capsys.readouterr()
    assert expected_diagnostic in captured.err
    assert json.loads(captured.out)["summary"]["warnings"] == (severity == "warning")
    assert json.loads(captured.out)["summary"]["errors"] == (severity == "error")


def test_out_writes_schema_version_one_json_without_stdout(tmp_path, monkeypatch, capsys):
    facade = _load_facade()
    out_path = tmp_path / "findings.json"
    monkeypatch.setattr(
        facade,
        "CHECKS",
        {"structure": lambda _repo: facade.CheckResult("structure")},
    )

    assert facade.main(
        ["--check", "structure", "--repo-root", str(tmp_path), "--out", str(out_path)]
    ) == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(out_path.read_text(encoding="utf-8"))["schema_version"] == 1


def test_phase_b_out_calls_only_export_function(tmp_path, monkeypatch, capsys):
    facade = _load_facade()
    out_path = tmp_path / "phase-b.json"
    calls = []

    def export(repo, out):
        calls.append((repo, out))
        return 3

    def unexpected_check(*_args):
        raise AssertionError("main check loop must be bypassed")

    monkeypatch.setattr(facade, "export_phase_b_candidates", export)
    monkeypatch.setattr(facade, "CHECKS", {"structure": unexpected_check})

    assert facade.main(["--repo-root", str(tmp_path), "--phase-b-out", str(out_path)]) == 0
    captured = capsys.readouterr()
    assert calls == [(tmp_path.resolve(), out_path)]
    assert captured.out == ""
    assert f"verify_repo: 3 Phase-B candidates → {out_path}" in captured.err
