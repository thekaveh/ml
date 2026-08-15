from __future__ import annotations

from pathlib import Path

import pytest

import scripts.verify_junit as junit_module
from scripts.verify_junit import JUnitVerificationError, verify_junit


def _write_junit(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "report.xml"
    path.write_text(text, encoding="utf-8")
    return path


def test_valid_multisuite_aggregates(tmp_path: Path) -> None:
    path = _write_junit(
        tmp_path,
        """<testsuites>
  <testsuite name="one" tests="1" failures="0" errors="0" skipped="0">
    <testcase classname="c1" name="t1"/>
  </testsuite>
  <testsuite name="two" tests="1" failures="0" errors="0" skipped="0">
    <testcase classname="c2" name="t2"/>
  </testsuite>
</testsuites>""",
    )
    assert verify_junit(path) == (2, 0, 0, 0)


@pytest.mark.parametrize(
    "xml",
    (
        "<other/>",
        "<testsuites/>",
        '<testsuite name="s" failures="0" errors="0" skipped="0"/>',
        '<testsuite name="s" tests="0" errors="0" skipped="0"/>',
        '<testsuite name="s" tests="0" failures="0" skipped="0"/>',
        '<testsuite name="s" tests="0" failures="0" errors="0"/>',
        '<testsuite name="s" tests="+1" failures="0" errors="0" skipped="0"/>',
        '<testsuite name="s" tests="-1" failures="0" errors="0" skipped="0"/>',
        '<testsuite name="s" tests="1.0" failures="0" errors="0" skipped="0"/>',
        '<testsuite name="s" tests="x" failures="0" errors="0" skipped="0"/>',
        '<testsuite name="s" tests="2" failures="0" errors="0" skipped="0"><testcase classname="c" name="t"/></testsuite>',
        '<testsuite name="s" tests="1" failures="1" errors="0" skipped="0"><testcase classname="c" name="t"/></testsuite>',
        '<testsuite name="s" tests="1" failures="0" errors="0" skipped="0"><testcase classname="c" name="t"><failure/><error/></testcase></testsuite>',
        '<testsuites><testsuite name="one" tests="1" failures="0" errors="0" skipped="0"><testcase classname="c" name="t"/></testsuite><testsuite name="two" tests="1" failures="0" errors="0" skipped="0"><testcase classname="c" name="t"/></testsuite></testsuites>',
        '<testsuites><testsuite name="same" tests="1" failures="0" errors="0" skipped="0"><testcase classname="c1" name="t1"/></testsuite><testsuite name="same" tests="1" failures="0" errors="0" skipped="0"><testcase classname="c2" name="t2"/></testsuite></testsuites>',
    ),
    ids=(
        "wrong-root",
        "no-suites",
        "missing-tests",
        "missing-failures",
        "missing-errors",
        "missing-skipped",
        "signed",
        "negative",
        "decimal",
        "nonnumeric",
        "test-mismatch",
        "failure-child-mismatch",
        "two-outcomes",
        "duplicate-testcase",
        "duplicate-suite",
    ),
)
def test_junit_schema_is_fail_closed(tmp_path: Path, xml: str) -> None:
    with pytest.raises(JUnitVerificationError, match=r": schema$"):
        verify_junit(_write_junit(tmp_path, xml))


@pytest.mark.parametrize("tag", ("failure", "error", "skipped"))
def test_junit_rejects_nonzero_outcomes(tmp_path: Path, tag: str) -> None:
    counts = {"failure": 0, "error": 0, "skipped": 0}
    counts[tag] = 1
    xml = (
        f'<testsuite name="s" tests="1" failures="{counts["failure"]}" '
        f'errors="{counts["error"]}" skipped="{counts["skipped"]}">'
        f'<testcase classname="c" name="t"><{tag}/></testcase></testsuite>'
    )
    with pytest.raises(JUnitVerificationError, match=r": outcome$"):
        verify_junit(_write_junit(tmp_path, xml))


def test_junit_rejects_zero_tests(tmp_path: Path) -> None:
    xml = '<testsuite name="s" tests="0" failures="0" errors="0" skipped="0"/>'
    with pytest.raises(JUnitVerificationError, match=r": outcome$"):
        verify_junit(_write_junit(tmp_path, xml))


def test_junit_missing_and_invalid_categories(tmp_path: Path) -> None:
    with pytest.raises(JUnitVerificationError, match=r": missing$"):
        verify_junit(tmp_path / "absent.xml")
    with pytest.raises(JUnitVerificationError, match=r": invalid$"):
        verify_junit(_write_junit(tmp_path, "<testsuite>"))


def test_cli_prints_only_success_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _write_junit(
        tmp_path,
        '<testsuite name="s" tests="1" failures="0" errors="0" skipped="0">'
        '<testcase classname="c" name="t"/></testsuite>',
    )

    assert junit_module.main([str(path)]) == 0
    captured = capsys.readouterr()
    assert captured.out == "junit verification ok: tests=1 failures=0 errors=0 skipped=0\n"
    assert captured.err == ""


def test_cli_failure_is_stable_and_redacted(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    sensitive_path = tmp_path / "credential=secret.xml"

    assert junit_module.main([str(sensitive_path)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "junit verification failed: missing\n"
    assert "credential=secret" not in captured.err


@pytest.mark.parametrize("argv", ((), ("one.xml", "two.xml")))
def test_cli_accepts_exactly_one_path(
    argv: tuple[str, ...], capsys: pytest.CaptureFixture[str]
) -> None:
    assert junit_module.main(list(argv)) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "junit verification failed: schema\n"
