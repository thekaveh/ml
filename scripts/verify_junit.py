"""Fail closed unless a JUnit report is complete, passing, and unskipped."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from xml.etree import ElementTree


_COUNT_FIELDS = ("tests", "failures", "errors", "skipped")
_OUTCOME_TAGS = ("failure", "error", "skipped")


class JUnitVerificationError(RuntimeError):
    """The JUnit report is missing, malformed, empty, failing, or skipping."""


def _count_attribute(suite: ElementTree.Element, field: str) -> int:
    value = suite.attrib.get(field)
    if value is None or re.fullmatch(r"0|[1-9][0-9]*", value) is None:
        raise JUnitVerificationError("junit verification failed: schema")
    return int(value)


def _suite_counts(
    suite: ElementTree.Element,
    identities: set[tuple[str, str]],
) -> tuple[int, int, int, int]:
    declared = tuple(_count_attribute(suite, field) for field in _COUNT_FIELDS)
    cases = tuple(suite.findall("testcase"))
    actual_outcomes = {tag: 0 for tag in _OUTCOME_TAGS}
    for case in cases:
        classname = case.attrib.get("classname")
        name = case.attrib.get("name")
        if not classname or not name or (classname, name) in identities:
            raise JUnitVerificationError("junit verification failed: schema")
        identities.add((classname, name))
        outcomes = [child.tag for child in case if child.tag in _OUTCOME_TAGS]
        if len(outcomes) > 1:
            raise JUnitVerificationError("junit verification failed: schema")
        if outcomes:
            actual_outcomes[outcomes[0]] += 1
    actual = (
        len(cases),
        actual_outcomes["failure"],
        actual_outcomes["error"],
        actual_outcomes["skipped"],
    )
    if declared != actual:
        raise JUnitVerificationError("junit verification failed: schema")
    return declared


def verify_junit(path: Path) -> tuple[int, int, int, int]:
    try:
        root = ElementTree.parse(path).getroot()
    except FileNotFoundError:
        raise JUnitVerificationError("junit verification failed: missing") from None
    except (OSError, ElementTree.ParseError):
        raise JUnitVerificationError("junit verification failed: invalid") from None
    if root.tag == "testsuite":
        suites = (root,)
    elif root.tag == "testsuites":
        suites = tuple(root.findall("testsuite"))
    else:
        raise JUnitVerificationError("junit verification failed: schema")
    suite_names = tuple(suite.attrib.get("name") for suite in suites)
    if (
        not suites
        or any(not name for name in suite_names)
        or len(set(suite_names)) != len(suite_names)
        or tuple(root.iter("testsuite")) != suites
    ):
        raise JUnitVerificationError("junit verification failed: schema")
    identities: set[tuple[str, str]] = set()
    per_suite = tuple(_suite_counts(suite, identities) for suite in suites)
    totals = tuple(sum(counts[index] for counts in per_suite) for index in range(4))
    tests, failures, errors, skipped = totals
    if tests <= 0 or failures or errors or skipped:
        raise JUnitVerificationError("junit verification failed: outcome")
    return totals


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("junit verification failed: schema", file=sys.stderr)
        return 1
    try:
        tests, failures, errors, skipped = verify_junit(Path(args[0]))
    except JUnitVerificationError as error:
        print(error, file=sys.stderr)
        return 1
    except BaseException:
        print("junit verification failed: invalid", file=sys.stderr)
        return 1
    print(
        f"junit verification ok: tests={tests} failures={failures} "
        f"errors={errors} skipped={skipped}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
