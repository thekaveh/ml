from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.verify_issue63_report import Issue63ReportError, main, verify_issue63_report


SHA_A = "a" * 40
SHA_B = "b" * 40
HASH_A = "a" * 64


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(repo: Path) -> tuple[Path, dict[str, object]]:
    evidence = repo / ".superpowers" / "sdd" / "issue63-evidence"
    evidence.mkdir(parents=True)
    evidence_paths: dict[str, dict[str, str]] = {}
    for name in (
        "repository.xml",
        "nnx.xml",
        "qat.xml",
        "advisory.json",
        "docker.json",
        "publication.json",
    ):
        path = evidence / name
        path.write_text(f"evidence:{name}\n", encoding="utf-8")
        relative = path.relative_to(repo).as_posix()
        evidence_paths[name] = {"path": relative, "sha256": _sha256(path)}

    report: dict[str, object] = {
        "schema_version": 1,
        "identities": {
            "final_sha": SHA_B,
            "final_tree": SHA_A,
            "design_sha256": HASH_A,
            "plan_sha256": "b" * 64,
            "commit_range_start": SHA_A,
            "commit_range_end": SHA_B,
            "develop_merge_sha": "c" * 40,
            "release_merge_sha": "d" * 40,
            "final_develop_sha": "e" * 40,
        },
        "locks": {
            "policy_sha256": "c" * 64,
            "image_lock_sha256": "d" * 64,
            "input_sha256": {"requirements.txt": "e" * 64},
            "lock_sha256": {"requirements/locks/bootstrap.txt": "f" * 64},
            "compiler": {
                "distribution": "uv",
                "version": "0.11.19",
                "executable_sha256": "1" * 64,
            },
            "local_python": {
                "version": "3.11.15",
                "executable": "/private/tmp/issue63/venv/bin/python",
            },
            "docker_python": {"version": "3.11.10", "architecture": "arm64"},
        },
        "tests": {
            "repository": {
                "total": 2500,
                "failures": 0,
                "errors": 0,
                "skips": 0,
                **evidence_paths["repository.xml"],
            },
            "nnx": {
                "total": 350,
                "failures": 0,
                "errors": 0,
                "skips": 0,
                **evidence_paths["nnx.xml"],
            },
            "qat": {
                "total": 1,
                "failures": 0,
                "errors": 0,
                "skips": 0,
                **evidence_paths["qat.xml"],
            },
        },
        "advisory": {
            "result": "accepted",
            "errors": [],
            "notices": [],
            "surfaces": {
                "combined-runtime": {"findings": 2},
                "torch": {"findings": 2},
                "docs": {"findings": 1},
                "atlas": {"findings": 1},
            },
            "non_pypi": [
                {"package": name, "version": version, "source": source}
                for name, version, source in (
                    ("en-core-web-sm", "3.8.0", "direct-url"),
                    ("pyg-lib", "0.8.0", "pyg-flat-index"),
                    ("torch-scatter", "2.1.2", "pyg-flat-index"),
                    ("torch-sparse", "0.6.18", "pyg-flat-index"),
                )
            ],
            **evidence_paths["advisory.json"],
        },
        "tiers": {
            "a": {"notebooks": 18, "code_cells": 210, "root_sha256": "2" * 64},
            "b": {"notebooks": 6, "code_cells": 75, "root_sha256": "3" * 64},
            "c": {"notebooks": 4, "code_cells": 56, "root_sha256": "4" * 64},
        },
        "docker": {
            "image_id": "sha256:" + "5" * 64,
            "architecture": "arm64",
            "base_reference": (
                "quay.io/jupyter/datascience-notebook:python-3.11@sha256:"
                + "6" * 64
            ),
            "probes": [
                {"name": name, "returncode": 0}
                for name in ("pip-check", "torch-stack", "nnx")
            ],
            **evidence_paths["docker.json"],
        },
        "pull_requests": {
            role: {
                "url": f"https://github.com/thekaveh/ml-eng-lab/pull/{number}",
                "source_sha": source_sha,
                "merge_sha": merge_sha,
                "run_urls": [
                    f"https://github.com/thekaveh/ml-eng-lab/actions/runs/{number}"
                ],
                "log_sha256": "7" * 64,
            }
            for role, number, source_sha, merge_sha in (
                ("feature", 120, SHA_B, "c" * 40),
                ("release", 121, "c" * 40, "d" * 40),
                ("sync", 122, "d" * 40, "e" * 40),
            )
        },
        "publication": {
            "pages_url": "https://thekaveh.github.io/ml-eng-lab/",
            "wiki_url": "https://github.com/thekaveh/ml-eng-lab/wiki",
            "pages_run_url": "https://github.com/thekaveh/ml-eng-lab/actions/runs/123",
            "live_sha256": "8" * 64,
            **evidence_paths["publication.json"],
        },
        "ruleset": {
            "before_sha256": "9" * 64,
            "after_sha256": "9" * 64,
            "contexts": [
                "atlas-consumer-policy",
                "dependency-audit",
                "pytest-repository",
            ],
        },
        "issues": {
            str(number): {
                "state": "OPEN",
                "before_sha256": character * 64,
                "after_sha256": character * 64,
            }
            for number, character in ((64, "a"), (65, "b"), (66, "c"))
        },
        "cleanup": {
            "feature_refs_absent": True,
            "temporary_root_absent": True,
            "image_absent": True,
            "noncompleted_runs": 0,
            "open_scoped_prs": 0,
        },
    }
    report_path = repo / ".superpowers" / "sdd" / "issue63-qualification-report.md"
    report_path.write_text(
        "# Issue #63 immutable qualification\n\n```json\n"
        + json.dumps(report, sort_keys=True, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    return report_path, report


def _write_report(path: Path, report: dict[str, object]) -> None:
    path.write_text(
        "# Issue #63 immutable qualification\n\n```json\n"
        + json.dumps(report, sort_keys=True, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )


def test_complete_issue63_report_passes(tmp_path: Path) -> None:
    report_path, _ = _fixture(tmp_path)

    verify_issue63_report(report_path, repo=tmp_path)


@pytest.mark.parametrize(
    "mutation",
    (
        lambda report: report.pop("docker"),
        lambda report: report.__setitem__("unexpected", {}),
        lambda report: report.__setitem__("schema_version", 2),
        lambda report: report["identities"].__setitem__("final_sha", "bad"),
        lambda report: report["identities"].__setitem__("commit_range_end", SHA_A),
        lambda report: report["locks"]["compiler"].__setitem__("version", "0.11.20"),
        lambda report: report["locks"]["local_python"].__setitem__("version", "3.11.14"),
        lambda report: report["locks"]["docker_python"].__setitem__("version", "3.11.15"),
        lambda report: report["tests"]["repository"].__setitem__("failures", 1),
        lambda report: report["tests"]["qat"].__setitem__("total", True),
        lambda report: report["tiers"]["b"].__setitem__("notebooks", 5),
        lambda report: report["docker"].__setitem__("architecture", "amd64"),
        lambda report: report["docker"]["probes"][0].__setitem__("returncode", 1),
        lambda report: report["advisory"].__setitem__("result", "rejected"),
        lambda report: report["advisory"]["non_pypi"].pop(),
        lambda report: report["pull_requests"]["feature"].__setitem__(
            "url", "https://example.invalid/pull/120"
        ),
        lambda report: report["ruleset"].__setitem__("after_sha256", "0" * 64),
        lambda report: report["issues"]["65"].__setitem__("state", "CLOSED"),
        lambda report: report["issues"]["66"].__setitem__("after_sha256", "0" * 64),
        lambda report: report["cleanup"].__setitem__("feature_refs_absent", False),
    ),
)
def test_report_mutations_fail_closed(tmp_path: Path, mutation) -> None:
    report_path, valid = _fixture(tmp_path)
    report = copy.deepcopy(valid)
    mutation(report)
    _write_report(report_path, report)

    with pytest.raises(Issue63ReportError, match=r"^Issue 63 report verification failed$"):
        verify_issue63_report(report_path, repo=tmp_path)


def test_report_rejects_evidence_path_escape(tmp_path: Path) -> None:
    report_path, valid = _fixture(tmp_path)
    report = copy.deepcopy(valid)
    report["tests"]["repository"]["path"] = "../outside.xml"
    _write_report(report_path, report)

    with pytest.raises(Issue63ReportError):
        verify_issue63_report(report_path, repo=tmp_path)


def test_report_rejects_wrong_evidence_hash(tmp_path: Path) -> None:
    report_path, valid = _fixture(tmp_path)
    report = copy.deepcopy(valid)
    report["tests"]["nnx"]["sha256"] = "0" * 64
    _write_report(report_path, report)

    with pytest.raises(Issue63ReportError):
        verify_issue63_report(report_path, repo=tmp_path)


@pytest.mark.parametrize(
    ("role", "field"),
    (("feature", "source_sha"), ("release", "source_sha"), ("sync", "merge_sha")),
)
def test_report_rejects_pr_sha_relationship_drift(
    tmp_path: Path,
    role: str,
    field: str,
) -> None:
    report_path, valid = _fixture(tmp_path)
    report = copy.deepcopy(valid)
    report["pull_requests"][role][field] = "f" * 40
    _write_report(report_path, report)

    with pytest.raises(Issue63ReportError):
        verify_issue63_report(report_path, repo=tmp_path)


def _structured_mappings(value: object, path: tuple[str, ...] = ()):
    if not isinstance(value, dict):
        return
    yield path, value
    for key, child in value.items():
        if path + (key,) in (("locks", "input_sha256"), ("locks", "lock_sha256")):
            continue
        yield from _structured_mappings(child, path + (key,))


def _at_path(value: dict[str, object], path: tuple[str, ...]) -> dict[str, object]:
    selected = value
    for component in path:
        selected = selected[component]
    return selected


def test_every_structured_report_field_is_required(tmp_path: Path) -> None:
    report_path, valid = _fixture(tmp_path)
    cases = tuple(
        (path, key)
        for path, mapping in _structured_mappings(valid)
        for key in mapping
        if not (path == ("locks", "input_sha256") or path == ("locks", "lock_sha256"))
    )
    assert len(cases) > 80

    for path, key in cases:
        report = copy.deepcopy(valid)
        _at_path(report, path).pop(key)
        _write_report(report_path, report)
        with pytest.raises(Issue63ReportError):
            verify_issue63_report(report_path, repo=tmp_path)


def test_every_structured_mapping_rejects_unknown_fields(tmp_path: Path) -> None:
    report_path, valid = _fixture(tmp_path)
    paths = tuple(path for path, _ in _structured_mappings(valid))
    assert len(paths) > 20

    for path in paths:
        report = copy.deepcopy(valid)
        _at_path(report, path)["unexpected"] = "value"
        _write_report(report_path, report)
        with pytest.raises(Issue63ReportError):
            verify_issue63_report(report_path, repo=tmp_path)


def test_cli_is_stable_and_redacted(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    report_path, valid = _fixture(tmp_path)
    report = copy.deepcopy(valid)
    report["identities"]["final_sha"] = "/Users/private/token=secret"
    _write_report(report_path, report)

    assert main([str(report_path), "--repo", str(tmp_path)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "Issue 63 report verification failed\n"
