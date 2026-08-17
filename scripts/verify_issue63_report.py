"""Fail closed unless an Issue #63 qualification report is internally complete."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NoReturn


REPO_ROOT = Path(__file__).resolve().parent.parent
_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
_JSON_FENCE = re.compile(r"```json\n(?P<payload>.*?)\n```", re.DOTALL)
_GITHUB_URL = re.compile(
    r"^https://github\.com/thekaveh/ml-eng-lab/(?:pull/[1-9][0-9]*|actions/runs/[1-9][0-9]*)$"
)


class Issue63ReportError(RuntimeError):
    """The immutable qualification report failed its stable contract."""

    def __init__(self) -> None:
        super().__init__("Issue 63 report verification failed")


def _fail() -> NoReturn:
    raise Issue63ReportError() from None


def _mapping(value: object, keys: set[str]) -> Mapping[str, object]:
    if not isinstance(value, dict) or set(value) != keys:
        _fail()
    return value


def _text(value: object) -> str:
    if not isinstance(value, str) or not value or "\n" in value or "\r" in value:
        _fail()
    return value


def _integer(value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail()
    return value


def _sha1(value: object) -> str:
    selected = _text(value)
    if _SHA1.fullmatch(selected) is None:
        _fail()
    return selected


def _sha256(value: object) -> str:
    selected = _text(value)
    if _SHA256.fullmatch(selected) is None:
        _fail()
    return selected


def _sequence(value: object, *, length: int | None = None) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, list):
        _fail()
    if length is not None and len(value) != length:
        _fail()
    return value


def _evidence(record: Mapping[str, object], repo: Path) -> None:
    relative = Path(_text(record["path"]))
    if relative.is_absolute() or ".." in relative.parts or relative == Path("."):
        _fail()
    try:
        root = repo.resolve(strict=True)
        path = (root / relative).resolve(strict=True)
        path.relative_to(root)
    except (OSError, ValueError):
        _fail()
    if path.is_symlink() or not path.is_file():
        _fail()
    if hashlib.sha256(path.read_bytes()).hexdigest() != _sha256(record["sha256"]):
        _fail()


def _verify_identities(value: object) -> Mapping[str, object]:
    record = _mapping(value, {
        "final_sha", "final_tree", "design_sha256", "plan_sha256",
        "commit_range_start", "commit_range_end", "develop_merge_sha",
        "release_merge_sha", "final_develop_sha",
    })
    final_sha = _sha1(record["final_sha"])
    _sha1(record["final_tree"])
    _sha256(record["design_sha256"])
    _sha256(record["plan_sha256"])
    _sha1(record["commit_range_start"])
    if _sha1(record["commit_range_end"]) != final_sha:
        _fail()
    for field in ("develop_merge_sha", "release_merge_sha", "final_develop_sha"):
        _sha1(record[field])
    return record


def _verify_locks(value: object) -> None:
    record = _mapping(value, {
        "policy_sha256", "image_lock_sha256", "input_sha256", "lock_sha256",
        "compiler", "local_python", "docker_python",
    })
    _sha256(record["policy_sha256"])
    _sha256(record["image_lock_sha256"])
    for field in ("input_sha256", "lock_sha256"):
        entries = record[field]
        if not isinstance(entries, dict) or not entries:
            _fail()
        for path, digest in entries.items():
            selected = Path(_text(path))
            if selected.is_absolute() or ".." in selected.parts:
                _fail()
            _sha256(digest)
    compiler = _mapping(
        record["compiler"], {"distribution", "version", "executable_sha256"}
    )
    if compiler["distribution"] != "uv" or compiler["version"] != "0.11.19":
        _fail()
    _sha256(compiler["executable_sha256"])
    local = _mapping(record["local_python"], {"version", "executable"})
    if local["version"] != "3.11.15" or not Path(_text(local["executable"])).is_absolute():
        _fail()
    docker = _mapping(record["docker_python"], {"version", "architecture"})
    if docker != {"version": "3.11.10", "architecture": "arm64"}:
        _fail()


def _verify_tests(value: object, repo: Path) -> None:
    tests = _mapping(value, {"repository", "nnx", "qat"})
    for name, minimum in (("repository", 1), ("nnx", 1), ("qat", 1)):
        record = _mapping(
            tests[name], {"total", "failures", "errors", "skips", "path", "sha256"}
        )
        _integer(record["total"], minimum=minimum)
        if any(_integer(record[field]) != 0 for field in ("failures", "errors", "skips")):
            _fail()
        _evidence(record, repo)
    if tests["qat"]["total"] != 1:
        _fail()


def _verify_advisory(value: object, repo: Path) -> None:
    record = _mapping(
        value, {"result", "errors", "notices", "surfaces", "non_pypi", "path", "sha256"}
    )
    if record["result"] != "accepted" or record["errors"] != [] or record["notices"] != []:
        _fail()
    surfaces = _mapping(record["surfaces"], {"combined-runtime", "torch", "docs", "atlas"})
    for surface in surfaces.values():
        selected = _mapping(surface, {"findings"})
        _integer(selected["findings"])
    observed: set[tuple[str, str, str]] = set()
    for entry in _sequence(record["non_pypi"], length=4):
        selected = _mapping(entry, {"package", "version", "source"})
        observed.add(tuple(_text(selected[field]) for field in ("package", "version", "source")))
    if observed != {
        ("en-core-web-sm", "3.8.0", "direct-url"),
        ("pyg-lib", "0.8.0", "pyg-flat-index"),
        ("torch-scatter", "2.1.2", "pyg-flat-index"),
        ("torch-sparse", "0.6.18", "pyg-flat-index"),
    }:
        _fail()
    _evidence(record, repo)


def _verify_tiers(value: object) -> None:
    tiers = _mapping(value, {"a", "b", "c"})
    expected = {"a": (18, 210), "b": (6, 75), "c": (4, 56)}
    for name, (notebooks, cells) in expected.items():
        record = _mapping(tiers[name], {"notebooks", "code_cells", "root_sha256"})
        if _integer(record["notebooks"]) != notebooks or _integer(record["code_cells"]) != cells:
            _fail()
        _sha256(record["root_sha256"])


def _verify_docker(value: object, repo: Path) -> None:
    record = _mapping(
        value, {"image_id", "architecture", "base_reference", "probes", "path", "sha256"}
    )
    if _IMAGE_ID.fullmatch(_text(record["image_id"])) is None:
        _fail()
    if record["architecture"] != "arm64":
        _fail()
    base = _text(record["base_reference"])
    if not base.startswith("quay.io/jupyter/datascience-notebook:python-3.11@sha256:"):
        _fail()
    if _SHA256.fullmatch(base.rpartition(":")[2]) is None:
        _fail()
    probes = _sequence(record["probes"], length=3)
    observed: set[str] = set()
    for probe in probes:
        selected = _mapping(probe, {"name", "returncode"})
        observed.add(_text(selected["name"]))
        if _integer(selected["returncode"]) != 0:
            _fail()
    if observed != {"pip-check", "torch-stack", "nnx"}:
        _fail()
    _evidence(record, repo)


def _github_url(value: object) -> str:
    selected = _text(value)
    if _GITHUB_URL.fullmatch(selected) is None:
        _fail()
    return selected


def _verify_pull_requests(value: object, identities: Mapping[str, object]) -> None:
    records = _mapping(value, {"feature", "release", "sync"})
    for record in records.values():
        selected = _mapping(
            record, {"url", "source_sha", "merge_sha", "run_urls", "log_sha256"}
        )
        _github_url(selected["url"])
        _sha1(selected["source_sha"])
        _sha1(selected["merge_sha"])
        run_urls = _sequence(selected["run_urls"])
        if not run_urls:
            _fail()
        for url in run_urls:
            _github_url(url)
        _sha256(selected["log_sha256"])
    expected = {
        "feature": (identities["final_sha"], identities["develop_merge_sha"]),
        "release": (identities["develop_merge_sha"], identities["release_merge_sha"]),
        "sync": (identities["release_merge_sha"], identities["final_develop_sha"]),
    }
    for role, (source_sha, merge_sha) in expected.items():
        if records[role]["source_sha"] != source_sha or records[role]["merge_sha"] != merge_sha:
            _fail()


def _verify_publication(value: object, repo: Path) -> None:
    record = _mapping(
        value,
        {"pages_url", "wiki_url", "pages_run_url", "live_sha256", "path", "sha256"},
    )
    if record["pages_url"] != "https://thekaveh.github.io/ml-eng-lab/":
        _fail()
    if record["wiki_url"] != "https://github.com/thekaveh/ml-eng-lab/wiki":
        _fail()
    _github_url(record["pages_run_url"])
    _sha256(record["live_sha256"])
    _evidence(record, repo)


def _verify_ruleset(value: object) -> None:
    record = _mapping(value, {"before_sha256", "after_sha256", "contexts"})
    before = _sha256(record["before_sha256"])
    if _sha256(record["after_sha256"]) != before:
        _fail()
    if record["contexts"] != [
        "atlas-consumer-policy", "dependency-audit", "pytest-repository"
    ]:
        _fail()


def _verify_issues(value: object) -> None:
    records = _mapping(value, {"64", "65", "66"})
    for record in records.values():
        selected = _mapping(record, {"state", "before_sha256", "after_sha256"})
        if selected["state"] != "OPEN":
            _fail()
        before = _sha256(selected["before_sha256"])
        if _sha256(selected["after_sha256"]) != before:
            _fail()


def _verify_cleanup(value: object) -> None:
    record = _mapping(value, {
        "feature_refs_absent", "temporary_root_absent", "image_absent",
        "noncompleted_runs", "open_scoped_prs",
    })
    for field in ("feature_refs_absent", "temporary_root_absent", "image_absent"):
        if record[field] is not True:
            _fail()
    if _integer(record["noncompleted_runs"]) != 0 or _integer(record["open_scoped_prs"]) != 0:
        _fail()


def verify_issue63_report(report_path: Path | str, *, repo: Path = REPO_ROOT) -> None:
    """Validate one fenced schema-1 report and every retained evidence digest."""
    try:
        path = Path(report_path)
        source = path.read_text(encoding="utf-8")
        matches = tuple(_JSON_FENCE.finditer(source))
        if len(matches) != 1:
            _fail()
        payload = json.loads(matches[0].group("payload"))
        report = _mapping(payload, {
            "schema_version", "identities", "locks", "tests", "advisory", "tiers",
            "docker", "pull_requests", "publication", "ruleset", "issues", "cleanup",
        })
        if report["schema_version"] != 1:
            _fail()
        identities = _verify_identities(report["identities"])
        _verify_locks(report["locks"])
        _verify_tests(report["tests"], repo)
        _verify_advisory(report["advisory"], repo)
        _verify_tiers(report["tiers"])
        _verify_docker(report["docker"], repo)
        _verify_pull_requests(report["pull_requests"], identities)
        _verify_publication(report["publication"], repo)
        _verify_ruleset(report["ruleset"])
        _verify_issues(report["issues"])
        _verify_cleanup(report["cleanup"])
    except Issue63ReportError:
        raise
    except BaseException:
        _fail()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    try:
        verify_issue63_report(args.report, repo=args.repo)
    except Issue63ReportError as error:
        print(error, file=sys.stderr)
        return 1
    print("Issue 63 qualification report verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
