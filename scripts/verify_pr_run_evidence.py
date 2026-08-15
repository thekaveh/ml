"""Verify GitHub pull-request source-head metadata and synthetic-merge checkout evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_SHA = re.compile(r"[0-9a-f]{40}")
_REPOSITORY = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")


class PrRunEvidenceError(RuntimeError):
    """Pull-request run evidence does not satisfy the dual-identity contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PrRunEvidenceError(message)


def _mapping(value: object, message: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), message)
    return value


def _sequence(value: object, message: str) -> Sequence[Any]:
    _require(isinstance(value, Sequence) and not isinstance(value, (str, bytes)), message)
    return value


def _string(value: object, message: str) -> str:
    _require(isinstance(value, str) and bool(value), message)
    return value


def _integer(value: object, message: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool), message)
    return value


def _timestamp(value: object, message: str) -> str:
    text = _string(value, message)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise PrRunEvidenceError(message) from error
    _require(parsed.tzinfo == timezone.utc and text.endswith("Z"), message)
    return text


def _sha(value: object, message: str) -> str:
    text = _string(value, message)
    _require(_SHA.fullmatch(text) is not None, message)
    return text


def _url(value: object, expected_prefix: str, message: str) -> str:
    text = _string(value, message)
    _require(text.startswith(expected_prefix), message)
    return text


def _checkout_evidence(
    log: str,
    *,
    jobs: Mapping[str, str],
    pr_number: int,
    merge_sha: str,
) -> None:
    _require(isinstance(log, str) and bool(log), "checkout log")
    merge_ref = f"refs/remotes/pull/{pr_number}/merge"
    fetch = f"+{merge_sha}:{merge_ref}"
    checkout = f"git checkout --progress --force {merge_ref}"
    full_head = re.compile(rf"\t[^\t\r\n]*Z {re.escape(merge_sha)}\r?$")
    for job, conclusion in jobs.items():
        if conclusion == "skipped":
            continue
        lines = tuple(line for line in log.splitlines() if line.startswith(f"{job}\t"))
        _require(bool(lines), f"{job} checkout log")
        _require(any(fetch in line for line in lines), f"{job} merge fetch")
        _require(any(checkout in line for line in lines), f"{job} merge checkout")
        _require(any(full_head.search(line) for line in lines), f"{job} full checkout SHA")


def _pull_request_association(
    rest: Mapping[str, Any],
    *,
    expected_repo: str,
    expected_pr_number: int,
    expected_head_ref: str,
    expected_head_sha: str,
    expected_base_ref: str,
    expected_base_sha: str,
) -> None:
    associations = _sequence(rest.get("pull_requests"), "run PR associations")
    _require(len(associations) == 1, "exact PR association")
    association = _mapping(associations[0], "PR association")
    _require(association.get("number") == expected_pr_number, "PR association number")
    head = _mapping(association.get("head"), "PR association head")
    base = _mapping(association.get("base"), "PR association base")
    _require(head.get("ref") == expected_head_ref, "PR association head ref")
    _require(head.get("sha") == expected_head_sha, "PR association head SHA")
    _require(base.get("ref") == expected_base_ref, "PR association base ref")
    _require(base.get("sha") == expected_base_sha, "PR association base SHA")
    head_repo = _mapping(head.get("repo"), "PR association head repository")
    base_repo = _mapping(base.get("repo"), "PR association base repository")
    run_repo = _mapping(rest.get("head_repository"), "run head repository")
    repository_id = _integer(run_repo.get("id"), "run repository id")
    repository_name = expected_repo.rsplit("/", 1)[1]
    repository_url = f"https://api.github.com/repos/{expected_repo}"
    for label, repository in (("head", head_repo), ("base", base_repo)):
        _require(repository.get("id") == repository_id, f"{label} repository id")
        _require(repository.get("name") == repository_name, f"{label} repository name")
        _require(repository.get("url") == repository_url, f"{label} repository URL")


def verify_pr_run_evidence(
    *,
    pr: object,
    checks: object,
    run_summaries: object,
    run_records: object,
    expected_repo: str,
    expected_pr_number: int,
    expected_head_ref: str,
    expected_head_sha: str,
    expected_base_ref: str,
    expected_base_sha: str,
    expected_merge_sha: str,
    merge_parents: object,
    merge_tree: object,
    head_tree: object,
) -> dict[str, object]:
    """Return schema-2 evidence only when both GitHub PR identities are exact."""
    _require(_REPOSITORY.fullmatch(expected_repo) is not None, "expected repository")
    _require(expected_pr_number > 0, "expected PR number")
    for value, label in (
        (expected_head_sha, "expected head SHA"),
        (expected_base_sha, "expected base SHA"),
        (expected_merge_sha, "expected merge SHA"),
    ):
        _sha(value, label)
    _require(expected_head_sha != expected_merge_sha, "distinct source and merge SHAs")

    check_rows = tuple(
        _mapping(item, "PR check") for item in _sequence(checks, "PR checks")
    )
    _require(bool(check_rows), "nonempty PR checks")
    checks_by_link: dict[str, Mapping[str, Any]] = {}
    for check in check_rows:
        _require(set(check) == {"name", "state", "bucket", "link"}, "PR check schema")
        link = _string(check.get("link"), "PR check link")
        _require(link not in checks_by_link, "unique PR check link")
        checks_by_link[link] = check

    pr_data = _mapping(pr, "PR document")
    _require(pr_data.get("number") == expected_pr_number, "PR number")
    _require(pr_data.get("state") == "OPEN", "PR state")
    _require(pr_data.get("isDraft") is False, "PR draft state")
    expected_pr_url = f"https://github.com/{expected_repo}/pull/{expected_pr_number}"
    _require(pr_data.get("url") == expected_pr_url, "PR URL")
    _require(pr_data.get("headRefName") == expected_head_ref, "PR head ref")
    _require(pr_data.get("headRefOid") == expected_head_sha, "PR head SHA")
    head_repository = _mapping(pr_data.get("headRepository"), "PR head repository")
    _require(head_repository.get("nameWithOwner") == expected_repo, "PR head repository name")
    _require(pr_data.get("baseRefName") == expected_base_ref, "PR base ref")
    _require(pr_data.get("baseRefOid") == expected_base_sha, "PR base SHA")
    potential = _mapping(pr_data.get("potentialMergeCommit"), "PR potential merge commit")
    _require(potential.get("oid") == expected_merge_sha, "PR potential merge SHA")

    parents = tuple(_sequence(merge_parents, "merge parents"))
    _require(parents == (expected_base_sha, expected_head_sha), "merge parent order")
    merge_tree_value = _sha(merge_tree, "merge tree")
    head_tree_value = _sha(head_tree, "head tree")
    _require(merge_tree_value == head_tree_value, "merge/head tree identity")

    summaries = tuple(_mapping(item, "run summary") for item in _sequence(
        run_summaries, "run summaries"
    ))
    records = tuple(_mapping(item, "run record") for item in _sequence(
        run_records, "run records"
    ))
    _require(bool(summaries) and len(summaries) == len(records), "run inventory")
    summary_by_id: dict[int, Mapping[str, Any]] = {}
    for summary in summaries:
        run_id = _integer(summary.get("databaseId"), "summary run id")
        _require(run_id not in summary_by_id, "unique summary run id")
        summary_by_id[run_id] = summary

    selected_workflows: set[str] = set()
    record_ids: set[int] = set()
    output_runs: list[dict[str, object]] = []
    contaminating_runs: list[dict[str, object]] = []
    selected_ci_jobs: dict[str, str] | None = None
    contaminating_ci_jobs: list[dict[str, str]] = []
    for record in records:
        _require(
            set(record) == {
                "workflow", "jobs", "action", "selected", "rest", "view", "log",
            },
            "run record schema",
        )
        workflow = _string(record.get("workflow"), "workflow name")
        selected = record.get("selected")
        _require(isinstance(selected, bool), "run selection")
        action = record.get("action")
        if workflow == "CI":
            action = _string(action, "CI action")
            _require(
                action in {"opened", "labeled", "synchronize"},
                "supported CI action",
            )
        else:
            _require(action is None, "non-CI action")
            _require(selected is True, "non-CI run selected")
        if selected:
            _require(workflow not in selected_workflows, "unique selected workflow")
            selected_workflows.add(workflow)
        else:
            _require(workflow == "CI" and action == "opened", "opened CI contaminant")
        expected_jobs_raw = _mapping(record.get("jobs"), "job policy")
        expected_jobs = {
            _string(name, "job name"): _string(conclusion, "job conclusion")
            for name, conclusion in expected_jobs_raw.items()
        }
        _require(bool(expected_jobs), "nonempty job policy")
        _require(set(expected_jobs.values()) <= {"success", "skipped"}, "job conclusions")
        _require("success" in expected_jobs.values(), "successful job policy")

        rest = _mapping(record.get("rest"), "REST run")
        view = _mapping(record.get("view"), "run view")
        run_id = _integer(rest.get("id"), "REST run id")
        _require(run_id not in record_ids, "unique run id")
        record_ids.add(run_id)
        summary = summary_by_id.get(run_id)
        _require(summary is not None, "run summary membership")
        _require(view.get("databaseId") == run_id, "run view id")
        _require(view.get("workflowName") == workflow, "run view workflow")
        _require(summary.get("workflowName") == workflow, "summary workflow")
        for source, label, snake_case in (
            (summary, "summary", False),
            (view, "view", False),
            (rest, "REST", True),
        ):
            head_key = "head_sha" if snake_case else "headSha"
            event_key = "event"
            status_key = "status"
            conclusion_key = "conclusion"
            url_key = "html_url" if snake_case else "url"
            _require(source.get(head_key) == expected_head_sha, f"{label} source head SHA")
            _require(source.get(event_key) == "pull_request", f"{label} event")
            _require(source.get(status_key) == "completed", f"{label} status")
            _require(source.get(conclusion_key) == "success", f"{label} conclusion")
            _url(source.get(url_key), f"https://github.com/{expected_repo}/actions/runs/", f"{label} URL")
        _require(summary.get("headBranch") == expected_head_ref, "summary head ref")
        _require(rest.get("head_branch") == expected_head_ref, "REST head ref")
        rest_repository = _mapping(rest.get("head_repository"), "REST head repository")
        _require(rest_repository.get("full_name") == expected_repo, "REST head repository name")
        run_attempt = _integer(rest.get("run_attempt"), "REST run attempt")
        _require(run_attempt == 1, "first run attempt")
        created_at = _timestamp(rest.get("created_at"), "REST created at")
        _require(summary.get("createdAt") == created_at, "summary created at")
        display_title = _string(rest.get("display_title"), "REST display title")
        _require(summary.get("displayTitle") == display_title, "summary display title")
        if workflow == "CI":
            _require(
                display_title
                == f"CI / pull_request / {action} / PR {expected_pr_number}",
                "CI action display title",
            )
        _pull_request_association(
            rest,
            expected_repo=expected_repo,
            expected_pr_number=expected_pr_number,
            expected_head_ref=expected_head_ref,
            expected_head_sha=expected_head_sha,
            expected_base_ref=expected_base_ref,
            expected_base_sha=expected_base_sha,
        )

        jobs = tuple(
            _mapping(item, "run job") for item in _sequence(view.get("jobs"), "run jobs")
        )
        jobs_by_name: dict[str, Mapping[str, Any]] = {}
        for job in jobs:
            name = _string(job.get("name"), "run job name")
            _require(name not in jobs_by_name, "unique run job")
            jobs_by_name[name] = job
        _require(set(jobs_by_name) == set(expected_jobs), "exact job set")
        check_urls: dict[str, str] = {}
        for name, conclusion in expected_jobs.items():
            job = jobs_by_name[name]
            _require(job.get("status") == "completed", f"{name} status")
            _require(job.get("conclusion") == conclusion, f"{name} conclusion")
            job_url = _url(
                job.get("url"),
                f"https://github.com/{expected_repo}/actions/runs/{run_id}/job/",
                f"{name} URL",
            )
            if selected:
                selected_check = checks_by_link.get(job_url)
                _require(selected_check is not None, f"{name} selected PR check")
                _require(selected_check.get("name") == name, f"{name} PR check name")
                expected_state, expected_bucket = {
                    "success": ("SUCCESS", "pass"),
                    "skipped": ("SKIPPED", "skipping"),
                }[conclusion]
                _require(
                    selected_check.get("state") == expected_state,
                    f"{name} PR check state",
                )
                _require(
                    selected_check.get("bucket") == expected_bucket,
                    f"{name} PR check bucket",
                )
                check_urls[name] = job_url

        log = _string(record.get("log"), "run log")
        _checkout_evidence(
            log,
            jobs=expected_jobs,
            pr_number=expected_pr_number,
            merge_sha=expected_merge_sha,
        )
        if selected:
            if workflow == "CI":
                tier_b = expected_jobs.get("smoke-tier-b")
                if tier_b == "success":
                    _require(action in {"labeled", "synchronize"}, "selected Tier B CI action")
                elif tier_b == "skipped":
                    _require(action in {"opened", "synchronize"}, "selected skipped Tier B CI action")
                else:
                    raise PrRunEvidenceError("selected CI Tier B policy")
                selected_ci_jobs = expected_jobs
            output_runs.append(
                {
                    "database_id": run_id,
                    "workflow": workflow,
                    "url": rest["html_url"],
                    "event": "pull_request",
                    "action": action,
                    "created_at": created_at,
                    "run_attempt": run_attempt,
                    "metadata_head_sha": expected_head_sha,
                    "checkout_sha": expected_merge_sha,
                    "jobs": dict(sorted(expected_jobs.items())),
                    "check_urls": dict(sorted(check_urls.items())),
                    "log_sha256": hashlib.sha256(log.encode("utf-8")).hexdigest(),
                }
            )
        else:
            contaminating_ci_jobs.append(expected_jobs)
            contaminating_runs.append(
                {
                    "database_id": run_id,
                    "workflow": workflow,
                    "url": rest["html_url"],
                    "event": "pull_request",
                    "action": action,
                    "created_at": created_at,
                    "run_attempt": run_attempt,
                    "metadata_head_sha": expected_head_sha,
                    "checkout_sha": expected_merge_sha,
                    "jobs": dict(sorted(expected_jobs.items())),
                    "log_sha256": hashlib.sha256(log.encode("utf-8")).hexdigest(),
                }
            )

    _require(record_ids == set(summary_by_id), "exact run ids")
    _require(selected_ci_jobs is not None, "selected CI run")
    if selected_ci_jobs.get("smoke-tier-b") == "success":
        _require(len(contaminating_ci_jobs) <= 1, "unique opened CI contaminant")
        opened_policy = dict(selected_ci_jobs)
        opened_policy["smoke-tier-b"] = "skipped"
        _require(
            all(policy == opened_policy for policy in contaminating_ci_jobs),
            "opened CI contaminant policy",
        )
    else:
        _require(not contaminating_ci_jobs, "unexpected CI contaminant")
    _require(
        selected_workflows == {"CI", "Docs gate", "Atlas contract"}
        or selected_workflows == {"CI"},
        "selected workflow set",
    )
    return {
        "schema": 2,
        "pull_request": {"number": expected_pr_number, "url": expected_pr_url},
        "source_head": {
            "repository": expected_repo,
            "ref": expected_head_ref,
            "sha": expected_head_sha,
        },
        "base": {
            "repository": expected_repo,
            "ref": expected_base_ref,
            "sha": expected_base_sha,
        },
        "synthetic_merge": {
            "sha": expected_merge_sha,
            "parents": [expected_base_sha, expected_head_sha],
            "tree": merge_tree_value,
        },
        "runs": sorted(output_runs, key=lambda item: str(item["workflow"])),
        "contaminating_runs": sorted(
            contaminating_runs, key=lambda item: int(item["database_id"])
        ),
    }


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_identity(
    root: Path,
    merge_sha: str,
    head_sha: str,
) -> tuple[tuple[str, ...], str, str]:
    revision = subprocess.check_output(
        ("git", "-C", str(root), "rev-list", "--parents", "-n", "1", merge_sha),
        text=True,
    ).strip().split()
    _require(revision and revision[0] == merge_sha, "merge revision")
    merge_tree = subprocess.check_output(
        ("git", "-C", str(root), "rev-parse", f"{merge_sha}^{{tree}}"), text=True
    ).strip()
    head_tree = subprocess.check_output(
        ("git", "-C", str(root), "rev-parse", f"{head_sha}^{{tree}}"), text=True
    ).strip()
    return tuple(revision[1:]), merge_tree, head_tree


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pr-json", type=Path, required=True)
    parser.add_argument("--checks-json", type=Path, required=True)
    parser.add_argument("--runs-json", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--git-root", type=Path, required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--pr-number", type=int, required=True)
    parser.add_argument("--head-ref", required=True)
    parser.add_argument("--head-sha", required=True)
    parser.add_argument("--base-ref", required=True)
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--merge-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        for path in (
            args.pr_json,
            args.checks_json,
            args.runs_json,
            args.manifest,
            args.git_root,
            args.output,
        ):
            _require(path.is_absolute(), "absolute paths")
        manifest = _mapping(_load_json(args.manifest), "manifest")
        _require(set(manifest) == {"schema", "runs"} and manifest["schema"] == 2, "manifest schema")
        records: list[dict[str, object]] = []
        for item in _sequence(manifest["runs"], "manifest runs"):
            record = _mapping(item, "manifest run")
            _require(
                set(record) == {
                    "workflow",
                    "jobs",
                    "action",
                    "selected",
                    "rest_path",
                    "view_path",
                    "log_path",
                },
                "manifest run schema",
            )
            rest_path = Path(_string(record["rest_path"], "REST path"))
            view_path = Path(_string(record["view_path"], "view path"))
            log_path = Path(_string(record["log_path"], "log path"))
            _require(all(path.is_absolute() for path in (rest_path, view_path, log_path)), "absolute evidence paths")
            records.append(
                {
                    "workflow": record["workflow"],
                    "jobs": record["jobs"],
                    "action": record["action"],
                    "selected": record["selected"],
                    "rest": _load_json(rest_path),
                    "view": _load_json(view_path),
                    "log": log_path.read_text(encoding="utf-8"),
                }
            )
        parents, merge_tree, head_tree = _git_identity(
            args.git_root, args.merge_sha, args.head_sha
        )
        evidence = verify_pr_run_evidence(
            pr=_load_json(args.pr_json),
            checks=_load_json(args.checks_json),
            run_summaries=_load_json(args.runs_json),
            run_records=records,
            expected_repo=args.repo,
            expected_pr_number=args.pr_number,
            expected_head_ref=args.head_ref,
            expected_head_sha=args.head_sha,
            expected_base_ref=args.base_ref,
            expected_base_sha=args.base_sha,
            expected_merge_sha=args.merge_sha,
            merge_parents=parents,
            merge_tree=merge_tree,
            head_tree=head_tree,
        )
        args.output.write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except (OSError, UnicodeError, json.JSONDecodeError, subprocess.SubprocessError, PrRunEvidenceError):
        print("PR run evidence verification failed", file=sys.stderr)
        return 1
    print(f"PR run evidence verification ok: {args.pr_number}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
