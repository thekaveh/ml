from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest


HEAD_SHA = "1" * 40
BASE_SHA = "2" * 40
MERGE_SHA = "3" * 40
TREE_SHA = "4" * 40
REPO = "thekaveh/ml-eng-lab"
PR_NUMBER = 110
HEAD_REF = "codex/issue-62-torch-stack-upgrade"
BASE_REF = "develop"


def _checkout_log(jobs: tuple[str, ...]) -> str:
    ref = f"refs/remotes/pull/{PR_NUMBER}/merge"
    return "\n".join(
        line
        for job in jobs
        for line in (
            f"{job}\tCheckout\t2026-08-15T00:00:00Z "
            f"[command]/usr/bin/git fetch origin +{MERGE_SHA}:{ref}",
            f"{job}\tCheckout\t2026-08-15T00:00:01Z "
            f"[command]/usr/bin/git checkout --progress --force {ref}",
            f"{job}\tCheckout\t2026-08-15T00:00:02Z {MERGE_SHA}",
        )
    ) + "\n"


def _pull_request() -> dict[str, object]:
    return {
        "number": PR_NUMBER,
        "state": "OPEN",
        "isDraft": False,
        "url": f"https://github.com/{REPO}/pull/{PR_NUMBER}",
        "headRefName": HEAD_REF,
        "headRefOid": HEAD_SHA,
        "headRepository": {"id": "R_repo", "nameWithOwner": REPO},
        "baseRefName": BASE_REF,
        "baseRefOid": BASE_SHA,
        "potentialMergeCommit": {"oid": MERGE_SHA},
    }


def _rest_run(run_id: int) -> dict[str, object]:
    return {
        "id": run_id,
        "event": "pull_request",
        "head_sha": HEAD_SHA,
        "head_branch": HEAD_REF,
        "head_repository": {"id": 619028479, "full_name": REPO},
        "run_attempt": 1,
        "status": "completed",
        "conclusion": "success",
        "html_url": f"https://github.com/{REPO}/actions/runs/{run_id}",
        "pull_requests": [
            {
                "number": PR_NUMBER,
                "head": {
                    "ref": HEAD_REF,
                    "sha": HEAD_SHA,
                    "repo": {
                        "id": 619028479,
                        "name": "ml-eng-lab",
                        "url": f"https://api.github.com/repos/{REPO}",
                    },
                },
                "base": {
                    "ref": BASE_REF,
                    "sha": BASE_SHA,
                    "repo": {
                        "id": 619028479,
                        "name": "ml-eng-lab",
                        "url": f"https://api.github.com/repos/{REPO}",
                    },
                },
            }
        ],
    }


def _run_record(
    run_id: int,
    workflow: str,
    jobs: dict[str, str],
) -> dict[str, object]:
    return {
        "workflow": workflow,
        "jobs": jobs,
        "rest": _rest_run(run_id),
        "view": {
            "databaseId": run_id,
            "workflowName": workflow,
            "event": "pull_request",
            "headSha": HEAD_SHA,
            "status": "completed",
            "conclusion": "success",
            "url": f"https://github.com/{REPO}/actions/runs/{run_id}",
            "jobs": [
                {
                    "name": name,
                    "status": "completed",
                    "conclusion": conclusion,
                    "url": f"https://github.com/{REPO}/actions/runs/{run_id}/job/{index}",
                }
                for index, (name, conclusion) in enumerate(jobs.items(), start=1)
            ],
        },
        "log": _checkout_log(
            tuple(name for name, conclusion in jobs.items() if conclusion == "success")
        ),
    }


def _control() -> dict[str, object]:
    records = [
        _run_record(
            101,
            "CI",
            {
                "pytest-repository": "success",
                "verify-repo": "success",
                "smoke-tier-c": "skipped",
            },
        ),
        _run_record(102, "Docs gate", {"check": "success"}),
        _run_record(103, "Atlas contract", {"atlas-contract": "success"}),
    ]
    return {
        "pr": _pull_request(),
        "run_summaries": [
            {
                "databaseId": record["rest"]["id"],
                "workflowName": record["workflow"],
                "event": "pull_request",
                "headSha": HEAD_SHA,
                "headBranch": HEAD_REF,
                "status": "completed",
                "conclusion": "success",
                "url": record["rest"]["html_url"],
            }
            for record in records
        ],
        "run_records": records,
        "merge_parents": (BASE_SHA, HEAD_SHA),
        "merge_tree": TREE_SHA,
        "head_tree": TREE_SHA,
    }


def _verify(control: dict[str, object]) -> dict[str, object]:
    from scripts.verify_pr_run_evidence import verify_pr_run_evidence

    return verify_pr_run_evidence(
        pr=control["pr"],
        run_summaries=control["run_summaries"],
        run_records=control["run_records"],
        expected_repo=REPO,
        expected_pr_number=PR_NUMBER,
        expected_head_ref=HEAD_REF,
        expected_head_sha=HEAD_SHA,
        expected_base_ref=BASE_REF,
        expected_base_sha=BASE_SHA,
        expected_merge_sha=MERGE_SHA,
        merge_parents=control["merge_parents"],
        merge_tree=control["merge_tree"],
        head_tree=control["head_tree"],
    )


def test_pr_run_evidence_binds_source_metadata_and_synthetic_checkout() -> None:
    evidence = _verify(_control())

    assert evidence["schema"] == 1
    assert evidence["pull_request"] == {
        "number": PR_NUMBER,
        "url": f"https://github.com/{REPO}/pull/{PR_NUMBER}",
    }
    assert evidence["source_head"] == {
        "repository": REPO,
        "ref": HEAD_REF,
        "sha": HEAD_SHA,
    }
    assert evidence["base"] == {
        "repository": REPO,
        "ref": BASE_REF,
        "sha": BASE_SHA,
    }
    assert evidence["synthetic_merge"] == {
        "sha": MERGE_SHA,
        "parents": [BASE_SHA, HEAD_SHA],
        "tree": TREE_SHA,
    }
    assert {run["workflow"] for run in evidence["runs"]} == {
        "CI",
        "Docs gate",
        "Atlas contract",
    }
    assert all(run["metadata_head_sha"] == HEAD_SHA for run in evidence["runs"])
    assert all(run["checkout_sha"] == MERGE_SHA for run in evidence["runs"])


@pytest.mark.parametrize(
    ("mutation", "value"),
    (
        ("pr-head-sha", MERGE_SHA),
        ("pr-base-sha", "5" * 40),
        ("pr-merge-sha", "6" * 40),
        ("pr-head-ref", "wrong-head"),
        ("pr-base-ref", "main"),
        ("pr-head-repo", "other/repo"),
        ("summary-head-sha", MERGE_SHA),
        ("view-head-sha", MERGE_SHA),
        ("rest-head-sha", MERGE_SHA),
        ("rest-head-ref", "wrong-head"),
        ("rest-head-repo", "other/repo"),
        ("association-number", 111),
        ("association-head-sha", MERGE_SHA),
        ("association-base-sha", "7" * 40),
        ("association-head-ref", "wrong-head"),
        ("association-base-ref", "main"),
        ("run-attempt", 2),
        ("run-event", "push"),
        ("run-status", "in_progress"),
        ("run-conclusion", "failure"),
        ("missing-workflow", None),
        ("missing-job", None),
        ("extra-job", None),
        ("wrong-job-conclusion", "neutral"),
        ("missing-log-job", None),
        ("wrong-log-fetch", "8" * 40),
        ("wrong-log-checkout", "refs/remotes/pull/111/merge"),
        ("missing-log-full-head", None),
        ("wrong-merge-parent", "9" * 40),
        ("wrong-merge-tree", "a" * 40),
    ),
)
def test_pr_run_evidence_rejects_identity_and_checkout_mutations(
    mutation: str,
    value: object,
) -> None:
    control = copy.deepcopy(_control())
    pr = control["pr"]
    summaries = control["run_summaries"]
    records = control["run_records"]
    first = records[0]
    rest = first["rest"]
    view = first["view"]
    association = rest["pull_requests"][0]

    if mutation == "pr-head-sha":
        pr["headRefOid"] = value
    elif mutation == "pr-base-sha":
        pr["baseRefOid"] = value
    elif mutation == "pr-merge-sha":
        pr["potentialMergeCommit"]["oid"] = value
    elif mutation == "pr-head-ref":
        pr["headRefName"] = value
    elif mutation == "pr-base-ref":
        pr["baseRefName"] = value
    elif mutation == "pr-head-repo":
        pr["headRepository"]["nameWithOwner"] = value
    elif mutation == "summary-head-sha":
        summaries[0]["headSha"] = value
    elif mutation == "view-head-sha":
        view["headSha"] = value
    elif mutation == "rest-head-sha":
        rest["head_sha"] = value
    elif mutation == "rest-head-ref":
        rest["head_branch"] = value
    elif mutation == "rest-head-repo":
        rest["head_repository"]["full_name"] = value
    elif mutation == "association-number":
        association["number"] = value
    elif mutation == "association-head-sha":
        association["head"]["sha"] = value
    elif mutation == "association-base-sha":
        association["base"]["sha"] = value
    elif mutation == "association-head-ref":
        association["head"]["ref"] = value
    elif mutation == "association-base-ref":
        association["base"]["ref"] = value
    elif mutation == "run-attempt":
        rest["run_attempt"] = value
    elif mutation == "run-event":
        rest["event"] = value
    elif mutation == "run-status":
        rest["status"] = value
    elif mutation == "run-conclusion":
        rest["conclusion"] = value
    elif mutation == "missing-workflow":
        records.pop()
    elif mutation == "missing-job":
        first["jobs"].pop("verify-repo")
    elif mutation == "extra-job":
        first["jobs"]["unexpected"] = "success"
    elif mutation == "wrong-job-conclusion":
        first["jobs"]["verify-repo"] = value
    elif mutation == "missing-log-job":
        first["log"] = "\n".join(
            line
            for line in first["log"].splitlines()
            if not line.startswith("verify-repo\t")
        )
    elif mutation == "wrong-log-fetch":
        first["log"] = first["log"].replace(
            f"+{MERGE_SHA}:refs/remotes/pull/{PR_NUMBER}/merge",
            f"+{value}:refs/remotes/pull/{PR_NUMBER}/merge",
            1,
        )
    elif mutation == "wrong-log-checkout":
        first["log"] = first["log"].replace(
            f"refs/remotes/pull/{PR_NUMBER}/merge",
            str(value),
            2,
        )
    elif mutation == "missing-log-full-head":
        first["log"] = "\n".join(
            line for line in first["log"].splitlines() if not line.endswith(MERGE_SHA)
        )
    elif mutation == "wrong-merge-parent":
        control["merge_parents"] = (value, HEAD_SHA)
    elif mutation == "wrong-merge-tree":
        control["merge_tree"] = value
    else:  # pragma: no cover - the parameter table is exhaustive
        raise AssertionError(mutation)

    from scripts.verify_pr_run_evidence import PrRunEvidenceError

    with pytest.raises(PrRunEvidenceError):
        _verify(control)


def test_cli_writes_validated_evidence_only_after_all_inputs_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import verify_pr_run_evidence

    control = _control()
    pr_path = tmp_path / "pr.json"
    runs_path = tmp_path / "runs.json"
    manifest_path = tmp_path / "manifest.json"
    output_path = tmp_path / "evidence.json"
    pr_path.write_text(json.dumps(control["pr"]), encoding="utf-8")
    runs_path.write_text(json.dumps(control["run_summaries"]), encoding="utf-8")

    manifest_runs = []
    for index, record in enumerate(control["run_records"]):
        rest_path = tmp_path / f"rest-{index}.json"
        view_path = tmp_path / f"view-{index}.json"
        log_path = tmp_path / f"log-{index}.txt"
        rest_path.write_text(json.dumps(record["rest"]), encoding="utf-8")
        view_path.write_text(json.dumps(record["view"]), encoding="utf-8")
        log_path.write_text(record["log"], encoding="utf-8")
        manifest_runs.append(
            {
                "workflow": record["workflow"],
                "jobs": record["jobs"],
                "rest_path": str(rest_path),
                "view_path": str(view_path),
                "log_path": str(log_path),
            }
        )
    manifest_path.write_text(
        json.dumps({"schema": 1, "runs": manifest_runs}), encoding="utf-8"
    )
    monkeypatch.setattr(
        verify_pr_run_evidence,
        "_git_identity",
        lambda _root, _merge, _head: ((BASE_SHA, HEAD_SHA), TREE_SHA, TREE_SHA),
    )

    result = verify_pr_run_evidence.main(
        [
            "--pr-json",
            str(pr_path),
            "--runs-json",
            str(runs_path),
            "--manifest",
            str(manifest_path),
            "--git-root",
            str(tmp_path),
            "--repo",
            REPO,
            "--pr-number",
            str(PR_NUMBER),
            "--head-ref",
            HEAD_REF,
            "--head-sha",
            HEAD_SHA,
            "--base-ref",
            BASE_REF,
            "--base-sha",
            BASE_SHA,
            "--merge-sha",
            MERGE_SHA,
            "--output",
            str(output_path),
        ]
    )

    assert result == 0
    assert json.loads(output_path.read_text(encoding="utf-8"))["schema"] == 1
