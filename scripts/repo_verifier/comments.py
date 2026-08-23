"""Comment-hygiene validation and Phase-B candidate export."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path

import nbformat

from .common import read_text as _read_text
from .models import CheckResult, Finding, VerifierConfig


_STATE_THE_WHAT_PATTERNS: tuple[tuple[re.Pattern, re.Pattern], ...] = (
    (re.compile(r"^\s*#\s*import\s+\S", re.IGNORECASE),
     re.compile(r"^\s*(?:from\s+\S+\s+)?import\s+\S")),
    (re.compile(r"^\s*#\s*loop\s+(over|through|across)\b", re.IGNORECASE),
     re.compile(r"^\s*(?:for|while)\s+")),
    (re.compile(r"^\s*#\s*return\b", re.IGNORECASE),
     re.compile(r"^\s*return\b")),
    (re.compile(r"^\s*#\s*(define|create|define the|declare)\b", re.IGNORECASE),
     re.compile(r"^\s*def\s+|^\s*class\s+|^\s*\w+\s*=")),
    (re.compile(r"^\s*#\s*(initialize|init|set|assign)\b", re.IGNORECASE),
     re.compile(r"^\s*\w+\s*=")),
    (re.compile(r"^\s*#\s*print\b", re.IGNORECASE),
     re.compile(r"^\s*print\s*\(")),
    (re.compile(r"^\s*#\s*(call|invoke|run)\s+\w+", re.IGNORECASE),
     re.compile(r"^\s*\w+\s*\(")),
)


def _scan_source_for_comments(source: str, location_prefix: str) -> list[Finding]:
    findings: list[Finding] = []
    lines = source.splitlines()
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if not stripped.startswith("#"):
            continue
        j = i + 1
        while j < len(lines):
            nxt = lines[j].strip()
            if nxt and not nxt.startswith("#"):
                break
            j += 1
        if j >= len(lines):
            continue
        nxt_line = lines[j]
        for comment_pat, code_pat in _STATE_THE_WHAT_PATTERNS:
            if comment_pat.match(line) and code_pat.match(nxt_line):
                findings.append(Finding(
                    id="C.state_the_what", check="comments", severity="warning",
                    location=f"{location_prefix}:{i+1}",
                    message=f"comment restates the next code line: {stripped[:80]!r}",
                    detail={"next_code": nxt_line.strip()[:80]},
                ))
                break
    return findings


def _iter_in_scope_code(
    repo: Path, config: VerifierConfig
) -> Iterator[tuple[Path, str]]:
    # verify_repo.py is the scanner itself; scanning its own source produces
    # spurious C.state_the_what hits on its rule-matching helpers. The other
    # scripts under scripts/ are in scope.
    for path in (repo / "scripts").glob("*.py"):
        if path.name == "verify_repo.py":
            continue
        yield path, _read_text(path)
    for task in config.active_task_dirs:
        for path in (repo / "notebooks" / task).glob("*.py"):
            yield path, _read_text(path)
    for notebook in (
        path
        for task in config.active_task_dirs
        for path in (repo / "notebooks" / task).glob("*.ipynb")
    ):
        try:
            document = nbformat.read(notebook, as_version=4)
        except Exception:
            continue
        for cell_index, cell in enumerate(document.cells):
            if cell.cell_type != "code":
                continue
            # Papermill `parameters`-tagged cells carry convention-bound
            # boilerplate (see scripts/inject_smoke_test_cell.py). Their
            # leading comments document the papermill -p invocation
            # contract — they're documentation, not state-the-what hits.
            # Same self-exclusion principle as the verify_repo.py skip
            # above.
            tags = cell.get("metadata", {}).get("tags") or []
            if "parameters" in tags:
                continue
            marker = notebook.with_name(f"{notebook.name}#cell[{cell_index}]")
            yield marker, cell.source


def check_comments(repo: Path, config: VerifierConfig) -> CheckResult:
    result = CheckResult(name="comments")
    for path_marker, source in _iter_in_scope_code(repo, config):
        try:
            location_prefix = str(path_marker.relative_to(repo))
        except (ValueError, AttributeError):
            location_prefix = str(path_marker)
        result.findings.extend(_scan_source_for_comments(source, location_prefix))
    return result


def export_phase_b_candidates(repo: Path, out_path: Path, config: VerifierConfig) -> int:
    """Export unflagged comment candidates for the Phase-B LLM judge."""
    candidates = []
    for path_marker, source in _iter_in_scope_code(repo, config):
        try:
            location_prefix = str(path_marker.relative_to(repo))
        except (ValueError, AttributeError):
            location_prefix = str(path_marker)
        a_flagged_lines = {
            int(finding.location.rsplit(":", 1)[-1])
            for finding in _scan_source_for_comments(source, location_prefix)
        }
        lines = source.splitlines()
        for i, line in enumerate(lines):
            stripped = line.lstrip()
            if not stripped.startswith("#") or (i + 1) in a_flagged_lines:
                continue
            start = max(0, i - 5)
            end = min(len(lines), i + 6)
            snippet = "\n".join(
                ("▶ " if j == i else "  ") + lines[j]
                for j in range(start, end)
            )
            candidates.append({
                "location": f"{location_prefix}:{i+1}",
                "comment": stripped,
                "snippet": snippet,
            })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({
            "schema_version": 1,
            "candidate_count": len(candidates),
            "candidates": candidates,
        }, indent=2),
        encoding="utf-8",
    )
    return len(candidates)
