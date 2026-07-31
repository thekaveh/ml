#!/usr/bin/env python3
"""CI gate: self-containment, completeness, placeholders, determinism (spec §6.3)."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

from scripts.docs.links import SITE_URL, WIKI_URL, find_links, is_forbidden
from scripts.docs.manifest import Manifest, load_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
_PLACEHOLDER_RE = re.compile(r"\b(TODO|TBD|FIXME|XXX)\b")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_NUMBER_PREFIX_RE = re.compile(r"^(\d+(?:\.\d+)*)(?:\.)?(?:\s+|$)")
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")


@dataclass(frozen=True)
class Finding:
    severity: str  # "error" | "warning"
    message: str


def check_self_containment(generated_root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for surface in ("site", "wiki"):
        d = generated_root / surface
        if not d.exists():
            continue
        for md in d.rglob("*.md"):
            for link in find_links(md.read_text(encoding="utf-8")):
                if is_forbidden(link.target, surface):
                    findings.append(Finding("error", f"{surface}: {md.relative_to(generated_root)} links cross-surface: {link.target}"))
    return findings


def check_repo_self_containment(repo_root: Path) -> list[Finding]:
    findings: list[Finding] = []
    candidates = [repo_root / "README.md", *(repo_root / "docs").rglob("*.md")]
    for md in candidates:
        if not md.exists():
            continue
        text = md.read_text(encoding="utf-8")
        for target in (SITE_URL, WIKI_URL):
            if target in text:
                findings.append(
                    Finding(
                        "error",
                        f"repo: {md.relative_to(repo_root)} references another documentation surface: {target}",
                    )
                )
    return findings


def manifest_markdown_sources(manifest: Manifest) -> set[str]:
    sources: set[str] = set()

    def visit(section) -> None:
        if section.source:
            sources.add(section.source)
        for child in section.children:
            visit(child)

    for section in manifest.sections:
        visit(section)
    sources.update(notebook.doc for notebook in manifest.notebooks)
    return sources


def _manifest_page_numbers(manifest: Manifest) -> dict[str, str]:
    numbers: dict[str, str] = {}

    def visit(section) -> None:
        if section.source:
            numbers[section.source] = section.number
        for child in section.children:
            visit(child)

    for section in manifest.sections:
        visit(section)
    numbers.update((notebook.doc, notebook.number) for notebook in manifest.notebooks)
    return numbers


def _markdown_headings(text: str) -> list[tuple[int, str]]:
    headings: list[tuple[int, str]] = []
    fence: str | None = None
    for line in text.splitlines():
        fence_match = _FENCE_RE.match(line)
        if fence_match:
            marker = fence_match.group(1)[0]
            if fence is None:
                fence = marker
            elif fence == marker:
                fence = None
            continue
        if fence is not None:
            continue
        heading_match = _HEADING_RE.match(line)
        if heading_match:
            headings.append((len(heading_match.group(1)), heading_match.group(2)))
    return headings


def check_numbering(manifest: Manifest, repo_root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for source, page_number in sorted(_manifest_page_numbers(manifest).items()):
        path = repo_root / source
        if not path.exists():
            continue
        headings = _markdown_headings(path.read_text(encoding="utf-8"))
        if not headings or headings[0][0] != 1:
            findings.append(Finding("error", f"{source}: missing H1 for manifest number {page_number}"))
            continue

        last_number: dict[int, str] = {}
        sibling_count: dict[int, int] = {}
        h1_count = 0
        for level, title in headings:
            number_match = _NUMBER_PREFIX_RE.match(title)
            actual = number_match.group(1) if number_match else None
            if level == 1:
                h1_count += 1
                if h1_count > 1:
                    findings.append(Finding("error", f"{source}: multiple H1 headings are not allowed"))
                expected = page_number
            else:
                parent = last_number.get(level - 1)
                if parent is None:
                    findings.append(Finding("error", f"{source}: H{level} has no H{level - 1} parent: {title}"))
                    continue
                sibling_count[level] = sibling_count.get(level, 0) + 1
                expected = f"{parent}.{sibling_count[level]}"

            if actual != expected:
                findings.append(
                    Finding(
                        "error",
                        f"{source}: H{level} must start with {expected} (got {title!r})",
                    )
                )
            last_number[level] = expected
            for deeper in range(level + 1, 7):
                last_number.pop(deeper, None)
                sibling_count.pop(deeper, None)
    return findings


def check_completeness(manifest: Manifest, repo_root: Path) -> list[Finding]:
    findings: list[Finding] = []
    declared = manifest_markdown_sources(manifest)
    actual = {
        str(path.relative_to(repo_root))
        for path in (repo_root / "docs").rglob("*.md")
    }
    for source in sorted(declared - actual):
        findings.append(Finding("error", f"manifest Markdown source missing: {source}"))
    for source in sorted(actual - declared):
        findings.append(Finding("error", f"canonical Markdown not declared in manifest: {source}"))
    for n in manifest.notebooks:
        if not (repo_root / n.spec).exists():
            findings.append(Finding("error", f"notebook spec missing: {n.spec}"))
    return findings


def check_notebook_infrastructure(manifest: Manifest, repo_root: Path) -> list[Finding]:
    from scripts.docs.notebook_infrastructure import (
        NotebookInfrastructureError,
        load_atlas_task_contracts,
        render_atlas_task_table,
        verify_atlas_task_table,
    )

    try:
        contracts = load_atlas_task_contracts(repo_root, manifest)
        verify_atlas_task_table(
            repo_root / "docs/notebook-infrastructure.md",
            render_atlas_task_table(contracts),
        )
    except (NotebookInfrastructureError, OSError) as error:
        return [Finding("error", f"notebook infrastructure: {error}")]
    return []


def check_placeholders(generated_root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for md in generated_root.rglob("*.md"):
        if md.parent.name == "superpowers":
            continue
        for m in _PLACEHOLDER_RE.finditer(md.read_text(encoding="utf-8")):
            findings.append(Finding("error", f"placeholder {m.group(0)!r} in {md.relative_to(generated_root)}"))
    return findings


def check(repo_root: Path, generated_root: Path) -> int:
    from scripts.docs.build_docs import build

    manifest = load_manifest(repo_root / "docs/manifest.yaml", repo_root)
    findings = check_notebook_infrastructure(manifest, repo_root)
    if findings:
        for finding in findings:
            print(f"[{finding.severity.upper()}] {finding.message}", file=sys.stderr)
        return 1 if any(finding.severity == "error" for finding in findings) else 0
    rc = build(repo_root / "docs/manifest.yaml", repo_root, check=True)
    if rc != 0:
        return rc
    findings = []
    findings += check_self_containment(generated_root)
    findings += check_repo_self_containment(repo_root)
    findings += check_completeness(manifest, repo_root)
    findings += check_numbering(manifest, repo_root)
    findings += check_placeholders(generated_root)
    errors = [f for f in findings if f.severity == "error"]
    for f in findings:
        print(f"[{f.severity.upper()}] {f.message}", file=sys.stderr)
    return 1 if errors else 0


def main(argv: list[str] | None = None) -> int:
    return check(REPO_ROOT, REPO_ROOT / "generated")


if __name__ == "__main__":
    sys.exit(main())
