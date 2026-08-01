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
_HTML_H1_RE = re.compile(r'^<h1\s+align=["\']center["\']>(.+?)</h1>$', re.IGNORECASE)
_NUMBER_PREFIX_RE = re.compile(r"^(\d+(?:\.\d+)*)(?:\.)?(?:\s+|$)")
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_NUMBERED_H2_RE = re.compile(r"^##\s+\d+(?:\.\d)*(?:\.)?(?:\s+|$)", re.MULTILINE)
_PROJECT_SUMMARY_RE = re.compile(
    r"<!-- project-summary:start -->\s*(.*?)\s*<!-- project-summary:end -->",
    re.DOTALL,
)
PROJECT_TAGLINE = "Local notebooks. Remote Atlas execution. Explicit infrastructure contracts."
PROJECT_SUMMARY_OPENING = (
    "ml-eng-lab is a portfolio of self-contained machine-learning notebook experiments built "
    "for local editing in VS Code and recommended remote execution through JupyterHub on Atlas's "
    "ML Engineering track."
)
_PROJECT_POSTER_MARKUP = {
    "README.md": (
        '<p align="center">\n'
        '  <img src="docs/assets/ml-eng-lab-poster.png" '
        'alt="ML Eng Lab — notebooks, systems, and reproducibility" width="100%">\n'
        "</p>"
    ),
    "docs/index.md": (
        '<p align="center">\n'
        '  <img src="assets/ml-eng-lab-poster.png" '
        'alt="ML Eng Lab — notebooks, systems, and reproducibility" width="100%">\n'
        "</p>"
    ),
}
_PROJECT_CENTERED_TITLES = {
    "README.md": '<h1 align="center">ML ENG LAB</h1>',
    "docs/index.md": '<h1 align="center">1 · ML ENG LAB</h1>',
}
_PROJECT_ASSET_PREFIXES = {
    "README.md": "docs/assets/",
    "docs/index.md": "assets/",
}
PROJECT_BADGE_GROUPS = (
    (
        "Core ML",
        (
            ("Python", "python.svg"),
            ("Jupyter", "jupyter.svg"),
            ("NumPy", "numpy.svg"),
            ("pandas", "pandas.svg"),
            ("PyTorch", "pytorch.svg"),
            ("PyTorch Geometric", "pytorch-geometric.svg"),
            ("scikit-learn", "scikit-learn.svg"),
        ),
    ),
    (
        "NLP and graphs",
        (
            ("spaCy", "spacy.svg"),
            ("NLTK", "nltk.svg"),
            ("NetworkX", "networkx.svg"),
        ),
    ),
    (
        "Runtime",
        (
            ("Atlas", "atlas.svg"),
            ("Docker", "docker.svg"),
            ("VS Code", "vscode.svg"),
            ("GitHub Codespaces", "github-codespaces.svg"),
        ),
    ),
    (
        "Engineering",
        (
            ("NNx", "nnx.svg"),
            ("Papermill", "papermill.svg"),
            ("pytest", "pytest.svg"),
            ("Ruff", "ruff.svg"),
            ("GitHub Actions", "github-actions.svg"),
        ),
    ),
)


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
    fence: tuple[str, int] | None = None
    in_html_comment = False
    for raw_line in text.splitlines():
        if fence is not None:
            marker, opening_length = fence
            stripped = raw_line.lstrip()
            closing_match = re.match(re.escape(marker) + r"+", stripped)
            if (
                closing_match
                and len(closing_match.group()) >= opening_length
                and not stripped[closing_match.end() :].strip()
            ):
                fence = None
            continue

        line, in_html_comment = _strip_html_comments(raw_line, in_html_comment)
        fence_match = _FENCE_RE.match(line)
        if fence_match:
            opening = fence_match.group(1)
            fence = (opening[0], len(opening))
            continue
        html_match = _HTML_H1_RE.match(line)
        if html_match:
            headings.append((1, html_match.group(1)))
            continue
        heading_match = _HEADING_RE.match(line)
        if heading_match:
            headings.append((len(heading_match.group(1)), heading_match.group(2)))
    return headings


def _strip_html_comments(line: str, in_comment: bool) -> tuple[str, bool]:
    while True:
        if in_comment:
            end = line.find("-->")
            if end == -1:
                return "", True
            line = line[end + 3 :]
            in_comment = False

        start = line.find("<!--")
        if start == -1:
            return line, False
        end = line.find("-->", start + 4)
        if end == -1:
            return line[:start], True
        line = line[:start] + line[end + 3 :]


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


def _normalize_prose(text: str) -> str:
    return " ".join(text.split())


def _badge_rows(asset_prefix: str) -> str:
    rows: list[str] = []
    for label, badges in PROJECT_BADGE_GROUPS:
        images = " ".join(
            f'<img alt="{alt}" src="{asset_prefix}badges/{filename}">'
            for alt, filename in badges
        )
        rows.append(
            '<p align="center">\n'
            f"  <sub><strong>{label}</strong></sub><br>\n"
            f"  {images}\n"
            "</p>"
        )
    return "\n\n".join(rows)


def _normalize_opener_structure(opener: str) -> str:
    return opener.replace("docs/assets/", "assets/").replace(
        '<h1 align="center">1 · ML ENG LAB</h1>',
        '<h1 align="center">ML ENG LAB</h1>',
    )


def check_project_opening(repo_root: Path) -> list[Finding]:
    findings: list[Finding] = []
    summaries: dict[str, str] = {}
    opener_structures: dict[str, str] = {}
    expected_tagline = (
        f'<p align="center"><strong>{PROJECT_TAGLINE}</strong></p>'
    )

    for relative_path, poster in _PROJECT_POSTER_MARKUP.items():
        path = repo_root / relative_path
        if not path.exists():
            findings.append(Finding("error", f"project opener source missing: {relative_path}"))
            continue
        text = path.read_text(encoding="utf-8")
        centered_title = _PROJECT_CENTERED_TITLES[relative_path]
        asset_prefix = _PROJECT_ASSET_PREFIXES[relative_path]
        expected_prefix = (
            f"{poster}\n\n"
            f"{centered_title}\n\n"
            f"{expected_tagline}\n\n"
            f"{_badge_rows(asset_prefix)}\n\n"
            "<!-- project-summary:start -->\n"
        )
        if not text.startswith(expected_prefix):
            findings.append(
                Finding(
                    "error",
                    f"project opener order or structure is invalid in {relative_path}",
                )
            )
        matches = _PROJECT_SUMMARY_RE.findall(text)
        if len(matches) != 1:
            findings.append(
                Finding("error", f"project summary markers must occur exactly once in {relative_path}")
            )
            continue
        summary_match = _PROJECT_SUMMARY_RE.search(text)
        assert summary_match is not None
        following_summary = text[summary_match.end() :]
        h2_match = _NUMBERED_H2_RE.search(following_summary)
        if h2_match is None:
            findings.append(
                Finding(
                    "error",
                    f"project summary must be followed by a numbered H2 in {relative_path}",
                )
            )
            opener = text
        else:
            summary_tail = following_summary[: h2_match.start()]
            opener = text[: summary_match.end() + h2_match.start()]
            if summary_tail.strip():
                findings.append(
                    Finding(
                        "error",
                        f"project opener tail after summary must contain only whitespace in {relative_path}",
                    )
                )
        if "runtime-flow" in opener:
            findings.append(
                Finding(
                    "error",
                    f"runtime-flow diagram cannot appear in project opener in {relative_path}",
                )
            )
        if centered_title not in opener:
            findings.append(
                Finding("error", f"centered HTML title missing from {relative_path}"))
        if poster not in opener:
            findings.append(Finding("error", f"project opener poster missing from {relative_path}"))
        if expected_tagline not in opener:
            findings.append(Finding("error", f"canonical project tagline missing from {relative_path}"))

        poster_source = re.search(r'<img src="([^"]+)"', poster)
        assert poster_source is not None
        resolved_poster = Path(relative_path).parent / poster_source.group(1)
        if not (repo_root / resolved_poster).is_file():
            findings.append(
                Finding("error", f"project poster asset missing: {resolved_poster}")
            )

        for _, badges in PROJECT_BADGE_GROUPS:
            for alt, filename in badges:
                source = f"{asset_prefix}badges/{filename}"
                badge = f'<img alt="{alt}" src="{source}">'
                if badge not in opener:
                    findings.append(
                        Finding(
                            "error",
                            f"project badge {alt!r} missing from {relative_path}",
                        )
                    )
                resolved_source = Path(relative_path).parent / source
                if not (repo_root / resolved_source).is_file():
                    findings.append(
                        Finding(
                            "error",
                            f"project badge asset missing: {resolved_source}",
                        )
                    )
        if len(re.split(r"\n\s*\n", matches[0].strip())) != 2:
            findings.append(
                Finding(
                    "error",
                    f"project summary in {relative_path} must contain exactly two paragraphs",
                )
            )
        summary = _normalize_prose(matches[0])
        summaries[relative_path] = summary
        normalized_opener = (
            opener[: summary_match.start(1)]
            + _normalize_prose(summary_match.group(1))
            + opener[summary_match.end(1) :]
        )
        opener_structures[relative_path] = _normalize_opener_structure(
            normalized_opener
        )
        if not summary.startswith(PROJECT_SUMMARY_OPENING):
            findings.append(
                Finding("error", f"canonical project summary opening missing from {relative_path}")
            )
        word_count = len(re.findall(r"\b[\w'-]+\b", summary))
        if not 100 <= word_count <= 150:
            findings.append(
                Finding(
                    "error",
                    f"project summary in {relative_path} must contain 100-150 words; found {word_count}",
                )
            )

    if len(summaries) == len(_PROJECT_POSTER_MARKUP) and len(set(summaries.values())) != 1:
        findings.append(Finding("error", "project summary differs between README.md and docs/index.md"))
    if (
        len(opener_structures) == len(_PROJECT_POSTER_MARKUP)
        and len(set(opener_structures.values())) != 1
    ):
        findings.append(
            Finding(
                "error",
                "project opener order or structure differs between README.md and docs/index.md",
            )
        )
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
    findings += check_project_opening(repo_root)
    errors = [f for f in findings if f.severity == "error"]
    for f in findings:
        print(f"[{f.severity.upper()}] {f.message}", file=sys.stderr)
    return 1 if errors else 0


def main(argv: list[str] | None = None) -> int:
    return check(REPO_ROOT, REPO_ROOT / "generated")


if __name__ == "__main__":
    sys.exit(main())
