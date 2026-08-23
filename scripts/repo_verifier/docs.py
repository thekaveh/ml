"""Documentation validation rules for the repository verifier."""
from __future__ import annotations

import ast
import hashlib
import re
import shlex
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path

import nbformat

try:
    import yaml as _yaml
except ImportError:
    _yaml = None

from . import common as _common
from .models import CheckResult, Finding, RunCommand, VerifierConfig

NOTEBOOK_ROOT = Path("notebooks")


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, FileNotFoundError):
        return ""


def _active_task_path(repo: Path, task: str) -> Path:
    return repo / NOTEBOOK_ROOT / task


def _notebook_rel(path: Path, repo: Path) -> str:
    return str(path.relative_to(repo))


README_REQUIRED_H2 = (
    "1. Task summary", "2. Why this exists", "3. What's in the notebook",
    "4. How to run", "5. Dependencies", "6. Known issues",
)

ROOT_README_REQUIRED_H2 = (
    "1. Overview", "2. Repository layout", "3. Quick start", "4. Tasks",
    "5. Notebook re-execution policy", "6. NNx library",
    "7. Repository conventions", "8. Roadmap", "9. License",
)

TERMINOLOGY_CANONICALS = {
    "JupyterHub": ("Jupyterhub", "Jupyter Hub", "jupyter hub"),
    "NumPy": ("Numpy", "NUMPY"),
    "PyTorch": ("Pytorch", "PYTORCH", "Py-Torch"),
    "PyG": ("PYG", "Pyg"),
}

def _strip_markdown_code(text: str, *, strip_inline: bool = True) -> str:
    def code_fragment(value: str, *, crosses_lines: bool = False) -> str:
        return " " * len(value) if strip_inline or crosses_lines else value

    def mask_html_comments(
        line: str, in_comment: bool, inline_marker: str | None
    ) -> tuple[str, bool, str | None]:
        masked: list[str] = []
        index = 0
        crosses_lines = inline_marker is not None
        while index < len(line):
            if in_comment:
                close = line.find("-->", index)
                if close == -1:
                    masked.append(" " * (len(line) - index))
                    return "".join(masked), True, inline_marker
                masked.append(" " * (close + 3 - index))
                index = close + 3
                in_comment = False
                continue
            if inline_marker is not None:
                code_span = re.search(r"`+", line[index:])
                if code_span is None:
                    masked.append(code_fragment(line[index:], crosses_lines=crosses_lines))
                    return "".join(masked), in_comment, inline_marker
                end = index + code_span.end()
                masked.append(code_fragment(line[index:end], crosses_lines=crosses_lines))
                index = end
                if len(code_span.group(0)) == len(inline_marker):
                    inline_marker = None
                continue
            code_span = re.match(r"`+", line[index:])
            if code_span:
                inline_marker = code_span.group(0)
                end = index + len(inline_marker)
                masked.append(code_fragment(line[index:end]))
                index = end
                continue
            if line.startswith("<!--", index):
                in_comment = True
                continue
            masked.append(line[index])
            index += 1
        return "".join(masked), in_comment, inline_marker

    stripped: list[str] = []
    fence: tuple[str, int] | None = None
    raw_html_block: str | None = None
    in_comment = False
    inline_marker: str | None = None
    for line in text.splitlines():
        opener = re.match(r"^ {0,3}(?P<marker>`{3,}|~{3,})", line)
        invalid_backtick_fence = bool(
            not in_comment
            and inline_marker is None
            and opener
            and opener["marker"].startswith("`")
            and "`" in line[opener.end():]
        )
        if (
            fence is None
            and not in_comment
            and inline_marker is None
            and opener
            and not invalid_backtick_fence
        ):
            marker = opener["marker"]
            fence = (marker[0], len(marker))
            stripped.append(" " * len(line))
            continue
        if fence is not None:
            marker, minimum_length = fence
            if re.fullmatch(rf" {{0,3}}{re.escape(marker)}{{{minimum_length},}}\s*", line):
                fence = None
            stripped.append(" " * len(line))
            continue
        if raw_html_block is not None:
            if re.search(rf"</{raw_html_block}\s*>", line, re.IGNORECASE):
                raw_html_block = None
            stripped.append(" " * len(line))
            continue
        if line.startswith("    "):
            stripped.append(" " * len(line))
            continue
        line, in_comment, inline_marker = mask_html_comments(
            line, in_comment, inline_marker
        )
        raw_html_opener = re.match(
            r"^ {0,3}<(script|style|pre|textarea)(?=\s|>|$)",
            line,
            re.IGNORECASE,
        )
        if raw_html_opener is not None:
            raw_html_block = raw_html_opener.group(1).lower()
            if re.search(rf"</{raw_html_block}\s*>", line[raw_html_opener.end():], re.IGNORECASE):
                raw_html_block = None
            stripped.append(" " * len(line))
            continue
        stripped.append(line)
    return "\n".join(stripped)


_H1_RE = re.compile(r"^# ([^\n]+)", re.MULTILINE)
_H2_RE = re.compile(r"^## ([^\n]+)", re.MULTILINE)
_MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_NUMBERED_HEADING_RE = re.compile(r"^(\d+(?:\.\d+)*)\.\s+\S")
_STALE_LAYOUT_PATTERNS: tuple[tuple[str, re.Pattern], ...] = (
    (
        "flat top-level task-folder guidance",
        re.compile(r"\b(?:top-level folder|top-level folders|flat top-level layout|<21 active task folders>)\b"),
    ),
    (
        "old root archive guidance",
        re.compile(r"(?<!notebooks/)archive/(?:README\.md)?"),
    ),
    (
        "old nbviewer placeholder without notebooks prefix",
        re.compile(r"nbviewer\.org/github/thekaveh/ml-eng-lab/(?:blob|tree)/main/<folder>"),
    ),
)
_STALE_ACTIVE_NOTEBOOK_PATHS: tuple[tuple[str, re.Pattern], ...] = (
    (
        "old local repo path",
        re.compile(r"/Users/[^/\s]+/repos/ml(?!-eng-lab)\b"),
    ),
    (
        "old JupyterHub repo path",
        re.compile(r"/home/jovyan/work/ml(?!-eng-lab)\b"),
    ),
    (
        "old Codespaces repo path",
        re.compile(r"/workspaces/ml(?!-eng-lab)\b"),
    ),
    (
        "old GitHub repo URL",
        re.compile(r"github\.com/thekaveh/ml-lab\b"),
    ),
    (
        "removed in-repo nnx source tree",
        re.compile(r"/(?:home/jovyan/work|workspaces)/ml-eng-lab/nnx/src/nnx\b"),
    ),
    (
        "host-local Python environment path",
        re.compile(r"/Users/[^/\s]+/\.pyenv\b"),
    ),
    (
        "absolute NNx saved-run output",
        re.compile(
            r"Run saved to (?:/(?:[^\r\n\"]*?/)?runs/|"
            r"(?:[A-Za-z]:[\\/]+|[\\/]{2,}[^\\/\s\"]+[\\/]+"
            r"[^\\/\s\"]+[\\/]+)(?:[^\r\n\"]*?[\\/]+)?runs[\\/]+)"
            r"[^\\/\s\"]+"
        ),
    ),
)


def _markdown_headings(text: str, level: int) -> list[str]:
    pat = _H1_RE if level == 1 else _H2_RE
    return [m.group(1).strip() for m in pat.finditer(text)]


def _iter_markdown_headings(text: str) -> Iterator[tuple[int, int, str]]:
    in_fence = False
    for line_no, line in enumerate(text.splitlines(), 1):
        if re.match(r"^\s*(?:```|~~~)", line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = _MARKDOWN_HEADING_RE.match(line)
        if not m:
            continue
        title = m.group(2).strip().rstrip("#").strip()
        yield line_no, len(m.group(1)), title


def _iter_numbered_doc_files(
    repo: Path, config: VerifierConfig
) -> Iterator[Path]:
    for rel in ("README.md", "CONTRIBUTING.md"):
        path = repo / rel
        if path.exists():
            yield path
    for rel in (
        "index.md",
        "architecture.md",
        "diagrams/README.md",
        "FINDINGS-NNX.md",
        "FINDINGS-ATLAS.md",
        "dependency-contracts.md",
        "atlas-pin-bump-runbook.md",
        "env-setup.md",
        "jupyterhub-integration.md",
        "vscode-remote-access.md",
        "notebook-infrastructure.md",
    ):
        path = repo / "docs" / rel
        if path.exists():
            yield path
    maintenance_dir = repo / "docs" / "maintenance"
    if maintenance_dir.exists():
        yield from sorted(maintenance_dir.glob("*.md"))
    for d in config.active_task_dirs:
        path = _active_task_path(repo, d) / "README.md"
        if path.exists():
            yield path


def _numbered_heading_findings(repo: Path, path: Path) -> list[Finding]:
    findings: list[Finding] = []
    for line_no, level, title in _iter_markdown_headings(_read_text(path)):
        if level == 1:
            continue
        m = _NUMBERED_HEADING_RE.match(title)
        if not m:
            findings.append(Finding(
                id="D9.numbered_heading", check="docs", severity="error",
                location=f"{path.relative_to(repo)}:{line_no}",
                message="numbered-doc heading must start with '<number>. '",
                detail={"heading": title},
            ))
            continue
        expected_depth = level - 1
        actual_depth = len(m.group(1).split("."))
        if actual_depth != expected_depth:
            findings.append(Finding(
                id="D9.numbered_heading", check="docs", severity="error",
                location=f"{path.relative_to(repo)}:{line_no}",
                message=(
                    f"numbered-doc H{level} must use {expected_depth} "
                    f"numeric component(s)"
                ),
                detail={"heading": title},
            ))
    return findings


def _stale_layout_guidance_findings(repo: Path) -> list[Finding]:
    findings: list[Finding] = []
    for rel in ("README.md", "CONTRIBUTING.md"):
        path = repo / rel
        if not path.exists():
            continue
        text = _strip_markdown_code(_read_text(path))
        for label, pattern in _STALE_LAYOUT_PATTERNS:
            for m in pattern.finditer(text):
                line_no = text.count("\n", 0, m.start()) + 1
                findings.append(Finding(
                    id="D11.stale_notebook_layout",
                    check="docs",
                    severity="error",
                    location=f"{rel}:{line_no}",
                    message=f"stale pre-notebooks/ layout guidance: {label}",
                    detail={"match": m.group(0)},
                ))
    return findings


_ATLAS_INFRA_LEDGER_SECTION_RE = re.compile(
    r"^##[ \t]+\d+(?:\.\d+)*[.]?[ \t]+Atlas Infra Submodule Contract[ \t]*\r?$"
    r"(?P<body>.*?)(?=^##[ \t]|\Z)",
    re.MULTILINE | re.DOTALL,
)
_ATLAS_INFRA_GITLINK_SHA_RE = re.compile(
    r"^Current Atlas `infra` gitlink SHA:[ \t]*`([0-9a-f]{40})`\.[ \t]*$",
    re.MULTILINE,
)
_ATLAS_CURRENT_PIN_DOCUMENTS = (
    "README.md",
    "docs/env-setup.md",
    "docs/atlas-pin-bump-runbook.md",
)
_ATLAS_CURRENT_PIN_MARKER_RE = re.compile(
    r"^Current reviewed Atlas pin: `([0-9a-f]{40})`\.[ \t]*$",
    re.MULTILINE,
)
_DEPENDENCY_CURRENT_SNAPSHOT_RE = re.compile(
    r"^###[ \t]+6[.]1[.]1[.]2[ \t]+Current[ \t]+Issue[ \t]+#63[ \t]+"
    r"locked[ \t]+four-surface[ \t]+audit[ \t]*\r?$"
    r"(?P<body>.*?)(?=^#{1,3}[ \t]|\Z)",
    re.MULTILINE | re.DOTALL,
)
_DEPENDENCY_HASH_HEADER = "| Input | SHA-256 |"
_DEPENDENCY_HASH_SEPARATOR = "| --- | --- |"
_DEPENDENCY_HASH_ROW_RE = re.compile(
    r"\| `(?P<path>[^`]+)` \| `(?P<sha256>[0-9a-f]{64})` \|"
)
_DEPENDENCY_HASH_INPUTS = (
    "vulnerability-audit-requirements.txt",
    "requirements.txt",
    "torch-core-requirements.txt",
    "torch-ecosystem-requirements.txt",
    "torch-requirements.txt",
    "torch-audit-requirements.txt",
    "pyg-extension-audit-requirements.txt",
    "docs-requirements.txt",
    "atlas-contract-requirements.txt",
    "security/accepted-advisories.json",
)
_DEPENDENCY_HTML_TYPE1_TAGS = ("pre", "script", "style", "textarea")
_DEPENDENCY_HTML_TYPE6_TAGS = (
    "address", "article", "aside", "base", "basefont", "blockquote", "body", "caption",
    "center", "col", "colgroup", "dd", "details", "dialog", "dir", "div", "dl", "dt",
    "fieldset", "figcaption", "figure", "footer", "form", "frame", "frameset", "h1", "h2",
    "h3", "h4", "h5", "h6", "head", "header", "hgroup", "hr", "html", "iframe", "legend", "li",
    "link", "main", "menu", "menuitem", "nav", "noframes", "ol", "optgroup", "option", "p",
    "param", "search", "section", "source", "summary", "table", "tbody", "td", "tfoot", "th", "thead",
    "title", "tr", "track", "ul",
)
_DEPENDENCY_HTML_TYPE1_OPEN_RE = re.compile(
    rf"^ {{0,3}}<(?P<tag>{'|'.join(_DEPENDENCY_HTML_TYPE1_TAGS)})(?:[ \t]|>|$)",
    re.IGNORECASE,
)
_DEPENDENCY_HTML_TYPE6_OPEN_RE = re.compile(
    rf"^ {{0,3}}</?(?P<tag>{'|'.join(_DEPENDENCY_HTML_TYPE6_TAGS)})(?:[ \t]|/?>|$)",
    re.IGNORECASE,
)
_DEPENDENCY_PYG_SUPPLEMENT_CONTRACT = (
    "The pre-resolved `pyg-extension-audit-requirements.txt` supplement contains exactly "
    "`torch-scatter==2.1.2` and `torch-sparse==0.6.18`; it contains neither "
    "`torch-cluster` nor `torch-spline-conv`."
)
_DEPENDENCY_PYG_LIB_LIMITATION = (
    "pyg-lib is an exact external-index wheel outside ordinary PyPI audit coverage; "
    "its version and provenance are verified by `verify_torch_stack`."
)
_DEPENDENCY_SUMMARY_HEADER = (
    "| Package | Manifest Constraint | Audited Resolved Version | Finding Count | "
    "Current Disposition |"
)
_DEPENDENCY_SUMMARY_SEPARATOR = "| --- | --- | ---: | ---: | --- |"
_DEPENDENCY_ADVISORY_HEADER = (
    "| Package | Advisory ID | Feed Records | Fix Versions | Audited Version | "
    "Aliases | Surface |"
)
_DEPENDENCY_ADVISORY_SEPARATOR = "| --- | --- | ---: | --- | ---: | --- | --- |"
_DEPENDENCY_SUMMARY_ROW_RE = re.compile(
    r"\| `([^`]+)` \| `[^`]+` \| `[^`]+` \| ([1-9]\d*) \| [^|]+ \|"
)
_DEPENDENCY_ADVISORY_ROW_RE = re.compile(
    r"\| `([^`]+)` \| `(?:PYSEC|CVE)-[^`]+` \| ([1-9]\d*) \| "
    r"(?:`[^`]+`|None listed) \| `[^`]+` \| "
    r"(?:`[^`]+`(?:, `[^`]+`)*|None listed) \| [^|]+ \|"
)
_DEPENDENCY_ADVISORY_IDENTITY_ROW_RE = re.compile(
    r"\| `(?P<package>[^`]+)` \| `(?P<advisory_id>(?:PYSEC|CVE)-[^`]+)` \| "
    r"[1-9]\d* \| (?:`[^`]+`|None listed) \| `(?P<accepted_version>[^`]+)` \| "
    r"(?:`[^`]+`(?:, `[^`]+`)*|None listed) \| (?P<surfaces>[^|]+) \|"
)
_DEPENDENCY_RESULT_RE = re.compile(
    r"Result: ([1-9]\d*) known vulnerabilities across "
    r"[1-9]\d* resolved packages?[.]"
)
_MARKDOWN_ADVISORY_SURFACES = {
    "Combined runtime": "combined-runtime",
    "Torch": "torch",
    "Documentation": "documentation",
    "Atlas contract": "atlas-contract",
}


def _masked_markdown_line(line: str) -> str:
    return "".join("\r" if char == "\r" else "\n" if char == "\n" else " " for char in line)


def _mask_dependency_raw_html(text: str) -> str:
    masked: list[str] = []
    type1_tag: str | None = None
    in_type6 = False
    for line in text.splitlines(keepends=True):
        if type1_tag is None and not in_type6:
            type1 = _DEPENDENCY_HTML_TYPE1_OPEN_RE.match(line)
            type6 = _DEPENDENCY_HTML_TYPE6_OPEN_RE.match(line)
            if type1 is not None:
                type1_tag = type1["tag"].lower()
            elif type6 is not None:
                in_type6 = True
            else:
                masked.append(line)
                continue
        masked.append(_masked_markdown_line(line))
        if type1_tag is not None and re.search(
            rf"</{re.escape(type1_tag)}[ \t]*>", line, re.IGNORECASE,
        ):
            type1_tag = None
        elif in_type6 and not line.strip():
            in_type6 = False
    return "".join(masked)


def _dependency_table_rows(
    body: str, *, header: str, separator: str
) -> list[str] | None:
    lines = body.splitlines()
    header_indexes = [index for index, line in enumerate(lines) if line == header]
    if len(header_indexes) != 1:
        return None
    header_index = header_indexes[0]
    if header_index + 1 >= len(lines) or lines[header_index + 1] != separator:
        return None
    rows: list[str] = []
    for line in lines[header_index + 2:]:
        if not line.startswith("|"):
            break
        rows.append(line)
    return rows or None


def _markdown_advisory_surfaces(value: str, *, canonical_order: tuple[str, ...]) -> tuple[str, ...]:
    labels = tuple(label.strip() for label in value.split(";"))
    if not labels or any(not label for label in labels):
        raise ValueError("advisory surface labels are malformed")
    try:
        surfaces = tuple(_MARKDOWN_ADVISORY_SURFACES[label] for label in labels)
    except KeyError as error:
        raise ValueError(f"unknown advisory surface label: {error.args[0]}") from error
    if len(set(surfaces)) != len(surfaces):
        raise ValueError("advisory surface labels must be unique")
    ordered_surfaces = tuple(surface for surface in canonical_order if surface in surfaces)
    if surfaces != ordered_surfaces:
        raise ValueError("advisory surface labels must use canonical order")
    return surfaces


def _format_advisory_identity(
    identity: tuple[str, str, str], surfaces: tuple[str, ...]
) -> str:
    package, advisory_id, accepted_version = identity
    return f"{package} {accepted_version} {advisory_id} on [{', '.join(surfaces)}]"


def _dependency_advisory_baseline_findings(
    repo: Path, advisory_lines: list[str] | None
) -> list[Finding]:
    location = "docs/dependency-contracts.md"
    try:
        from scripts.advisory_baseline import (
            AdvisoryBaselineError,
            advisory_identity,
            load_baseline,
            normalize_package_name,
        )
    except (ImportError, OSError):
        return [Finding(
            id="D10.dependency_advisory_baseline",
            check="docs",
            severity="error",
            location="security/accepted-advisories.json",
            message="accepted advisory baseline loader is unavailable",
        )]
    try:
        baseline = load_baseline(repo / "security" / "accepted-advisories.json")
    except (AdvisoryBaselineError, OSError) as error:
        return [Finding(
            id="D10.dependency_advisory_baseline",
            check="docs",
            severity="error",
            location="security/accepted-advisories.json",
            message=f"accepted advisory baseline is invalid: {error}",
        )]

    if not advisory_lines:
        return [Finding(
            id="D10.dependency_advisory_baseline",
            check="docs",
            severity="error",
            location=location,
            message="current accepted-advisories table cannot be compared to the baseline",
        )]

    markdown_items: set[tuple[tuple[str, str, str], tuple[str, ...]]] = set()
    for line in advisory_lines:
        row = _DEPENDENCY_ADVISORY_IDENTITY_ROW_RE.fullmatch(line)
        if row is None:
            return [Finding(
                id="D10.dependency_advisory_baseline",
                check="docs",
                severity="error",
                location=location,
                message="current accepted-advisories row cannot be compared to the baseline",
            )]
        try:
            surfaces = _markdown_advisory_surfaces(
                row["surfaces"], canonical_order=baseline.audited_surfaces
            )
        except ValueError as error:
            return [Finding(
                id="D10.dependency_advisory_baseline",
                check="docs",
                severity="error",
                location=location,
                message=f"current accepted-advisories row is malformed: {error}",
            )]
        identity = (
            normalize_package_name(row["package"]),
            row["advisory_id"],
            row["accepted_version"],
        )
        markdown_items.add((identity, surfaces))

    baseline_items = {
        (advisory_identity(item), item.surfaces)
        for item in baseline.accepted_advisories
    }
    findings: list[Finding] = []
    for identity, surfaces in sorted(baseline_items - markdown_items):
        findings.append(Finding(
            id="D10.dependency_advisory_baseline",
            check="docs",
            severity="error",
            location=location,
            message=(
                "accepted advisory baseline identity is missing from the current Markdown ledger: "
                f"{_format_advisory_identity(identity, surfaces)}"
            ),
        ))
    for identity, surfaces in sorted(markdown_items - baseline_items):
        findings.append(Finding(
            id="D10.dependency_advisory_baseline",
            check="docs",
            severity="error",
            location=location,
            message=(
                "current Markdown ledger identity is missing from accepted advisory baseline JSON: "
                f"{_format_advisory_identity(identity, surfaces)}"
            ),
        ))
    return findings


def _dependency_input_hash_findings(repo: Path, body: str) -> list[Finding]:
    location = "docs/dependency-contracts.md"
    lines = _dependency_table_rows(
        body,
        header=_DEPENDENCY_HASH_HEADER,
        separator=_DEPENDENCY_HASH_SEPARATOR,
    )
    if lines is None:
        return [Finding(
            id="D10.dependency_input_hash",
            check="docs",
            severity="error",
            location=location,
            message="current Issue #63 input-hash table is missing or malformed",
        )]
    rows = [_DEPENDENCY_HASH_ROW_RE.fullmatch(line) for line in lines]
    if not all(rows):
        return [Finding(
            id="D10.dependency_input_hash",
            check="docs",
            severity="error",
            location=location,
            message="current Issue #63 input-hash row is malformed",
        )]
    parsed = [(row["path"], row["sha256"]) for row in rows if row is not None]
    names = [name for name, _ in parsed]
    if len(names) != len(set(names)):
        return [Finding(
            id="D10.dependency_input_hash",
            check="docs",
            severity="error",
            location=location,
            message="current Issue #63 input-hash table has duplicate paths",
        )]
    if tuple(names) != _DEPENDENCY_HASH_INPUTS:
        return [Finding(
            id="D10.dependency_input_hash",
            check="docs",
            severity="error",
            location=location,
            message="current Issue #63 input-hash paths or order drifted",
            detail={"expected": list(_DEPENDENCY_HASH_INPUTS), "actual": names},
        )]
    findings: list[Finding] = []
    for relative_path, recorded in parsed:
        source = repo / relative_path
        if not source.is_file():
            findings.append(Finding(
                id="D10.dependency_input_hash",
                check="docs",
                severity="error",
                location=relative_path,
                message="current Issue #63 hashed input is missing",
            ))
            continue
        actual = hashlib.sha256(source.read_bytes()).hexdigest()
        if actual != recorded:
            findings.append(Finding(
                id="D10.dependency_input_hash",
                check="docs",
                severity="error",
                location=relative_path,
                message="current Issue #63 recorded input hash is stale",
                detail={"expected": recorded, "actual": actual},
            ))
    return findings


def _atlas_current_pin_projection_findings(
    repo: Path, *, gitlink_sha: str
) -> list[Finding]:
    present = [
        relative
        for relative in _ATLAS_CURRENT_PIN_DOCUMENTS
        if (repo / relative).exists()
    ]
    if not present:
        return []

    findings: list[Finding] = []
    for relative in _ATLAS_CURRENT_PIN_DOCUMENTS:
        path = repo / relative
        matches = (
            _ATLAS_CURRENT_PIN_MARKER_RE.findall(_read_text(path))
            if path.exists()
            else []
        )
        if len(matches) != 1 or matches[0] != gitlink_sha:
            findings.append(Finding(
                id="D10.atlas_current_pin_projection",
                check="docs",
                severity="error",
                location=relative,
                message=(
                    "current Atlas pin projection must occur once and equal "
                    "the infra gitlink"
                ),
                detail={"matches": matches, "gitlink_sha": gitlink_sha},
            ))
    return findings


def dependency_ledger_findings(
    repo: Path, *, run: RunCommand
) -> list[Finding]:
    path = repo / "docs" / "dependency-contracts.md"
    infra_exists = (repo / "infra").exists()
    if not path.exists():
        baseline_finding = Finding(
            id="D10.dependency_advisory_baseline",
            check="docs",
            severity="error",
            location="docs/dependency-contracts.md",
            message="current accepted-advisories section is missing",
        )
        if not infra_exists:
            return [baseline_finding]
        return [baseline_finding, Finding(
            id="D10.dependency_ledger_submodule_sha",
            check="docs",
            severity="error",
            location="docs/dependency-contracts.md",
            message=(
                "Atlas Infra Submodule Contract ledger entry is required while "
                "infra exists"
            ),
        )]
    text = _read_text(path)
    findings: list[Finding] = []
    published_text = _mask_dependency_raw_html(
        _strip_markdown_code(text, strip_inline=False)
    )
    snapshot_matches = list(_DEPENDENCY_CURRENT_SNAPSHOT_RE.finditer(published_text))
    if not snapshot_matches:
        findings.append(Finding(
            id="D10.dependency_ledger_count", check="docs", severity="error",
            location="docs/dependency-contracts.md",
            message="current accepted-advisories section is missing",
        ))
        findings.append(Finding(
            id="D10.dependency_advisory_baseline", check="docs", severity="error",
            location="docs/dependency-contracts.md",
            message="current accepted-advisories section is missing",
        ))
    elif len(snapshot_matches) != 1:
        findings.append(Finding(
            id="D10.dependency_ledger_count", check="docs", severity="error",
            location="docs/dependency-contracts.md",
            message=(
                "current accepted-advisories heading must appear exactly once; "
                f"found {len(snapshot_matches)}"
            ),
        ))
        findings.append(Finding(
            id="D10.dependency_advisory_baseline", check="docs", severity="error",
            location="docs/dependency-contracts.md",
            message=(
                "current accepted-advisories heading must appear exactly once; "
                f"found {len(snapshot_matches)}"
            ),
        ))
    else:
        body = snapshot_matches[0].group("body")
        summary_lines = _dependency_table_rows(
            body,
            header=_DEPENDENCY_SUMMARY_HEADER,
            separator=_DEPENDENCY_SUMMARY_SEPARATOR,
        )
        advisory_lines = _dependency_table_rows(
            body,
            header=_DEPENDENCY_ADVISORY_HEADER,
            separator=_DEPENDENCY_ADVISORY_SEPARATOR,
        )
        findings.extend(_dependency_advisory_baseline_findings(repo, advisory_lines))
        findings.extend(_dependency_input_hash_findings(repo, body))
        normalized_body = " ".join(body.split())
        if "zero vulnerabilities" in body:
            findings.append(Finding(
                id="D10.dependency_ledger_contract",
                check="docs",
                severity="error",
                location="docs/dependency-contracts.md",
                message="current Issue #62 audit must not claim zero vulnerabilities",
            ))
        if _DEPENDENCY_PYG_SUPPLEMENT_CONTRACT not in normalized_body:
            findings.append(Finding(
                id="D10.dependency_ledger_contract",
                check="docs",
                severity="error",
                location="docs/dependency-contracts.md",
                message="current Issue #62 PyG supplement contract is missing or stale",
            ))
        if _DEPENDENCY_PYG_LIB_LIMITATION not in normalized_body:
            findings.append(Finding(
                id="D10.dependency_pyg_lib_limitation",
                check="docs",
                severity="error",
                location="docs/dependency-contracts.md",
                message="current Issue #62 pyg-lib external-index audit limitation is missing",
            ))
        package_rows = (
            [_DEPENDENCY_SUMMARY_ROW_RE.fullmatch(line) for line in summary_lines]
            if summary_lines
            else []
        )
        advisory_rows = (
            [_DEPENDENCY_ADVISORY_ROW_RE.fullmatch(line) for line in advisory_lines]
            if advisory_lines
            else []
        )
        summary_valid = bool(package_rows) and all(package_rows)
        advisory_valid = bool(advisory_rows) and all(advisory_rows)
        if not summary_valid:
            findings.append(Finding(
                id="D10.dependency_ledger_count", check="docs", severity="error",
                location="docs/dependency-contracts.md",
                message="current accepted-advisories summary table is malformed",
            ))
        if not advisory_valid:
            findings.append(Finding(
                id="D10.dependency_ledger_count", check="docs", severity="error",
                location="docs/dependency-contracts.md",
                message="current accepted-advisories advisory table is malformed",
            ))
        advisory_counts: dict[str, int] = {}
        if advisory_valid:
            for row in advisory_rows:
                assert row is not None
                package, count = row.groups()
                advisory_counts[package] = advisory_counts.get(package, 0) + int(count)
        if summary_valid and advisory_valid:
            parsed_package_rows = [row.groups() for row in package_rows if row]
            package_names = [package for package, _ in parsed_package_rows]
            duplicate_packages = sorted({
                package for package in package_names if package_names.count(package) > 1
            })
            if duplicate_packages:
                findings.append(Finding(
                    id="D10.dependency_ledger_count", check="docs", severity="error",
                    location="docs/dependency-contracts.md",
                    message=(
                        "current accepted-advisories summary table has duplicate "
                        f"package rows: {', '.join(duplicate_packages)}"
                    ),
                    detail={"packages": duplicate_packages},
                ))
            package_counts = {
                package: int(count) for package, count in parsed_package_rows
            }
            for package, expected in package_counts.items():
                actual = advisory_counts.get(package, 0)
                if actual != expected:
                    findings.append(Finding(
                        id="D10.dependency_ledger_count", check="docs", severity="error",
                        location="docs/dependency-contracts.md",
                        message=(
                            f"{package} advisory feed-record count is {actual}; "
                            f"expected {expected} from audit summary"
                        ),
                        detail={"package": package, "expected": expected, "actual": actual},
                    ))

            for package in sorted(advisory_counts.keys() - package_counts.keys()):
                findings.append(Finding(
                    id="D10.dependency_ledger_count", check="docs", severity="error",
                    location="docs/dependency-contracts.md",
                    message=f"{package} advisory package is absent from audit summary",
                    detail={"package": package},
                ))

        result_lines = [
            line for line in body.splitlines() if line.startswith("Result:")
        ]
        if len(result_lines) != 1:
            findings.append(Finding(
                id="D10.dependency_ledger_count", check="docs", severity="error",
                location="docs/dependency-contracts.md",
                message=(
                    "current accepted-advisories section requires exactly one Result "
                    f"line; found {len(result_lines)}"
                ),
            ))
        else:
            total_match = _DEPENDENCY_RESULT_RE.fullmatch(result_lines[0])
            if not total_match:
                findings.append(Finding(
                    id="D10.dependency_ledger_count", check="docs", severity="error",
                    location="docs/dependency-contracts.md",
                    message="current accepted-advisories Result line is malformed",
                ))
            elif advisory_valid:
                expected_total = int(total_match.group(1))
                actual_total = sum(advisory_counts.values())
                if actual_total != expected_total:
                    findings.append(Finding(
                        id="D10.dependency_ledger_count", check="docs", severity="error",
                        location="docs/dependency-contracts.md",
                        message=(
                            f"advisory feed-record total is {actual_total}; "
                            f"expected {expected_total} from audit summary"
                        ),
                        detail={"expected": expected_total, "actual": actual_total},
                    ))
    if not infra_exists:
        return findings

    section_match = _ATLAS_INFRA_LEDGER_SECTION_RE.search(text)
    if not section_match:
        findings.append(Finding(
            id="D10.dependency_ledger_submodule_sha",
            check="docs",
            severity="error",
            location="docs/dependency-contracts.md",
            message=(
                "Atlas Infra Submodule Contract ledger entry is required while "
                "infra exists"
            ),
        ))
        return findings

    ledger_sha_match = _ATLAS_INFRA_GITLINK_SHA_RE.search(section_match.group("body"))
    if not ledger_sha_match:
        findings.append(Finding(
            id="D10.dependency_ledger_submodule_sha",
            check="docs",
            severity="error",
            location="docs/dependency-contracts.md",
            message=(
                "Atlas Infra Submodule Contract must include a parseable "
                "40-character infra gitlink SHA"
            ),
        ))
        return findings

    ledger_sha = ledger_sha_match.group(1)
    rc, out, _err = run(["git", "ls-files", "--stage", "--", "infra"], repo)
    gitlink_match = re.search(r"160000 ([0-9a-f]{40}) \d+\s+infra", out)
    if rc != 0 or not gitlink_match:
        findings.append(Finding(
            id="D10.dependency_ledger_submodule_sha",
            check="docs",
            severity="error",
            location="docs/dependency-contracts.md",
            message="Atlas ledger SHA cannot be compared to a parseable infra gitlink",
            detail={"ledger_sha": ledger_sha, "gitlink_sha": None},
        ))
        return findings
    gitlink_sha = gitlink_match.group(1)
    if ledger_sha != gitlink_sha:
        findings.append(Finding(
            id="D10.dependency_ledger_submodule_sha",
            check="docs",
            severity="error",
            location="docs/dependency-contracts.md",
            message="Atlas ledger SHA does not match the superproject infra gitlink",
            detail={"ledger_sha": ledger_sha, "gitlink_sha": gitlink_sha},
        ))
    findings.extend(
        _atlas_current_pin_projection_findings(repo, gitlink_sha=gitlink_sha)
    )
    return findings


def _workflow_action_pin_ledger(text: str) -> dict[str, tuple[str, str]]:
    return {
        action: (tag, sha)
        for action, tag, sha in re.findall(
            r"^\| `([^`]+)` \| `([^`]+)` \| `([0-9a-f]{40})` \|", text, re.M
        )
    }


def _workflow_action_pin_findings(repo: Path) -> list[Finding]:
    ledger_path = repo / "docs" / "dependency-contracts.md"
    if not ledger_path.exists():
        return []
    ledger = _workflow_action_pin_ledger(_read_text(ledger_path))
    findings: list[Finding] = []
    workflow_dir = repo / ".github" / "workflows"
    workflows = sorted((*workflow_dir.glob("*.yml"), *workflow_dir.glob("*.yaml")))
    for workflow in workflows:
        for line_no, line in enumerate(_read_text(workflow).splitlines(), start=1):
            m = re.search(r"\buses:\s*([^\s#]+)(?:\s*#\s*(\S+))?", line)
            if not m:
                continue
            uses_ref = m.group(1).strip("'\"")
            tag_comment = (m.group(2) or "").strip()
            if uses_ref.startswith(("./", "../")):
                continue
            if "@" not in uses_ref:
                findings.append(Finding(
                    id="D10.workflow_action_pin",
                    check="docs",
                    severity="error",
                    location=f"{workflow.relative_to(repo)}:{line_no}",
                    message=f"workflow action reference must include @ref: {uses_ref}",
                ))
                continue
            action, ref = uses_ref.rsplit("@", 1)
            if not re.fullmatch(r"[0-9a-f]{40}", ref):
                findings.append(Finding(
                    id="D10.workflow_action_pin",
                    check="docs",
                    severity="error",
                    location=f"{workflow.relative_to(repo)}:{line_no}",
                    message=f"workflow action reference must be pinned to a full SHA: {uses_ref}",
                    detail={"action": action, "ref": ref},
                ))
                continue
            if action not in ledger:
                findings.append(Finding(
                    id="D10.workflow_action_pin",
                    check="docs",
                    severity="error",
                    location=f"{workflow.relative_to(repo)}:{line_no}",
                    message=f"workflow action is SHA-pinned but missing from dependency ledger: {action}",
                    detail={"action": action, "sha": ref},
                ))
                continue
            ledger_tag, ledger_sha = ledger[action]
            if ref != ledger_sha or tag_comment != ledger_tag:
                findings.append(Finding(
                    id="D10.workflow_action_pin",
                    check="docs",
                    severity="error",
                    location=f"{workflow.relative_to(repo)}:{line_no}",
                    message=(
                        "workflow action SHA/comment must match dependency ledger "
                        f"for {action}"
                    ),
                    detail={
                        "action": action,
                        "workflow_sha": ref,
                        "workflow_tag_comment": tag_comment,
                        "ledger_sha": ledger_sha,
                        "ledger_tag": ledger_tag,
                    },
                ))
    return findings


_EXPECTED_TORCH_STACK_IMPORTS = {
    "torch": "torch",
    "torchvision": "torchvision",
    "torchaudio": "torchaudio",
    "pytorch-lightning": "pytorch_lightning",
    "torchmetrics": "torchmetrics",
    "torchao": "torchao",
    "torch-geometric": "torch_geometric",
    "pyg-lib": "pyg_lib",
    "torch-scatter": "torch_scatter",
    "torch-sparse": "torch_sparse",
}
_EXPECTED_RUNTIME_ONLY_MODULES = frozenset({
    "numpy", "torch", "torchvision", "torch_geometric", "torch_sparse",
    "torch_scatter", "pyg_lib", "matplotlib", "seaborn", "pandas", "sklearn",
    "scipy", "networkx", "community", "nnx", "tqdm",
})
_EXPECTED_RUNTIME_AVAILABLE_IMPORTS = (
    "torch", "torch_geometric", "pyg_lib", "torch_scatter", "torch_sparse",
)
_TORCH_RUNTIME_IMPORTS = frozenset(_EXPECTED_RUNTIME_AVAILABLE_IMPORTS)
_FORBIDDEN_TORCH_RUNTIME_IMPORTS = frozenset(("torch_cluster", "torch_spline_conv"))
_MUTATING_DECLARATION_METHODS = frozenset({
    "update", "clear", "pop", "popitem", "setdefault", "add", "discard", "remove",
    "append", "extend", "insert", "reverse", "sort", "__setitem__", "__delitem__",
})
_INPLACE_DECLARATION_DUNDERS = frozenset({
    "__iadd__", "__iand__", "__idivmod__", "__ifloordiv__", "__ilshift__",
    "__imatmul__", "__imod__", "__imul__", "__ior__", "__ipow__", "__irshift__",
    "__isub__", "__itruediv__", "__ixor__",
})
_UNBOUND_MUTATOR_TYPES = frozenset({"dict", "frozenset", "list", "set", "tuple"})
_OPERATOR_MUTATION_FUNCTIONS = frozenset(
    method[2:-2] for method in _INPLACE_DECLARATION_DUNDERS
) | frozenset({"iconcat", "setitem", "delitem"})


def _protected_name_bindings(tree: ast.AST, name: str) -> list[ast.AST]:
    bindings: list[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id == name:
            bindings.append(node)
        elif isinstance(node, ast.arg) and node.arg == name:
            bindings.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
            bindings.append(node)
        elif isinstance(node, ast.alias) and (node.asname == name or node.name == name):
            bindings.append(node)
        elif isinstance(node, ast.ExceptHandler) and node.name == name:
            bindings.append(node)
        elif isinstance(node, ast.MatchAs) and node.name == name:
            bindings.append(node)
    return bindings


def _protected_mutation_root(node: ast.AST) -> str | None:
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _is_mutating_declaration_method(method: str) -> bool:
    return method in _MUTATING_DECLARATION_METHODS or method in _INPLACE_DECLARATION_DUNDERS


def _operator_module_aliases(tree: ast.AST) -> frozenset[str]:
    return frozenset(
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name == "operator"
    )


def _operator_function_aliases(tree: ast.AST) -> frozenset[str]:
    return frozenset(
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "operator" and node.level == 0
        for alias in node.names
        if alias.name in _OPERATOR_MUTATION_FUNCTIONS
    )


def _operator_star_imports(tree: ast.AST) -> list[ast.alias]:
    return [
        alias
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "operator" and node.level == 0
        for alias in node.names
        if alias.name == "*"
    ]


def _is_unbound_mutator_type(node: ast.AST) -> bool:
    return isinstance(node, ast.Name) and node.id in _UNBOUND_MUTATOR_TYPES


def _is_operator_mutator_call(
    function: ast.Attribute,
    operator_aliases: frozenset[str],
) -> bool:
    return (
        isinstance(function.value, ast.Name)
        and function.value.id in operator_aliases
        and function.attr in _OPERATOR_MUTATION_FUNCTIONS
    )


def _protected_name_mutations(
    tree: ast.AST,
    name: str,
    allowed_target: ast.Name,
) -> list[ast.AST]:
    mutations: list[ast.AST] = []
    operator_aliases = _operator_module_aliases(tree)
    operator_function_aliases = _operator_function_aliases(tree)
    mutations.extend(_operator_star_imports(tree))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = (node.target,)
        elif isinstance(node, ast.Delete):
            targets = node.targets
        else:
            targets = ()
        for target in targets:
            if target is not allowed_target and _protected_mutation_root(target) == name:
                mutations.append(target)
        if not isinstance(node, ast.Call):
            continue
        protected_first_argument = (
            bool(node.args) and _protected_mutation_root(node.args[0]) == name
        )
        if isinstance(node.func, ast.Name):
            if protected_first_argument and node.func.id in operator_function_aliases:
                mutations.append(node)
            continue
        if isinstance(node.func, ast.Attribute):
            direct_receiver = _protected_mutation_root(node.func.value) == name
            qualified_mutator = protected_first_argument and (
                (
                    _is_unbound_mutator_type(node.func.value)
                    and _is_mutating_declaration_method(node.func.attr)
                )
                or _is_operator_mutator_call(node.func, operator_aliases)
            )
            if (
                direct_receiver and _is_mutating_declaration_method(node.func.attr)
            ) or qualified_mutator:
                mutations.append(node)
    return mutations


def _literal_assignment_values(source: str, name: str) -> object:
    tree = ast.parse(source)
    assignments = [
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == name
    ]
    if len(assignments) != 1:
        raise ValueError(f"{name} must have exactly one direct assignment")
    assignment = assignments[0]
    if (
        _protected_name_bindings(tree, name) != [assignment.targets[0]]
        or _protected_name_mutations(tree, name, assignment.targets[0])
    ):
        raise ValueError(f"{name} must not be rebound")
    value = assignment.value
    if name == "IMPORTS":
        if not isinstance(value, ast.Dict) or any(key is None for key in value.keys):
            raise ValueError("IMPORTS must be a plain literal dictionary")
        pairs = [
            (ast.literal_eval(key), ast.literal_eval(item))
            for key, item in zip(value.keys, value.values, strict=True)
        ]
        if not all(isinstance(key, str) and isinstance(item, str) for key, item in pairs):
            raise ValueError("IMPORTS must map strings to strings")
        if len({key for key, _ in pairs}) != len(pairs):
            raise ValueError("IMPORTS must not repeat keys")
        return dict(pairs)
    if name == "_RUNTIME_ONLY_MODULES":
        if not (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "frozenset"
            and len(value.args) == 1
            and not value.keywords
            and isinstance(value.args[0], ast.Set)
        ):
            raise ValueError("_RUNTIME_ONLY_MODULES must be a frozenset literal")
        items = [ast.literal_eval(item) for item in value.args[0].elts]
        if not all(isinstance(item, str) for item in items) or len(set(items)) != len(items):
            raise ValueError("_RUNTIME_ONLY_MODULES must contain unique strings")
        return frozenset(items)
    if name == "_RUNTIME_AVAILABLE_IMPORTS":
        if not isinstance(value, ast.Tuple):
            raise ValueError("_RUNTIME_AVAILABLE_IMPORTS must be a tuple literal")
        items = tuple(ast.literal_eval(item) for item in value.elts)
        if not all(isinstance(item, str) for item in items) or len(set(items)) != len(items):
            raise ValueError("_RUNTIME_AVAILABLE_IMPORTS must contain unique strings")
        return items
    raise ValueError(f"unknown protected declaration: {name}")


def _workflow_run_commands(source: str) -> tuple[str, ...]:
    if _yaml is None:
        raise ValueError("PyYAML is required to parse workflow commands")
    parsed = _yaml.safe_load(source)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("jobs"), dict):
        raise ValueError("workflow jobs are invalid")
    commands: list[str] = []
    for job in parsed["jobs"].values():
        if not isinstance(job, dict):
            continue
        steps = job.get("steps", ())
        if not isinstance(steps, list):
            raise ValueError("workflow steps are invalid")
        for step in steps:
            if isinstance(step, dict) and "run" in step:
                if not isinstance(step["run"], str):
                    raise ValueError("workflow run command is invalid")
                commands.append(step["run"])
    return tuple(commands)


def _docker_run_commands(source: str) -> tuple[str, ...]:
    commands: list[str] = []
    current: list[str] = []
    continuing = False
    for raw_line in source.splitlines():
        line = raw_line.strip()
        if not current:
            if not line or line.startswith("#") or not line.upper().startswith("RUN "):
                continue
            line = line[4:].strip()
        current.append(line[:-1].rstrip() if line.endswith("\\") else line)
        continuing = line.endswith("\\")
        if not continuing:
            commands.append("\n".join(current))
            current = []
    if current or continuing:
        raise ValueError("Docker RUN instruction has an unterminated continuation")
    return tuple(commands)


def _has_python_candidate(command: str) -> bool:
    return any(
        not line.lstrip().startswith("#")
        and re.search(r"(?:^|[;&|]\s*)python(?:3(?:\.\d+)?)?\s+-c\b", line)
        for line in command.splitlines()
    )


def _python_command_imports(commands: Iterable[str]) -> set[str]:
    imports: set[str] = set()
    for command in commands:
        try:
            argv = shlex.split(command, comments=True)
        except ValueError:
            if _has_python_candidate(command):
                raise ValueError("python -c shell command is malformed") from None
            continue
        for index, value in enumerate(argv[:-2]):
            if not re.fullmatch(r"python(?:3(?:\.\d+)?)?", value) or argv[index + 1] != "-c":
                continue
            try:
                tree = ast.parse(argv[index + 2])
            except SyntaxError:
                raise ValueError("python -c source is invalid") from None
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.update(alias.name.split(".", 1)[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.add(node.module.split(".", 1)[0])
    return imports


def _torch_runtime_contract_findings(repo: Path) -> list[Finding]:
    declarations = (
        ("scripts/verify_torch_stack.py", "IMPORTS", _EXPECTED_TORCH_STACK_IMPORTS),
        ("scripts/verify_repo.py", "_RUNTIME_ONLY_MODULES", _EXPECTED_RUNTIME_ONLY_MODULES),
        ("scripts/verify_repo.py", "_RUNTIME_AVAILABLE_IMPORTS", _EXPECTED_RUNTIME_AVAILABLE_IMPORTS),
    )
    findings: list[Finding] = []
    for location, name, required in declarations:
        try:
            values = _literal_assignment_values(
                (repo / location).read_text(encoding="utf-8"), name
            )
            if isinstance(values, dict):
                import_values = set(values.values())
            else:
                import_values = set(values)
            missing = _TORCH_RUNTIME_IMPORTS - import_values
            forbidden = import_values & _FORBIDDEN_TORCH_RUNTIME_IMPORTS
            drift = values != required
            if missing or forbidden or drift:
                findings.append(Finding(
                    id="D10.torch_runtime_contract", check="docs", severity="error",
                    location=f"{location}:{name}",
                    message="Torch runtime imports drift from the selected graph contract",
                    detail={
                        "missing": sorted(missing),
                        "forbidden": sorted(forbidden),
                        "actual": sorted(import_values),
                    },
                ))
        except (OSError, SyntaxError, StopIteration, ValueError):
            findings.append(Finding(
                id="D10.torch_runtime_contract", check="docs", severity="error",
                location=f"{location}:{name}",
                message="Torch runtime import declaration is unreadable or invalid",
            ))

    command_sources = (
        (".github/workflows/ci.yml", _workflow_run_commands),
        ("Dockerfile", _docker_run_commands),
    )
    for location, extract_commands in command_sources:
        try:
            imports = _python_command_imports(extract_commands(
                (repo / location).read_text(encoding="utf-8")
            ))
        except (OSError, ValueError, TypeError):
            findings.append(Finding(
                id="D10.torch_runtime_contract", check="docs", severity="error",
                location=location,
                message="Torch runtime availability declaration is unreadable or invalid",
            ))
            continue
        forbidden = imports & _FORBIDDEN_TORCH_RUNTIME_IMPORTS
        if forbidden:
            findings.append(Finding(
                id="D10.torch_runtime_contract", check="docs", severity="error",
                location=location,
                message="CI/Docker runtime availability imports a retired graph module",
                detail={"forbidden": sorted(forbidden)},
            ))
    return findings


def _notebook_markdown_text(nb_path: Path) -> str:
    try:
        doc = nbformat.read(nb_path, as_version=4)
    except Exception:
        return ""
    return "\n\n".join(c.source for c in doc.cells if c.cell_type == "markdown")


def _ordered_contains(required: tuple[str, ...], actual: list[str]) -> tuple[bool, list[str]]:
    """Returns (ok, missing). `actual` must contain `required` as an ordered subsequence."""
    missing: list[str] = []
    needed_idx = 0
    actual_idx = 0
    while needed_idx < len(required) and actual_idx < len(actual):
        if required[needed_idx].lower() in actual[actual_idx].lower():
            needed_idx += 1
        actual_idx += 1
    while needed_idx < len(required):
        missing.append(required[needed_idx])
        needed_idx += 1
    return (not missing, missing)


def dependency_lock_findings(repo: Path) -> list[Finding]:
    if not (repo / "requirements" / "lock-policy.toml").is_file():
        return []
    from scripts.verify_dependency_locks import verify_dependency_locks

    return [
        Finding(
            id="D10.dependency_locks",
            check="docs",
            severity="error",
            location=finding.path,
            message=f"dependency lock contract failed: {finding.category}: {finding.detail}",
        )
        for finding in verify_dependency_locks(repo)
    ]


def dependency_advisory_lock_findings(repo: Path) -> list[Finding]:
    if not (repo / "requirements" / "lock-policy.toml").is_file():
        return []
    location = "scripts/advisory_baseline.py"
    try:
        from scripts.advisory_baseline import AdvisoryBaselineError, derive_lock_audit_surfaces

        with tempfile.TemporaryDirectory(prefix="verify-advisory-locks-") as temporary:
            prepared = derive_lock_audit_surfaces(repo, Path(temporary))
        identity = tuple(
            (item.surface.name, item.surface.projection_kind, item.surface.output_name)
            for item in prepared
        )
        expected = (
            ("combined-runtime", "main", "combined-runtime-resolver"),
            ("combined-runtime", "pyg-extensions", "combined-runtime-pyg-extensions"),
            ("torch", "main", "torch-resolver"),
            ("torch", "pyg-extensions", "torch-pyg-extensions"),
            ("documentation", "main", "documentation"),
            ("atlas-contract", "main", "atlas-contract"),
        )
        if identity != expected:
            raise AdvisoryBaselineError("audit projection inventory is invalid")
        if any(
            not item.expected_versions
            or not item.lock_inputs
            or any(record.audited or record.reason != "non-pypi" for record in item.non_pypi)
            for item in prepared
        ):
            raise AdvisoryBaselineError("audit projection evidence is invalid")
    except (ImportError, OSError, ValueError, RuntimeError) as error:
        return [Finding(
            id="D10.dependency_advisory_locks",
            check="docs",
            severity="error",
            location=location,
            message=f"lock-derived advisory contract failed: {error}",
        )]
    return []




def check_docs(repo: Path, config: VerifierConfig, run: RunCommand) -> CheckResult:
    result = CheckResult(name="docs")
    result.findings.extend(dependency_lock_findings(repo))
    result.findings.extend(dependency_advisory_lock_findings(repo))
    canonical_doc_sources: set[str] = set()

    manifest_path = repo / "docs" / "manifest.yaml"
    if manifest_path.exists():
        from scripts.docs.check_docs import (
            check_notebook_infrastructure,
            check_numbering,
            manifest_markdown_sources,
        )
        from scripts.docs.manifest import ManifestError, load_manifest

        try:
            manifest = load_manifest(manifest_path, repo)
        except (ManifestError, OSError) as error:
            result.findings.append(Finding(
                id="D9.invalid_manifest",
                check="docs",
                severity="error",
                location="docs/manifest.yaml",
                message=f"documentation manifest is invalid: {error}",
            ))
        else:
            canonical_doc_sources = manifest_markdown_sources(manifest)
            for finding in check_notebook_infrastructure(manifest, repo):
                result.findings.append(Finding(
                    id="D10.notebook_infrastructure",
                    check="docs",
                    severity=finding.severity,
                    location="docs/notebook-infrastructure.md",
                    message=finding.message,
                ))
            for finding in check_numbering(manifest, repo):
                result.findings.append(Finding(
                    id="D9.numbered_heading",
                    check="docs",
                    severity=finding.severity,
                    location=finding.message.split(":", 1)[0],
                    message=finding.message,
                ))

    configured_notebooks = set(config.required_sections)
    for nb in _common.iter_notebooks(repo, config, NOTEBOOK_ROOT):
        rel = _notebook_rel(nb, repo)
        if rel not in configured_notebooks:
            result.findings.append(Finding(
                id="D1.unconfigured_notebook", check="docs", severity="error",
                location=rel,
                message=(
                    "active notebook is missing from verify_repo_config.yaml "
                    "required_sections; docs and papermill-parameter checks "
                    "would otherwise skip it"
                ),
            ))

    for rel, required in config.required_sections.items():
        nb = repo / rel
        if not nb.exists():
            result.findings.append(Finding(
                id="D1.missing_notebook", check="docs", severity="error",
                location=rel, message="referenced in REQUIRED_SECTIONS but file missing",
            ))
            continue
        text = _notebook_markdown_text(nb)
        h1s = _markdown_headings(text, level=1)
        ok, missing = _ordered_contains(required, h1s)
        if not ok:
            result.findings.append(Finding(
                id="D1.missing_sections", check="docs", severity="error",
                location=rel,
                message=f"missing or out-of-order top-level sections: {missing}",
                detail={"found": h1s, "required": list(required)},
            ))

    for rel in config.required_sections:
        nb = repo / rel
        if not nb.exists():
            continue
        try:
            doc = nbformat.read(nb, as_version=4)
        except Exception:
            continue
        if not doc.cells:
            result.findings.append(Finding(
                id="D2.empty_notebook", check="docs", severity="error",
                location=rel, message="notebook has no cells",
            ))
            continue
        first = doc.cells[0]
        if first.cell_type != "markdown":
            result.findings.append(Finding(
                id="D2.first_cell_not_markdown", check="docs", severity="error",
                location=rel, message="first cell must be a markdown title/purpose cell",
            ))

    for d in config.active_task_dirs:
        readme = _active_task_path(repo, d) / "README.md"
        readme_rel = f"{NOTEBOOK_ROOT.as_posix()}/{d}/README.md"
        if not readme.exists():
            result.findings.append(Finding(
                id="D3.missing_readme", check="docs", severity="error",
                location=readme_rel, message="per-task README missing",
            ))
            continue
        h2s = _markdown_headings(_read_text(readme), level=2)
        ok, missing = _ordered_contains(README_REQUIRED_H2, h2s)
        if not ok:
            result.findings.append(Finding(
                id="D3.missing_sections", check="docs", severity="error",
                location=readme_rel,
                message=f"per-task README missing required H2s: {missing}",
                detail={"found": h2s, "required": list(README_REQUIRED_H2)},
            ))

    root_readme = repo / "README.md"
    root_h2s = _markdown_headings(_read_text(root_readme), level=2)
    ok, missing = _ordered_contains(ROOT_README_REQUIRED_H2, root_h2s)
    if not ok:
        result.findings.append(Finding(
            id="D4.missing_sections", check="docs", severity="error",
            location="README.md",
            message=f"root README missing required H2s: {missing}",
            detail={"found": root_h2s, "required": list(ROOT_README_REQUIRED_H2)},
        ))

    root_text = _read_text(root_readme)
    table_rows = sum(
        1 for line in root_text.splitlines()
        if line.startswith("| [") and "/](" in line
    )
    active_count = sum(1 for d in config.active_task_dirs if _active_task_path(repo, d).is_dir())
    if table_rows < active_count:
        result.findings.append(Finding(
            id="D5.task_table_mismatch", check="docs", severity="error",
            location="README.md",
            message=f"task table has {table_rows} rows; expected >= {active_count} active",
        ))

    roadmap_marker = None
    for candidate in ("## 8. Roadmap", "## Roadmap"):
        if candidate in root_text:
            roadmap_marker = candidate
            break
    if roadmap_marker is None:
        result.findings.append(Finding(
            id="D6.missing_roadmap", check="docs", severity="error",
            location="README.md", message="Roadmap section absent",
        ))
    else:
        body = root_text.split(roadmap_marker, 1)[1]
        body = body.split("\n## ", 1)[0]
        if not re.search(r"-\s*\[\s*[xX ]\s*\]\s+\S", body):
            result.findings.append(Finding(
                id="D6.empty_roadmap", check="docs", severity="warning",
                location="README.md",
                message="Roadmap section present but has no checklist items",
            ))

    for required_doc in ("env-setup.md", "jupyterhub-integration.md", "vscode-remote-access.md"):
        p = repo / "docs" / required_doc
        if not p.exists():
            result.findings.append(Finding(
                id="D7.missing_doc", check="docs", severity="error",
                location=f"docs/{required_doc}", message="required doc missing",
            ))
            continue
        if not _markdown_headings(_read_text(p), level=2):
            result.findings.append(Finding(
                id="D7.no_sections", check="docs", severity="warning",
                location=f"docs/{required_doc}", message="doc has no H2 sections",
            ))

    for path in _common.iter_in_scope_text_files(
        repo, config, NOTEBOOK_ROOT, _read_text
    ):
        text = _read_text(path)
        for canonical, deviations in TERMINOLOGY_CANONICALS.items():
            for dev in deviations:
                for m in re.finditer(rf"\b{re.escape(dev)}\b", text):
                    line_no = text.count("\n", 0, m.start()) + 1
                    result.findings.append(Finding(
                        id="D8.terminology", check="docs", severity="warning",
                        location=f"{path.relative_to(repo)}:{line_no}",
                        message=f"non-canonical spelling {dev!r}; use {canonical!r}",
                    ))

    for path in _iter_numbered_doc_files(repo, config):
        if str(path.relative_to(repo)) in canonical_doc_sources:
            continue
        result.findings.extend(_numbered_heading_findings(repo, path))

    result.findings.extend(dependency_ledger_findings(repo, run=run))
    result.findings.extend(_workflow_action_pin_findings(repo))
    result.findings.extend(_torch_runtime_contract_findings(repo))
    result.findings.extend(_stale_layout_guidance_findings(repo))

    return result
