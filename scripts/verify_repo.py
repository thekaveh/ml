#!/usr/bin/env python3
"""Repo verification oracle.

Runs five orthogonal checks (structure, assets, execution, docs, comments) and emits
machine-readable findings JSON + a human-readable report. Exit code 0 = no
error-severity findings (warnings are allowed and reported but don't fail
the run); 1 = at least one error finding (counts on stderr).
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import io
import json
import re
import shlex
import sys
import tempfile
import tokenize
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict
from pathlib import Path

import nbformat

try:
    import yaml as _yaml  # PyYAML
except ImportError:
    _yaml = None

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
CONFIG_PATH = Path(__file__).resolve().parent / "verify_repo_config.yaml"
_HELP_REQUESTED = any(arg in ("-h", "--help") for arg in sys.argv[1:])


def _load_config(config_path: Path = CONFIG_PATH) -> dict:
    if _yaml is None or not config_path.exists():
        if _HELP_REQUESTED:
            return {}
        raise RuntimeError(
            "verify_repo_config.yaml is required; install PyYAML and ensure "
            "the file exists."
        )
    return _yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}


def _active_task_dirs_from_config(config: dict) -> tuple[str, ...]:
    raw = config.get("active_task_dirs")
    if not raw:
        if _HELP_REQUESTED:
            raw = ()
        else:
            raise RuntimeError(
                "verify_repo_config.yaml is missing the required 'active_task_dirs' key."
            )
    return tuple(raw)


def _required_sections_from_config(config: dict) -> dict[str, tuple[str, ...]]:
    raw = config.get("required_sections")
    if not raw:
        return {}
    return {k: tuple(v) for k, v in raw.items()}


def _tier_a_notebooks_from_config(config: dict) -> tuple[str, ...]:
    raw = config.get("tier_a_notebooks")
    if not raw:
        if _HELP_REQUESTED:
            raw = ()
        else:
            raise RuntimeError(
                "verify_repo_config.yaml is missing the required 'tier_a_notebooks' key."
            )
    return tuple(raw)


def _apply_config(config: dict) -> None:
    global _CONFIG, ACTIVE_TASK_DIRS, REQUIRED_SECTIONS, TIER_A_NOTEBOOKS
    _CONFIG = config
    ACTIVE_TASK_DIRS = _active_task_dirs_from_config(config)
    REQUIRED_SECTIONS = _required_sections_from_config(config)
    TIER_A_NOTEBOOKS = _tier_a_notebooks_from_config(config)


_CONFIG: dict = {}
NOTEBOOK_ROOT = Path("notebooks")
DEFAULT_SUBPROCESS_TIMEOUT = 120
ACTIVE_TASK_DIRS: tuple[str, ...] = ()
REQUIRED_SECTIONS: dict[str, tuple[str, ...]] = {}
TIER_A_NOTEBOOKS: tuple[str, ...] = ()
TIER_A_CI_OUTPUT_ROOT = "/tmp/ml-tier-a"
TIER_C_CODE_BASELINE_TAG = "tier-c-deterministic-seeding-atlas-baseline-2026-08-22"
TIER_C_CODE_BASELINE_COMMIT = "35e7903afe45f60e5e30bf8fbd49f7d6463caa6a"
_apply_config(_load_config())

if not _HELP_REQUESTED:
    from scripts.repo_verifier import assets as _assets_validator
    from scripts.repo_verifier import common as _common
    from scripts.repo_verifier.models import CheckResult, Finding, VerifierConfig
    from scripts.repo_verifier import structure as _structure_validator


def _config_snapshot() -> VerifierConfig:
    return VerifierConfig(
        active_task_dirs=tuple(ACTIVE_TASK_DIRS),
        required_sections={key: tuple(value) for key, value in REQUIRED_SECTIONS.items()},
        tier_a_notebooks=tuple(TIER_A_NOTEBOOKS),
    )

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


if not _HELP_REQUESTED:
    ImportedModule = _structure_validator.ImportedModule
    _cell_magic_name = _structure_validator._cell_magic_name
    _literal_dynamic_import = _structure_validator._literal_dynamic_import
    _paren_balance_delta = _structure_validator._paren_balance_delta
    _fallback_statement = _structure_validator._fallback_statement
    _imported_modules_from_source = _structure_validator._imported_modules_from_source
    _importlib_aliases = _structure_validator._importlib_aliases
    _blank_multiline_string_lines = _structure_validator._blank_multiline_string_lines
    _git_ls_files = _structure_validator._git_ls_files
    _is_allowed_tracked_superpowers_doc = _structure_validator._is_allowed_tracked_superpowers_doc
    _read_text = _structure_validator._read_text
    _strip_markdown_code = _structure_validator._strip_markdown_code
    _github_markdown_slug = _structure_validator._github_markdown_slug
    _markdown_heading_slugs = _structure_validator._markdown_heading_slugs
    _split_markdown_link_target = _structure_validator._split_markdown_link_target
    _iter_notebook_schema_files = _structure_validator._iter_notebook_schema_files
    _active_task_path = _structure_validator._active_task_path
    _notebook_rel = _structure_validator._notebook_rel
    _required_shellcheck_targets = _structure_validator._required_shellcheck_targets
    _shellcheck_targets = _structure_validator._shellcheck_targets
    _required_submodule_paths = _structure_validator._required_submodule_paths
    _MARKDOWN_LINK_RE = _structure_validator._MARKDOWN_LINK_RE
    _INLINE_CODE_RE = _structure_validator._INLINE_CODE_RE
    _IMPORT_RE = _structure_validator._IMPORT_RE
    _NON_PYTHON_CELL_MAGICS = _structure_validator._NON_PYTHON_CELL_MAGICS
    _GITIGNORE_REQUIRED_PATTERNS = _structure_validator._GITIGNORE_REQUIRED_PATTERNS
    _TRACKED_SUPERPOWERS_DOC_PREFIXES = _structure_validator._TRACKED_SUPERPOWERS_DOC_PREFIXES
    _BLOAT_PATTERNS = _structure_validator._BLOAT_PATTERNS
    _FORBIDDEN_TOPLEVEL_DIRS = _structure_validator._FORBIDDEN_TOPLEVEL_DIRS
    _RUNTIME_ONLY_MODULES = _structure_validator._RUNTIME_ONLY_MODULES


def _iter_notebooks(repo: Path) -> Iterator[Path]:
    yield from _common.iter_notebooks(repo, _config_snapshot(), NOTEBOOK_ROOT)


def _iter_in_scope_text_files(repo: Path) -> Iterator[Path]:
    yield from _common.iter_in_scope_text_files(
        repo, _config_snapshot(), NOTEBOOK_ROOT, _read_text
    )


def _iter_in_scope_markdown_documents(repo: Path) -> Iterator[tuple[Path, Path, str]]:
    yield from _common.iter_in_scope_markdown_documents(
        repo, _config_snapshot(), NOTEBOOK_ROOT, _read_text
    )



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


def _iter_numbered_doc_files(repo: Path) -> Iterator[Path]:
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
    for d in ACTIVE_TASK_DIRS:
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


def _dependency_ledger_findings(repo: Path) -> list[Finding]:
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
    rc, out, _err = _run(["git", "ls-files", "--stage", "--", "infra"], repo)
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


def _dependency_lock_findings(repo: Path) -> list[Finding]:
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


def _dependency_advisory_lock_findings(repo: Path) -> list[Finding]:
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




if not _HELP_REQUESTED:
    _qualified_name = _assets_validator._qualified_name
    _notebook_nlp_asset_contract = _assets_validator._notebook_nlp_asset_contract
    _nlp_asset_contract_findings = _assets_validator._nlp_asset_contract_findings


def check_structure(repo: Path) -> CheckResult:
    return _structure_validator.check_structure(repo, _config_snapshot())


def check_assets(repo: Path) -> CheckResult:
    return _assets_validator.check_assets(repo, _config_snapshot())


def check_docs(repo: Path) -> CheckResult:
    result = CheckResult(name="docs")
    result.findings.extend(_dependency_lock_findings(repo))
    result.findings.extend(_dependency_advisory_lock_findings(repo))
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

    configured_notebooks = set(REQUIRED_SECTIONS)
    for nb in _iter_notebooks(repo):
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

    for rel, required in REQUIRED_SECTIONS.items():
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

    for rel in REQUIRED_SECTIONS:
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

    for d in ACTIVE_TASK_DIRS:
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
    active_count = sum(1 for d in ACTIVE_TASK_DIRS if _active_task_path(repo, d).is_dir())
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

    for path in _iter_in_scope_text_files(repo):
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

    for path in _iter_numbered_doc_files(repo):
        if str(path.relative_to(repo)) in canonical_doc_sources:
            continue
        result.findings.extend(_numbered_heading_findings(repo, path))

    result.findings.extend(_dependency_ledger_findings(repo))
    result.findings.extend(_workflow_action_pin_findings(repo))
    result.findings.extend(_torch_runtime_contract_findings(repo))
    result.findings.extend(_stale_layout_guidance_findings(repo))

    return result


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


def _iter_in_scope_code(repo: Path):
    # verify_repo.py is the scanner itself; scanning its own source produces
    # spurious C.state_the_what hits on its rule-matching helpers. The other
    # scripts under scripts/ are in scope.
    for p in (repo / "scripts").glob("*.py"):
        if p.name == "verify_repo.py":
            continue
        yield p, _read_text(p)
    for d in ACTIVE_TASK_DIRS:
        for p in _active_task_path(repo, d).glob("*.py"):
            yield p, _read_text(p)
    for nb in _iter_notebooks(repo):
        try:
            doc = nbformat.read(nb, as_version=4)
        except Exception:
            continue
        for ci, cell in enumerate(doc.cells):
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
            marker = nb.with_name(f"{nb.name}#cell[{ci}]")
            yield marker, cell.source


def check_comments(repo: Path) -> CheckResult:
    result = CheckResult(name="comments")
    for path_marker, source in _iter_in_scope_code(repo):
        try:
            rel = path_marker.relative_to(repo)
            location_prefix = str(rel)
        except (ValueError, AttributeError):
            location_prefix = str(path_marker)
        for f in _scan_source_for_comments(source, location_prefix):
            result.findings.append(f)
    return result


def export_phase_b_candidates(repo: Path, out_path: Path) -> int:
    """Phase-B LLM judge input.

    Phase A (the deterministic heuristic above) catches obvious state-the-what
    comments. Phase B is meant to send the *survivors* — comments that look
    plausible but might still be redundant — to an LLM judge.

    This function exports the candidates: for each comment line that survived
    Phase A, the 5 lines before, the comment line, and the 5 lines after.
    A calling agent (or the /goal loop) reads this JSON, dispatches a subagent
    per file with the prompt below, and applies the verdict.

    Judge prompt template:

        You are reviewing a Python source snippet to enforce a strict
        comment-hygiene rule: comments are allowed ONLY if they explain WHY
        (a non-obvious choice), note a hidden CONSTRAINT or workaround, or
        cite an external reference. Comments that merely restate WHAT the
        code does must be removed.

        Source path: <path>
        Context (5 before, comment, 5 after; comment marked ▶):
        <snippet>

        Respond with: "KEEP" or "DELETE", a colon, then a 12-word-max
        justification.

    Returns the candidate count.
    """
    candidates = []
    for path_marker, source in _iter_in_scope_code(repo):
        try:
            rel = path_marker.relative_to(repo)
            location_prefix = str(rel)
        except (ValueError, AttributeError):
            location_prefix = str(path_marker)
        # Phase A scanner reports state-the-what — opposite filter wanted here:
        # comments that DIDN'T match the heuristic but exist anyway.
        a_flagged_lines = {
            int(f.location.rsplit(":", 1)[-1])
            for f in _scan_source_for_comments(source, location_prefix)
        }
        lines = source.splitlines()
        for i, line in enumerate(lines):
            stripped = line.lstrip()
            if not stripped.startswith("#"):
                continue
            if (i + 1) in a_flagged_lines:
                continue  # already flagged by Phase A
            # 5 lines of context on each side.
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


if not _HELP_REQUESTED:
    _subprocess_text = _common._subprocess_text


def _cell_tags(cell) -> set[str]:
    return set(cell.get("metadata", {}).get("tags") or [])


def _is_parameters_cell(cell) -> bool:
    tags = _cell_tags(cell)
    return "parameters" in tags or "injected-parameters" in tags


def _code_cell_sources_for_baseline(doc) -> list[str]:
    return [
        cell.source
        for cell in doc.cells
        if cell.cell_type == "code" and not _is_parameters_cell(cell)
    ]


def _parameter_trailing_comment_findings(doc, rel: str) -> list[Finding]:
    findings: list[Finding] = []
    bad_line_re = re.compile(r"^\s*[A-Za-z_]\w*\s*=.*#.*=")
    for ci, cell in enumerate(doc.cells):
        if cell.cell_type != "code" or not _is_parameters_cell(cell):
            continue
        for li, line in enumerate(cell.source.splitlines(), start=1):
            if bad_line_re.search(line):
                findings.append(Finding(
                    id="E9.parameter_trailing_comment",
                    check="execution",
                    severity="error",
                    location=f"{rel}:cell[{ci}]:line[{li}]",
                    message=(
                        "parameters cell assignment has a trailing comment with "
                        "'='; papermill 2.7 cannot inspect it reliably"
                    ),
                ))
    return findings


def _assignment_names(source: str) -> set[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    names: set[str] = set()

    def collect(target: ast.expr) -> None:
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                collect(elt)

    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets.extend(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets.append(node.target)
        elif isinstance(node, ast.AugAssign):
            targets.append(node.target)
        for target in targets:
            collect(target)
    return names


def _parameters_assignment_names(doc) -> set[str]:
    names: set[str] = set()
    for cell in doc.cells:
        if cell.cell_type == "code" and _is_parameters_cell(cell):
            names.update(_assignment_names(cell.source))
    return names


def _makefile_variable_items(repo: Path, name: str) -> tuple[str, ...]:
    lines = _read_text(repo / "Makefile").splitlines()
    items: list[str] = []
    collecting = False
    prefix = f"{name} :="
    for line in lines:
        stripped = line.strip()
        if not collecting:
            if not stripped.startswith(prefix):
                continue
            collecting = True
            stripped = stripped[len(prefix):].strip()
        if stripped.endswith("\\"):
            stripped = stripped[:-1].strip()
            keep_collecting = True
        else:
            keep_collecting = False
        if stripped:
            items.extend(stripped.split())
        if collecting and not keep_collecting:
            break
    return tuple(items)


def _ci_tier_a_artifact_paths(repo: Path) -> tuple[str, ...]:
    workflow = repo / ".github" / "workflows" / "ci.yml"
    if _yaml is None or not workflow.exists():
        return ()
    data = _yaml.safe_load(workflow.read_text(encoding="utf-8")) or {}
    steps = data.get("jobs", {}).get("tier-a-papermill", {}).get("steps", [])
    for step in steps:
        if step.get("name") != "Upload refreshed notebook outputs as artifact":
            continue
        raw_path = step.get("with", {}).get("path", "")
        return tuple(
            line.strip()
            for line in str(raw_path).splitlines()
            if line.strip()
        )
    return ()


if not _HELP_REQUESTED:
    _run = _common.run_command


_ATLAS_MANIFEST_CONTRACT = {
    "name": "ml-eng-lab",
    "project_name": "ml-eng-lab",
    "profile": "dev",
    "brand": {"name": "ML Eng Lab"},
    "env": {
        "file": "./atlas.env.user",
        "values": {
            "BASE_PORT": "auto",
            "JUPYTERHUB_SOURCE": "container",
            "LLM_PROVIDER_SOURCE": "ollama-localhost",
        },
    },
    "compose_overlays": ["./compose/ml-eng-lab-atlas.yml"],
}
_ATLAS_REQUIRED_PARENT_FILES = (
    "atlas.env.user.example",
    "compose/ml-eng-lab-atlas.yml",
    "scripts/atlas-up.sh",
    "scripts/atlas-down.sh",
    "scripts/atlas-connect.sh",
)
_ATLAS_ENDPOINT_RE = re.compile(
    r"(?<![\w.-])(?:(?:[a-z][a-z0-9+.-]*)://)?"
    r"(?:localhost|127\.0\.0\.1):\d{1,5}\b",
    re.IGNORECASE,
)
_ATLAS_INTEGRATION_SOURCE_GLOBS = (
    "scripts/atlas-*.py",
    "scripts/atlas-*.sh",
)


def _atlas_manifest_findings(repo: Path) -> list[Finding]:
    manifest_path = repo / "atlas.consumer.yml"
    findings: list[Finding] = []
    if not manifest_path.exists():
        findings.append(Finding(
            id="E15.atlas_manifest",
            check="execution",
            severity="error",
            location="atlas.consumer.yml",
            message="required Atlas consumer manifest is missing",
        ))
    else:
        try:
            manifest = _yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        except _yaml.YAMLError as error:
            findings.append(Finding(
                id="E15.atlas_manifest",
                check="execution",
                severity="error",
                location="atlas.consumer.yml",
                message=f"Atlas consumer manifest is not valid YAML: {error}",
            ))
        else:
            if not isinstance(manifest, dict):
                findings.append(Finding(
                    id="E15.atlas_manifest",
                    check="execution",
                    severity="error",
                    location="atlas.consumer.yml",
                    message="Atlas consumer manifest must be a YAML mapping",
                ))
            elif "track" in manifest:
                findings.append(Finding(
                    id="E15.atlas_manifest",
                    check="execution",
                    severity="error",
                    location="atlas.consumer.yml",
                    message=(
                        "Atlas consumer manifest must not contain track; "
                        "the lifecycle wrapper owns --track ml-eng"
                    ),
                ))
            elif manifest != _ATLAS_MANIFEST_CONTRACT:
                findings.append(Finding(
                    id="E15.atlas_manifest",
                    check="execution",
                    severity="error",
                    location="atlas.consumer.yml",
                    message="Atlas consumer manifest drifted from the parent repository contract",
                ))

    lifecycle_scripts = set(_required_shellcheck_targets(repo))
    for rel in _ATLAS_REQUIRED_PARENT_FILES:
        path = repo / rel
        if not path.exists():
            findings.append(Finding(
                id="E15.atlas_manifest",
                check="execution",
                severity="error",
                location=rel,
                message="required Atlas parent contract file is missing",
            ))
        elif path in lifecycle_scripts and not path.stat().st_mode & 0o111:
            findings.append(Finding(
                id="E15.atlas_manifest",
                check="execution",
                severity="error",
                location=rel,
                message="required Atlas lifecycle script is not executable",
            ))
    return findings


def _atlas_task_metadata_findings(repo: Path) -> list[Finding]:
    manifest_path = repo / "docs" / "manifest.yaml"
    if not manifest_path.exists():
        return []

    from scripts.docs.manifest import ManifestError, load_manifest
    from scripts.docs.notebook_infrastructure import (
        NotebookInfrastructureError,
        load_atlas_task_contracts,
    )

    try:
        manifest = load_manifest(manifest_path, repo)
        load_atlas_task_contracts(repo, manifest)
    except (ManifestError, NotebookInfrastructureError, OSError) as error:
        return [Finding(
            id="E16.atlas_task_metadata",
            check="execution",
            severity="error",
            location="notebooks/**/docs/spec.yaml",
            message=f"invalid Atlas active-task metadata: {error}",
        )]
    return []


def _is_ipython_magic_or_help_line(line: str) -> bool:
    stripped = line.lstrip()
    return (
        stripped.startswith(("%", "!", "?"))
        or (bool(stripped) and stripped.rstrip().endswith("?"))
    )


def _blank_source_span(
    lines: list[str],
    start_line: int,
    start_column: int,
    end_line: int,
    end_column: int,
) -> None:
    for line_index in range(max(start_line, 0), min(end_line + 1, len(lines))):
        line = lines[line_index]
        first = start_column if line_index == start_line else 0
        last = end_column if line_index == end_line else len(line)
        first = max(0, min(first, len(line)))
        last = max(first, min(last, len(line)))
        lines[line_index] = line[:first] + " " * (last - first) + line[last:]


def _ast_column_to_character_index(line: str, byte_offset: int) -> int:
    return len(line.encode("utf-8")[:byte_offset].decode("utf-8"))


def _blank_ast_span(lines: list[str], node: ast.Expr) -> None:
    start_line = node.lineno - 1
    end_line = (node.end_lineno or node.lineno) - 1
    _blank_source_span(
        lines,
        start_line,
        _ast_column_to_character_index(lines[start_line], node.col_offset),
        end_line,
        _ast_column_to_character_index(lines[end_line], node.end_col_offset),
    )


def _python_lines_without_comments(source: str) -> list[str]:
    lines = [
        " " * len(line) if _is_ipython_magic_or_help_line(line) else line
        for line in source.splitlines()
    ]
    parse_source = "\n".join(lines)
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(parse_source).readline))
    except (IndentationError, tokenize.TokenError):
        return ["" for _line in lines]
    for token in reversed(tokens):
        if token.type != tokenize.COMMENT:
            continue
        _blank_source_span(
            lines,
            token.start[0] - 1,
            token.start[1],
            token.end[0] - 1,
            token.end[1],
        )

    try:
        tree = ast.parse(parse_source)
    except SyntaxError:
        for token in reversed(tokens):
            if token.type == tokenize.STRING:
                _blank_source_span(
                    lines,
                    token.start[0] - 1,
                    token.start[1],
                    token.end[0] - 1,
                    token.end[1],
                )
        return lines
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.body:
            continue
        first_statement = node.body[0]
        if not (
            isinstance(first_statement, ast.Expr)
            and isinstance(first_statement.value, ast.Constant)
            and isinstance(first_statement.value.value, str)
        ):
            continue
        _blank_ast_span(lines, first_statement)
    return lines


def _shell_line_without_comment(line: str) -> str:
    quote: str | None = None
    escaped = False
    for index, char in enumerate(line):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote != "'":
            escaped = True
            continue
        if quote:
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char == "#":
            return line[:index]
    return line


def _endpoint_findings_for_lines(
    lines: list[str], location: Callable[[int], str]
) -> list[Finding]:
    findings: list[Finding] = []
    for line_no, line in enumerate(lines, start=1):
        for match in _ATLAS_ENDPOINT_RE.finditer(line):
            findings.append(Finding(
                id="E17.atlas_hardcoded_endpoint",
                check="execution",
                severity="error",
                location=location(line_no),
                message=(
                    "executable Atlas integration code must use an injected "
                    "service endpoint instead of a hard-coded host port"
                ),
                detail={"endpoint": match.group(0)},
            ))
    return findings


def _atlas_hardcoded_endpoint_findings(repo: Path) -> list[Finding]:
    findings: list[Finding] = []
    source_paths = sorted(
        path
        for pattern in _ATLAS_INTEGRATION_SOURCE_GLOBS
        for path in repo.glob(pattern)
        if path.is_file()
    )
    for path in source_paths:
        rel = str(path.relative_to(repo))
        source = _read_text(path)
        if path.suffix == ".py":
            lines = _python_lines_without_comments(source)
        else:
            lines = [_shell_line_without_comment(line) for line in source.splitlines()]
        findings.extend(_endpoint_findings_for_lines(
            lines, lambda line_no, rel=rel: f"{rel}:{line_no}"
        ))

    for notebook_path in _iter_notebooks(repo):
        try:
            notebook = nbformat.read(notebook_path, as_version=4)
        except Exception:
            continue
        rel = _notebook_rel(notebook_path, repo)
        for cell_index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            findings.extend(_endpoint_findings_for_lines(
                _python_lines_without_comments(cell.source),
                lambda line_no, rel=rel, cell_index=cell_index: (
                    f"{rel}:cell[{cell_index}]:line[{line_no}]"
                ),
            ))
    return findings


def _phase3_code_cells_unchanged(repo: Path) -> list[Finding]:
    findings: list[Finding] = []
    baseline_ref = f"refs/tags/{TIER_C_CODE_BASELINE_TAG}"
    rc, object_type, _ = _run(
        ["git", "cat-file", "-t", baseline_ref],
        repo,
    )
    if rc != 0:
        findings.append(Finding(
            id="E5.no_baseline", check="execution", severity="error",
            location="<git>",
            message=f"{baseline_ref} is missing; E5 cannot enforce Tier-C source equality",
        ))
        return findings
    if object_type.strip() != "tag":
        findings.append(Finding(
            id="E5.baseline_not_annotated", check="execution", severity="error",
            location="<git>",
            message=f"{baseline_ref} must be an annotated tag object",
            detail={"actual_object_type": object_type.strip()},
        ))
        return findings
    rc, peeled_commit, err = _run(
        ["git", "rev-parse", "--verify", f"{baseline_ref}^{{}}"],
        repo,
    )
    if rc != 0:
        findings.append(Finding(
            id="E5.baseline_target_unreadable", check="execution", severity="error",
            location="<git>",
            message=f"could not resolve the annotated Tier-C baseline target: {err.strip()[:120]}",
        ))
        return findings
    if peeled_commit.strip() != TIER_C_CODE_BASELINE_COMMIT:
        findings.append(Finding(
            id="E5.baseline_target_changed", check="execution", severity="error",
            location="<git>",
            message=f"{baseline_ref} no longer targets the reviewed Tier-C baseline commit",
            detail={
                "expected_commit": TIER_C_CODE_BASELINE_COMMIT,
                "actual_commit": peeled_commit.strip(),
            },
        ))
        return findings
    phase3 = list(_active_task_path(repo, "node_classification-reddit-gnn-pyg").glob("phase3-*.ipynb"))
    for nb in phase3:
        rel = str(nb.relative_to(repo))
        try:
            head_doc = nbformat.read(nb, as_version=4)
        except Exception as e:
            findings.append(Finding(
                id="E5.head_parse_failed", check="execution", severity="error",
                location=rel,
                message=f"HEAD notebook unparseable: {e}",
            ))
            continue
        rc, raw, err = _run(
            ["git", "show", f"{baseline_ref}:{rel}"],
            repo,
        )
        if rc != 0:
            findings.append(Finding(
                id="E5.baseline_read_failed", check="execution", severity="error",
                location=rel,
                message=f"could not read baseline: {err.strip()[:120]}",
            ))
            continue
        try:
            base_doc = nbformat.reads(raw, as_version=4)
        except Exception as e:
            findings.append(Finding(
                id="E5.baseline_parse_failed", check="execution", severity="error",
                location=rel,
                message=f"baseline notebook unparseable: {e}",
            ))
            continue
        head_codes = _code_cell_sources_for_baseline(head_doc)
        base_codes = _code_cell_sources_for_baseline(base_doc)
        if head_codes != base_codes:
            findings.append(Finding(
                id="E5.code_cells_changed", check="execution", severity="error",
                location=rel,
                message="Tier-C code cells diverged from baseline",
                detail={"head_count": len(head_codes), "base_count": len(base_codes)},
            ))
    return findings


_RUNTIME_AVAILABLE_IMPORTS = (
    "torch",
    "torch_geometric",
    "pyg_lib",
    "torch_scatter",
    "torch_sparse",
)
_TORCH_RUNTIME_IMPORTS = frozenset(_RUNTIME_AVAILABLE_IMPORTS)
_FORBIDDEN_TORCH_RUNTIME_IMPORTS = frozenset(("torch_cluster", "torch_spline_conv"))


def _runtime_available() -> bool:
    """True when the heavyweight ML runtime (torch, PyG) is importable in this env.

    The Tier-A/B/C papermill targets exercise notebooks that import torch,
    torch_geometric, and PyG's compiled extension stack. When these are
    missing, running the make targets fails with environment errors that have
    nothing to do with the notebooks' correctness — so we downgrade E1-E3 to
    env-limited skips (warning), not errors. The full execution check is
    meaningful only in the Atlas JupyterHub runtime or an equivalent
    fully-provisioned env.
    """
    return all(
        importlib.util.find_spec(import_name) is not None
        for import_name in _RUNTIME_AVAILABLE_IMPORTS
    )


def _source_hash_findings(repo: Path) -> list[Finding]:
    """Require current source hashes for retained active notebook outputs."""
    from scripts.stamp_notebook_source_hashes import compute_source_hash

    findings: list[Finding] = []
    for notebook in _iter_notebooks(repo):
        rel = _notebook_rel(notebook, repo)
        try:
            document = nbformat.read(notebook, as_version=4)
        except Exception:
            continue
        for cell_index, cell in enumerate(document.cells):
            location = f"{rel}:cell[{cell_index}]"
            metadata = cell.get("metadata", {})
            has_marker = "source_hash" in metadata
            if cell.cell_type != "code":
                if has_marker:
                    findings.append(Finding(
                        id="E8.source_hash_orphan", check="execution", severity="error",
                        location=location,
                        message=(
                            f"{cell.cell_type} cell carries metadata.source_hash; "
                            "remove it or re-run the source-hash stamper"
                        ),
                    ))
                continue
            outputs = cell.get("outputs", [])
            if not outputs:
                if has_marker:
                    findings.append(Finding(
                        id="E8.source_hash_orphan", check="execution", severity="error",
                        location=location,
                        message=(
                            "outputless code cell carries metadata.source_hash; "
                            "remove it or re-run the source-hash stamper"
                        ),
                    ))
                continue
            if not has_marker:
                findings.append(Finding(
                    id="E8.source_hash_missing", check="execution", severity="error",
                    location=location,
                    message=(
                        "output-bearing code cell is missing metadata.source_hash; "
                        "re-run the notebook and stamp its outputs"
                    ),
                ))
                continue
            marker = metadata["source_hash"]
            if not isinstance(marker, str) or re.fullmatch(r"[0-9a-f]{64}", marker) is None:
                findings.append(Finding(
                    id="E8.source_hash_invalid", check="execution", severity="error",
                    location=location,
                    message=(
                        "metadata.source_hash must be a bare lowercase SHA-256 digest; "
                        "re-run the notebook and stamp its outputs"
                    ),
                ))
                continue
            if marker != compute_source_hash(cell.source):
                findings.append(Finding(
                    id="E8.stale_output", check="execution", severity="error",
                    location=location,
                    message="cell source changed since last execution; re-run to refresh outputs",
                ))
    return findings


def check_execution(repo: Path, fast: bool) -> CheckResult:
    result = CheckResult(name="execution")
    result.findings.extend(_atlas_manifest_findings(repo))
    result.findings.extend(_atlas_task_metadata_findings(repo))
    result.findings.extend(_atlas_hardcoded_endpoint_findings(repo))

    make_tier_a = _makefile_variable_items(repo, "TIER_A")
    if not make_tier_a:
        result.findings.append(Finding(
            id="E11.tier_a_makefile_missing",
            check="execution",
            severity="error",
            location="Makefile:TIER_A",
            message="Makefile TIER_A is missing or empty; Tier-A execution contract is unenforceable",
        ))
    elif make_tier_a != TIER_A_NOTEBOOKS:
        result.findings.append(Finding(
            id="E11.tier_a_config_drift",
            check="execution",
            severity="error",
            location="Makefile:TIER_A",
            message="Makefile TIER_A drifted from scripts/verify_repo_config.yaml tier_a_notebooks",
            detail={
                "makefile_only": sorted(set(make_tier_a) - set(TIER_A_NOTEBOOKS)),
                "config_only": sorted(set(TIER_A_NOTEBOOKS) - set(make_tier_a)),
            },
        ))

    ci_tier_a_artifacts = _ci_tier_a_artifact_paths(repo)
    if not ci_tier_a_artifacts:
        result.findings.append(Finding(
            id="E12.tier_a_artifact_paths_missing",
            check="execution",
            severity="error",
            location=".github/workflows/ci.yml:tier-a-papermill",
            message="Tier-A artifact upload paths are missing or empty",
        ))
    elif ci_tier_a_artifacts != tuple(
        f"{TIER_A_CI_OUTPUT_ROOT}/{notebook}" for notebook in TIER_A_NOTEBOOKS
    ):
        result.findings.append(Finding(
            id="E12.tier_a_artifact_paths_drift",
            check="execution",
            severity="error",
            location=".github/workflows/ci.yml:tier-a-papermill",
            message="Tier-A temporary artifact paths drifted from verifier config",
            detail={
                "artifact_only": sorted(
                    set(ci_tier_a_artifacts)
                    - {f"{TIER_A_CI_OUTPUT_ROOT}/{notebook}" for notebook in TIER_A_NOTEBOOKS}
                ),
                "config_only": sorted(
                    {f"{TIER_A_CI_OUTPUT_ROOT}/{notebook}" for notebook in TIER_A_NOTEBOOKS}
                    - set(ci_tier_a_artifacts)
                ),
            },
        ))

    if not fast:
        if not _runtime_available():
            result.findings.append(Finding(
                id="E1-3.runtime_unavailable", check="execution", severity="warning",
                location="<env>",
                message=(
                    "torch / torch_geometric not importable in verifier env; "
                    "Tier-A/B/C papermill targets skipped. Run verify inside "
                    "the Atlas JupyterHub runtime for full execution coverage."
                ),
            ))
        else:
            # Timeouts mirror the CI caps in .github/workflows/ci.yml
            # (tier-a-papermill 90 min, smoke-tier-b/c 180 min each). Without
            # local caps a hung papermill cell blocks the verifier indefinitely.
            rc, _, err = _run(["make", "smoke-tier-a"], repo, timeout=5400)
            if rc != 0:
                result.findings.append(Finding(
                    id="E1.tier_a_failed", check="execution", severity="error",
                    location="Makefile:smoke-tier-a",
                    message=f"failed: {err.strip()[-300:]}",
                ))
            else:
                rc, _, err = _run(["make", "check-tier-a-artifacts"], repo)
                if rc != 0:
                    result.findings.append(Finding(
                        id="E1.tier_a_failed", check="execution", severity="error",
                        location="Makefile:check-tier-a-artifacts",
                        message=f"output check failed: {err.strip()[-300:]}",
                    ))
                else:
                    rc, _, err = _run(["make", "check-tier-a-clean"], repo)
                    if rc != 0:
                        result.findings.append(Finding(
                            id="E1.tier_a_failed", check="execution", severity="error",
                            location="Makefile:check-tier-a-clean",
                            message=f"source clean check failed: {err.strip()[-300:]}",
                        ))
            rc, _, err = _run(["make", "smoke-tier-b"], repo, timeout=10800)
            if rc != 0:
                result.findings.append(Finding(
                    id="E2.tier_b_smoke_failed", check="execution", severity="error",
                    location="Makefile:smoke-tier-b",
                    message=f"failed: {err.strip()[-300:]}",
                ))
            rc, _, err = _run(["make", "smoke-tier-c"], repo, timeout=10800)
            if rc != 0:
                result.findings.append(Finding(
                    id="E3.tier_c_smoke_failed", check="execution", severity="error",
                    location="Makefile:smoke-tier-c",
                    message=f"failed: {err.strip()[-300:]}",
                ))

    for rel in TIER_A_NOTEBOOKS:
        nb = repo / rel
        if not nb.exists():
            continue
        try:
            doc = nbformat.read(nb, as_version=4)
        except Exception:
            continue
        for ci, cell in enumerate(doc.cells):
            if cell.cell_type != "code":
                continue
            for out in cell.get("outputs", []):
                if out.get("output_type") == "error":
                    result.findings.append(Finding(
                        id="E4.cell_error", check="execution", severity="error",
                        location=f"{rel}:cell[{ci}]",
                        message=(
                            f"errored output: {out.get('ename', '?')}: "
                            f"{str(out.get('evalue', ''))[:120]}"
                        ),
                    ))

    # V7: every notebook scheduled in REQUIRED_SECTIONS that's also a
    # papermill target (Tier-A/B/C) must have a cell tagged 'parameters'.
    # Without the tag, `papermill -p NAME val` silently no-ops.
    for rel in REQUIRED_SECTIONS:
        nb = repo / rel
        if not nb.exists():
            continue
        try:
            doc = nbformat.read(nb, as_version=4)
        except Exception:
            continue
        has_params_tag = any(
            "parameters" in (c.get("metadata", {}).get("tags") or [])
            for c in doc.cells
        )
        if not has_params_tag:
            # Tag missing → papermill parameterization won't work for this
            # notebook. Warning rather than error because some notebooks
            # legitimately don't accept parameters.
            result.findings.append(Finding(
                id="E7.no_papermill_params_tag", check="execution", severity="warning",
                location=rel,
                message=(
                    "no cell tagged 'parameters'; papermill -p will silently "
                    "no-op against this notebook"
                ),
            ))
        elif "SMOKE_TEST" not in _parameters_assignment_names(doc):
            result.findings.append(Finding(
                id="E10.missing_smoke_test_parameter",
                check="execution",
                severity="error",
                location=rel,
                message=(
                    "parameters-tagged cell does not assign SMOKE_TEST; "
                    "make smoke targets pass `-p SMOKE_TEST 1`"
                ),
            ))
        result.findings.extend(_parameter_trailing_comment_findings(doc, rel))

    result.findings.extend(_source_hash_findings(repo))

    for nb in _iter_notebooks(repo):
        rel = _notebook_rel(nb, repo)
        text = _read_text(nb)
        for label, pattern in _STALE_ACTIVE_NOTEBOOK_PATHS:
            for match in pattern.finditer(text):
                line_no = text.count("\n", 0, match.start()) + 1
                result.findings.append(Finding(
                    id="E13.stale_active_notebook_path",
                    check="execution",
                    severity="warning",
                    location=f"{rel}:line[{line_no}]",
                    message=f"stale active-notebook path artifact: {label}",
                ))
        try:
            doc = nbformat.read(nb, as_version=4)
        except Exception:
            continue
        papermill_meta = doc.get("metadata", {}).get("papermill") or {}
        output_path = str(papermill_meta.get("output_path", ""))
        if papermill_meta:
            result.findings.append(Finding(
                id="E14.source_papermill_metadata",
                check="execution",
                severity="warning",
                location=rel,
                message=(
                    "active source notebook carries top-level papermill metadata; "
                    "strip generated-run metadata before committing"
                ),
            ))
        if output_path.startswith("/tmp/"):
            result.findings.append(Finding(
                id="E14.tmp_papermill_output_path",
                check="execution",
                severity="warning",
                location=rel,
                message=(
                    "notebook metadata.papermill.output_path points at /tmp; "
                    "strip or refresh papermill metadata before committing"
                ),
            ))

    result.findings.extend(_phase3_code_cells_unchanged(repo))

    for submodule in _required_submodule_paths():
        rc, out, err = _run(["git", "submodule", "status", "--", submodule], repo)
        if rc != 0:
            result.findings.append(Finding(
                id="E6.submodule_status",
                check="execution",
                severity="warning",
                location=submodule,
                message=f"could not inspect required submodule status: {(out + err).strip()[-300:]}",
            ))
            continue
        status = out.strip()
        if status.startswith(("+", "-", "U")):
            result.findings.append(Finding(
                id="E6.submodule_dirty",
                check="execution",
                severity="error",
                location=submodule,
                message=(
                    "required submodule checkout does not match the superproject "
                    "gitlink; stage the intended gitlink or run git submodule update"
                ),
                detail={"status": status},
            ))
            continue
        submodule_repo = repo / submodule
        rc, out, err = _run(["git", "status", "--porcelain", "--", "."], submodule_repo)
        if rc != 0:
            result.findings.append(Finding(
                id="E6.submodule_status",
                check="execution",
                severity="warning",
                location=submodule,
                message=f"could not inspect required submodule worktree: {(out + err).strip()[-300:]}",
            ))
            continue
        worktree_status = out.strip()
        if worktree_status:
            result.findings.append(Finding(
                id="E6.submodule_dirty",
                check="execution",
                severity="error",
                location=submodule,
                message=(
                    "required submodule checkout has local modifications; commit, "
                    "stash, or discard them before recording consumed-contract parity"
                ),
                detail={"status": worktree_status},
            ))

    for sh in _required_shellcheck_targets(repo):
        if not sh.exists():
            result.findings.append(Finding(
                id="E6.shellcheck_target_missing",
                check="execution",
                severity="error",
                location=str(sh.relative_to(repo)),
                message=(
                    "required parent-owned Atlas shellcheck target is missing"
                ),
            ))
        elif not sh.stat().st_mode & 0o111:
            result.findings.append(Finding(
                id="E6.shellcheck_target_not_executable",
                check="execution",
                severity="error",
                location=str(sh.relative_to(repo)),
                message="required parent-owned Atlas shellcheck target is not executable",
            ))

    rc_shellcheck, _, _ = _run(["which", "shellcheck"], repo)
    if rc_shellcheck != 0:
        result.findings.append(Finding(
            id="E6.shellcheck_missing", check="execution", severity="warning",
            location="<env>",
            message="shellcheck not on PATH; install with `brew install shellcheck`",
        ))
    else:
        for sh in _shellcheck_targets(repo):
            rc, out, err = _run(["shellcheck", str(sh)], repo)
            if rc != 0:
                result.findings.append(Finding(
                    id="E6.shellcheck", check="execution", severity="error",
                    location=str(sh.relative_to(repo)),
                    message=(out + err).strip()[-300:],
                ))

    return result


CHECKS: dict[str, Callable[..., CheckResult]] = {
    "structure": check_structure,
    "assets": check_assets,
    "docs": check_docs,
    "comments": check_comments,
    "execution": check_execution,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Repo verification oracle. Runs one or all checks: structure, "
            "assets, execution, docs, comments, all."
        )
    )
    parser.add_argument(
        "--check",
        choices=("structure", "assets", "execution", "docs", "comments", "all"),
        help="Which check to run.",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Skip slow checks (E1-E3 in execution). Required when only "
             "non-executable areas changed in the round.",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Path to write findings JSON. Default: print to stdout.",
    )
    parser.add_argument(
        "--repo-root", type=Path, default=REPO_ROOT,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--phase-b-out", type=Path, default=None,
        help=(
            "Path to write Phase-B comment-hygiene candidates JSON (the input "
            "to the LLM judge subagent). When set, only this is produced; "
            "the main check loop is skipped."
        ),
    )
    args = parser.parse_args(argv)

    if args.check is None and args.phase_b_out is None:
        parser.error("--check is required unless --phase-b-out is used")

    repo_root = args.repo_root.resolve()
    repo_config_path = repo_root / "scripts" / "verify_repo_config.yaml"
    if repo_config_path.exists() and repo_config_path.resolve() != CONFIG_PATH.resolve():
        _apply_config(_load_config(repo_config_path))

    if args.phase_b_out is not None:
        count = export_phase_b_candidates(repo_root, args.phase_b_out)
        print(f"verify_repo: {count} Phase-B candidates → {args.phase_b_out}", file=sys.stderr)
        return 0

    if args.check == "all":
        checks_to_run = list(CHECKS.keys())
    else:
        checks_to_run = [args.check]

    # Only check_execution respects --fast; the other checks never read it.
    results = [
        CHECKS[name](repo_root, args.fast) if name == "execution" else CHECKS[name](repo_root)
        for name in checks_to_run
    ]

    all_findings = [asdict(f) for r in results for f in r.findings]
    error_count = sum(1 for f in all_findings if f["severity"] == "error")
    warning_count = sum(1 for f in all_findings if f["severity"] == "warning")
    payload = {
        "schema_version": 1,
        "summary": {
            "checks_run": checks_to_run,
            "skipped": [r.name for r in results if r.skipped],
            "total_findings": len(all_findings),
            "errors": error_count,
            "warnings": warning_count,
            "by_check": {r.name: len(r.findings) for r in results},
            "by_check_errors": {
                r.name: sum(1 for f in r.findings if f.severity == "error")
                for r in results
            },
        },
        "findings": all_findings,
    }

    out_text = json.dumps(payload, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(out_text, encoding="utf-8")
    else:
        print(out_text)

    if error_count:
        print(
            f"verify_repo: {error_count} errors, {warning_count} warnings",
            file=sys.stderr,
        )
        return 1
    if warning_count:
        print(f"verify_repo: 0 errors, {warning_count} warnings", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
