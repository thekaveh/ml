"""Documentation validation rules for the repository verifier."""
from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import nbformat

from . import common as _common
from .docs_dependencies import (
    _DEPENDENCY_HASH_INPUTS,
    _atlas_current_pin_projection_findings,
    _mask_dependency_raw_html,
    _strip_markdown_code,
    _torch_runtime_contract_findings,
    _workflow_action_pin_findings,
    _read_text,
    dependency_advisory_lock_findings,
    dependency_ledger_findings,
    dependency_lock_findings,
)
from .models import CheckResult, Finding, RunCommand, VerifierConfig

NOTEBOOK_ROOT = Path("notebooks")

__all__ = (
    "_DEPENDENCY_HASH_INPUTS",
    "_atlas_current_pin_projection_findings",
    "_mask_dependency_raw_html",
    "_strip_markdown_code",
    "check_docs",
    "dependency_advisory_lock_findings",
    "dependency_ledger_findings",
    "dependency_lock_findings",
)


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
