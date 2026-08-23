"""Execution validation rules and runtime hooks."""
from __future__ import annotations

import ast
import importlib.util
import io
import re
import tokenize
from collections.abc import Callable
from pathlib import Path

import nbformat

try:
    import yaml as _yaml
except ImportError:
    _yaml = None

from . import common as _common
from .docs import _STALE_ACTIVE_NOTEBOOK_PATHS
from .models import CheckResult, Finding, RunCommand, VerifierConfig
from .structure import (
    _active_task_path,
    _notebook_rel,
    _read_text,
    _required_shellcheck_targets,
    _required_submodule_paths,
    _shellcheck_targets,
)

NOTEBOOK_ROOT = Path("notebooks")
TIER_A_CI_OUTPUT_ROOT = "/tmp/ml-tier-a"

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


def _atlas_hardcoded_endpoint_findings(
    repo: Path, config: VerifierConfig
) -> list[Finding]:
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

    for notebook_path in _common.iter_notebooks(repo, config, NOTEBOOK_ROOT):
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


def phase3_code_cells_unchanged(
    repo: Path,
    *,
    run: RunCommand,
    baseline_tag: str,
    baseline_commit: str,
) -> list[Finding]:
    findings: list[Finding] = []
    baseline_ref = f"refs/tags/{baseline_tag}"
    rc, object_type, _ = run(
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
    rc, peeled_commit, err = run(
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
    if peeled_commit.strip() != baseline_commit:
        findings.append(Finding(
            id="E5.baseline_target_changed", check="execution", severity="error",
            location="<git>",
            message=f"{baseline_ref} no longer targets the reviewed Tier-C baseline commit",
            detail={
                "expected_commit": baseline_commit,
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
        rc, raw, err = run(
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


def runtime_available() -> bool:
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


def source_hash_findings(repo: Path, config: VerifierConfig) -> list[Finding]:
    """Require current source hashes for retained active notebook outputs."""
    from scripts.stamp_notebook_source_hashes import compute_source_hash

    findings: list[Finding] = []
    for notebook in _common.iter_notebooks(repo, config, NOTEBOOK_ROOT):
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


def check_execution(
    repo: Path,
    fast: bool,
    config: VerifierConfig,
    run: RunCommand,
    runtime_available: Callable[[], bool],
    phase3_check: Callable[[Path], list[Finding]],
) -> CheckResult:
    result = CheckResult(name="execution")
    result.findings.extend(_atlas_manifest_findings(repo))
    result.findings.extend(_atlas_task_metadata_findings(repo))
    result.findings.extend(_atlas_hardcoded_endpoint_findings(repo, config))

    make_tier_a = _makefile_variable_items(repo, "TIER_A")
    if not make_tier_a:
        result.findings.append(Finding(
            id="E11.tier_a_makefile_missing",
            check="execution",
            severity="error",
            location="Makefile:TIER_A",
            message="Makefile TIER_A is missing or empty; Tier-A execution contract is unenforceable",
        ))
    elif make_tier_a != config.tier_a_notebooks:
        result.findings.append(Finding(
            id="E11.tier_a_config_drift",
            check="execution",
            severity="error",
            location="Makefile:TIER_A",
            message="Makefile TIER_A drifted from scripts/verify_repo_config.yaml tier_a_notebooks",
            detail={
                "makefile_only": sorted(set(make_tier_a) - set(config.tier_a_notebooks)),
                "config_only": sorted(set(config.tier_a_notebooks) - set(make_tier_a)),
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
        f"{TIER_A_CI_OUTPUT_ROOT}/{notebook}" for notebook in config.tier_a_notebooks
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
                    - {f"{TIER_A_CI_OUTPUT_ROOT}/{notebook}" for notebook in config.tier_a_notebooks}
                ),
                "config_only": sorted(
                    {f"{TIER_A_CI_OUTPUT_ROOT}/{notebook}" for notebook in config.tier_a_notebooks}
                    - set(ci_tier_a_artifacts)
                ),
            },
        ))

    if not fast:
        if not runtime_available():
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
            rc, _, err = run(["make", "smoke-tier-a"], repo, timeout=5400)
            if rc != 0:
                result.findings.append(Finding(
                    id="E1.tier_a_failed", check="execution", severity="error",
                    location="Makefile:smoke-tier-a",
                    message=f"failed: {err.strip()[-300:]}",
                ))
            else:
                rc, _, err = run(["make", "check-tier-a-artifacts"], repo)
                if rc != 0:
                    result.findings.append(Finding(
                        id="E1.tier_a_failed", check="execution", severity="error",
                        location="Makefile:check-tier-a-artifacts",
                        message=f"output check failed: {err.strip()[-300:]}",
                    ))
                else:
                    rc, _, err = run(["make", "check-tier-a-clean"], repo)
                    if rc != 0:
                        result.findings.append(Finding(
                            id="E1.tier_a_failed", check="execution", severity="error",
                            location="Makefile:check-tier-a-clean",
                            message=f"source clean check failed: {err.strip()[-300:]}",
                        ))
            rc, _, err = run(["make", "smoke-tier-b"], repo, timeout=10800)
            if rc != 0:
                result.findings.append(Finding(
                    id="E2.tier_b_smoke_failed", check="execution", severity="error",
                    location="Makefile:smoke-tier-b",
                    message=f"failed: {err.strip()[-300:]}",
                ))
            rc, _, err = run(["make", "smoke-tier-c"], repo, timeout=10800)
            if rc != 0:
                result.findings.append(Finding(
                    id="E3.tier_c_smoke_failed", check="execution", severity="error",
                    location="Makefile:smoke-tier-c",
                    message=f"failed: {err.strip()[-300:]}",
                ))

    for rel in config.tier_a_notebooks:
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
    for rel in config.required_sections:
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

    result.findings.extend(source_hash_findings(repo, config))

    for nb in _common.iter_notebooks(repo, config, NOTEBOOK_ROOT):
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

    result.findings.extend(phase3_check(repo))

    for submodule in _required_submodule_paths():
        rc, out, err = run(["git", "submodule", "status", "--", submodule], repo)
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
        rc, out, err = run(["git", "status", "--porcelain", "--", "."], submodule_repo)
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

    rc_shellcheck, _, _ = run(["which", "shellcheck"], repo)
    if rc_shellcheck != 0:
        result.findings.append(Finding(
            id="E6.shellcheck_missing", check="execution", severity="warning",
            location="<env>",
            message="shellcheck not on PATH; install with `brew install shellcheck`",
        ))
    else:
        for sh in _shellcheck_targets(repo):
            rc, out, err = run(["shellcheck", str(sh)], repo)
            if rc != 0:
                result.findings.append(Finding(
                    id="E6.shellcheck", check="execution", severity="error",
                    location=str(sh.relative_to(repo)),
                    message=(out + err).strip()[-300:],
                ))

    return result
