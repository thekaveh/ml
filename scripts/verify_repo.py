#!/usr/bin/env python3
"""Repo verification oracle.

Runs five orthogonal checks (structure, assets, execution, docs, comments) and emits
machine-readable findings JSON + a human-readable report. Exit code 0 = no
error-severity findings (warnings are allowed and reported but don't fail
the run); 1 = at least one error finding (counts on stderr).
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Callable, Iterator
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
CONFIG_PATH = Path(__file__).resolve().parent / "verify_repo_config.yaml"
_HELP_REQUESTED = any(arg in ("-h", "--help") for arg in sys.argv[1:])

_yaml = None
if not _HELP_REQUESTED:
    try:
        import yaml as _yaml  # PyYAML
    except ImportError:
        pass


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
if not _HELP_REQUESTED:
    _apply_config(_load_config())

if not _HELP_REQUESTED:
    from scripts.repo_verifier import assets as _assets_validator
    from scripts.repo_verifier import comments as _comments_validator
    from scripts.repo_verifier import common as _common
    from scripts.repo_verifier import docs as _docs_validator
    from scripts.repo_verifier import execution as _execution_validator
    from scripts.repo_verifier.models import CheckResult, Finding, VerifierConfig
    from scripts.repo_verifier import structure as _structure_validator


def _config_snapshot() -> VerifierConfig:
    return VerifierConfig(
        active_task_dirs=tuple(ACTIVE_TASK_DIRS),
        required_sections={key: tuple(value) for key, value in REQUIRED_SECTIONS.items()},
        tier_a_notebooks=tuple(TIER_A_NOTEBOOKS),
    )

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
    _strip_markdown_code = _docs_validator._strip_markdown_code
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

_RUNTIME_ONLY_MODULES = frozenset({
    "numpy", "torch", "torchvision", "torch_geometric", "torch_sparse",
    "torch_scatter", "pyg_lib", "matplotlib", "seaborn", "pandas", "sklearn",
    "scipy", "networkx", "community", "nnx", "tqdm",
})

# This declaration is intentionally facade-owned: the documentation validator
# statically verifies the public runtime contract in this executable script.
_RUNTIME_AVAILABLE_IMPORTS = (
    "torch",
    "torch_geometric",
    "pyg_lib",
    "torch_scatter",
    "torch_sparse",
)
_TORCH_RUNTIME_IMPORTS = frozenset(_RUNTIME_AVAILABLE_IMPORTS)
_FORBIDDEN_TORCH_RUNTIME_IMPORTS = frozenset(("torch_cluster", "torch_spline_conv"))


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


if not _HELP_REQUESTED:
    # Compatibility export for the JupyterHub, NumPy, and PyTorch terminology contract.
    TERMINOLOGY_CANONICALS = _docs_validator.TERMINOLOGY_CANONICALS
    _qualified_name = _assets_validator._qualified_name
    _notebook_nlp_asset_contract = _assets_validator._notebook_nlp_asset_contract
    _nlp_asset_contract_findings = _assets_validator._nlp_asset_contract_findings
    _mask_dependency_raw_html = _docs_validator._mask_dependency_raw_html
    _DEPENDENCY_HASH_INPUTS = _docs_validator._DEPENDENCY_HASH_INPUTS
    _STALE_ACTIVE_NOTEBOOK_PATHS = _docs_validator._STALE_ACTIVE_NOTEBOOK_PATHS
    _atlas_current_pin_projection_findings = (
        _docs_validator._atlas_current_pin_projection_findings
    )
    _STATE_THE_WHAT_PATTERNS = _comments_validator._STATE_THE_WHAT_PATTERNS
    _scan_source_for_comments = _comments_validator._scan_source_for_comments


def _iter_in_scope_code(repo: Path):
    yield from _comments_validator._iter_in_scope_code(repo, _config_snapshot())


def _dependency_ledger_findings(repo: Path) -> list[Finding]:
    return _docs_validator.dependency_ledger_findings(repo, run=_run)


def _dependency_lock_findings(repo: Path) -> list[Finding]:
    return _docs_validator.dependency_lock_findings(repo)


def _dependency_advisory_lock_findings(repo: Path) -> list[Finding]:
    return _docs_validator.dependency_advisory_lock_findings(repo)


def check_structure(repo: Path) -> CheckResult:
    return _structure_validator.check_structure(repo, _config_snapshot())


def check_assets(repo: Path) -> CheckResult:
    return _assets_validator.check_assets(repo, _config_snapshot())


def check_docs(repo: Path) -> CheckResult:
    return _docs_validator.check_docs(repo, _config_snapshot(), _run)



def check_comments(repo: Path) -> CheckResult:
    return _comments_validator.check_comments(repo, _config_snapshot())


def export_phase_b_candidates(repo: Path, out_path: Path) -> int:
    return _comments_validator.export_phase_b_candidates(
        repo, out_path, _config_snapshot()
    )


if not _HELP_REQUESTED:
    _subprocess_text = _common._subprocess_text
    subprocess = _common.subprocess
    _run = _common.run_command
    _runtime_importlib = importlib

    _cell_tags = _execution_validator._cell_tags
    _is_parameters_cell = _execution_validator._is_parameters_cell
    _code_cell_sources_for_baseline = _execution_validator._code_cell_sources_for_baseline
    _parameter_trailing_comment_findings = (
        _execution_validator._parameter_trailing_comment_findings
    )
    _assignment_names = _execution_validator._assignment_names
    _parameters_assignment_names = _execution_validator._parameters_assignment_names
    _makefile_variable_items = _execution_validator._makefile_variable_items
    _ci_tier_a_artifact_paths = _execution_validator._ci_tier_a_artifact_paths
    _ATLAS_MANIFEST_CONTRACT = _execution_validator._ATLAS_MANIFEST_CONTRACT
    _ATLAS_REQUIRED_PARENT_FILES = _execution_validator._ATLAS_REQUIRED_PARENT_FILES
    _ATLAS_ENDPOINT_RE = _execution_validator._ATLAS_ENDPOINT_RE
    _ATLAS_INTEGRATION_SOURCE_GLOBS = _execution_validator._ATLAS_INTEGRATION_SOURCE_GLOBS
    _atlas_manifest_findings = _execution_validator._atlas_manifest_findings
    _atlas_task_metadata_findings = _execution_validator._atlas_task_metadata_findings
    _is_ipython_magic_or_help_line = _execution_validator._is_ipython_magic_or_help_line
    _blank_source_span = _execution_validator._blank_source_span
    _ast_column_to_character_index = _execution_validator._ast_column_to_character_index
    _blank_ast_span = _execution_validator._blank_ast_span
    _python_lines_without_comments = _execution_validator._python_lines_without_comments
    _shell_line_without_comment = _execution_validator._shell_line_without_comment
    _endpoint_findings_for_lines = _execution_validator._endpoint_findings_for_lines
    _atlas_hardcoded_endpoint_findings = _execution_validator._atlas_hardcoded_endpoint_findings
    _runtime_available = _execution_validator.runtime_available

def _source_hash_findings(repo: Path) -> list[Finding]:
    return _execution_validator.source_hash_findings(repo, _config_snapshot())

def _phase3_code_cells_unchanged(repo: Path) -> list[Finding]:
    return _execution_validator.phase3_code_cells_unchanged(
        repo,
        run=_run,
        baseline_tag=TIER_C_CODE_BASELINE_TAG,
        baseline_commit=TIER_C_CODE_BASELINE_COMMIT,
    )

def check_execution(repo: Path, fast: bool) -> CheckResult:
    return _execution_validator.check_execution(
        repo,
        fast,
        _config_snapshot(),
        _run,
        _runtime_available,
        _phase3_code_cells_unchanged,
    )


CHECKS: dict[str, Callable[..., CheckResult]] = {
    "structure": check_structure,
    # D11.nlp_asset_contract remains a public assets-check contract; its
    # implementation is owned by repo_verifier.assets.
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
