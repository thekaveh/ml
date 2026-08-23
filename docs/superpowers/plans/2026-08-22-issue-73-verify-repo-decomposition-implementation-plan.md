# Repository Verifier Decomposition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decompose `scripts/verify_repo.py` into cohesive validator modules while preserving every existing CLI, finding, configuration, timeout, and test contract.

**Architecture:** Keep `scripts/verify_repo.py` as a deferred-import CLI/configuration facade. Move data models and shared primitives into `scripts/repo_verifier/`, then move structure, assets, docs, comments, and execution responsibilities into separate modules whose config and runtime dependencies are explicit arguments. Facade wrappers snapshot current globals and inject currently bound hooks so existing imports and monkeypatches remain compatible.

**Tech Stack:** Python 3.11, argparse, dataclasses, pathlib, nbformat, PyYAML, pytest, Ruff, Git/GitHub Actions.

## Global Constraints

- Preserve CLI arguments, help text, check choices/order, JSON schema, finding identifiers, severities, messages, output routing, diagnostics, and exit codes.
- Preserve `DEFAULT_SUBPROCESS_TIMEOUT = 120` and timeout return code `124`, including partial byte-stream normalization.
- Preserve facade-level monkeypatch behavior for `ACTIVE_TASK_DIRS`, `REQUIRED_SECTIONS`, `TIER_A_NOTEBOOKS`, `_run`, `_runtime_available`, `_phase3_code_cells_unchanged`, and Tier-C baseline constants.
- Do not add verifier features, finding types, plugins, parallelism, notebook changes, dependency-policy changes, or Atlas startup.
- Each extraction commit must be independently revertible and must keep focused tests green.
- Final `scripts/verify_repo.py` must be at most 450 lines; no extracted module may exceed 1,200 lines.

---

### Task 1: Characterize the Existing Facade

**Files:**
- Create: `tests/test_verify_repo_facade.py`
- Test: `tests/test_verify_repo.py`

**Interfaces:**
- Consumes: current `scripts/verify_repo.py` public `main(argv)` and `CHECKS` mapping.
- Produces: exact regression coverage for dispatch, aggregation, exit diagnostics, output routing, and Phase-B bypass before production movement.

- [ ] **Step 1: Add facade characterization helpers and dispatch tests**

Create a fresh module loader and validator stubs. The core test must use the current public objects rather than copied verifier logic:

```python
def _load_facade():
    name = "verify_repo_facade_contract"
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_all_dispatch_preserves_order_and_forwards_fast_only_to_execution(
    tmp_path, monkeypatch, capsys
):
    facade = _load_facade()
    calls = []

    def ordinary(name):
        def check(repo):
            calls.append((name, repo, None))
            return facade.CheckResult(name)
        return check

    def execution(repo, fast):
        calls.append(("execution", repo, fast))
        return facade.CheckResult("execution")

    monkeypatch.setattr(facade, "CHECKS", {
        "structure": ordinary("structure"),
        "assets": ordinary("assets"),
        "docs": ordinary("docs"),
        "comments": ordinary("comments"),
        "execution": execution,
    })
    assert facade.main(["--check", "all", "--fast", "--repo-root", str(tmp_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [name for name, _, _ in calls] == [
        "structure", "assets", "docs", "comments", "execution"
    ]
    assert calls[-1][2] is True
    assert payload["summary"]["checks_run"] == [
        "structure", "assets", "docs", "comments", "execution"
    ]
```

- [ ] **Step 2: Characterize warning/error aggregation and output routing**

Add separate tests using real `Finding` and `CheckResult` instances. Assert warning-only returns 0 with `verify_repo: 0 errors, 1 warnings`, error returns 1 with `verify_repo: 1 errors, 0 warnings`, `--out` writes schema-version-1 JSON without stdout, and `--phase-b-out` calls only the export function.

```python
finding = facade.Finding(
    id="X1.test", check="structure", severity="warning",
    location="fixture", message="characterized",
)
monkeypatch.setattr(
    facade, "CHECKS", {"structure": lambda _repo: facade.CheckResult("structure", [finding])}
)
```

- [ ] **Step 3: Run the new characterization tests against the monolith**

Run: `pytest tests/test_verify_repo_facade.py -q`

Expected: PASS. These tests characterize existing behavior before extraction; a failure is a test mistake to correct before production changes.

- [ ] **Step 4: Confirm the existing timeout characterization remains green**

Run: `pytest tests/test_verify_repo.py -q -k 'run_helper_timeout or run_helper_supplies_default_timeout'`

Expected: all selected tests PASS.

- [ ] **Step 5: Commit the characterization baseline**

```bash
git add tests/test_verify_repo_facade.py
git commit -m "test: characterize verifier facade contract"
```

---

### Task 2: Establish Models, Shared Primitives, and the Architecture Contract

**Files:**
- Create: `scripts/repo_verifier/__init__.py`
- Create: `scripts/repo_verifier/models.py`
- Create: `scripts/repo_verifier/common.py`
- Create: `tests/test_verify_repo_architecture.py`
- Modify: `scripts/verify_repo.py`

**Interfaces:**
- Produces: frozen `VerifierConfig`, `Finding`, `CheckResult`, shared `run_command`, and parameterized repository iteration.
- Preserves: facade exports `Finding`, `CheckResult`, `_run`, and config globals.

- [ ] **Step 1: Write the failing architecture test**

```python
def test_verifier_config_is_an_immutable_value_snapshot():
    from scripts.repo_verifier.models import VerifierConfig

    config = VerifierConfig(
        active_task_dirs=("task",),
        required_sections={"notebooks/task/main.ipynb": ("1. Overview",)},
        tier_a_notebooks=("notebooks/task/main.ipynb",),
    )
    assert config.active_task_dirs == ("task",)
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.active_task_dirs = ()
    with pytest.raises(TypeError):
        config.required_sections["new"] = ()
```

- [ ] **Step 2: Run the architecture test to verify RED**

Run: `pytest tests/test_verify_repo_architecture.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.repo_verifier'`.

- [ ] **Step 3: Add the minimal models and common interfaces**

```python
@dataclass(frozen=True)
class VerifierConfig:
    active_task_dirs: tuple[str, ...]
    required_sections: Mapping[str, tuple[str, ...]]
    tier_a_notebooks: tuple[str, ...]

    def __post_init__(self) -> None:
        snapshot = {
            key: tuple(value) for key, value in self.required_sections.items()
        }
        object.__setattr__(self, "required_sections", MappingProxyType(snapshot))


class RunCommand(Protocol):
    def __call__(
        self, cmd: list[str], cwd: Path, timeout: int | None = None
    ) -> tuple[int, str, str]: ...
```

Move `Finding`, `CheckResult`, `_subprocess_text`, and `_run` without changing their bodies; rename the package implementation to `run_command` and retain `_run = run_command` in the facade. Parameterize shared notebook/text iterators with `VerifierConfig` rather than module globals.

- [ ] **Step 4: Make the facade snapshot current globals**

```python
def _config_snapshot() -> VerifierConfig:
    return VerifierConfig(
        active_task_dirs=tuple(ACTIVE_TASK_DIRS),
        required_sections={key: tuple(value) for key, value in REQUIRED_SECTIONS.items()},
        tier_a_notebooks=tuple(TIER_A_NOTEBOOKS),
    )
```

- [ ] **Step 5: Run architecture, facade, and selected existing tests**

Run: `pytest tests/test_verify_repo_architecture.py tests/test_verify_repo_facade.py tests/test_verify_repo.py -q -k 'run_helper or iter_notebooks or facade or VerifierConfig'`

Expected: all selected tests PASS.

- [ ] **Step 6: Commit the foundation**

```bash
git add scripts/repo_verifier scripts/verify_repo.py tests/test_verify_repo_architecture.py
git commit -m "refactor(verifier): add explicit shared context"
```

---

### Task 3: Extract Structure and Asset Validators

**Files:**
- Create: `scripts/repo_verifier/structure.py`
- Create: `scripts/repo_verifier/assets.py`
- Modify: `scripts/verify_repo.py`
- Modify: `tests/test_verify_repo_architecture.py`

**Interfaces:**
- Produces: `structure.check_structure(repo, config)` and `assets.check_assets(repo, config)`.
- Preserves facade: `check_structure(repo)`, `check_assets(repo)`, `_strip_markdown_code`, `_mask_dependency_raw_html` remains docs-owned later, and `_nlp_asset_contract_findings(repo)`.

- [ ] **Step 1: Add failing ownership and delegation tests**

```python
def test_structure_facade_passes_current_config(monkeypatch, tmp_path):
    facade = load_facade()
    monkeypatch.setattr(facade, "ACTIVE_TASK_DIRS", ("changed",))
    seen = {}
    monkeypatch.setattr(
        facade._structure_validator,
        "check_structure",
        lambda repo, config: seen.update(repo=repo, config=config) or facade.CheckResult("structure"),
    )
    facade.check_structure(tmp_path)
    assert seen["config"].active_task_dirs == ("changed",)
```

Run: `pytest tests/test_verify_repo_architecture.py -q`

Expected: FAIL because the validator modules/delegates do not exist.

- [ ] **Step 2: Move structure symbols without algorithm edits**

Move import parsing, Markdown masking/link helpers, structure constants, and `check_structure` to `structure.py`. Change only config/global reads to `config.active_task_dirs` or shared helpers. The facade wrapper is:

```python
def check_structure(repo: Path) -> CheckResult:
    return _structure_validator.check_structure(repo, _config_snapshot())
```

- [ ] **Step 3: Move NLP asset symbols without algorithm edits**

Move `_qualified_name`, `_notebook_nlp_asset_contract`, `_nlp_asset_contract_findings`, and `check_assets` to `assets.py`. Preserve direct helper aliases and use a facade wrapper for config-sensitive calls.

- [ ] **Step 4: Run red/green focused suites**

Run: `pytest tests/test_verify_repo_architecture.py tests/test_verify_repo_facade.py tests/test_verify_repo.py -q -k 'structure or imported_modules or markdown or nlp_asset or facade'`

Expected: all selected tests PASS.

- [ ] **Step 5: Verify both real checks**

Run: `python scripts/verify_repo.py --check structure --fast && python scripts/verify_repo.py --check assets --fast`

Expected: exit 0 and no error findings.

- [ ] **Step 6: Commit the extraction**

```bash
git add scripts/repo_verifier/structure.py scripts/repo_verifier/assets.py scripts/verify_repo.py tests/test_verify_repo_architecture.py
git commit -m "refactor(verifier): extract structure and asset checks"
```

---

### Task 4: Extract the Documentation Validator

**Files:**
- Create: `scripts/repo_verifier/docs.py`
- Modify: `scripts/verify_repo.py`
- Modify: `tests/test_verify_repo_architecture.py`

**Interfaces:**
- Produces: `docs.check_docs(repo, config, run)` and existing docs helper functions.
- Preserves facade: `check_docs(repo)`, `_dependency_ledger_findings(repo)`, `_dependency_lock_findings(repo)`, `_dependency_advisory_lock_findings(repo)`, `_mask_dependency_raw_html`, `_strip_markdown_code`, `_DEPENDENCY_HASH_INPUTS`, and facade `_run` injection.

- [ ] **Step 1: Add a failing run-hook delegation test**

```python
def test_docs_facade_injects_current_run_hook(monkeypatch, tmp_path):
    facade = load_facade()
    sentinel = lambda cmd, cwd, timeout=None: (0, "sentinel", "")
    monkeypatch.setattr(facade, "_run", sentinel)
    seen = {}
    monkeypatch.setattr(
        facade._docs_validator,
        "check_docs",
        lambda repo, config, run: seen.update(run=run) or facade.CheckResult("docs"),
    )
    facade.check_docs(tmp_path)
    assert seen["run"] is sentinel
```

Run: `pytest tests/test_verify_repo_architecture.py -q`

Expected: FAIL before `_docs_validator` exists.

- [ ] **Step 2: Move the complete docs responsibility**

Move numbered-heading, stale-layout, dependency-ledger/advisory/hash, workflow-action-pin, protected-binding, Torch runtime-contract helpers, docs constants, and `check_docs`. Replace internal `_run` reads with the injected `run` argument. Keep finding construction and ordering byte-for-byte equivalent.

```python
def check_docs(repo: Path) -> CheckResult:
    return _docs_validator.check_docs(repo, _config_snapshot(), _run)


def _dependency_ledger_findings(repo: Path) -> list[Finding]:
    return _docs_validator.dependency_ledger_findings(repo, run=_run)
```

- [ ] **Step 3: Run focused docs and injection tests**

Run: `pytest tests/test_verify_repo_architecture.py tests/test_verify_repo.py -q -k 'docs or dependency_ledger or dependency_lock or workflow_action_pin or torch_runtime_contract'`

Expected: all selected tests PASS, including facade-level `_run` monkeypatch cases.

- [ ] **Step 4: Verify the real docs check**

Run: `python scripts/verify_repo.py --check docs --fast`

Expected: exit 0 and no error findings.

- [ ] **Step 5: Commit the extraction**

```bash
git add scripts/repo_verifier/docs.py scripts/verify_repo.py tests/test_verify_repo_architecture.py
git commit -m "refactor(verifier): extract documentation checks"
```

---

### Task 5: Extract Comment Hygiene and Phase-B Export

**Files:**
- Create: `scripts/repo_verifier/comments.py`
- Modify: `scripts/verify_repo.py`
- Modify: `tests/test_verify_repo_architecture.py`

**Interfaces:**
- Produces: `comments.check_comments(repo, config)` and `comments.export_phase_b_candidates(repo, out_path, config)`.
- Preserves facade and CLI Phase-B output schema/diagnostic.

- [ ] **Step 1: Add failing comment/export delegation tests**

Patch the future module functions, change facade `ACTIVE_TASK_DIRS`, and assert both wrappers receive a snapshot containing the changed value.

```python
assert facade.export_phase_b_candidates(tmp_path, tmp_path / "out.json") == 7
assert seen["config"].active_task_dirs == ("changed",)
```

Run: `pytest tests/test_verify_repo_architecture.py -q`

Expected: FAIL before `_comments_validator` exists.

- [ ] **Step 2: Move comment symbols and add wrappers**

Move `_STATE_THE_WHAT_PATTERNS`, `_scan_source_for_comments`, `_iter_in_scope_code`, `check_comments`, and `export_phase_b_candidates`. Preserve JSON key order and the source/snippet extraction algorithm.

- [ ] **Step 3: Run focused comment and facade tests**

Run: `pytest tests/test_verify_repo_architecture.py tests/test_verify_repo_facade.py tests/test_verify_repo.py -q -k 'comment or phase_b or facade'`

Expected: all selected tests PASS.

- [ ] **Step 4: Commit the extraction**

```bash
git add scripts/repo_verifier/comments.py scripts/verify_repo.py tests/test_verify_repo_architecture.py
git commit -m "refactor(verifier): extract comment hygiene checks"
```

---

### Task 6: Extract the Execution Validator and Runtime Hooks

**Files:**
- Create: `scripts/repo_verifier/execution.py`
- Modify: `scripts/verify_repo.py`
- Modify: `tests/test_verify_repo_architecture.py`

**Interfaces:**
- Produces: execution helpers and `execution.check_execution(repo, fast, config, run, runtime_available, phase3_check)`.
- Preserves facade: all execution helper names used by tests, Tier-C baseline constants, `_run`, `_runtime_available`, `_phase3_code_cells_unchanged`, and `check_execution(repo, fast)`.

- [ ] **Step 1: Add failing runtime-hook delegation tests**

```python
def test_execution_facade_injects_current_runtime_hooks(monkeypatch, tmp_path):
    facade = load_facade()
    run = lambda cmd, cwd, timeout=None: (0, "", "")
    runtime = lambda: True
    phase3 = lambda repo: []
    monkeypatch.setattr(facade, "_run", run)
    monkeypatch.setattr(facade, "_runtime_available", runtime)
    monkeypatch.setattr(facade, "_phase3_code_cells_unchanged", phase3)
    seen = {}
    monkeypatch.setattr(
        facade._execution_validator,
        "check_execution",
        lambda repo, fast, config, run, runtime_available, phase3_check:
            seen.update(run=run, runtime=runtime_available, phase3=phase3_check)
            or facade.CheckResult("execution"),
    )
    facade.check_execution(tmp_path, fast=True)
    assert (seen["run"], seen["runtime"], seen["phase3"]) == (run, runtime, phase3)
```

Run: `pytest tests/test_verify_repo_architecture.py -q`

Expected: FAIL before `_execution_validator` exists.

- [ ] **Step 2: Move execution helpers and constants**

Move cell-tag/parameter helpers, Makefile/workflow inventory parsing, Atlas static checks, IPython/comment masking, endpoint checks, Tier-C baseline comparison, runtime probe, source-hash checks, and `check_execution`. Convert reads of task inventories to `config`, commands to injected `run`, runtime probing to injected `runtime_available`, and the baseline call to injected `phase3_check`.

Keep facade wrappers for config-sensitive helpers:

```python
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
        repo, fast, _config_snapshot(), _run, _runtime_available,
        _phase3_code_cells_unchanged,
    )
```

- [ ] **Step 3: Run focused execution and hook tests**

Run: `pytest tests/test_verify_repo_architecture.py tests/test_verify_repo.py -q -k 'execution or source_hash or tier_a or parameter or atlas_manifest or hardcoded_endpoint or shellcheck or submodule or run_helper'`

Expected: all selected tests PASS, including every facade monkeypatch case.

- [ ] **Step 4: Verify the real execution check**

Run: `python scripts/verify_repo.py --check execution --fast`

Expected: exit 0 with current repository findings unchanged from the baseline.

- [ ] **Step 5: Commit the extraction**

```bash
git add scripts/repo_verifier/execution.py scripts/verify_repo.py tests/test_verify_repo_architecture.py
git commit -m "refactor(verifier): extract execution checks"
```

---

### Task 7: Finalize the Facade, Record Metrics, and Qualify

**Files:**
- Modify: `scripts/verify_repo.py`
- Modify: `scripts/repo_verifier/__init__.py`
- Modify: `tests/test_verify_repo_architecture.py`
- Modify: `docs/maintenance/overnight-2026-07-04.md`

**Interfaces:**
- Produces: deferred-import help-safe facade no larger than 450 lines and documented before/after evidence.
- Preserves: all public/compatibility interfaces from Tasks 1–6.

- [ ] **Step 1: Make copied-script help independent of the package**

Keep parser construction in the facade and skip package/config imports when
`_HELP_REQUESTED` is true, allowing `argparse` to exit from `--help` first.
Verify the existing copied-script help test before deleting any legacy
definitions.

Run: `pytest tests/test_verify_repo.py::test_help_does_not_require_adjacent_config tests/test_verify_repo_facade.py -q`

Expected: PASS.

- [ ] **Step 2: Delete superseded monolith definitions and enforce architecture limits**

Add an AST/line-count architecture test:

```python
def test_verifier_modules_stay_within_decomposition_boundaries():
    facade = REPO / "scripts" / "verify_repo.py"
    modules = sorted((REPO / "scripts" / "repo_verifier").glob("*.py"))
    assert len(facade.read_text().splitlines()) <= 450
    assert max(len(path.read_text().splitlines()) for path in modules) <= 1200
    assert tuple(FACADE.CHECKS) == (
        "structure", "assets", "docs", "comments", "execution"
    )
```

Run: `pytest tests/test_verify_repo_architecture.py -q`

Expected: PASS only after all duplicate legacy definitions are removed.

- [ ] **Step 3: Record exact before/after metrics and resolve OM-022 for this script**

Run an AST measurement over the facade and package, record line counts and per-file branch sums, and update OM-022 to say issue #73 resolved `verify_repo.py` while `rewrite_imports.py` remains tracked by issue #74.

```bash
python - <<'PY'
import ast
from pathlib import Path
for path in [Path("scripts/verify_repo.py"), *sorted(Path("scripts/repo_verifier").glob("*.py"))]:
    tree = ast.parse(path.read_text())
    branches = sum(isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.Match, ast.BoolOp, ast.comprehension)) for node in ast.walk(tree))
    print(path, len(path.read_text().splitlines()), branches)
PY
```

- [ ] **Step 4: Run complete focused verifier qualification**

Run: `pytest tests/test_verify_repo.py tests/test_verify_repo_facade.py tests/test_verify_repo_architecture.py -q`

Expected: all tests PASS.

- [ ] **Step 5: Run the repository verifier and Ruff**

Run: `python scripts/verify_repo.py --check all --fast && ruff check .`

Expected: verifier exit 0 with zero errors; Ruff exit 0.

- [ ] **Step 6: Run strict docs and full repository tests**

Run the repository's canonical strict docs check/build/wiki commands, then `pytest tests/ -q` and `python -m pip check`.

Expected: all commands exit 0; full pytest reports zero failures.

- [ ] **Step 7: Verify the final diff and requirements**

Run: `git diff --check && git status --short && git diff --stat origin/develop...HEAD`

Expected: only issue #73 code, tests, design/plan, and maintenance documentation are changed; no notebook, config, dependency, or submodule gitlink changes.

- [ ] **Step 8: Commit final metrics/documentation**

```bash
git add scripts/verify_repo.py scripts/repo_verifier tests/test_verify_repo_architecture.py docs/maintenance/overnight-2026-07-04.md
git commit -m "docs: record verifier decomposition evidence"
```

- [ ] **Step 9: Request independent review and address findings test-first**

Review the complete range from `411a04af23c3bf6be24c6246e93ef3e7ce14ba24` through feature HEAD against the design and this plan. Fix every Critical and Important issue, evaluate Minor issues technically, rerun affected tests, and then repeat the complete qualification before push/PR.
