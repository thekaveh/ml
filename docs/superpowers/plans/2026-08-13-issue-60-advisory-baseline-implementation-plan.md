# 12.18 Issue 60 Machine-Readable Advisory Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enforce the reviewed dependency-advisory policy with a deterministic JSON baseline, a
fail-closed four-surface live comparator, an offline JSON-to-ledger parity check, and a dedicated
required GitHub Actions context.

**Architecture:** `scripts/advisory_baseline.py` owns canonical JSON parsing, pip-audit
normalization, four-surface execution, and live comparison. `scripts/verify_repo.py` remains
offline and enforces parity between `security/accepted-advisories.json` and the current Markdown
ledger. An unconditional `dependency-audit` job runs exact `make audit-advisories` and becomes the
third required ruleset context after its first live PR run.

**Tech Stack:** Python 3.11 standard library, `pip-audit==2.10.0`, JSON, pytest, GNU Make, GitHub
Actions, existing MkDocs/wiki documentation pipeline.

## 12.18.1 Global Constraints

- The accepted identity key is exactly `(normalized package, primary advisory ID, accepted version)`.
- The baseline contains 21 unique identities; the Markdown ledger continues to preserve 23 raw feed records.
- The canonical surfaces, in order, are `combined-runtime`, `torch`, `documentation`, and `atlas-contract`.
- New identities, accepted-version drift, package/surface drift, invalid observations, and policy/ledger drift fail closed.
- A disappeared accepted identity produces a reconciliation notice and does not fail the live comparison.
- Aliases, fix versions, raw-record duplication, rationale, and history remain human-ledger evidence rather than baseline keys.
- Never use `--ignore-vuln`, broad suppression, automatic acceptance, or count-only comparison.
- `make verify` must remain offline and network-independent.
- Pin only the direct audit tool at `pip-audit==2.10.0`; Issue #63 owns complete dependency locking.
- Do not change NNx, Torch, PyG, Lightning, torchao, documentation, or Atlas dependency versions.
- The Atlas audit covers only `atlas-contract-requirements.txt`; do not initialize or start Atlas or any service.
- Never start containerized Ollama. No Atlas, JupyterHub, Docker, Docker Compose, Ollama, or ComfyUI operation is required.
- Preserve the existing `pytest-repository` and `atlas-consumer-policy` required contexts and every other ruleset parameter.
- Use TDD: capture the intended RED before each production change, then the focused GREEN.
- Do not commit generated `site/`, `generated/site/`, or `generated/wiki/` output.

---

## 12.18.2 Task 1: Canonical policy and live comparator

**Files:**

- Create: `security/accepted-advisories.json`
- Create: `vulnerability-audit-requirements.txt`
- Create: `scripts/advisory_baseline.py`
- Create: `tests/test_advisory_baseline.py`
- Modify: `Makefile`
- Modify: `tests/test_makefile_contract.py`

**Interfaces:**

- Produces `Baseline`, `AcceptedAdvisory`, `Observation`, `Comparison`, and `AuditSurface` immutable dataclasses.
- Produces `load_baseline(path: Path) -> Baseline`.
- Produces `normalize_pip_audit(surface: str, payload: object) -> Observation`.
- Produces `compare_baseline(baseline: Baseline, observations: Sequence[Observation]) -> Comparison`.
- Produces `run_audit_surfaces(repo: Path, runner: AuditRunner = subprocess.run) -> tuple[Observation, ...]`.
- Produces CLI `python -m scripts.advisory_baseline --repo-root PATH` and Make target `audit-advisories`.
- Task 2 consumes `load_baseline`, canonical identity/surface helpers, and the baseline dataclasses.
- Task 3 consumes the exact Make target and tool manifest.

- [x] **Step 1: Write canonical-schema RED tests**

Create tests that write temporary JSON and assert:

```python
def test_load_baseline_accepts_canonical_policy(tmp_path): ...
def test_load_baseline_rejects_unknown_root_and_item_keys(tmp_path): ...
def test_load_baseline_rejects_unsupported_schema_version(tmp_path): ...
def test_load_baseline_rejects_missing_extra_duplicate_or_reordered_surfaces(tmp_path): ...
def test_load_baseline_rejects_duplicate_identity_after_pep503_normalization(tmp_path): ...
def test_load_baseline_rejects_unknown_or_duplicate_item_surfaces(tmp_path): ...
def test_load_baseline_rejects_unsorted_items_and_noncanonical_serialization(tmp_path): ...
def test_real_baseline_contains_21_unique_reviewed_identities(): ...
```

The real-identity assertion must compare exactly this set:

```python
{
    ("pytorch-lightning", "PYSEC-2026-3043", "2.4.0"),
    ("torch", "CVE-2025-2148", "2.4.1"),
    ("torch", "CVE-2025-2149", "2.4.1"),
    ("torch", "CVE-2025-2998", "2.4.1"),
    ("torch", "CVE-2025-2999", "2.4.1"),
    ("torch", "CVE-2025-3001", "2.4.1"),
    ("torch", "PYSEC-2024-259", "2.4.1"),
    ("torch", "PYSEC-2025-191", "2.4.1"),
    ("torch", "PYSEC-2025-194", "2.4.1"),
    ("torch", "PYSEC-2025-198", "2.4.1"),
    ("torch", "PYSEC-2025-203", "2.4.1"),
    ("torch", "PYSEC-2025-204", "2.4.1"),
    ("torch", "PYSEC-2025-205", "2.4.1"),
    ("torch", "PYSEC-2025-206", "2.4.1"),
    ("torch", "PYSEC-2025-207", "2.4.1"),
    ("torch", "PYSEC-2025-208", "2.4.1"),
    ("torch", "PYSEC-2025-209", "2.4.1"),
    ("torch", "PYSEC-2025-41", "2.4.1"),
    ("torch", "PYSEC-2026-139", "2.4.1"),
    ("torch", "PYSEC-2026-1970", "2.4.1"),
    ("torch", "PYSEC-2026-2286", "2.4.1"),
}
```

- [x] **Step 2: Run schema tests and capture RED**

Run:

```bash
pytest -p no:cacheprovider tests/test_advisory_baseline.py -q -k 'load_baseline or real_baseline'
```

Expected: collection or assertions fail because the module, policy, and interfaces do not exist.

- [x] **Step 3: Implement the canonical baseline loader**

Use frozen dataclasses and exact constants:

```python
SCHEMA_VERSION = 1
SURFACE_ORDER = (
    "combined-runtime",
    "torch",
    "documentation",
    "atlas-contract",
)

@dataclass(frozen=True, order=True)
class AcceptedAdvisory:
    package: str
    advisory_id: str
    accepted_version: str
    surfaces: tuple[str, ...]

@dataclass(frozen=True)
class Baseline:
    schema_version: int
    audited_surfaces: tuple[str, ...]
    accepted_advisories: tuple[AcceptedAdvisory, ...]
```

Normalize package names with `re.sub(r"[-_.]+", "-", value).lower()`. Parse with
`json.loads`, validate exact keys/types/order, and compare the original bytes with
`json.dumps(document, indent=2, ensure_ascii=False) + "\n"`. Raise one domain exception such as
`AdvisoryBaselineError` with stable, credential-free messages.

Populate the checked JSON with the 21 identities from Step 1. Every entry currently has surfaces
`["combined-runtime", "torch"]`.

- [x] **Step 4: Run schema tests and capture GREEN**

Run the Step 2 command. Expected: all selected tests pass.

- [x] **Step 5: Write observation and comparator RED tests**

Add focused tests for:

```python
def test_normalize_accepts_clean_and_vulnerable_pip_audit_payloads(): ...
def test_normalize_collapses_duplicate_raw_records_to_one_identity(): ...
def test_normalize_rejects_malformed_payload_and_dependency_records(): ...
def test_compare_accepts_exact_observations(): ...
def test_compare_fails_on_new_primary_id(): ...
def test_compare_fails_on_package_or_surface_drift(): ...
def test_compare_fails_on_accepted_version_drift_with_advisory_present(): ...
def test_compare_fails_on_accepted_version_drift_with_empty_vulnerability_list(): ...
def test_compare_reports_removed_identity_without_failing(): ...
def test_compare_reports_removed_package_without_failing(): ...
def test_alias_only_and_fix_version_changes_do_not_change_policy_identity(): ...
```

The `Comparison` object must expose deterministic `errors` and `notices` tuples; success is
`not errors`. Removed identities belong only in `notices`.

- [x] **Step 6: Run comparator tests and capture RED**

Run:

```bash
pytest -p no:cacheprovider tests/test_advisory_baseline.py -q -k 'normalize or compare or alias'
```

Expected: failures identify the missing observation/comparison behavior.

- [x] **Step 7: Implement observation normalization and comparison**

Track both resolved versions and observed identity/surface membership. Version drift is detected
from the full dependency list, not only vulnerable rows, so a package upgraded to a clean but
unreviewed version still fails.

Use deterministic diagnostic categories:

```text
new advisory: <package> <version> <id> on <surface>
accepted version drift: <package> expected <old>; observed <new>
surface drift: <package> <version> <id> expected [...]; observed [...]
reconcile removed advisory: <package> <version> <id>
```

- [x] **Step 8: Write audit-runner and CLI RED tests**

Use a fake subprocess runner to assert:

```python
def test_audit_commands_cover_exact_four_surfaces_and_flags(): ...
def test_documentation_audit_uses_disable_pip(): ...
def test_audit_exit_zero_and_one_are_completed_observations(): ...
def test_audit_other_exit_missing_output_and_malformed_json_fail_closed(): ...
def test_cli_returns_nonzero_for_comparison_errors_and_zero_for_notices(): ...
def test_cli_has_no_accept_or_in_place_write_option(): ...
def test_cli_diagnostics_do_not_disclose_index_credentials_or_temp_paths(): ...
```

- [x] **Step 9: Run audit-runner tests and capture RED**

Run:

```bash
pytest -p no:cacheprovider tests/test_advisory_baseline.py -q -k 'audit or cli'
```

Expected: failures identify missing commands, exit handling, or CLI behavior.

- [x] **Step 10: Implement exact audit surfaces and CLI**

Each command must begin with `sys.executable -m pip_audit`, write JSON to a unique temporary
directory, retain the Issue #59 strict flags, and never include `--ignore-vuln`. Delete temporary
outputs through `TemporaryDirectory` cleanup after parsing.

- [x] **Step 11: Add focused tool manifest and Make contract with RED first**

Add tests asserting:

```python
assert Path("vulnerability-audit-requirements.txt").read_text() == "pip-audit==2.10.0\n"
assert "audit-advisories:" in makefile
assert "$(PYTHON) -m scripts.advisory_baseline" in makefile
```

Run the selected tests before editing the manifest/Makefile, then add the exact pin and target.

- [x] **Step 12: Run Task 1 GREEN gates**

Run:

```bash
pytest -p no:cacheprovider tests/test_advisory_baseline.py tests/test_makefile_contract.py -q
ruff check scripts/advisory_baseline.py tests/test_advisory_baseline.py tests/test_makefile_contract.py
python -m scripts.advisory_baseline --help
git diff --check
```

- [x] **Step 13: Commit Task 1**

```bash
git add security/accepted-advisories.json vulnerability-audit-requirements.txt \
  scripts/advisory_baseline.py tests/test_advisory_baseline.py \
  Makefile tests/test_makefile_contract.py
git commit -m "security: add accepted advisory baseline"
```

## 12.18.3 Task 2: Offline policy-to-ledger parity

**Files:**

- Modify: `scripts/verify_repo.py`
- Modify: `tests/test_verify_repo.py`

**Interfaces:**

- Consumes Task 1 `load_baseline` and canonical identity/surface semantics.
- Produces D10 findings under `D10.dependency_advisory_baseline`.
- Retains existing `D10.dependency_ledger_count`, Atlas gitlink, and workflow action-pin behavior.

- [x] **Step 1: Write JSON-policy integrity RED tests**

Create temporary-repository tests for missing policy, malformed JSON, unsupported schema, unknown
keys, duplicate or unsorted identities, wrong package/version/ID, and non-canonical bytes. Assert
all produce `D10.dependency_advisory_baseline` and that the current repository produces none.

- [x] **Step 2: Run focused verifier tests and capture RED**

```bash
pytest -p no:cacheprovider tests/test_verify_repo.py -q \
  -k 'dependency_advisory_baseline or dependency_ledger_counts_match_current_doc'
```

- [x] **Step 3: Add offline policy loading to D10**

Import Task 1's public loader rather than duplicating schema logic. Convert loader failures to
structured D10 findings; do not propagate exceptions and do not execute any network audit.

- [x] **Step 4: Write unique Markdown identity parity RED tests**

Mutate the current accepted table one dimension at a time:

```python
def test_docs_d10_flags_baseline_advisory_id_drift(tmp_path): ...
def test_docs_d10_flags_baseline_package_drift(tmp_path): ...
def test_docs_d10_flags_baseline_accepted_version_drift(tmp_path): ...
def test_docs_d10_flags_baseline_surface_drift(tmp_path): ...
def test_docs_d10_collapses_duplicate_raw_rows_for_identity_parity(tmp_path): ...
def test_docs_d10_excludes_historical_rows_from_baseline_parity(tmp_path): ...
```

Map Markdown `Combined runtime; Torch` to `("combined-runtime", "torch")`. Parse the audited
version and primary advisory ID from every current advisory row. The two duplicated raw IDs must
collapse to one unique identity before comparing with JSON.

- [x] **Step 5: Run parity tests and capture RED**

Run the Step 2 selection. Expected: new mutation tests fail because D10 does not yet compare
unique identities.

- [x] **Step 6: Implement exact two-way parity**

Report identities missing from JSON and identities missing from Markdown separately. A current
Markdown row with an unknown surface label is malformed rather than silently normalized. Keep all
existing raw-count reconciliation intact.

- [x] **Step 7: Mutation-audit the fail-closed boundary**

Temporarily mutate each of JSON ID, package, version, surface, duplicate key, ordering, and
serialization in isolated copies. Confirm each mutation produces at least one D10 error and restore
the original bytes after every probe.

- [x] **Step 8: Run Task 2 GREEN gates**

```bash
pytest -p no:cacheprovider tests/test_verify_repo.py -q \
  -k 'dependency_ledger or dependency_advisory_baseline'
pytest -p no:cacheprovider tests/test_verify_repo.py -q
make verify
ruff check scripts/verify_repo.py tests/test_verify_repo.py
git diff --check
```

- [x] **Step 9: Commit Task 2**

```bash
git add scripts/verify_repo.py tests/test_verify_repo.py
git commit -m "test: enforce advisory policy ledger parity"
```

## 12.18.4 Task 3: Isolated required CI gate

**Files:**

- Modify: `.github/workflows/ci.yml`
- Modify: `.github/workflows/docs.yml`
- Modify: `tests/test_verify_repo.py`

**Interfaces:**

- Consumes exact `make audit-advisories` and `vulnerability-audit-requirements.txt` from Task 1.
- Produces unconditional job/context `dependency-audit`.
- Produces explicit positive workflow-contract node ID executed by `verify-repo`.
- Controller updates GitHub ruleset only after a live PR establishes the context.

- [x] **Step 1: Write exact job-contract RED test**

Assert job key/name `dependency-audit`, `ubuntu-24.04`, Python `3.11`, timeout `20`, no job-level
`if`, `needs`, `services`, `container`, `env`, `defaults`, or `continue-on-error`, and exact steps:

```text
Checkout
Set up Python 3.11
Install vulnerability audit tool
Compare dependency advisories with accepted baseline
```

Checkout must disable persisted credentials and omit submodules. Setup-python must cache pip against:

```text
vulnerability-audit-requirements.txt
requirements.txt
torch-core-requirements.txt
torch-requirements.txt
docs-requirements.txt
atlas-contract-requirements.txt
```

Install command is exact `python -m pip install -r vulnerability-audit-requirements.txt`; comparison
is exact `make audit-advisories`.

- [x] **Step 2: Run job-contract test and capture RED**

```bash
pytest -p no:cacheprovider tests/test_verify_repo.py -q \
  -k 'dependency_audit_job_contract'
```

- [x] **Step 3: Add the minimal CI job**

Add it alongside existing unconditional jobs without modifying their semantics or workflow triggers.

- [x] **Step 4: Write mutation and self-contract RED tests**

Reject independently:

- deleted/renamed job or positive test;
- omitted/reordered/extra steps;
- condition, failure mask, alternate shell, job container, or services;
- removed audited manifest from the cache key;
- unpinned or alternate audit tool install;
- altered Make target, `--ignore-vuln`, or direct inline audit commands;
- Docker/Compose, Atlas lifecycle, JupyterHub, Ollama, ComfyUI, or localhost commands; and
- removal of the positive node ID from `verify-repo`'s first explicit pytest invocation.

- [x] **Step 5: Run mutation tests and capture RED**

Run the Step 2 selection plus the existing repository-workflow self-contract tests.

- [x] **Step 6: Complete the workflow contract**

Add exact
`tests/test_verify_repo.py::test_ci_dependency_audit_job_contract`
to the explicit node-ID command before the broad selector. Add `dependency_audit` keywords to the
supplemental selector without relying on that selector as the only guard.

- [x] **Step 7: Add docs-workflow path coverage with RED first**

Require the docs workflow to watch:

```text
security/accepted-advisories.json
vulnerability-audit-requirements.txt
scripts/advisory_baseline.py
tests/test_advisory_baseline.py
```

Extend the existing workflow path regression, observe RED, then update `.github/workflows/docs.yml`.

- [x] **Step 8: Run Task 3 GREEN gates**

```bash
pytest -p no:cacheprovider tests/test_verify_repo.py -q \
  -k 'dependency_audit or documentation_workflow'
pytest -p no:cacheprovider tests/test_verify_repo.py -q
python -c "from pathlib import Path; import yaml; yaml.safe_load(Path('.github/workflows/ci.yml').read_text()); yaml.safe_load(Path('.github/workflows/docs.yml').read_text())"
ruff check tests/test_verify_repo.py
git diff --check
```

- [x] **Step 9: Commit Task 3**

```bash
git add .github/workflows/ci.yml .github/workflows/docs.yml tests/test_verify_repo.py
git commit -m "ci: require dependency advisory audit"
```

## 12.18.5 Task 4: Canonical documentation and projection

**Files:**

- Modify: `README.md`
- Modify: `CONTRIBUTING.md`
- Modify: `SECURITY.md`
- Modify: `CHANGELOG.md`
- Modify: `docs/dependency-contracts.md`
- Modify: `docs/conventions.md`
- Modify: `docs/architecture.md`
- Modify: `docs/maintenance/overnight-2026-07-04.md`
- Modify: `docs/manifest.yaml`
- Modify: `tests/test_check_docs.py`
- Modify: `tests/test_build_docs.py`
- Modify: `tests/test_wiki.py`

**Interfaces:**

- Documents Task 1's policy, Task 2's parity boundary, and Task 3's required context.
- Produces manifest entry `12.18` immediately after design entry `12.17`.
- Produces identical canonical enforcement/removal language on repository, site, and wiki surfaces.

- [x] **Step 1: Write documentation-contract RED tests**

Require the canonical docs to state:

- `security/accepted-advisories.json` is the policy artifact;
- `make audit-advisories` runs all four surfaces without suppression;
- new primary IDs and accepted-version drift fail;
- disappeared IDs are reconciliation evidence, not proof of remediation;
- JSON and current Markdown rows change together through review;
- `dependency-audit` is required alongside the existing two contexts;
- Issue #63 still owns full locks; and
- Atlas/service startup is not involved.

Require manifest entries `12.17` and `12.18` to be consecutive and require the same marker language
in generated site `dependency-contracts.md` and wiki `6-1-Dependency-ledger.md` (use the actual
transform filename returned by the existing test helpers).

- [x] **Step 2: Run documentation tests and capture RED**

```bash
pytest -p no:cacheprovider tests/test_check_docs.py tests/test_build_docs.py tests/test_wiki.py -q \
  -k 'advisory or manifest or dependency'
```

- [x] **Step 3: Update canonical documents**

Make these exact semantic changes:

- README repository tree adds `security/` with the accepted-advisory policy.
- CONTRIBUTING workflow adds the local command and required CI signal.
- SECURITY §13.6 replaces the stale no-gate claim with the implemented boundary.
- Dependency contracts §6.1.1.4 documents comparison/removal; §6.1.12 no longer defers advisory comparison.
- Conventions adds an isolated dependency-audit subsection and pre-PR command.
- Architecture identifies the isolated security signal without changing an architecture diagram unless the canonical prose test proves one is required.
- OM-047 becomes `Fixed` with Issue #60's policy/job/ruleset evidence boundary.
- CHANGELOG adds the new gate and removes the Issue #59 entry's stale future-Issue-#60 sentence.
- Historical Issue #59 design/plan records remain unchanged.

- [x] **Step 4: Add manifest entry 12.18**

```yaml
- id: issue-60-advisory-baseline-implementation
  number: "12.18"
  title: Issue 60 machine-readable advisory baseline implementation plan
  source: docs/superpowers/plans/2026-08-13-issue-60-advisory-baseline-implementation-plan.md
```

- [x] **Step 5: Run documentation GREEN gates and inspect projections**

```bash
make docs-check
make docs-wiki
pytest -p no:cacheprovider tests/test_check_docs.py tests/test_build_docs.py tests/test_wiki.py -q
python scripts/verify_repo.py --check docs --fast
ruff check tests/test_check_docs.py tests/test_build_docs.py tests/test_wiki.py
git diff --check
```

Inspect generated site/wiki dependency-contract pages and navigation, then leave generated trees
ignored and uncommitted.

- [x] **Step 6: Commit Task 4**

```bash
git add README.md CONTRIBUTING.md SECURITY.md CHANGELOG.md \
  docs/dependency-contracts.md docs/conventions.md docs/architecture.md \
  docs/maintenance/overnight-2026-07-04.md docs/manifest.yaml \
  docs/superpowers/plans/2026-08-13-issue-60-advisory-baseline-implementation-plan.md \
  tests/test_check_docs.py tests/test_build_docs.py tests/test_wiki.py
git commit -m "docs: document required advisory baseline gate"
```

## 12.18.6 Task 5: Broad verification, required context, GitFlow, and cleanup

**Files:**

- No planned production-file changes; review fixes must be separate, focused commits.
- Update ignored report/ledger files under `.superpowers/sdd/` only for execution bookkeeping.

**Interfaces:**

- Consumes all prior tasks.
- Produces reviewed feature branch, live `dependency-audit` context, updated GitHub ruleset, feature-to-develop PR, develop-to-main PR, main-to-develop sync PR when ancestry requires it, and clean local/remote state.

- [x] **Step 1: Run the real live comparator**

Use the exact current manifests and fixed PyPI service:

```bash
make audit-advisories
```

Expected: exit 0, all four surfaces observed, no new identity/version/surface error. Reconciliation
notices are acceptable only if their corresponding ledger update is reviewed before merge.

- [x] **Step 2: Run full local verification**

In a canonical NNx wheel environment:

```bash
make test
make verify
make lint
make docs-check
make docs-wiki
python -c "from pathlib import Path; import yaml; yaml.safe_load(Path('.github/workflows/ci.yml').read_text()); yaml.safe_load(Path('.github/workflows/docs.yml').read_text())"
git diff --check
git status --short
```

Record exact pass/skip counts and warnings. An editable NNx run is supplemental only.

- [x] **Step 3: Perform task and broad review**

Use the subagent-driven workflow's per-task review packages and final whole-branch review. Fix every
Critical or Important finding in one focused final-review fix wave, rerun affected tests, and obtain
a clean re-review before publishing.

- [ ] **Step 4: Push and open feature-to-develop PR**

Push `codex/issue-60-advisory-baseline`, open a ready PR to `develop`, and include Issue #60 scope,
tests, no-service statement, baseline semantics, and rollback boundary.

- [ ] **Step 5: Establish and require the live context**

Wait for the feature PR's `dependency-audit` job to appear and pass. Read ruleset `18620095`, then
update only `required_status_checks` so the exact contexts are:

```text
pytest-repository
atlas-consumer-policy
dependency-audit
```

Preserve `strict_required_status_checks_policy`, `do_not_enforce_on_create`, pull-request settings,
deletion protection, non-fast-forward protection, enforcement, conditions, and bypass actors.
Read the ruleset back and compare every preserved field.

- [ ] **Step 6: Merge feature to develop**

Require all applicable checks green, merge by PR, verify the remote feature branch is deleted, and
fast-forward local `develop`.

- [ ] **Step 7: Promote develop to main**

Open a separate ready `develop` to `main` PR that closes #60. Require `dependency-audit`,
`pytest-repository`, `atlas-consumer-policy`, Tier-A, docs, verifier, NNx, Docker build, Pages/wiki,
and all other applicable checks green before merging.

- [ ] **Step 8: Synchronize main back into develop**

If the release merge makes `main` not an ancestor of `develop`, open a content-neutral `main` to
`develop` PR. Require the same applicable checks, merge it, then prove:

```bash
git diff --exit-code origin/main origin/develop
git merge-base --is-ancestor origin/main origin/develop
```

- [ ] **Step 9: Close bookkeeping and clean state**

Check every Issue #60 acceptance box, add PR/main/develop/test/ruleset evidence, mark its project
item Done, check #60 in parent #53, and leave #53 open. Fetch/prune, remove merged feature branches,
confirm one worktree, no open/superseded PRs, a clean tracked tree, and no scoped service containers.

- [ ] **Step 10: Continue the goal**

Move Issue #61 to In Progress, create and push its feature branch from synchronized `develop`, then
begin its design/research cycle without reopening completed Issue #60 work.

## 12.18.7 Portability Amendment: Selector-Free PyG Audit Projection

Linux PyG wheel resolution can produce local `+pt24cpu` versions that are valid runtime artifacts
but unsupported by strict PyPI pip-audit resolution. Keep `torch-requirements.txt` unchanged for
runtime installation. Split audit input into resolver-safe `torch-audit-requirements.txt` (core
include plus `torch_geometric`) and pre-resolved `pyg-extension-audit-requirements.txt` (the four
compiled extension pins). Each logical combined-runtime/Torch surface runs strict normal resolution
then strict `--disable-pip --no-deps` supplement resolution and merges the observations. Before
network, canonical semantic partition validation reconstructs the exact runtime manifest after
removing its selector; missing, extra, changed, duplicate, ambiguous, option, continuation, or
non-UTF-8 input fails closed. Cache and docs workflow path contracts include both manifests.
