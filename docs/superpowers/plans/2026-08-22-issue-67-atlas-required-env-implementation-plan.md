# 12.32 Issue 67 Atlas Required-Environment Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fail-closed, machine-readable Atlas environment-requirement contract and project it deterministically across repository, site, and wiki documentation without admitting a new service.

**Architecture:** Extend the existing notebook-infrastructure parser with a frozen `AtlasEnvironmentRequirement` value object and a required `atlas.required_env` list of exact `{name, service}` mappings. Link every non-JupyterHub service to at least one injected variable, migrate all current JupyterHub-only task specs to explicit empty lists, and add a value-free deterministic matrix column.

**Tech Stack:** Python 3.11, frozen dataclasses, PyYAML, pytest, Markdown generation, MkDocs, GitHub Actions, GitFlow.

**Spec:** `docs/superpowers/specs/2026-08-22-issue-67-atlas-required-env-design.md`

## 12.32.1 Global Constraints

- Do not admit, enable, start, or contact a new Atlas service.
- Keep `infra` pinned unchanged at `41ba856f7cd35f0b559d6875e08443eac3e98a98`.
- `atlas.required_env` is mandatory; all twenty-one current specs explicitly use `required_env: []`.
- Environment entries contain exactly `name` and `service`; actual values, endpoints, credentials, tokens, and host paths are forbidden.
- Environment names match `^[A-Z][A-Z0-9_]*$`; service IDs match `^[a-z][a-z0-9-]*$`.
- Environment names are unique, every binding references a declared service, and every non-JupyterHub service has at least one binding.
- Canonical storage and rendering sort requirements by `(name, service)`.
- Service availability and variable injection are not notebook authorization or runtime-health evidence.
- Follow red-green-refactor for every behavior change and keep commits independently reviewable.

---

### 12.32.1.1 Task 1: Add the strict environment-requirement parser model

**Files:**
- Modify: `tests/test_notebook_infrastructure.py`
- Modify: `scripts/docs/notebook_infrastructure.py`

**Interfaces:**
- Produces: `AtlasEnvironmentRequirement(name: str, service: str)`.
- Produces: `AtlasTaskContract.required_env: tuple[AtlasEnvironmentRequirement, ...]`.
- Produces: `_parse_required_env(task: str, atlas: dict[object, object], services: Sequence[str]) -> tuple[AtlasEnvironmentRequirement, ...]`.

- [ ] **Step 1: Write valid-load and canonical-order tests**

Add `AtlasEnvironmentRequirement` to the test import and split the fixtures so the baseline contract is JupyterHub-only while a future-service fixture carries bindings:

```python
VALID_ATLAS = """  executor: jupyterhub
  default_mode: vscode-remote
  required_services: [jupyterhub]
  required_env: []
  workspace_access: remote
  artifact_policy: atlas-jupyter-volume
  constraints: ["Use a dedicated kernel", "Keep data private"]"""

FUTURE_SERVICE_ATLAS = VALID_ATLAS.replace(
    "required_services: [jupyterhub]\n  required_env: []",
    """required_services: [jupyterhub, postgres]
  required_env:
    - {name: POSTGRES_USER, service: postgres}
    - {name: POSTGRES_HOST, service: postgres}""",
)
```

Use `FUTURE_SERVICE_ATLAS` in `test_loads_contracts_in_manifest_order` and assert canonical sorting:

```python
assert contracts[0].required_env == (
    AtlasEnvironmentRequirement(name="POSTGRES_HOST", service="postgres"),
    AtlasEnvironmentRequirement(name="POSTGRES_USER", service="postgres"),
)
```

- [ ] **Step 2: Run the valid-load test and verify RED**

Run:

```bash
/private/tmp/ml-eng-lab-issue67-venv/bin/python -m pytest \
  tests/test_notebook_infrastructure.py::test_loads_contracts_in_manifest_order -q
```

Expected: collection or assertion failure because `AtlasEnvironmentRequirement` and `required_env` do not exist.

- [ ] **Step 3: Add invalid-shape and relationship tests**

Add one parameterized test whose cases independently cover:

```python
(
    VALID_ATLAS.replace("  required_env: []\n", ""),
    "required_env is required",
),
(VALID_ATLAS.replace("required_env: []", "required_env: no"), "must be a list"),
(VALID_ATLAS.replace("required_env: []", "required_env: [SPARK_REMOTE]"), "entry 1 must be a mapping"),
(VALID_ATLAS.replace("required_env: []", "required_env: [{service: jupyterhub}]"), "missing keys: name"),
(VALID_ATLAS.replace("required_env: []", "required_env: [{name: JUPYTER_URL}]"), "missing keys: service"),
(VALID_ATLAS.replace("required_env: []", "required_env: [{name: JUPYTER_URL, service: jupyterhub, value: secret}]"), "unexpected keys: value"),
(VALID_ATLAS.replace("required_env: []", "required_env: [{name: lower_name, service: jupyterhub}]"), "valid environment name"),
(VALID_ATLAS.replace("required_env: []", "required_env: [{name: JUPYTER_URL, service: bad_service}]"), "valid service ID"),
(VALID_ATLAS.replace("required_env: []", "required_env: [{name: JUPYTER_URL, service: jupyterhub}, {name: JUPYTER_URL, service: jupyterhub}]"), "names must be unique"),
(VALID_ATLAS.replace("required_env: []", "required_env: [{name: SPARK_REMOTE, service: spark-connect}]"), "must reference required_services"),
(
    VALID_ATLAS.replace("required_services: [jupyterhub]", "required_services: [jupyterhub, spark-connect]"),
    "services missing required_env bindings: spark-connect",
),
```

Each case writes a one-task spec and asserts `NotebookInfrastructureError` with the stated message.

- [ ] **Step 4: Run the invalid-contract test and verify RED**

Run the new parameterized test directly. Expected: every case fails because the current parser ignores `required_env`.

- [ ] **Step 5: Implement the minimal frozen model and parser**

In `scripts/docs/notebook_infrastructure.py`, add:

```python
_ENV_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


@dataclass(frozen=True, order=True)
class AtlasEnvironmentRequirement:
    name: str
    service: str
```

Add `required_env` to `AtlasTaskContract`, then implement `_parse_required_env`. It must:

```python
if "required_env" not in atlas:
    raise NotebookInfrastructureError(f"{task}: atlas.required_env is required")
raw_requirements = atlas["required_env"]
if not isinstance(raw_requirements, list):
    raise NotebookInfrastructureError(f"{task}: atlas.required_env must be a list")
```

For each one-based entry index, require a mapping and exact keys. Validate name/service syntax, reject duplicate names, reject undeclared service references, and after parsing compute:

```python
bound_services = {requirement.service for requirement in requirements}
missing_services = sorted(set(services) - {"jupyterhub"} - bound_services)
```

Raise on a non-empty `missing_services`; otherwise return
`tuple(sorted(requirements))`. Call the helper after `required_services`
validation and store its result in `AtlasTaskContract`.

- [ ] **Step 6: Run focused parser tests and verify GREEN**

Run:

```bash
/private/tmp/ml-eng-lab-issue67-venv/bin/python -m pytest tests/test_notebook_infrastructure.py -q
```

Expected: parser/unit tests pass except repository-spec and renderer expectations intentionally left for Tasks 2 and 3.

- [ ] **Step 7: Commit the parser contract**

```bash
git add scripts/docs/notebook_infrastructure.py tests/test_notebook_infrastructure.py
git commit -m "feat: validate Atlas environment requirements"
```

### 12.32.1.2 Task 2: Migrate every active task explicitly

**Files:**
- Modify: `notebooks/*/docs/spec.yaml` (all twenty-one active task specs)
- Modify: `tests/test_notebook_infrastructure.py`
- Modify: `tests/test_check_docs.py`
- Modify: `tests/test_verify_repo.py`

**Interfaces:**
- Consumes: mandatory `atlas.required_env` parser from Task 1.
- Produces: every active JupyterHub-only spec declares `required_env: []` immediately after `required_services`.

- [ ] **Step 1: Strengthen the repository migration test**

In `test_repository_declares_contracts_for_every_manifest_notebook`, add:

```python
assert all(contract.required_services == ("jupyterhub",) for contract in contracts)
assert all(contract.required_env == () for contract in contracts)
```

- [ ] **Step 2: Run the repository-contract test and verify RED**

Expected: `NotebookInfrastructureError` reports the first spec missing `atlas.required_env`.

- [ ] **Step 3: Add explicit empty requirements to all specs**

Mechanically insert:

```yaml
  required_env: []
```

immediately after `required_services: [jupyterhub]` in all twenty-one active
`notebooks/*/docs/spec.yaml` files. Confirm exact coverage:

```bash
test "$(rg -l '^  required_env: \[\]$' notebooks/*/docs/spec.yaml | wc -l | tr -d ' ')" = 21
```

- [ ] **Step 4: Update synthetic spec fixtures outside the focused test**

Add `required_env: []` after `required_services` in every synthetic Atlas spec
constructed by `tests/test_check_docs.py` and `tests/test_verify_repo.py`.
Use `rg -n "required_services:" tests` to prove no relevant fixture remains
unmigrated.

- [ ] **Step 5: Run migration-focused tests and verify GREEN**

```bash
/private/tmp/ml-eng-lab-issue67-venv/bin/python -m pytest \
  tests/test_notebook_infrastructure.py \
  tests/test_check_docs.py \
  tests/test_verify_repo.py -q
```

- [ ] **Step 6: Commit the explicit schema migration**

```bash
git add notebooks tests/test_notebook_infrastructure.py tests/test_check_docs.py tests/test_verify_repo.py
git commit -m "docs: declare empty Atlas environment contracts"
```

### 12.32.1.3 Task 3: Render required environment bindings deterministically

**Files:**
- Modify: `tests/test_notebook_infrastructure.py`
- Modify: `scripts/docs/notebook_infrastructure.py`
- Modify: `docs/notebook-infrastructure.md`
- Modify: `tests/test_verify_repo.py`

**Interfaces:**
- Consumes: `AtlasTaskContract.required_env` from Task 1.
- Produces: table column `Required environment` with `—` or sorted `` `NAME` (service) `` entries.

- [ ] **Step 1: Write the new renderer expectation**

Construct contracts with keyword arguments. Give `task-b` unsorted bindings:

```python
required_env=(
    AtlasEnvironmentRequirement("MLFLOW_TRACKING_URI", "mlflow"),
    AtlasEnvironmentRequirement("MLFLOW_EXPERIMENT_NAME", "mlflow"),
)
```

Assert the exact header and rows:

```markdown
| Task | Tier | Default mode | Workspace access | Required Atlas services | Required environment | Artifact policy | Constraints |
| --- | --- | --- | --- | --- | --- | --- | --- |
| task-b | B | mounted-workspace | mounted-required | jupyterhub, mlflow | `MLFLOW_EXPERIMENT_NAME` (mlflow)<br>`MLFLOW_TRACKING_URI` (mlflow) | task-local-ignored-paths | — |
| task-a | A | vscode-remote | remote | jupyterhub | — | atlas-jupyter-volume | One<br>Two |
```

- [ ] **Step 2: Run the renderer test and verify RED**

Expected: the current seven-column table lacks `Required environment`.

- [ ] **Step 3: Implement the value-free renderer**

Add:

```python
def _render_required_env(requirements: Sequence[AtlasEnvironmentRequirement]) -> str:
    if not requirements:
        return "—"
    return "<br>".join(
        f"`{requirement.name}` ({requirement.service})"
        for requirement in sorted(requirements)
    )
```

Insert the new header/separator cell and `_render_required_env(contract.required_env)`
immediately after the required-services cell.

- [ ] **Step 4: Regenerate the canonical matrix**

```bash
/private/tmp/ml-eng-lab-issue67-venv/bin/python -m scripts.docs.notebook_infrastructure --write
```

Verify all twenty-one rows contain `jupyterhub | — |` and no variable value,
endpoint, credential, token, or host path appears in the new column.

- [ ] **Step 5: Update repository-verifier table fixtures and run focused tests**

Update exact table headers/rows in `tests/test_verify_repo.py`, then run:

```bash
/private/tmp/ml-eng-lab-issue67-venv/bin/python -m pytest \
  tests/test_notebook_infrastructure.py tests/test_verify_repo.py -q
```

- [ ] **Step 6: Commit deterministic rendering**

```bash
git add scripts/docs/notebook_infrastructure.py tests/test_notebook_infrastructure.py \
  tests/test_verify_repo.py docs/notebook-infrastructure.md
git commit -m "docs: render Atlas environment requirements"
```

### 12.32.1.4 Task 4: Synchronize policy documentation and regression assertions

**Files:**
- Modify: `tests/test_check_docs.py`
- Modify: `docs/notebook-infrastructure.md`
- Modify: `docs/atlas-pin-bump-runbook.md`
- Modify: `CONTRIBUTING.md`
- Modify: `CHANGELOG.md`
- Modify: `docs/manifest.yaml`
- Modify: `docs/superpowers/plans/2026-08-22-issue-67-atlas-required-env-implementation-plan.md`

**Interfaces:**
- Produces: one canonical policy across repository, generated site, and wiki.
- Produces: manifest entries `12.31` and `12.32` for the Issue #67 design and plan.

- [ ] **Step 1: Write documentation-policy tests first**

Add assertions in `tests/test_check_docs.py` that the canonical infrastructure
guide contains all of these exact concepts:

```python
for phrase in (
    "required_env",
    "does not authorize notebook use",
    "does not prove that the service is enabled or healthy",
    "must not contain environment values",
):
    assert phrase in infrastructure
```

Assert the runbook names the `{name, service}` contract and the targeted
JupyterHub smoke. Assert `CONTRIBUTING.md` requires exact task metadata and
central source enablement.

- [ ] **Step 2: Run the new documentation test and verify RED**

Run the exact new test; expected failure is the first missing `required_env`
policy phrase.

- [ ] **Step 3: Update canonical policy prose**

In `docs/notebook-infrastructure.md`, explain the schema with one value-free
example and the availability → declaration → injection → doctor → targeted
smoke sequence. State explicitly that actual values are prohibited.

In `docs/atlas-pin-bump-runbook.md`, replace the generic “required environment
values” wording with exact `{name, service}` metadata and clarify that the
binding names an injected variable but never commits its value.

In `CONTRIBUTING.md`, document the same future-service admission checklist.
In `CHANGELOG.md`, add one Unreleased bullet for Issue #67 without claiming a
new service was admitted.

- [ ] **Step 4: Register the implementation plan**

Append this manifest entry after design `12.31`:

```yaml
      - id: issue-67-atlas-required-env-implementation
        number: "12.32"
        title: Issue 67 Atlas required-environment contract implementation plan
        source: docs/superpowers/plans/2026-08-22-issue-67-atlas-required-env-implementation-plan.md
```

- [ ] **Step 5: Run focused docs tests and strict projections**

```bash
/private/tmp/ml-eng-lab-issue67-venv/bin/python -m pytest \
  tests/test_check_docs.py tests/test_manifest.py tests/test_transforms.py \
  tests/test_build_docs.py tests/test_wiki.py -q
source /private/tmp/ml-eng-lab-issue67-venv/bin/activate
make docs-check
make docs-wiki
```

- [ ] **Step 6: Commit synchronized documentation**

```bash
git add CHANGELOG.md CONTRIBUTING.md docs tests/test_check_docs.py
git commit -m "docs: define Atlas service admission bindings"
```

### 12.32.1.5 Task 5: Exact-SHA qualification, review, and GitFlow publication

**Files:**
- Verify only: repository worktree and remote checks.
- Create ignored report: `.superpowers/sdd/issue67-qualification-report.md`.

**Interfaces:**
- Consumes: completed Tasks 1–4.
- Produces: independently reviewed immutable feature SHA, green local/remote evidence, synchronized main/develop tree, closed Issue #67, and clean task state.

- [ ] **Step 1: Run complete local qualification**

```bash
source /private/tmp/ml-eng-lab-issue67-venv/bin/activate
python -m pip check
make verify-torch-stack
make verify-nnx-install
python -m pytest tests/test_notebook_infrastructure.py -q
python -m pytest tests/test_check_docs.py tests/test_verify_repo.py -q
make test
make verify
make lint
make docs-check
make docs-wiki
git diff --check
git status --short
```

Expected: all commands pass, the feature worktree and `infra` submodule are
clean, and the gitlink remains unchanged.

- [ ] **Step 2: Request independent code review and remediate with TDD**

Provide the reviewer the Issue #67 acceptance criteria, base SHA, feature SHA,
design, and implementation plan. Treat Critical/Important findings as blockers.
For each accepted finding, first add a failing regression test, then the
minimal fix, rerun focused/full validation, and request re-review.

- [ ] **Step 3: Freeze, push, and qualify the feature PR**

Push `codex/issue-67-atlas-env-contract`, open a feature PR to `develop`, and
record its immutable head/tree. Require fresh green `pytest-repository`,
`atlas-consumer-policy`, `dependency-audit`, Docs gate, Atlas contract, docs,
verifier, NNx surface, Docker, and security checks. Tier B/C may skip because
the issue does not change executable notebooks.

- [ ] **Step 4: Publish and synchronize through GitFlow**

Merge the feature PR to `develop` and delete its remote feature branch. After
the required develop-push gates pass, open and merge a `develop` to `main`
release PR. Require main Pages and native wiki publication, then open and merge
a content-neutral `main` to `develop` sync PR. Prove main is an ancestor of
develop and both tips share one tree.

- [ ] **Step 5: Close tracking and clean only task-owned state**

Check every Issue #67 acceptance item, post PR/SHA/test/docs/cleanup evidence,
move its project card to Done, update parent Issue #53, and close #67. Remove
the local/remote feature branch, Issue #67 worktree, external venv, ignored
reports/probes, and task images if any. Preserve Atlas volumes and confirm zero
ml-eng-lab project containers. Leave the primary checkout on clean synchronized
`develop`; Issue #68 remains next.
