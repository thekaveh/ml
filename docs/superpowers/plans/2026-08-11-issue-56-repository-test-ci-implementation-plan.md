# 12.10 Issue 56 complete repository test CI implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `subagent-driven-development` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Apply test-driven development to every behavior change and complete independent requirements and quality reviews before integration.

**Goal:** Make the complete repository-owned `make test` contract a required pull-request check while preserving the existing focused CI signals and documenting the live behavior on all three documentation surfaces.

**Architecture:** Add an independent `pytest-repository` job to the existing CI workflow and lock its command, collection boundary, dependencies, timeout, cache inputs, and failure semantics with repository tests. Keep the NNx, verifier, documentation, and notebook jobs separate; align canonical prose and the system diagram; then require the stable check context in the existing GitHub ruleset after its first successful live run.

**Tech Stack:** GitHub Actions, YAML, Python 3.11, pytest, Make, `tomllib`, CairoSVG/`libcairo2`, Ruff, MkDocs Material, GitHub repository rulesets, and the repository's three-surface documentation pipeline.

## 12.10.1 Global constraints

- Use feature → `develop` → `main` GitFlow and a final `main` → `develop` synchronization PR when merge topology requires it.
- The new job key and explicit display name are both `pytest-repository`; this is the stable required-check context.
- Run the exact `make test` command. Do not copy pytest selectors into the workflow or hard-code a test count.
- Use `ubuntu-24.04`, Python `3.11`, and `timeout-minutes: 15`.
- Install `libcairo2`, the pinned Torch stack, `requirements.txt`, and `docs-requirements.txt` so diagram tests execute instead of import-skipping.
- Cache pip downloads against `requirements.txt`, `torch-core-requirements.txt`, `torch-requirements.txt`, and `docs-requirements.txt`; do not describe the cache as an installed environment.
- Checkout must set `persist-credentials: false` and must not initialize the Atlas submodule.
- Preserve `pytest-nnx-surface`, `verify-repo`, documentation, Docker, and notebook jobs with their current responsibilities.
- Accept only explicit, reason-bearing platform/dependency/backend skips. Do not add blanket ignores, `continue-on-error`, or failure-masking shell logic.
- Do not start Atlas, JupyterHub, Ollama, ComfyUI, Docker Compose, or any unrelated container. Never use containerized Ollama.
- Edit diagram HTML masters and canonical Markdown only; regenerate committed PNGs and derived site/wiki output through the pipeline.
- Preserve every existing GitHub ruleset target and rule when adding the required status check.

---

## 12.10.2 Task 1: Encode and implement the complete pytest workflow contract

**Files:**
- Modify: `tests/test_verify_repo.py`
- Modify: `.github/workflows/ci.yml`
- Modify: `docs/superpowers/plans/2026-08-11-issue-56-repository-test-ci-implementation-plan.md`

**Interfaces:**
- Consumes: `Makefile` target `test`, `pyproject.toml` pytest collection settings, four dependency manifests, and the existing workflow trigger matrix.
- Produces: unconditional workflow job `pytest-repository`, plus regression tests `test_ci_runs_complete_repository_test_contract` and `test_repository_test_collection_boundary_is_explicit` that later reviewers and CI use as the executable contract.

- [x] **Step 1: Add the failing workflow and collection-boundary tests**

Add `import tomllib` beside the standard-library imports in `tests/test_verify_repo.py`, then add these tests beside the existing CI workflow-contract tests:

```python
def test_ci_runs_complete_repository_test_contract():
    workflow = _load_workflow(REPO / ".github/workflows/ci.yml")
    job = workflow["jobs"]["pytest-repository"]

    assert job["name"] == "pytest-repository"
    assert job["runs-on"] == "ubuntu-24.04"
    assert job["timeout-minutes"] == "15"
    assert "if" not in job
    assert "continue-on-error" not in job

    steps = job["steps"]
    checkout = next(step for step in steps if step.get("name") == "Checkout")
    assert checkout["with"]["persist-credentials"] == "false"
    assert "submodules" not in checkout["with"]

    cairo = next(
        step for step in steps
        if step.get("name") == "Install system dependencies for cairosvg"
    )
    assert "libcairo2" in cairo["run"]

    python = next(step for step in steps if step.get("name") == "Set up Python 3.11")
    assert python["with"]["python-version"] == "3.11"
    assert python["with"]["cache"] == "pip"
    assert set(python["with"]["cache-dependency-path"].splitlines()) == {
        "requirements.txt",
        "torch-core-requirements.txt",
        "torch-requirements.txt",
        "docs-requirements.txt",
    }

    install = next(step for step in steps if step.get("name") == "Install dependencies")
    assert install["run"].splitlines() == [
        "make install-torch-stack",
        "pip install -r requirements.txt",
        "pip install -r docs-requirements.txt",
    ]
    complete = next(step for step in steps if step.get("name") == "Run complete repository tests")
    assert complete["run"] == "make test"
    assert all("continue-on-error" not in step for step in steps)


def test_repository_test_collection_boundary_is_explicit():
    config = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    pytest_config = config["tool"]["pytest"]["ini_options"]

    assert pytest_config["testpaths"] == ["tests"]
    assert {"infra", "notebooks/archive", ".venv"} <= set(
        pytest_config["norecursedirs"]
    )
    assert "\ntest:\n\tpytest tests/ -v\n" in (REPO / "Makefile").read_text(
        encoding="utf-8"
    )
```

Extend `test_ci_runs_repository_workflow_contract_tests` so its exact `-k` expression ends with:

```python
"ci_runs_repository_workflow_contract_tests or "
"ci_runs_complete_repository_test_contract or "
"repository_test_collection_boundary_is_explicit'"
```

- [x] **Step 2: Run the tests and capture the intended red state**

Run:

```bash
pytest -p no:cacheprovider tests/test_verify_repo.py -q \
  -k 'ci_runs_complete_repository_test_contract or repository_test_collection_boundary_is_explicit or ci_runs_repository_workflow_contract_tests'
```

Expected: the collection-boundary test passes, while the workflow test fails with a missing
`pytest-repository` job and the existing narrow-selector test fails because it does not yet invoke
the new contract tests. Record the exact pass/fail count in the implementation report.

- [x] **Step 3: Add the minimal unconditional job**

Insert this job before `pytest-nnx-surface` in `.github/workflows/ci.yml`:

```yaml
  pytest-repository:
    name: pytest-repository
    runs-on: ubuntu-24.04
    timeout-minutes: 15
    steps:
      - name: Checkout
        uses: actions/checkout@9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0 # v7
        with:
          persist-credentials: false

      - name: Install system dependencies for cairosvg
        run: sudo apt-get update && sudo apt-get install -y libcairo2

      - name: Set up Python 3.11
        uses: actions/setup-python@ece7cb06caefa5fff74198d8649806c4678c61a1 # v6
        with:
          python-version: "3.11"
          cache: pip
          cache-dependency-path: |
            requirements.txt
            torch-core-requirements.txt
            torch-requirements.txt
            docs-requirements.txt

      - name: Install dependencies
        run: |
          make install-torch-stack
          pip install -r requirements.txt
          pip install -r docs-requirements.txt

      - name: Run complete repository tests
        run: make test
```

Append the two new test names to the `verify-repo` job's `Test repository workflow contracts` `-k`
expression exactly as asserted in Step 1. Do not change the workflow triggers or any existing job.

- [x] **Step 4: Verify green and inspect the parsed workflow**

Run:

```bash
pytest -p no:cacheprovider tests/test_verify_repo.py -q \
  -k 'ci_runs_complete_repository_test_contract or repository_test_collection_boundary_is_explicit or ci_runs_repository_workflow_contract_tests or ci_covers_gitflow_pr_targets'
python - <<'PY'
from pathlib import Path
import yaml
workflow = yaml.load(
    Path('.github/workflows/ci.yml').read_text(encoding='utf-8'),
    Loader=yaml.BaseLoader,
)
job = workflow['jobs']['pytest-repository']
print(job['name'], job['runs-on'], job['timeout-minutes'])
print([step['name'] for step in job['steps']])
PY
ruff check --no-cache tests/test_verify_repo.py
git diff --check
```

Expected: all selected tests pass; the parsed summary reports `pytest-repository`,
`ubuntu-24.04`, and `15`; Ruff and the diff check exit zero.

- [x] **Step 5: Mark Task 1 complete and commit**

Change Task 1's checkboxes to `[x]`, rerun the focused green command, then commit only the workflow,
tests, and updated plan:

```bash
git add .github/workflows/ci.yml tests/test_verify_repo.py \
  docs/superpowers/plans/2026-08-11-issue-56-repository-test-ci-implementation-plan.md
git diff --cached --check
git commit -m "ci: run complete repository pytest contract"
```

## 12.10.3 Task 2: Align canonical documentation and the system diagram

**Files:**
- Modify: `tests/test_render_diagrams.py`
- Modify: `CONTRIBUTING.md`
- Modify: `docs/conventions.md`
- Modify: `docs/architecture.md`
- Modify: `docs/maintenance/overnight-2026-07-04.md`
- Modify: `CHANGELOG.md`
- Modify: `docs/diagrams/ml-eng-lab-system.html`
- Regenerate: `docs/diagrams/img/system.png`
- Modify: `docs/superpowers/plans/2026-08-11-issue-56-repository-test-ci-implementation-plan.md`

**Interfaces:**
- Consumes: the Task 1 workflow job, the established Makefile/pytest boundary, the three-surface manifest, and diagram renderer.
- Produces: current contributor guidance, validation-gate prose, resolved OM-068 evidence, durable history, and a system diagram that visibly distinguishes complete pytest from focused CI signals.

- [x] **Step 1: Add the failing diagram-content regression**

Add this focused test to `tests/test_render_diagrams.py`:

```python
def test_system_master_names_complete_and_focused_ci_signals():
    master = REPO_ROOT / "docs/diagrams/ml-eng-lab-system.html"
    visible_text = _visible_svg_text(master)
    source = master.read_text(encoding="utf-8")

    assert "Full pytest + Tier-A" in visible_text
    assert "Focused NNx + verifier" in visible_text
    assert "CI runs complete pytest plus focused NNx, verifier, and notebook gates." in source
```

- [x] **Step 2: Run the focused test and capture the intended red state**

Run:

```bash
pytest -p no:cacheprovider tests/test_render_diagrams.py -q \
  -k system_master_names_complete_and_focused_ci_signals
```

Expected: one failure because the current system master mentions smoke tiers and Docker/docs but
does not show the complete pytest job or the retained focused signals.

- [x] **Step 3: Replace stale contributor and validation prose with the implemented contract**

In `CONTRIBUTING.md`, replace workflow steps 4–6 with prose that states:

```markdown
4. Run `make test` (wraps `pytest tests/`) locally. CI runs the same complete contract on every PR as the required `pytest-repository` job; `pytest-nnx-surface` remains the faster focused NNx/PyPI compatibility and Ruff signal.
5. If you touched a notebook, re-run it at its tier (Tier-A: `make run-tier-a` only when deliberately refreshing a committed snapshot; `make smoke-tier-a` is the non-mutating CI-equivalent target; Tier-B: `make smoke-tier-b`; Tier-C: `make smoke-tier-c`). Tier-C **code cells** must remain identical to the `pre-cleanup-baseline` tag — verify check E5 enforces this (markdown and embedded outputs are not compared).
6. Open a PR. CI runs complete pytest and Tier-A automatically; Tier-B runs on schedule, on `workflow_dispatch`, and on PRs labeled `tier-b-smoke`; Tier-C runs on schedule and on `workflow_dispatch`.
```

In `docs/conventions.md`:

- add an every-PR bullet under §5.2.2 stating that `pytest-repository` installs the runtime and
  documentation dependency sets, runs exact `make test`, has a 15-minute cap, does not initialize
  Atlas, and accepts only explicit reason-bearing skips;
- replace §5.3's claim that four gates each map to separate jobs with: `A change is not ready until
  the repository verifier, complete pytest contract, Ruff, and documentation gate pass. CI keeps
  complete pytest separate from faster focused and publication signals so failures remain
  attributable.`;
- replace the final paragraph of §5.3.2 with prose stating that `pytest-repository` runs the entire
  tree on every PR, its setup includes `libcairo2` plus runtime and locked documentation
  dependencies, its pip cache keys all four manifests, and the existing NNx job remains focused;
- state explicitly that `testpaths = ["tests"]` and the `infra`, `notebooks/archive`, and `.venv`
  exclusions define collection, and that no fixed test count is contractual; and
- change pre-PR item 2 to `make test locally; CI reruns that complete contract as
  pytest-repository and also retains the focused pytest-nnx-surface signal.`

In `docs/architecture.md` §2.1.3, replace the final sentence with:

```markdown
`scripts/verify_repo.py`, the complete `make test` / `pytest-repository` contract, the focused NNx and Ruff job, documentation checks, and notebook execution tiers verify structure, documentation, library surfaces, and executable notebook behavior before changes are merged.
```

Replace the OM-068 row's `Deferred` state with `Fixed` and use this resolution text:

```markdown
Added the unconditional `pytest-repository` job to run exact `make test` with the pinned runtime and documentation dependencies, native Cairo support, a 15-minute timeout, and a four-manifest pip download cache. Workflow-contract tests lock the job and collection boundary; the focused NNx, verifier, and notebook jobs retain their separate roles.
```

Add this Unreleased Changed entry to `CHANGELOG.md`:

```markdown
- **Complete repository pytest CI:** pull requests now run the exact `make test` contract in the required `pytest-repository` job with pinned runtime and documentation dependencies, native Cairo support, a 15-minute timeout, and a four-manifest pip download cache. Focused NNx/Ruff, verifier, documentation, and notebook-execution jobs remain separate diagnostic and reproducibility signals.
```

- [x] **Step 4: Update and render the system diagram**

In the `CI Workflows` box of `docs/diagrams/ml-eng-lab-system.html`, replace the two detail labels
with these exact visible strings:

```html
<text x="389" y="458" fill="#94a3b8" font-size="8" text-anchor="middle">Full pytest + Tier-A</text>
<text x="389" y="472" fill="#94a3b8" font-size="8" text-anchor="middle">Focused NNx + verifier</text>
```

Replace the validation-boundary card's last bullet with:

```html
<li>CI runs complete pytest plus focused NNx, verifier, and notebook gates.</li>
```

Run the renderer, then visually inspect the exact committed PNG at original resolution:

```bash
python -m scripts.docs.render_diagrams
```

Confirm the CI labels are readable, no text overlaps or clips, every arrow remains behind its boxes,
and no unsupported glyph renders as tofu. If layout changes are required, edit the HTML master and
rerender; never patch the PNG directly.

- [x] **Step 5: Verify all canonical and projected documentation**

Run:

```bash
pytest -p no:cacheprovider tests/test_render_diagrams.py -q
make docs-check
make docs-wiki
python scripts/verify_repo.py --check docs --fast
ruff check --no-cache tests/test_render_diagrams.py
git diff --check
```

Expected: diagram tests pass; strict MkDocs and wiki checks succeed; the docs verifier reports zero
findings; Ruff and the diff check exit zero. Inspect `generated/site` and `generated/wiki` to confirm
the conventions, architecture, OM-068, plan, and regenerated system diagram appear on both derived
surfaces without editing them.

- [x] **Step 6: Mark Task 2 complete and commit**

Change Task 2's checkboxes to `[x]`, rerun its focused green commands, then commit:

```bash
git add CONTRIBUTING.md CHANGELOG.md docs/conventions.md docs/architecture.md \
  docs/maintenance/overnight-2026-07-04.md \
  docs/diagrams/ml-eng-lab-system.html docs/diagrams/img/system.png \
  tests/test_render_diagrams.py \
  docs/superpowers/plans/2026-08-11-issue-56-repository-test-ci-implementation-plan.md
git diff --cached --check
git commit -m "docs: document complete repository test CI"
```

## 12.10.4 Task 3: Complete local verification and independent reviews

**Files:**
- Modify: only files already in Issue #56 scope when a reviewer finds a concrete defect.
- Modify: `docs/superpowers/plans/2026-08-11-issue-56-repository-test-ci-implementation-plan.md`

**Interfaces:**
- Consumes: the independently committed Task 1 workflow and Task 2 documentation deliverables.
- Produces: requirements-reviewed and quality-reviewed commits with fresh complete local evidence.

- [ ] **Step 1: Run the complete local gate from a clean task boundary**

Run sequentially so diagram rasterization cannot race inspection:

```bash
make test
make lint
make verify
make docs-check
make docs-wiki
git diff --check
git status --short --branch
```

Record pytest's actual collected, passed, failed, and skipped results plus every skip reason. Expected:
all commands exit zero, no unexplained skip exists, and status shows only intentional plan-checkbox
changes if they have not yet been committed.

- [ ] **Step 2: Dispatch a fresh requirements reviewer**

Give the reviewer the approved design, this plan, Issue #56 acceptance criteria, and the complete
branch diff from `origin/develop`. Require a line-by-line PASS/FAIL assessment of workflow command,
runner, Python, timeout, dependency installation, Cairo, cache inputs, collection boundary,
submodule exclusion, skip/failure semantics, preserved job roles, canonical docs, diagram, and no
Atlas/Ollama/ComfyUI runtime activity.

- [ ] **Step 3: Fix every requirements finding test-first**

For each Critical or Important finding, add or tighten the smallest focused regression, demonstrate
red where behavior is missing, apply the minimal correction, rerun the focused test, and commit with
a concern-specific conventional message. If the review reports no findings, record that result and
do not create an empty commit.

- [ ] **Step 4: Dispatch a fresh code-quality reviewer**

Require review of workflow maintainability, regression strength, selector self-protection,
YAML parsing behavior, documentation accuracy, diagram legibility, scope discipline, and generated
artifact hygiene. The reviewer must distinguish blocking findings from optional suggestions.

- [ ] **Step 5: Fix every quality finding and rerun the complete gate**

Use the same focused red/green loop for any Critical or Important finding, then rerun:

```bash
make test
make lint
make verify
make docs-check
make docs-wiki
git diff --check
git status --short --branch
```

Mark Task 3 complete only after fresh output proves every command passes and the worktree is clean.
Commit the checked Task 3 record with the final substantive review fix; if there was no fix, include
the checkbox update in the final documentation commit rather than making a plan-only commit.

## 12.10.5 Task 4: Enforce the live check, complete GitFlow, and clean up

**Files:**
- Modify: `docs/superpowers/plans/2026-08-11-issue-56-repository-test-ci-implementation-plan.md` before the feature PR if Task 3 did not already commit the final record.
- External state: GitHub feature PR, `gitflow` ruleset `18620095`, release PR, synchronization PR, issue #56, project item, and tracker #53.

**Interfaces:**
- Consumes: clean reviewed feature branch and stable `pytest-repository` workflow context.
- Produces: GitHub-enforced status check, merged and synchronized long-lived branches, live evidence, closed issue/project item, and no dangling feature state.

- [ ] **Step 1: Publish the reviewed feature tip and open the feature PR**

Push the branch, create a PR targeting `develop`, link Issue #56 without auto-closing it before the
default-branch release, and include the red/green record, local counts/skips, no-service statement,
and documentation verification:

```bash
git push origin codex/issue-56-repository-test-ci
gh pr create --base develop --head codex/issue-56-repository-test-ci \
  --title "ci: run complete repository tests on pull requests" \
  --body-file - <<'EOF'
## Summary
- run the complete repository-owned pytest contract on every pull request
- preserve focused NNx, verifier, documentation, and notebook execution signals
- align canonical documentation and the system diagram

## Verification
- focused workflow regressions: RED before implementation, GREEN afterward
- `make test`
- `make lint`
- `make verify`
- `make docs-check`
- `make docs-wiki`

## Runtime safety
- no Atlas, JupyterHub, Ollama, ComfyUI, or Docker Compose service was started

Refs #56
EOF
```

Create the PR body as a temporary file outside the repository, or pass it through standard input,
so no untracked PR artifact appears in the worktree. Wait for every required and applicable check;
inspect failures with `gh run view --log-failed` and fix them on the feature branch before merging.

- [ ] **Step 2: Require the established live check without weakening the ruleset**

After the feature PR's `pytest-repository` context has completed successfully, read ruleset
`18620095` through the GitHub API and save its targets, enforcement, bypass actors, and rule types as
evidence. Update that same ruleset in place: preserve every existing field and non-status rule,
merge `{context: "pytest-repository"}` into its required-status-check contexts, use non-strict
branch synchronization unless the existing rule already requires strict mode, and do not duplicate
the rule.

Use this read-modify-write pipeline so the API's current representation, rather than a reconstructed
ruleset, supplies every preserved field:

```bash
gh api repos/thekaveh/ml-eng-lab/rulesets/18620095 \
  --jq '{name,target,enforcement,bypass_actors,conditions,rule_types:[.rules[].type]}'

gh api repos/thekaveh/ml-eng-lab/rulesets/18620095 |
jq '
  (.rules | map(select(.type == "required_status_checks")) | first) as $status_rule
  | {
      name,
      target,
      enforcement,
      bypass_actors,
      conditions,
      rules: (
        [.rules[] | select(.type != "required_status_checks")]
        + [
            ($status_rule // {
              type: "required_status_checks",
              parameters: {
                strict_required_status_checks_policy: false,
                do_not_enforce_on_create: false,
                required_status_checks: []
              }
            }
            | .parameters.required_status_checks = (
                (.parameters.required_status_checks + [{context: "pytest-repository"}])
                | unique_by(.context)
              ))
          ]
      )
    }
' |
gh api --method PUT repos/thekaveh/ml-eng-lab/rulesets/18620095 --input -

gh api repos/thekaveh/ml-eng-lab/rulesets/18620095 \
  --jq '{conditions,rules:[.rules[] | {type,parameters}]}'
```

Re-read the ruleset and assert:

```text
target branches: main and develop (unchanged)
existing pull-request/deletion/non-fast-forward rules: present
required status context: pytest-repository
```

Then refresh the feature PR and prove GitHub lists `pytest-repository` as required and green.

- [ ] **Step 3: Merge feature to `develop` and promote `develop` to `main`**

Merge the green feature PR with a merge commit. Update local `develop`, open a separate release PR
from `develop` to `main`, wait for `pytest-repository` and every other applicable check, then merge
with a merge commit. Do not bypass the required check or use an administrator override.

- [ ] **Step 4: Synchronize `main` back to `develop`**

Fetch and compare the long-lived branches. When their merge topology differs, open a content-neutral
`main` → `develop` synchronization PR, wait for all required checks including
`pytest-repository`, and merge normally. Verify:

```bash
git fetch origin --prune
git diff --exit-code origin/main origin/develop
git merge-base --is-ancestor origin/main origin/develop
```

Expected: no content diff and `main` is an ancestor of synchronized `develop`.

- [ ] **Step 5: Record live evidence and close Issue #56**

Post an issue comment containing the feature, release, and sync PR links; final `main` and `develop`
SHAs; local full-suite result and explicit skip reasons; live `pytest-repository` run URLs,
conclusions, and durations on both merge directions; ruleset verification; documentation gates; and
the statement that no Atlas/JupyterHub/Ollama/ComfyUI service or container was started. Close Issue
#56, set its project item to Done, and check the Issue #56 entry in tracker #53.

- [ ] **Step 6: Delete conclusively merged temporary state and verify final hygiene**

Delete the feature branch locally and remotely only after both promotions and synchronization are
complete. Prune refs and verify:

```bash
git fetch origin --prune
git worktree list
git branch --format='%(refname:short)'
git branch -r --format='%(refname:short)'
gh pr list --state open --json number,title,headRefName,baseRefName
git status --short --branch
git diff --exit-code origin/main origin/develop
docker ps --format '{{.Names}}'
```

Expected: one clean worktree; only intended long-lived local and remote branches; no open/dangling
PR; exact `main`/`develop` content parity; and no ml-eng-lab Atlas, JupyterHub, Ollama, or ComfyUI
container. Do not stop or alter unrelated projects' containers.
