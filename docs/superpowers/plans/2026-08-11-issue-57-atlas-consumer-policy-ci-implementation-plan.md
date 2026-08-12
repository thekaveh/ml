# 12.12 Issue 57 Atlas consumer policy CI implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the parent-owned Atlas consumer and native host-service policy a fast, required pull-request gate while retaining separate non-live validation of the pinned Atlas submodule.

**Architecture:** Add an unconditional `atlas-consumer-policy` job to the main CI workflow and back it with a minimal pinned dependency file plus a public focused Make target. Keep the path-scoped `atlas-contract` workflow responsible for recursive-submodule validation, expand its input inventory, and lock both workflow contracts with mutation-resistant repository tests. Align canonical documentation and the system diagram, then require the stable check through the existing GitHub ruleset after its first live success.

**Tech Stack:** GitHub Actions, YAML, Python 3.11, pytest, PyYAML, GNU Make, ShellCheck, Bash, GitHub rulesets, CairoSVG, MkDocs Material, and the repository's three-surface documentation pipeline.

## 12.12.1 Global constraints

- Use feature → `develop` → `main` GitFlow and a final `main` → `develop` synchronization PR when merge topology requires it.
- The unconditional job key and display name are both exactly `atlas-consumer-policy`; this is the stable required-check context.
- Keep `.github/workflows/atlas-contract.yml` path-scoped and non-required; it remains the direct recursive-submodule validator.
- Use `ubuntu-24.04`, Python `3.11`, and `timeout-minutes: 15` for the focused job.
- Checkout must set `persist-credentials: false` and must not initialize submodules in the focused job.
- Install only `atlas-contract-requirements.txt`; its exact pytest and PyYAML pins must equal the locked documentation requirement versions.
- Run exact `make test-atlas-consumer` plus ShellCheck for the three Atlas wrappers and `scripts/lib/atlas-dotenv.sh`.
- Do not use `-k`, fixed test counts, blanket skips, `continue-on-error`, failure masks, job/workflow environment overrides, services, or job containers.
- Do not start, stop, contact, or modify Atlas, JupyterHub, Ollama, ComfyUI, Docker Compose, or unrelated containers. Never use containerized Ollama.
- The workflow must not contain live `docker`, `ollama serve`, `make atlas-up`, `make atlas-down`, `curl`, or localhost probe commands.
- Preserve the complete `pytest-repository`, focused NNx/Ruff, verifier, documentation, and notebook-execution job responsibilities.
- Apply red-green-refactor TDD to every behavioral or contract change and commit each reviewed task independently.
- Edit canonical Markdown and diagram HTML masters only; regenerate the committed PNG and derived site/wiki output through repository tooling.
- Preserve every existing GitHub ruleset target, bypass setting, enforcement state, and rule when adding `atlas-consumer-policy` beside `pytest-repository`.

---

## 12.12.2 Task 1: Create the focused local Atlas consumer contract

**Files:**
- Create: `atlas-contract-requirements.txt`
- Create: `tests/test_atlas_makefile_contract.py`
- Modify: `tests/test_makefile_contract.py`
- Modify: `tests/test_verify_repo.py`
- Modify: `Makefile`
- Modify: this plan

**Interfaces:**
- Consumes: the existing Atlas consumer, lifecycle, runtime-probe, and public Make target tests.
- Produces: exact Make target `test-atlas-consumer`, a four-module focused pytest boundary, and a two-package pinned manifest consumed by Task 2.

- [x] **Step 1: Move the three Atlas Makefile tests into a focused module**

Move `test_atlas_targets_expose_exact_lifecycle_commands`, `test_atlas_down_only_requests_cold_shutdown_when_explicit`, and `test_atlas_targets_are_documented_and_phony`, with their needed imports/constants, from `tests/test_makefile_contract.py` to new `tests/test_atlas_makefile_contract.py`. Keep assertions unchanged. Run:

```bash
pytest -p no:cacheprovider tests/test_makefile_contract.py tests/test_atlas_makefile_contract.py -q
```

Expected: green, with exactly three tests in the new module.

- [x] **Step 2: Add failing focused-target and dependency-contract tests**

Add tests that define this exact boundary:

```python
ATLAS_TEST_FILES = (
    "tests/test_atlas_consumer_contract.py",
    "tests/test_atlas_lifecycle.py",
    "tests/test_atlas_runtime_probe.py",
    "tests/test_atlas_makefile_contract.py",
)
```

Assert `make --no-print-directory -n test-atlas-consumer` emits exactly `pytest <four files> -v`; assert the target appears in `.PHONY` and `make help`; and assert `atlas-contract-requirements.txt` contains only exact pytest and PyYAML pins matching `docs-requirements.txt`. Reuse existing requirement parsers when possible; otherwise add narrow helpers that reject options, ranges, duplicates, continuations, and extra packages.

- [x] **Step 3: Run and record RED**

```bash
pytest -p no:cacheprovider tests/test_verify_repo.py tests/test_atlas_makefile_contract.py -q -k 'atlas_consumer_make_target or atlas_contract_direct_dependencies'
```

Expected: fail because the target and manifest do not exist. Record exact failures before production edits.

- [x] **Step 4: Add the minimal implementation**

Create `atlas-contract-requirements.txt`:

```text
pytest==9.0.3
pyyaml==6.0.3
```

Add to `Makefile`:

```make
ATLAS_CONSUMER_TESTS := tests/test_atlas_consumer_contract.py \
	tests/test_atlas_lifecycle.py \
	tests/test_atlas_runtime_probe.py \
	tests/test_atlas_makefile_contract.py

test-atlas-consumer:
	pytest $(ATLAS_CONSUMER_TESTS) -v
```

Add the target to `.PHONY` and help output. Keep `make test` unchanged.

- [x] **Step 5: Verify GREEN**

```bash
pytest -p no:cacheprovider tests/test_verify_repo.py tests/test_atlas_makefile_contract.py -q -k 'atlas_consumer_make_target or atlas_contract_direct_dependencies'
make test-atlas-consumer
pytest -p no:cacheprovider tests/test_makefile_contract.py tests/test_atlas_makefile_contract.py -q
ruff check --no-cache tests/test_verify_repo.py tests/test_makefile_contract.py tests/test_atlas_makefile_contract.py
git diff --check
```

- [x] **Step 6: Mark Task 1 complete and commit**

Mark Task 1 only, rerun Step 5, stage its files, run `git diff --cached --check`, and commit `test: define focused Atlas consumer contract`.

## 12.12.3 Task 2: Add and lock the two Atlas CI boundaries

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `.github/workflows/atlas-contract.yml`
- Modify: `tests/test_verify_repo.py`
- Modify: this plan

**Interfaces:**
- Consumes: Task 1's dependency file and `make test-atlas-consumer`.
- Produces: unconditional `atlas-consumer-policy` and complete input paths for the separate direct Atlas validator.

- [x] **Step 1: Add failing workflow-contract tests**

Require this exact five-step job:

```yaml
atlas-consumer-policy:
  name: atlas-consumer-policy
  runs-on: ubuntu-24.04
  timeout-minutes: 15
  steps:
    - name: Checkout
      uses: actions/checkout@9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0
      with: {persist-credentials: 'false'}
    - name: Set up Python 3.11
      uses: actions/setup-python@ece7cb06caefa5fff74198d8649806c4678c61a1
      with: {python-version: '3.11', cache: pip, cache-dependency-path: atlas-contract-requirements.txt}
    - name: Install focused Atlas contract dependencies
      run: pip install -r atlas-contract-requirements.txt
    - name: ShellCheck parent-owned Atlas wrappers
      run: shellcheck scripts/atlas-up.sh scripts/atlas-down.sh scripts/atlas-connect.sh scripts/lib/atlas-dotenv.sh
    - name: Run Atlas consumer policy tests
      run: make test-atlas-consumer
```

The assertion helper must reject workflow/job `defaults` or `env`; job `if`, `needs`, `services`, `container`, or `continue-on-error`; extra/reordered steps; step `if`, `env`, `continue-on-error`, `timeout-minutes`, or shell overrides; checkout submodules; and mutations adding `docker`, `ollama serve`, `make atlas-up`, `make atlas-down`, `curl`, or localhost endpoints. Add parametrized mutation tests proving rejection.

Replace the Atlas path expectation with the previous set plus exactly these inputs:

```text
atlas-contract-requirements.txt
scripts/atlas_runtime_probe.py
scripts/lib/atlas-dotenv.sh
tests/test_atlas_*.py
tests/test_makefile_contract.py
```

Extend the verifier job's self-referential selector with `atlas_consumer_policy_contract`.

- [x] **Step 2: Run and record RED**

```bash
pytest -p no:cacheprovider tests/test_verify_repo.py -q -k 'atlas_consumer_policy_contract or atlas_contract_workflow or ci_runs_repository_workflow_contract_tests'
```

Expected: failures for the missing job, incomplete paths, and stale selector.

- [x] **Step 3: Implement the workflow changes**

Insert the exact job above in `.github/workflows/ci.yml` without changing triggers or existing jobs. Expand only the Atlas path list and append the selector keyword without removing another contract.

- [x] **Step 4: Verify GREEN**

```bash
pytest -p no:cacheprovider tests/test_verify_repo.py -q -k 'atlas_consumer_policy_contract or atlas_contract_workflow or ci_runs_repository_workflow_contract_tests or ci_covers_gitflow_pr_targets'
python -c "from pathlib import Path; import yaml; [yaml.load(Path(p).read_text(encoding='utf-8'), Loader=yaml.BaseLoader) for p in ('.github/workflows/ci.yml', '.github/workflows/atlas-contract.yml')]"
shellcheck scripts/atlas-up.sh scripts/atlas-down.sh scripts/atlas-connect.sh scripts/lib/atlas-dotenv.sh
make test-atlas-consumer
ruff check --no-cache tests/test_verify_repo.py
git diff --check
```

- [x] **Step 5: Mark Task 2 complete and commit**

Mark Task 2 only, rerun Step 4, stage its files, run `git diff --cached --check`, and commit `ci: require Atlas consumer policy tests`.

## 12.12.4 Task 3: Synchronize documentation and the system diagram

**Files:**
- Modify: `CONTRIBUTING.md`
- Modify: `CHANGELOG.md`
- Modify: `docs/conventions.md`
- Modify: `docs/jupyterhub-integration.md`
- Modify: `docs/architecture.md`
- Modify: `docs/diagrams/ml-eng-lab-system.html`
- Regenerate: `docs/diagrams/img/system.png`
- Modify: `tests/test_render_diagrams.py`
- Modify: `tests/test_verify_repo.py`
- Modify: this plan

**Interfaces:**
- Consumes: Task 1's focused command and Task 2's two-job role split.
- Produces: synchronized repository/site/wiki guidance and a diagram naming the policy signal intended to become required in Task 4.

- [x] **Step 1: Add failing prose and diagram regressions**

Assert canonical prose contains `make test-atlas-consumer`, `atlas-consumer-policy`, the separate path-scoped `atlas-contract` responsibility, and the non-live/native-only restriction. Extend the system-master regression so visible SVG text includes `Atlas policy + contract` and the HTML summary names the policy job intended to become required without replacing `Full pytest + Tier-A` or `Focused NNx + verifier`.

- [x] **Step 2: Run and record RED**

```bash
pytest -p no:cacheprovider tests/test_render_diagrams.py tests/test_verify_repo.py -q -k 'atlas_consumer_policy_docs or system_master_names'
```

- [x] **Step 3: Update canonical prose**

- `CONTRIBUTING.md`: add the focused local command and say `atlas-consumer-policy` runs on every PR and is intended to become required in Task 4, while `atlas-contract` is path-scoped.
- `docs/conventions.md`: document the five-step focused job, two-package dependency file, ShellCheck inputs, exact Make target, non-live restriction, and relation to complete pytest.
- `docs/jupyterhub-integration.md`: map parent wrapper/probe/helper changes to both checks; reiterate native-only Ollama and non-live CI.
- `docs/architecture.md`: include the policy job intended to become required in Task 4 while preserving the direct submodule boundary.
- `CHANGELOG.md`: record the parent-policy job intended to become required in Task 4, expanded direct-validator paths, focused command, and no-live-service contract.

Do not claim path-scoped `atlas-contract` is required.

- [x] **Step 4: Update and inspect the system diagram**

Edit the HTML/SVG master so the CI box retains existing signals and visibly adds `Atlas policy + contract`; update accessible summary text. Run `python -m scripts.docs.render_diagrams`, inspect the PNG at original resolution, and confirm no overlap, clipping, tofu, or disconnected flow.

- [x] **Step 5: Verify all surfaces GREEN**

```bash
pytest -p no:cacheprovider tests/test_render_diagrams.py tests/test_verify_repo.py -q -k 'atlas_consumer_policy_docs or system_master_names'
make docs-check
make docs-wiki
python scripts/verify_repo.py --check docs --fast
ruff check --no-cache tests/test_render_diagrams.py tests/test_verify_repo.py scripts/docs
git diff --check
```

Expected wiki count after manifest addition: 56 entries. Inspect generated copies of the changed canonical pages.

- [x] **Step 6: Mark Task 3 complete and commit**

Mark Task 3 only, rerun Step 5, stage canonical sources and regenerated PNG, run `git diff --cached --check`, and commit `docs: document Atlas consumer policy gate`.

## 12.12.5 Task 4: Review, verify, publish, enforce, and close

**Files:**
- Modify only for review fixes: Task 1–3 files
- External evidence: GitHub PRs, Actions, ruleset, issue #57, parent #53, and project item

**Interfaces:**
- Consumes: all reviewed commits and existing `gitflow` ruleset.
- Produces: merged PRs, required enforcement, synchronized branches, closed records, and clean Git state.

- [x] **Step 1: Complete broad review**

Package the full branch diff from base `67882ac80b686f21e86b5037f03635b3207686c4`, dispatch a high-capability reviewer, send all findings to one fix subagent, require covering test evidence, and re-review until clean.

- [x] **Step 2: Run fresh final verification**

```bash
make test-atlas-consumer
shellcheck scripts/atlas-up.sh scripts/atlas-down.sh scripts/atlas-connect.sh scripts/lib/atlas-dotenv.sh
make test
make lint
make verify
make docs-check
make docs-wiki
git diff --check
git status --short
```

- [ ] **Step 3: Open and validate feature → develop PR**

Push, open a ready PR closing #57, and wait for green `atlas-consumer-policy`, `atlas-contract`, and `pytest-repository`. Record live URLs, conclusions, durations, focused results, and skip reasons.

- [ ] **Step 4: Require the stable context**

Update the existing `gitflow` ruleset in place, preserving all targets, bypass actors, enforcement, and rules, so required contexts are exactly `pytest-repository` and `atlas-consumer-policy`. Read it back and verify. Never require path-scoped `atlas-contract`.

- [ ] **Step 5: Merge through GitFlow and synchronize**

Merge feature → `develop`, delete the feature branch, open and merge `develop` → `main` after all checks, then merge a content-neutral `main` → `develop` sync PR if topology requires it. Verify empty content diff and that `main` is an ancestor of `develop`.

- [ ] **Step 6: Close records and clean state**

Complete issue #57 checkboxes/evidence, close it, move its project card to Done, update parent #53, fetch/prune, and verify one clean worktree, no temporary branches/PRs, synchronized long-lived branches, and no ml-eng-lab/Atlas/Ollama/ComfyUI container. Do not touch unrelated containers.
