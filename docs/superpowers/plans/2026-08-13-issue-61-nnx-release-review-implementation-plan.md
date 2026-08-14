# 12.20 Issue 61 NNx Release Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decide with released-wheel, consumer, notebook, security, and GitFlow evidence whether
`ml-eng-lab` can adopt `thekaveh-nnx[lm]==0.2.2`; retain that pin only when every mandatory
contract passes.

**Architecture:** Keep `requirements.txt` as the single production version source and keep the
Issue #58 verifier as the canonical provenance oracle. Strengthen all NNx-executing CI tiers to
install the binary distribution and verify it immediately before their workload, add focused
consumer tests for the newly relevant 0.2.2 surface, refresh the advisory observation without
weakening its accepted baseline, and project one canonical documentation truth to the repository,
site, and wiki.

**Tech Stack:** Python 3.11, pytest, PyTorch 2.4.1, PyG, `thekaveh-nnx[lm]`, papermill,
GitHub Actions, pip-audit, MkDocs Material, GitHub wiki, Git, and GitHub CLI.

**Spec:**
`docs/superpowers/specs/2026-08-13-issue-61-nnx-release-review-design.md`

## 12.20.1 Global Constraints

- Work only on `codex/issue-61-nnx-release-review`, created from the synchronized `develop`
  branch. Preserve unrelated user changes if any appear.
- Use test-driven development: add a focused failing test or mutation first, capture the intended
  RED, make the smallest production change, and rerun the focused GREEN gate before broad tests.
- Keep `requirements.txt` as the only production NNx version source. Do not add a candidate
  manifest, version constant, compatibility flag, editable bypass, direct URL, or downstream shim.
- Require `python -m pip install --only-binary=thekaveh-nnx -r requirements.txt` wherever CI
  claims NNx released-wheel evidence. Run `make verify-nnx-install` after the final install and
  immediately before the NNx workload.
- Do not weaken `scripts/verify_nnx_install.py`, workflow failure propagation, notebook
  acceptance thresholds, audit comparison, provenance checks, or skip guards to make 0.2.2 pass.
- Treat the active editable checkout as development context only. It is not release evidence even
  if its source reports 0.2.2. Use disposable clean environments and GitHub-hosted jobs for
  acceptance.
- Keep the root Torch/PyG/Lightning/torchao pins unchanged. Issue #62 owns their coordinated
  upgrade; Issue #66 owns restored quantization execution and tiering.
- Do not edit the `infra/` gitlink or Atlas-owned requirements. Preserve the observed Atlas NNx
  0.2.0 evidence as historical runtime truth. Do not launch Atlas, JupyterHub, Docker Compose,
  Ollama, or ComfyUI for this issue.
- Keep the active tabular-regression notebook's manual loader and existing results. Document that
  0.2.2 now offers `target_dtype=torch.float32` without changing the experiment split.
- Do not add `NNGraphDataset(seed=...)` to locked notebooks, migrate `NNDataset(batch_sizes=...)`,
  replace deep imports, change the MoE architecture, or introduce a Conv notebook. Issues #68,
  #69, and #70 own those changes.
- Preserve synthetic 0.2.0 values where they are generic verifier fixtures and preserve all
  historical Issue #58, changelog, maintenance, and Atlas evidence. Change only current-pin and
  current-user-contract references.
- Never auto-accept a new advisory or accepted-version drift. A new identity, changed accepted
  version, unexplained surface drift, or failed audit blocks adoption and requires explicit
  security review.
- Edit canonical documentation only. Regenerate and inspect `generated/site` and
  `generated/wiki`, but do not commit either ignored tree.
- Use `apply_patch` for tracked text edits. Use repository formatters or provided scripts only for
  mechanical formatting or generated derivatives.
- Every task receives an implementation review. After all tasks, request an independent broad
  review and fix every Critical, Important, and Minor finding with fresh RED-to-GREEN evidence.
- Merge through a ready feature-to-`develop` PR and a separate `develop`-to-`main` PR. Run the
  labeled Tier B job and manually dispatched Tier C job on the feature ref before the first merge.
- Finish with main/develop synchronization, Issue #61 and parent #53 bookkeeping, project-board
  completion, ruleset readback, and local/remote branch, PR, worktree, and repository-container
  cleanup.

## 12.20.2 File and Interface Map

### 12.20.2.1 Production and workflow contracts

- Modify `requirements.txt`: change the single root pin to `thekaveh-nnx[lm]==0.2.2` and retain
  its binary-wheel ownership commentary.
- Modify `Makefile`: update only current release commentary; preserve the version-deriving
  `verify-nnx-install` target and all tier membership.
- Modify `.github/workflows/ci.yml`: strengthen `tier-a-papermill`, `smoke-tier-b`, and
  `smoke-tier-c` with binary-only selection and canonical verification. Preserve triggers,
  conditions, job names, permissions, timeouts, artifacts, and required contexts.
- Do not modify `scripts/verify_nnx_install.py` unless a test demonstrates a version-independent
  defect. Its public contract remains `parse_requirement`, `discover_distribution`, and
  fail-closed `main` verification derived from `requirements.txt`.

### 12.20.2.2 Consumer tests

- Modify `tests/test_verify_nnx_install.py`: add or update only the current-repository 0.2.2
  expectation; retain generic 0.2.0 mismatch fixtures.
- Modify `tests/test_verify_repo.py`: encode exact Tier A/B/C install, provenance, ordering,
  failure-propagation, and no-override workflow contracts with isolated mutations.
- Create `tests/nnx_surface/test_tabular_regression_diabetes_mlp_pytorch.py`: validate floating
  regression targets, `(batch, 1)` shape, `output_dim == 1`, default classification behavior, and
  a tiny MSE training pass.
- Create `tests/nnx_surface/test_nnx_022_release_surface.py`: validate the public Conv and
  first-class MoE facade, constructor acceptance, and stable enum members without importing deep
  implementation modules.
- Modify `tests/nnx_surface/test_notebook_api_surface.py`: recognize released
  `NNGraphDataset(seed=..., sampler=...)` kwargs while leaving notebook source unchanged.
- Modify `tests/nnx_surface/test_quantization_mnist_ffnn_pytorch.py` only if the existing exact
  QAT signatures do not already prove the 0.2.2 facade. Keep the Torch 2.4.1 backend skip honest.

### 12.20.2.3 Current documentation truth

- Modify `README.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, and `docs/assets/badges/nnx.svg` for the
  current 0.2.2 adoption and immutable wheel/license evidence. Do not rewrite historical entries.
- Modify `docs/nnx-library.md`, `docs/dependency-contracts.md`, `docs/architecture.md`,
  `docs/FINDINGS-NNX.md`, and `docs/concepts.md` for the current release, evidence boundary,
  resolved regression support, and valid checkpoint contract.
- Modify `docs/notebooks/tabular_regression-diabetes-mlp-pytorch.md`,
  `notebooks/tabular_regression-diabetes-mlp-pytorch/README.md`,
  `notebooks/tabular_regression-diabetes-mlp-pytorch/docs/spec.yaml`, and only the relevant
  Markdown cell in `notebooks/tabular_regression-diabetes-mlp-pytorch/notebook.ipynb`. Preserve
  code, output, split, and recorded metrics.
- Modify current-pin references in
  `docs/notebooks/text_generation-tinyshakespeare-transformer-pytorch.md`,
  `notebooks/text_generation-tinyshakespeare-transformer-pytorch/README.md`, and
  `notebooks/preference_alignment-toy-dpo-pytorch/README.md`.
- Reconcile every current canonical `NNRun.load("best")` claim, including the Iris and
  dimensionality-reduction canonical pages and Iris task spec, to
  `NNCheckpoint.load(run=RUN_ID, type=Checkpoints.BEST)`. Preserve historical descriptions of the
  former bug.
- Modify `tests/test_check_docs.py` to enforce current pin parity, release-boundary wording,
  regression-finding status, valid checkpoint wording, and consecutive manifest entries 12.19
  and 12.20.
- Modify `docs/manifest.yaml` only in the plan commit to register this page as 12.20.

### 12.20.2.4 Security evidence and rollout

- Refresh the current observation in `docs/dependency-contracts.md` only after the exact
  `make audit-advisories` result is captured. Do not change
  `security/accepted-advisories.json` unless the controller separately approves a reviewed policy
  decision.
- Update `.github/workflows/docs.yml` paths only if a newly changed canonical source is not
  already covered. Do not add redundant literals beneath the existing root `*.md` coverage.
- Record implementation evidence in ignored `.superpowers/sdd/` reports; never stage those
  reports.

## 12.20.3 Task 1: Adopt the Canonical 0.2.2 Pin and Strengthen Tier Provenance

**Files:** `requirements.txt`, `Makefile`, `.github/workflows/ci.yml`,
`tests/test_verify_nnx_install.py`, `tests/test_verify_repo.py`

- [ ] **Step 1: Revalidate the release before editing**

  Query PyPI and the immutable GitHub release for 0.2.2. Confirm it is still the latest stable,
  non-yanked universal wheel; confirm tag commit
  `edfd197f3f54d4eb67313d46a80e823e6239c5b6`, wheel SHA-256
  `ee56474926fdfd5329721f067cf1b8ae31955627c6949844e09ee4a7bb2bb9d7`, `[lm]` extra,
  Python/Torch/PyG floors, and Apache-2.0 license. Stop on any mismatch.

- [ ] **Step 2: Write current-pin and Tier A/B/C workflow RED tests**

  Require the root requirement to parse as exactly `thekaveh-nnx[lm]==0.2.2`. For each of
  `tier-a-papermill`, `smoke-tier-b`, and `smoke-tier-c`, require this ordered boundary:

  ```text
  make install-torch-stack
  python -m pip install --only-binary=thekaveh-nnx -r requirements.txt
  make verify-nnx-install
  make smoke-tier-a  # or the exact smoke-tier-b / smoke-tier-c workload for that job
  ```

  Tier A may retain its two asset downloads between dependency installation and provenance, but
  no package installation may occur after provenance. Require no editable/path/direct-URL
  override, `PYTHONPATH`, `NNX_ALLOW_EDITABLE`, `continue-on-error`, job container, or service.
  Require the verifier step immediately before the tier workload.

- [ ] **Step 3: Run focused tests and capture RED**

  ```bash
  pytest -p no:cacheprovider tests/test_verify_nnx_install.py tests/test_verify_repo.py -q \
    -k 'current_requirement or tier_a_nnx or tier_b_nnx or tier_c_nnx'
  ```

  Expected: the current pin remains 0.2.0 and all three tier jobs lack the binary/provenance
  contract.

- [ ] **Step 4: Apply the minimal pin and workflow change**

  Change the sole root pin to 0.2.2. Replace each tier job's plain runtime install with the exact
  binary-only command, add a named canonical-verifier step, and make it the final step before the
  workload. Do not modify verifier implementation or unrelated jobs.

- [ ] **Step 5: Mutation-audit the workflow contract**

  In isolated workflow copies, prove RED for removed verifier, reversed verifier/workload order,
  plain `pip install`, editable install, late install, alternate requirements file, job-level
  `PYTHONPATH`, `NNX_ALLOW_EDITABLE`, failure masking, extra Atlas/service steps, and renamed
  workload. Restore the clean workflow after every mutation.

- [ ] **Step 6: Run Task 1 GREEN gates**

  ```bash
  pytest -p no:cacheprovider tests/test_verify_nnx_install.py tests/test_verify_repo.py -q \
    -k 'current_requirement or tier_a_nnx or tier_b_nnx or tier_c_nnx'
  python -c "from pathlib import Path; import yaml; yaml.safe_load(Path('.github/workflows/ci.yml').read_text())"
  ruff check tests/test_verify_nnx_install.py tests/test_verify_repo.py
  git diff --check
  ```

- [ ] **Step 7: Commit Task 1**

  ```bash
  git add requirements.txt Makefile .github/workflows/ci.yml \
    tests/test_verify_nnx_install.py tests/test_verify_repo.py
  git commit -m "deps: trial NNx 0.2.2 canonical wheel"
  ```

## 12.20.4 Task 2: Add Consumer-Facing 0.2.2 Compatibility Coverage

**Files:** `tests/nnx_surface/test_tabular_regression_diabetes_mlp_pytorch.py`,
`tests/nnx_surface/test_nnx_022_release_surface.py`,
`tests/nnx_surface/test_notebook_api_surface.py`, and, only if needed,
`tests/nnx_surface/test_quantization_mnist_ffnn_pytorch.py`

- [ ] **Step 1: Write the regression-dataset contract**

  Build a deterministic tiny pandas frame and require:

  ```python
  regression = NNTabularDataset(..., target_dtype=torch.float32)
  assert regression.output_dim == 1
  assert target.dtype == torch.float32
  assert target.shape[1:] == (1,)
  ```

  Run one tiny MSE optimization through the public NNx facade and assert finite loss/prediction
  shape. Add a paired default-`None` test proving Iris-style integer class targets remain
  contiguous `torch.long` values and retain multi-class `output_dim`.

- [ ] **Step 2: Write additive facade contracts**

  Import `ConvNN`, `NNConvParams`, `FeedFwdMoENN`, `NNMoEParams`, and `Nets` from `nnx` only.
  Construct the smallest valid Conv and MoE configurations, assert `Nets.CONV` and
  `Nets.FEED_FWD_MOE`, and execute one shape-level forward pass where the public model contract
  permits it. Do not migrate any notebook to these architectures.

- [ ] **Step 3: Extend the graph signature guard**

  Add synthetic accepted calls with `seed=0` and `sampler="full"` and make the accepted-signature
  fixture include both names. Do not edit Reddit notebooks; the live signature-derived scan must
  continue to validate their existing kwargs.

- [ ] **Step 4: Capture release-delta RED in a disposable 0.2.0 environment**

  Create a temporary Python 3.11 virtual environment outside the repository, install the released
  0.2.0 binary wheel plus only the focused test dependencies, and run the new release-surface
  tests. Expected RED: regression target dtype and/or new facade/graph contracts are absent.
  Delete the disposable environment after recording the sanitized result.

- [ ] **Step 5: Install and verify the clean 0.2.2 wheel in a disposable environment**

  Install the repository manifests using the existing Torch-first order and exact binary-only NNx
  command. Run `make verify-nnx-install` with the disposable interpreter and record version,
  non-editable provenance, WHEEL/RECORD ownership, and import origin without disclosing host paths.

- [ ] **Step 6: Run focused GREEN tests**

  ```bash
  make verify-nnx-install
  pytest -p no:cacheprovider \
    tests/nnx_surface/test_tabular_regression_diabetes_mlp_pytorch.py \
    tests/nnx_surface/test_nnx_022_release_surface.py \
    tests/nnx_surface/test_notebook_api_surface.py \
    tests/nnx_surface/test_quantization_mnist_ffnn_pytorch.py -q
  make test-nnx-surface
  ruff check tests/nnx_surface
  git diff --check
  ```

  If the host interpreter is editable, run these gates inside the verified disposable wheel
  environment; do not set `NNX_ALLOW_EDITABLE=1` for release evidence.

- [ ] **Step 7: Commit Task 2**

  ```bash
  git add tests/nnx_surface/test_tabular_regression_diabetes_mlp_pytorch.py \
    tests/nnx_surface/test_nnx_022_release_surface.py \
    tests/nnx_surface/test_notebook_api_surface.py \
    tests/nnx_surface/test_quantization_mnist_ffnn_pytorch.py
  git commit -m "test: cover NNx 0.2.2 consumer surface"
  ```

  Stage the quantization file only if it changed.

## 12.20.5 Task 3: Refresh the Fail-Closed Advisory Observation

**Files:** `docs/dependency-contracts.md`, `tests/test_verify_repo.py`, and only if an existing
generic parser defect is proven, `scripts/verify_repo.py` or `scripts/advisory_baseline.py`

- [ ] **Step 1: Capture the pre-refresh offline baseline state**

  Run D10 and record the expected input-hash drift caused by the new `requirements.txt`. Confirm
  the accepted-policy file remains canonical and unchanged.

- [ ] **Step 2: Run the live four-surface audit**

  ```bash
  make audit-advisories
  ```

  Capture tool version, UTC timestamp, branch commit, Python/platform, input SHA-256 values,
  resolved versions, raw feed records, unique identities, and per-surface attribution. Sanitize
  temporary paths and credentials.

- [ ] **Step 3: Enforce the adoption stop rule**

  Compare the new observation with `security/accepted-advisories.json` and the current ledger.
  Stop without changing policy if there is any new advisory identity, accepted-version drift,
  unexplained surface drift, missing required package, invalid schema, resolver failure, or
  service failure. A disappeared advisory is reconciliation evidence, not proof of remediation.

- [ ] **Step 4: Write ledger-refresh RED tests**

  Require the current snapshot to use the fresh timestamp, commit, manifest hashes, result count,
  and surface rows. Require the historical Issue #59 snapshot and Atlas 0.2.0 observation to remain
  byte-semantically historical. Add mutations for stale runtime hash/count and accidental Atlas
  rewrite.

- [ ] **Step 5: Update only the current human observation**

  Reconcile `docs/dependency-contracts.md` with the exact audit JSON. Do not edit
  `security/accepted-advisories.json` when identity/version/surface parity is unchanged. Do not
  claim a package is remediated solely because it is absent.

- [ ] **Step 6: Run Task 3 GREEN gates**

  ```bash
  pytest -p no:cacheprovider tests/test_verify_repo.py -q -k 'dependency_ledger or advisory'
  python scripts/verify_repo.py --check docs --fast
  make audit-advisories
  ruff check scripts/verify_repo.py scripts/advisory_baseline.py tests/test_verify_repo.py
  git diff --check
  ```

- [ ] **Step 7: Commit Task 3**

  ```bash
  git add docs/dependency-contracts.md tests/test_verify_repo.py
  git commit -m "security: refresh NNx dependency evidence"
  ```

  Add a production parser file only when its new test proved a version-independent defect. Never
  stage the accepted-policy JSON without a separate reviewed security decision.

## 12.20.6 Task 4: Reconcile Current Documentation Across All Three Surfaces

**Files:** the current documentation files listed in §12.20.2, `tests/test_check_docs.py`, and
`docs/manifest.yaml` only if a manifest correction is genuinely needed

- [ ] **Step 1: Write documentation-contract RED tests**

  Require:

  - exactly one current root requirement, `thekaveh-nnx[lm]==0.2.2`, and agreement among all
    current user-facing pin references;
  - wheel SHA, immutable release/tag evidence, and Apache-2.0 transition on the dependency page;
  - current Tier A/B/C canonical-wheel evidence and best-effort quantization boundary;
  - the tabular-regression finding marked resolved in 0.2.2 while the manual loader remains an
    intentional split-preservation choice;
  - current user guidance to use
    `NNCheckpoint.load(run=RUN_ID, type=Checkpoints.BEST)`, with no active
    `NNRun.load("best")` claim;
  - Atlas image evidence remaining 0.2.0 and explicitly Atlas-owned;
  - Issue #58/historical release records remaining historical; and
  - consecutive projected design/plan entries 12.19 and 12.20.

- [ ] **Step 2: Run focused docs tests and capture RED**

  ```bash
  pytest -p no:cacheprovider tests/test_check_docs.py -q \
    -k 'nnx or manifest or tabular_regression or checkpoint'
  ```

- [ ] **Step 3: Update current release entry points**

  Update current 0.2.0 references in README, contributor guidance, Makefile commentary, badge,
  NNx overview, architecture, dependency contracts, TinyShakespeare docs/README, and DPO README.
  Add one Unreleased changelog entry describing adoption and evidence; do not rewrite older entries.

- [ ] **Step 4: Resolve tabular-regression prose without changing execution**

  In the finding ledger, task README, task spec, canonical notebook page, and the notebook's
  relevant Markdown cell, state that 0.2.2 supports floating regression targets through
  `target_dtype=torch.float32`. State that this notebook intentionally retains manual loaders to
  reuse the established sklearn/NNx split. Do not change code, execution counts, outputs, or
  metrics.

- [ ] **Step 5: Correct every current checkpoint claim**

  Replace active `NNRun.load("best")` guidance in NNx overview, concepts, Iris canonical page,
  dimensionality-reduction canonical page, and Iris spec with the valid checkpoint API. Preserve
  historical changelog and maintenance references describing the prior defect.

- [ ] **Step 6: Regenerate and inspect all three surfaces**

  ```bash
  make docs-check
  make docs-wiki
  ```

  Inspect the generated site and wiki copies of the NNx overview, dependency ledger, finding
  ledger, and tabular-regression page. Confirm current markers and links are equivalent after
  transformation, design/plan nav entries are consecutive, and generated trees remain ignored.

- [ ] **Step 7: Run Task 4 GREEN gates**

  ```bash
  pytest -p no:cacheprovider tests/test_check_docs.py tests/test_build_docs.py \
    tests/test_wiki.py tests/test_manifest.py tests/test_transforms.py -q
  make docs-check
  make docs-wiki
  python scripts/verify_repo.py --check docs --fast
  ruff check .
  git diff --check
  ```

- [ ] **Step 8: Commit Task 4**

  Stage the exact changed canonical files and tests, then:

  ```bash
  git commit -m "docs: document NNx 0.2.2 adoption"
  ```

## 12.20.7 Task 5: Run the Complete Release-Acceptance Matrix

**Files:** no intended tracked changes; write sanitized ignored evidence under `.superpowers/sdd/`

- [ ] **Step 1: Establish a clean released-wheel environment**

  From a disposable Python 3.11 environment, install the pinned Torch stack, runtime manifest,
  and docs manifest with the repository's documented order. Require exact NNx 0.2.2,
  non-editable/non-direct-url metadata, and canonical wheel ownership before any acceptance test.

- [ ] **Step 2: Run focused and complete test gates**

  ```bash
  make verify-nnx-install
  make test-nnx-surface
  make test
  make verify
  make lint
  make docs-check
  make docs-wiki
  git diff --check
  ```

  Record exact pass/skip/warning counts. Only documented Torch 2.4.1 torchao backend skips and
  platform-specific PyG loader skips may remain, and every skip must execute its guard.

- [ ] **Step 3: Run Tier A in full**

  ```bash
  make smoke-tier-a
  make check-tier-a-artifacts
  make check-tier-a-clean
  ```

  Require all 18 Tier A outputs, including 17 NNx consumers plus the NumPy control. Confirm source
  notebooks and tracked files remain unchanged.

- [ ] **Step 4: Run Tier B and Tier C in full**

  ```bash
  make smoke-tier-b
  make smoke-tier-c
  ```

  Require all six Tier B and all four Tier C outputs in temporary directories. Do not overwrite
  committed notebooks or adopt graph seed changes.

- [ ] **Step 5: Run the best-effort QAT checkpoint probe**

  In a separate disposable Python 3.11 environment with Torch >=2.5 and compatible torchao,
  install the 0.2.2 binary wheel and run a tiny QAT prepare/train/convert/LAST-checkpoint reload
  through `NNModel.from_checkpoint`, then verify prediction shape/type. Record inability to create
  the environment as a concern; do not alter root manifests or weaken the main acceptance matrix.

- [ ] **Step 6: Request task-level and broad reviews**

  Have fresh reviewers compare every task commit with this plan and the approved design. Then
  request an independent broad review of the entire branch, including mutation resistance,
  released-wheel provenance, notebook semantics, security parity, and three-surface docs.

- [ ] **Step 7: Fix every review finding with TDD**

  Validate each finding, add a failing regression or mutation, apply the narrow fix, rerun focused
  and affected broad gates, and commit separately. Repeat review until no Critical, Important, or
  Minor finding remains.

- [ ] **Step 8: Decide adopt or retain**

  Retain 0.2.2 only when all mandatory local gates and task reviews pass. If a blocker is inherent
  to 0.2.2, revert the root pin and only release-coupled current changes, reinstall 0.2.0, rerun
  the complete matrix, and publish the precise incompatibility. Never partially retain new-only
  consumer calls under a reverted pin.

## 12.20.8 Task 6: GitFlow Rollout, Bookkeeping, and Cleanup

**Files:** GitHub pull requests, Issue #61, parent Issue #53, project board, and local/remote Git
state; no intended product-file changes

- [ ] **Step 1: Push the verified feature branch**

  Confirm a clean tracked tree, exact commit scope, no generated docs, no ignored evidence staged,
  no unintended gitlink change, and no repository service containers. Push
  `codex/issue-61-nnx-release-review` to origin.

- [ ] **Step 2: Open the ready feature-to-develop PR**

  Create a non-draft PR with Issue #61 linkage, adopt-or-retain decision, wheel provenance,
  security result, local test/tier matrix, rollback boundary, and explicit statement that Atlas
  was not launched. Add the `tier-b-smoke` label before waiting for checks.

- [ ] **Step 3: Dispatch Tier C on the feature ref**

  Run `.github/workflows/ci.yml` through `workflow_dispatch` for the exact feature-branch SHA.
  Wait for `smoke-tier-c` and record its run/job URL and conclusion. Reject evidence from a stale
  SHA.

- [ ] **Step 4: Wait for and review all feature PR checks**

  Require the protected contexts `pytest-repository`, `atlas-consumer-policy`, and
  `dependency-audit`; focused `pytest-nnx-surface`; Tier A; labeled Tier B; manually dispatched
  Tier C; docs; and all other triggered jobs to conclude successfully. Inspect failures rather
  than rerunning blindly. Address every review thread.

- [ ] **Step 5: Read back the active ruleset and merge into develop**

  Verify the `gitflow` ruleset still protects `main` and `develop` and requires exactly the three
  intended contexts. Merge the ready PR through GitHub, record its URL and merge SHA, fetch, and
  fast-forward local `develop`.

- [ ] **Step 6: Open and merge the develop-to-main release PR**

  Confirm `develop` contains only intended post-main commits, create a separate ready PR into
  `main`, wait for the full required and triggered matrix, resolve every thread, merge, and record
  the release PR URL and main merge SHA.

- [ ] **Step 7: Synchronize main back into develop**

  If the main merge commit is not already an ancestor of develop, create the normal main-to-develop
  sync PR, wait for required checks, and merge it. Prove `git diff origin/main..origin/develop` is
  empty and main is an ancestor of develop.

- [ ] **Step 8: Complete issue and project bookkeeping**

  Check every Issue #61 acceptance item with exact PR/run/commit evidence, post the closeout
  comment, close Issue #61, move its project item to Done, and check Issue #61 in parent #53. Do
  not mark #53 complete while later backlog issues remain.

- [ ] **Step 9: Clean local and remote state**

  Delete the merged feature branch locally and remotely, remove merged temporary sync branches,
  prune remote refs, close or delete dangling PRs, remove stale worktrees, and prove only intended
  long-lived refs remain. Do not delete unrelated user branches or containers.

- [ ] **Step 10: Prove the final state**

  Record:

  ```bash
  git status --short --branch
  git branch -a
  git worktree list
  git diff origin/main..origin/develop
  git merge-base --is-ancestor origin/main origin/develop
  gh pr list --state open
  docker ps --format '{{.Names}} {{.Status}}'
  ```

  Confirm no ml-eng-lab Atlas, JupyterHub, Ollama, ComfyUI, or issue-specific container remains.
  Leave unrelated project containers untouched.

## 12.20.9 Completion Criteria

- [ ] The latest stable release was revalidated and the adopted or retained version is supported
  by immutable upstream evidence.
- [ ] One exact root NNx pin owns the production version and every NNx-executing CI tier proves the
  released binary wheel immediately before its workload.
- [ ] Consumer tests cover regression dtype/shape, classification preservation, Conv/MoE facade,
  graph signature, and exact QAT boundary.
- [ ] Complete tests plus Tier A, Tier B, and Tier C pass in a canonical released-wheel
  environment; QAT side-environment evidence or its explicit limitation is recorded.
- [ ] The live advisory observation reconciles with the accepted policy without automatic risk
  acceptance.
- [ ] Current repository, site, and wiki documentation agree on the release, resolved limitation,
  checkpoint API, evidence boundary, and historical Atlas ownership.
- [ ] Independent task and broad reviews report no unresolved Critical, Important, or Minor
  findings.
- [ ] Feature-to-develop, develop-to-main, and any required main-to-develop synchronization are
  merged with green checks and the active ruleset remains correct.
- [ ] Issue #61, project board, and parent #53 bookkeeping is complete.
- [ ] Local and remote branches, PRs, worktrees, generated outputs, and repository service
  containers are clean.
