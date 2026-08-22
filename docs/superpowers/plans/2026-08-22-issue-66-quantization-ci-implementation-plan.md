# 12.30 Issue 66 Quantization Execution and CI Implementation Plan

**Goal:** Restore complete supported PTQ/QAT execution, prove QAT checkpoint
reconstruction, promote the notebook to Tier B, and publish Issue #66 through
the full GitFlow cycle.

**Architecture:** The notebook owns runtime assertions and a stable completion
marker. Focused tests protect its source contract, the smoke-output verifier
protects executed semantics, and task metadata drives Tier B inventory and
documentation projections.

**Tech stack:** Python 3.11; Jupyter/papermill; PyTorch 2.11; TorchVision 0.26;
torchao 0.18; NNx 0.2; pytest; YAML; GitHub Actions; Atlas JupyterHub.

## 12.30.1 Constraints

- Keep the canonical dependency locks unchanged.
- Preserve full three-epoch execution and deterministic one-epoch smoke.
- Exercise real NNx PTQ and QAT APIs; import-only coverage is insufficient.
- Reconstruct the supported pre-conversion QAT shadow checkpoint, require exact
  state/metadata parity, and require finite reconstructed evaluation.
- Prove final in-memory QAT conversion independently from checkpoint reload.
- Use existing Tier B triggers and artifact handling; do not add a duplicate
  workflow.
- Use only the Atlas JupyterHub `ml-eng` track for remote notebook execution.
- Preserve Atlas volumes and use ordinary shutdown.
- Do not mutate tracked files after the final qualification commit.

## 12.30.2 Task 1: Protect the notebook lifecycle with RED tests

**Files:**

- Modify: `tests/nnx_surface/test_quantization_mnist_ffnn_pytorch.py`
- Modify: `tests/test_notebook_infrastructure.py`
- Modify: `tests/test_verify_smoke_outputs.py`

- [ ] Add focused tests for QAT checkpoint discovery, safe explicit loading,
  `NNModel.from_checkpoint`, state/metadata parity, finite reconstructed
  evaluation, converted-module assertions, and the stable completion marker.
- [ ] Add task-spec tests requiring `tier: b` and rejecting a manual Issue #66
  exception.
- [ ] Add output-verifier tests for a valid marker and mutations covering a
  missing marker, false field, duplicate marker, and malformed payload.
- [ ] Run the focused selections and record the expected RED failures.

## 12.30.3 Task 2: Implement checkpoint reconstruction and output semantics

**Files:**

- Modify: `notebooks/quantization-mnist-ffnn-pytorch/notebook.ipynb`
- Modify: `scripts/verify_smoke_outputs.py`

- [ ] Add deterministic QAT checkpoint path resolution from the returned run
  identity.
- [ ] Load the saved `NNCheckpoint` explicitly, reconstruct the model, assert
  exact state/metadata parity, and require finite evaluation.
- [ ] Assert PTQ conversion, QAT callback lifecycle, converted integer-linear
  presence, finite metrics, and non-empty serialized artifacts.
- [ ] Emit one stable JSON completion marker.
- [ ] Teach the smoke-output verifier to validate that marker for this exact
  Tier B notebook.
- [ ] Map the quantization smoke to a stable unique output filename so its
  generic `notebook.ipynb` basename cannot collide with image classification.
- [ ] Run the focused tests to GREEN and execute a one-epoch notebook smoke.
- [ ] Commit the lifecycle slice.

## 12.30.4 Task 3: Restore Tier B and synchronize documentation

**Files:**

- Modify: `notebooks/quantization-mnist-ffnn-pytorch/docs/spec.yaml`
- Modify: `notebooks/quantization-mnist-ffnn-pytorch/README.md`
- Modify: `Makefile`
- Modify: `.github/workflows/ci.yml`
- Modify: `README.md`
- Modify: `CHANGELOG.md`
- Modify: `docs/dependency-contracts.md`
- Modify: `docs/notebook-infrastructure.md`
- Regenerate: `docs/notebooks/quantization-mnist-ffnn-pytorch.md`

- [ ] Change the authoritative task specification from manual to Tier B and
  document the smoke/checkpoint/output contract.
- [ ] Remove manual-only projections and add the notebook to Tier B commands
  and workflow inventory.
- [ ] Update source, repository, dependency, infrastructure, and changelog
  narratives.
- [ ] Rebuild generated notebook documentation through the repository tool.
- [ ] Run spec, docs, workflow, and reproducibility tests to GREEN.
- [ ] Commit the tier/documentation slice.

## 12.30.5 Task 4: Canonical local qualification

- [ ] Build a clean Python 3.11 environment from canonical locks.
- [ ] Verify pip, Torch/TorchVision/torchao, NNx provenance, and the focused
  quantization test file.
- [ ] Run Tier B smoke and validate all output artifacts.
- [ ] Run the full notebook with `SMOKE_TEST=0` and validate its marker.
- [ ] Run complete tests, repository verification, lint, strict docs build,
  wiki generation, and `git diff --check`.
- [ ] Confirm clean tracked state, then freeze one feature SHA/tree.

## 12.30.6 Task 5: Review and remote qualification

- [ ] Review the complete feature diff against Issue #66 and this design.
- [ ] Push the feature branch and open the feature -> `develop` pull request.
- [ ] Require all branch checks, including Tier B, to pass on the frozen SHA.
- [ ] Start Atlas with the retained manifest and prove JupyterHub is the only
  required execution service.
- [ ] Execute the full notebook inside Atlas JupyterHub, validate the output
  marker/checkpoint contract, and retain immutable evidence.
- [ ] Stop Atlas normally, prove volumes remain, and confirm clean parent and
  submodule state.

## 12.30.7 Task 6: GitFlow publication and cleanup

- [ ] Merge the feature PR to `develop` and qualify the merge commit.
- [ ] Open and merge `develop` -> `main`; require release checks and Pages/wiki
  publication to pass.
- [ ] Sync `main` -> `develop`, merge it, and prove equal trees with `main` an
  ancestor of `develop`.
- [ ] Publish the immutable qualification report to the feature PR, release PR,
  and Issue #66.
- [ ] Remove only Issue #66 worktrees, refs, generated outputs, and temporary
  environments; preserve unrelated state and Atlas volumes.
- [ ] Set Issue #66 project status to Done and close it as completed.
