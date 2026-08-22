# 12.34 Issue 68 NNx Public-Facade Notebook Migration Implementation Plan

> **For agentic workers:** Execute each task with red-green-refactor, exact
> preservation checks, independent review, and verification before completion.

**Goal:** Replace all supported deep NNx imports in active notebooks with the
public top-level facade while retaining notebook behavior and preserved output.

**Architecture:** Add a syntax-aware inventory guard and public-object identity
contract, migrate only classified imports, establish an immutable successor
Tier-C code baseline, and synchronize canonical documentation.

**Spec:** `docs/superpowers/specs/2026-08-22-issue-68-nnx-public-facade-design.md`

## 12.34.1 Global constraints

- Migrate exactly the sixteen classified symbols from `nnx.nn.*`, `nnx.utils`,
  `nnx.vis_utils`, and `nnx.seeding` to `nnx`.
- Do not change notebook algorithms, metrics, prose, outputs, execution counts,
  or unrelated source and metadata.
- Do not execute a committed notebook in place.
- Do not move or delete `pre-cleanup-baseline`.
- The new Tier-C baseline tag is immutable after publication.
- Use the pinned `thekaveh-nnx[lm]==0.2.0` wheel, not a sibling editable tree.
- Follow feature → `develop` → `main` → `develop` GitFlow.

## 12.34.2 Task 1: Prove the public facade and inventory guard

**Files:**

- Modify: `tests/nnx_surface/test_notebook_api_surface.py`

- [ ] Add the exact public-name/deep-path mapping and assert identity through
  the installed NNx 0.2.0 wheel.
- [ ] Add an AST-based active-notebook scanner for the classified deep imports,
  including IPython-line masking and synthetic detection tests.
- [ ] Run the focused tests and capture RED with 139 candidates in 15 notebooks.
- [ ] Commit the failing contract separately.

## 12.34.3 Task 2: Migrate notebook imports only

**Files:**

- Modify: the 15 classified active notebooks.

- [ ] Replace or consolidate classified imports into top-level `nnx` blocks.
- [ ] Run the focused guard and public-identity tests to GREEN.
- [ ] Compare feature base and working tree structurally. Require unchanged
  outputs, execution counts, notebook metadata, non-source cell fields, and
  source after normalizing only the classified import statements.
- [ ] Confirm 139 candidates became zero and commit the notebook migration.
- [ ] Create the local annotated Tier-C successor tag on this commit.

## 12.34.4 Task 3: Move E5 to the successor baseline

**Files:**

- Modify: `scripts/verify_repo.py`
- Modify: `tests/test_verify_repo.py`

- [ ] Add tests that require the new baseline constant in tag lookup,
  diagnostics, and Tier-C `git show` reads; run RED.
- [ ] Implement the single baseline constant and run verifier tests GREEN.
- [ ] Prove E5 passes against the local successor tag.
- [ ] Commit the verifier transition without changing notebook source.

## 12.34.5 Task 4: Teach the public API and baseline lifecycle

**Files:**

- Modify: `README.md`, `CONTRIBUTING.md`, `CHANGELOG.md`
- Modify: `docs/conventions.md`, `docs/env-setup.md`, `docs/nnx-library.md`
- Modify: `docs/notebooks/node_classification-reddit-gnn-pyg.md`
- Modify: `docs/maintenance/overnight-2026-07-04.md`
- Modify: `docs/manifest.yaml`

- [ ] Add or update documentation assertions and run them RED.
- [ ] Replace stale deep-import guidance, document the immutable successor tag,
  and mark OM-069 resolved.
- [ ] Register this design and plan and regenerate/check documentation.
- [ ] Commit documentation separately.

## 12.34.6 Task 5: Qualify, review, and publish

- [ ] Run focused NNx and verifier suites.
- [ ] Run `make verify-torch-stack`, `make verify-nnx-install`, `make test`,
  `make verify`, `make lint`, `make docs-check`, and `make docs-wiki`.
- [ ] Start the required Atlas JupyterHub path, run non-mutating Tier A, B, and
  C smoke targets and output-contract checks, then shut Atlas down while
  preserving volumes.
- [ ] Obtain independent review; reproduce and fix only verified findings.
- [ ] Re-run exact-SHA qualification and preservation checks.
- [ ] Push the frozen feature branch and annotated baseline tag; merge through
  GitFlow, publish Pages/wiki, close #68, update the roadmap/project, delete the
  feature branch, and remove only Issue #68 temporary artifacts.
