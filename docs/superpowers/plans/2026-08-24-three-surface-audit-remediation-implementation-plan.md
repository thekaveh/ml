# 12.41 Three-Surface Audit Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove every important and minor finding from the 2026-08-24 three-surface documentation audit.

**Architecture:** Edit only canonical sources: the four diagram HTML masters, the root README, and the published historical implementation record. Add regression assertions beside the diagram-rendering and documentation-integrity tests, then regenerate derived SVG, PNG, site, and wiki artifacts through the existing pipeline.

**Tech Stack:** HTML/SVG, Markdown, Python 3.11, pytest, CairoSVG, MkDocs Material, repository documentation scripts.

## 12.41.1 Global Constraints

- Treat `docs/manifest.yaml` and committed canonical Markdown/HTML as the only hand-edited documentation sources.
- Keep generated site/wiki trees and root `mkdocs.yml` untracked.
- Match diagram labels to `thekaveh-nnx[lm]==0.2.0` and `torch-geometric==2.8.0.post1`.
- Preserve the existing diagram geometry unless a label needs additional room.
- Run `make docs-check`, `make docs-wiki`, and repository verification before integration.

---

## 12.41.2 Task 1: Lock diagram and prose accuracy with regression tests

**Files:**
- Modify: `tests/test_render_diagrams.py`
- Modify: `tests/test_check_docs.py`

**Interfaces:**
- Consumes: canonical diagram masters and published Markdown.
- Produces: content assertions that reject each audited stale or ungrounded claim.

- [x] **Step 1: Add assertions for the corrected architecture vocabulary**

Assert that the transformer master names RMSNorm, pre-normalization, and SwiGLU; GraphConvNN identifies its PyG GCNConv implementation and symmetric normalization; the MLP master uses symbolic dimensions and a generic optimizer update; and the DDPM master names MNIST only.

- [x] **Step 2: Add assertions for clinical, evidence-based prose**

Assert that the README does not promise a fixed Codespaces setup duration and the opener visual record does not publish the production image prompt.

- [x] **Step 3: Run the focused tests and observe the audited claims fail**

```bash
python -m pytest tests/test_render_diagrams.py tests/test_check_docs.py -q
```

Expected: the new assertions fail on the stale diagram labels, fixed-duration promise, and published production prompt.

## 12.41.3 Task 2: Correct canonical diagram masters and prose

**Files:**
- Modify: `docs/diagrams/ml-eng-lab-transformer.html`
- Modify: `docs/diagrams/ml-eng-lab-gnn.html`
- Modify: `docs/diagrams/ml-eng-lab-mlp.html`
- Modify: `docs/diagrams/ml-eng-lab-ddpm.html`
- Modify: `README.md`
- Modify: `docs/superpowers/plans/2026-08-01-opener-visual-remediation-implementation-plan.md`

**Interfaces:**
- Consumes: the pinned NNx and PyG runtime contracts.
- Produces: accurate canonical content for every published surface.

- [x] **Step 1: Update the transformer master**

Represent pre-normalized RMSNorm attention and SwiGLU feed-forward blocks, plus the final RMSNorm used by the pinned NNx transformer.

- [x] **Step 2: Update the GraphConv master**

Represent NNx `GraphConvNN` as its actual `pyg.nn.GCNConv` implementation: self-loops, an additive neighborhood sum, symmetric degree normalization, and Kipf-Welling attribution. Do not describe PyG's separate `GraphConv` operator.

- [x] **Step 3: Generalize the shared MLP master**

Use symbolic input/output dimensions, task-dependent objective text, and a generic parameter update so the diagram remains true for classification, regression, NLP, PyTorch, and NumPy consumers.

- [x] **Step 4: Correct DDPM and prose claims**

Remove Fashion-MNIST from the MNIST-only DDPM master, remove the unbenchmarked Codespaces duration, and replace the historical production prompt with concise artifact provenance.

- [x] **Step 5: Run focused tests until green**

```bash
python -m pytest tests/test_render_diagrams.py tests/test_check_docs.py -q
```

Expected: all focused tests pass.

## 12.41.4 Task 3: Regenerate and verify all three surfaces

**Files:**
- Regenerate: `docs/diagrams/img/*.png`
- Regenerate: ignored `generated/site/`, `generated/wiki/`, `mkdocs.yml`, and `site/`

**Interfaces:**
- Consumes: corrected canonical sources.
- Produces: matching repository PNGs, site SVGs, wiki PNGs, and deterministic generated trees.

- [x] **Step 1: Run the complete documentation builds**

```bash
make docs-check
make docs-wiki
```

Expected: strict MkDocs build and wiki dry-run pass without warnings.

- [x] **Step 2: Run repository verification**

```bash
make verify
```

Expected: repository structure and documentation checks return zero findings.

- [x] **Step 3: Inspect the final diff and worktree**

```bash
git diff --check
git status --short
```

Expected: only canonical sources, tests, the plan/manifest entry, and regenerated committed diagram PNGs are modified.
