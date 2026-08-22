# 12.33 Issue 68 NNx Public-Facade Notebook Migration Design

## 12.33.1 Purpose and observed state

Fifteen active notebooks contain 139 imports through implementation-oriented
`nnx.nn`, `nnx.utils`, `nnx.vis_utils`, or `nnx.seeding` paths even though NNx
0.2.0 publishes every imported symbol from its supported top-level facade.
The candidates comprise sixteen symbols: `Activations`, `Devices`,
`FeedFwdNN`, `Losses`, `NNDataset`, `NNGraphDataset`, `NNModel`,
`NNModelParams`, `NNOptimParams`, `NNParams`, `NNTrainParams`, `Nets`,
`Optims`, `Utils`, `VisUtils`, and `set_seed`.

The affected set is one diffusion import; eleven model-surgery imports; two
mixture-of-experts imports; eleven PEFT imports; eleven pruning imports;
eleven quantization imports; one Reddit phase-1 import; forty-three Reddit
phase-2 imports; and forty-eight Reddit phase-3 imports. The four phase-3
notebooks are Tier C and preserve August 2023 outputs.

## 12.33.2 Decision and rejected alternatives

Every candidate is migrated to `from nnx import ...`. Existing top-level NNx
blocks are extended where practical; otherwise one top-level block replaces
the deep imports. Names, aliases, use sites, algorithms, metrics, markdown,
cell order, execution counts, outputs, and unrelated metadata remain
unchanged.

The selected Tier-C strategy creates the immutable annotated tag
`tier-c-public-facade-baseline-2026-08-22` on the dedicated notebook-migration
commit. E5 then compares phase-3 code cells against that successor baseline.
The historical `pre-cleanup-baseline` tag is neither moved nor deleted.

Two alternatives are rejected:

1. Leaving the forty-eight Tier-C imports unchanged would keep active examples
   on deep paths and fail the issue's migration purpose.
2. Adding import exceptions or source canonicalization to E5 would weaken a
   deliberately simple source lock and create a second interpretation of the
   committed notebooks.

If the migration commit changes before the new tag is published, its local tag
is recreated. Once pushed, the tag is immutable; a later source migration must
use a new versioned tag.

## 12.33.3 Public API and inventory contracts

The canonical wheel is `thekaveh-nnx[lm]==0.2.0`, matching the repository and
Atlas pin. Tests prove that all sixteen names exist on `nnx` and are identical
to their current deep-path objects. A syntax-aware scan of every tracked active
notebook rejects imports of those names from the four legacy path families.
Synthetic cases prove that the scanner detects both single-line and
parenthesized imports while ignoring comments and strings.

The scan is intentionally symbol-scoped. NNx submodules that expose no
equivalent supported top-level symbol remain outside this issue.

## 12.33.4 Output and Tier-C preservation

No source notebook is executed in place. The migration is import-only, so no
committed output refresh is required. A structural comparison against the
feature base proves that every cell's outputs and execution count, all
non-source cell fields, notebook metadata, and non-import source text are
unchanged. Tier A, B, and C smoke targets write only to their configured
temporary destinations.

E5 retains exact code-cell equality for Tier C, but reads the named successor
baseline through a single constant used by tag verification, `git show`, and
diagnostics. Documentation names the new tag and records that the old tag is a
historical rollback anchor.

## 12.33.5 Documentation and release contract

README, CONTRIBUTING, conventions, environment setup, the Reddit task guide,
and the NNx library guide teach `from nnx import Symbol` as the supported
notebook style. The maintenance finding is marked resolved and the changelog
records the migration and baseline transition. Design and plan are registered
in the documentation manifest so repository, site, and wiki stay synchronized.

Qualification uses the pinned wheel in an isolated Python 3.11 environment:
focused scanner and NNx-surface tests, verifier tests, complete pytest, Ruff,
repository verification, docs/site/wiki checks, and non-mutating Tier A/B/C
smokes on Atlas JupyterHub. Host-native Ollama remains the only allowed Ollama
service. Atlas is shut down afterward without deleting retained volumes.

After independent review and exact-SHA qualification, publish through feature
to `develop`, `develop` to `main`, and content-neutral `main` back to
`develop`. Rollback reverts the migration and E5/doc changes together; the old
baseline tag remains available throughout.
