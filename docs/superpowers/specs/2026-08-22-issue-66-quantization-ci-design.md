# 12.29 Issue 66 Quantization Execution and CI Design

## 12.29.1 Purpose and observed state

Issue #66 requires the active MNIST quantization notebook to execute complete
post-training quantization (PTQ) and quantization-aware training (QAT), prove
that training checkpoints reconstruct, and move into an automated tier only
when the supported dependency stack can sustain the workflow.

The notebook is currently manual-only because the pre-Issue-62 Torch/torchao
stack could not execute its required APIs. That reason is no longer true. A
clean macOS arm64 environment installed only from the canonical locks
(`torch==2.11.0`, `torchvision==0.26.0`, `torchao==0.18.0`, and
`thekaveh-nnx==0.2.0`) produced these baseline results on 2026-08-22:

| Probe | Result |
| --- | --- |
| Canonical stack verification | passed |
| Focused quantization surface | 31 tests passed |
| One-epoch smoke notebook | passed in 24 seconds |
| Three-epoch full notebook | passed in 31 seconds |
| PTQ artifact | converted weight-only int8 model, 112.6 KiB state |
| QAT artifact | converted `Int8DynActInt4WeightLinear` model, 406.7 KiB state |

Both notebook executions completed FP32, PTQ, and QAT evaluation without an
error cell. The existing Tier B budget is therefore sufficient and the old
manual-only exception must be removed.

## 12.29.2 Decision and alternatives

The selected design restores `quantization-mnist-ffnn-pytorch` to the existing
Tier B job. Tier B already supplies the intended weekly, pull-request-label,
and manual-dispatch triggers; a separate workflow would duplicate the same
locked installation and notebook runner.

Two alternatives are rejected:

1. **Keep the notebook manual-only.** Canonical smoke and full execution are
   both bounded and successful, so the old incompatibility evidence no longer
   supports this exception.
2. **Add a dedicated quantization workflow.** Isolation would be marginally
   clearer, but it would duplicate the Tier B setup, lock verification, cache,
   artifact upload, and trigger policy for one short notebook.

The tier assignment is metadata-driven. The task specification is the
authority; the Makefile, CI inventory, generated notebook documentation, and
verification output are projections of that specification.

Tier B historically flattens notebook outputs by basename, and both the image
classification and quantization sources are named `notebook.ipynb`. The
quantization projection therefore uses the explicit stable output name
`quantization-mnist-ffnn-pytorch.ipynb`; all six existing Tier B output names
remain unchanged. The verifier owns the same exact mapping and rejects any
duplicate mapped output.

## 12.29.3 Notebook execution contract

The notebook retains its full three-epoch path and its deterministic
one-epoch smoke parameter. Both paths must:

1. seed NNx and Torch through `nnx.set_seed(0)`;
2. train and evaluate the FP32 baseline;
3. construct and evaluate the NNx PTQ weight-only int8 model;
4. train QAT through `QATLifecycleCallback("8da4w")`, then prove the callback
   prepared and converted the model;
5. prove the converted network contains the expected torchao integer-linear
   implementation;
6. locate the QAT `last.pt` checkpoint, load the `NNCheckpoint`, reconstruct a
   model with `NNModel.from_checkpoint`, and evaluate it;
7. require reconstructed state tensors and checkpoint metadata to match the
   saved checkpoint exactly, and require reconstructed evaluation metrics to
   be finite; and
8. emit a stable machine-readable completion marker summarizing PTQ, QAT,
   checkpoint, and determinism status.

NNx writes FP32 and QAT runs to the same deterministic run identifier because
their model and training parameters are equal. The QAT run intentionally
replaces the earlier checkpoint set. The notebook does not claim the PTQ model
is an NNx training checkpoint; it proves PTQ through conversion, evaluation,
and serialized state size, while checkpoint reconstruction belongs to the QAT
training lifecycle.

The saved QAT checkpoint contains the trained floating-point shadow state
immediately before the lifecycle callback performs final conversion. This is
the supported NNx persistence boundary. The checkpoint is written while
fake-quant modules are active, so its recorded validation metrics describe the
fake-quant forward path; `NNModel.from_checkpoint` reconstructs the same
learned tensors in the plain shadow architecture and is not expected to
reproduce those fake-quant metrics. Reload correctness is therefore exact
tensor and checkpoint-metadata parity plus finite reconstructed evaluation.
The separate in-memory assertion proves that the post-training model was
converted for quantized inference.

## 12.29.4 Regression and output verification

The existing focused tests remain the API-level regression layer. Additional
tests exercise the notebook contract without training MNIST:

- the notebook declares bounded smoke parameters and the stable completion
  marker;
- checkpoint discovery, `torch.load(..., weights_only=False)`,
  `NNModel.from_checkpoint`, and metric-parity assertions remain present;
- PTQ and QAT conversion assertions cannot be removed independently; and
- the task specification assigns exactly Tier B and no manual exception.

The smoke-output verifier gains a quantization-specific semantic check. For a
successfully executed Tier B output it requires the completion marker and
rejects missing, false, duplicated, or malformed fields. General notebook
execution checks still reject error cells, missing outputs, and source/output
inventory drift.

## 12.29.5 Documentation contract

All three documentation surfaces must describe the same final state:

- the source task README explains PTQ, QAT, shadow-checkpoint reconstruction,
  the one-epoch smoke bound, and Tier B triggers;
- repository documentation and the root README remove the manual-only Issue
  #66 exception and name Tier B;
- generated notebook documentation is rebuilt from the task specification and
  remains byte-reproducible; and
- the Unreleased changelog records the restored workflow and CI placement.

The dependency ledger keeps Torch/torchao/NNx ownership with Issues #61-#65
and records Issue #66 as execution evidence, not as a new dependency bump.

## 12.29.6 Validation and release contract

Local qualification uses a clean canonical-wheel environment and runs:

```bash
make install-torch-stack
make verify-torch-stack
make verify-nnx-stack
python -m pytest tests/nnx_surface/test_quantization_mnist_ffnn_pytorch.py -q
make smoke-tier-b
make test
make verify
make lint
make docs-check
make docs-wiki
```

The full notebook also runs once with `SMOKE_TEST=0`. Remote qualification
runs the focused tests, full repository gates, and Tier B CI. Atlas then starts
the retained JupyterHub `ml-eng` track, executes the full notebook inside the
JupyterHub environment, verifies its output contract, and shuts down through
ordinary `make atlas-down`. Volumes are preserved; no `COLD=1` shutdown is
allowed. Quantization does not require Ollama, so no Ollama container or host
request is introduced.

After a clean immutable feature SHA is qualified, publish through
feature -> `develop` -> `main` -> `develop`. Any tracked change after the
qualification SHA invalidates the evidence. Publish the report, remove only
Issue #66 temporary refs/worktrees/artifacts, set the project item to Done,
and close Issue #66 as the final issue mutation.
