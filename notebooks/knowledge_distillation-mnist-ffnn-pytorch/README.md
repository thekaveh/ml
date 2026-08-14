# knowledge_distillation-mnist-ffnn-pytorch

## 1. Task summary

- **Task:** Knowledge distillation — born-again (self-distillation) training.
- **Dataset:** MNIST handwritten digits via `nnx.NNDataset`.
- **Model:** `nnx.FeedFwdNN` (`Nets.FEED_FWD`), `Activations.RELU`, `hidden_dims=[128, 64]`.
- **Framework:** PyTorch (via [`thekaveh-nnx`](https://github.com/thekaveh/NNx)).

> **Historical output notice:** The committed notebook outputs and results are a historical NNx 0.2.0 snapshot retained under the release-review preservation plan. They are not current NNx 0.2.2 acceptance evidence; current evidence requires a clean released-wheel execution and the complete acceptance matrix.

## 2. Why this exists

**Born-again** training (Furlanello et al., 2018) is the surprising special case of knowledge distillation where student and teacher have the *same* architecture: each generation takes the previous generation's frozen weights as its teacher. The student often beats the teacher despite the identical parameter count — the gain comes from the soft-label regularizer.

`nnx.born_again_train(model, generations=N, train_params=...)` wires the iteration: gen 0 trains plain on hard labels, gen k>0 distills against a frozen deepcopy of gen k-1's model. The notebook shows the per-generation val-loss trajectory on MNIST and the accuracy delta vs a single-gen reference.

The independent control uses the stable `single-gen-reference` identity salt so it keeps a separate run history from born-again generation 0 while retaining the same seeded training configuration.

Under NNx 0.2.2, each later student is reset to the caller-provided initial state, trains against a frozen prior-generation teacher, and records `parent_run_id` lineage to that teacher generation. The wrapper is reused, but each student begins from the same initial weights.

## 3. What's in the notebook

> **Tip:** GitHub may show "Unable to render code block" on output cells with large matplotlib PNGs. [View this notebook on nbviewer](https://nbviewer.org/github/thekaveh/ml-eng-lab/blob/main/notebooks/knowledge_distillation-mnist-ffnn-pytorch/notebook.ipynb) for full rendering.

- §1 Overview — distillation intuition, born-again as same-arch self-distillation, dataset, libraries.
- §2 Environment & Setup — imports, `N_GENERATIONS=3`, `N_EPOCHS=2` per gen, `nnx.set_seed(0)`.
- §3 Data — `NNDataset` on MNIST.
- §4 Model — small FFN baseline, born-again contract spelled out.
- §5 Training — train single-gen reference; separately run `born_again_train(generations=3)` from the same fresh init.
- §6 Evaluation & Results — comparison table (single-gen vs born-again last gen), per-generation trajectory table, discussion of why soft labels help.

## 4. How to run

In the recommended runtime ([../docs/jupyterhub-integration.md](../../docs/jupyterhub-integration.md)):

```bash
# Open the notebook in VS Code attached to the container, or in browser jupyter.
```

Or via the Tier-A `make` target:

```bash
make run-tier-a
```

**Tier-A** (cheap, ~25 s on CPU). Re-executed in CI on every PR. Accepts `SMOKE_TEST=1` (default 0 = full run) via the papermill `parameters` cell.

## 5. Dependencies

- `torch`, `torchvision` — MNIST + tensors.
- `nnx` (PyPI: `thekaveh-nnx`) — `FeedFwdNN`, `NNModel`, `NNDataset`, `born_again_train`.
- `prettytable` — comparison tables.

All in the root `requirements.txt` + `torch-requirements.txt`.

## 6. Known issues

- **Short training budget.** Each independently initialized student gets only 2 epochs, so a single trajectory is noisy. The retained approximately 21-point difference belongs only to the historical NNx 0.2.0 snapshot; do not cite it as a current 0.2.2 result.
- **Diminishing returns.** The original paper shows gains taper off past BA-4 on ResNet-110 / CIFAR-100. We run only 3 generations to stay Tier-A, so the curve doesn't have a chance to plateau visibly.
- **Same-arch only.** This notebook intentionally fixes the student to the *same* architecture as the teacher (the born-again setting). The classical compression case (small student, big teacher) uses the same `nnx.kd_train_step_factory(teacher=...)` plumbing but with different `NNParams` — outside this notebook's scope.
