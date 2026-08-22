# 9.1 NNx (`thekaveh-nnx`) findings

> **Note (2026-06-14)**: ml-eng-lab switched from a git submodule at `./nnx` to the `thekaveh-nnx` PyPI distribution. Source paths cited below (e.g. `nnx/src/nnx/nn/dataset/nn_dataset.py:24`) refer to the upstream [`thekaveh/NNx`](https://github.com/thekaveh/NNx) repo, not a local submodule.

Issues found by the verify_repo.py loop in the `nnx` (PyPI: `thekaveh-nnx`) library. These are
NOT fixed by this loop (per spec §1.3); they are surfaced here for an upstream
PR follow-up to [thekaveh/NNx](https://github.com/thekaveh/NNx).

## 9.1.1 Findings

### 9.1.1.0 Triage summary

Last triaged against NNx releases `v0.2.0` through `v0.2.3` and upstream
`main` on 2026-08-22. "Upstream resolved" means the behavior is available in a
released NNx wheel; it does not imply availability under ml-eng-lab's retained
`thekaveh-nnx[lm]==0.2.0` runtime.

| ID | Finding | Upstream disposition | Affected notebooks | Local status / remaining work |
| --- | --- | --- | --- | --- |
| F1 | `NNDataset` uses one full-split batch for each omitted batch size | Open documentation/API-default issue [NNx #188](https://github.com/thekaveh/NNx/issues/188); unchanged through `v0.2.3` | `diffusion-mnist-ddpm-pytorch`, `moe-fmnist-mixture-of-experts-pytorch`, `self_supervised-fmnist-jepa-pytorch` | Resolved locally by Issue #69 with explicit public `batch_sizes=` values; no follow-up |
| F2 | `nnx.deepen` is function-preserving only for ReLU | Released, enforced, documented design constraint; this is not an open upstream bug | `model_surgery-mnist-ffnn-pytorch` | Notebook deliberately uses ReLU and documents the constraint; no follow-up |
| F3 | NNx 0.2.0 `NNTabularDataset` is classification-only | Resolved by [NNx #81](https://github.com/thekaveh/NNx/issues/81) in `v0.2.2` with `target_dtype` | `tabular_regression-diabetes-mlp-pytorch` | Manual float-target loader remains under 0.2.0; coordinated upgrade/migration tracked by [ml-eng-lab #146](https://github.com/thekaveh/ml-eng-lab/issues/146) |
| F4 | `EarlyStopping()` defaults to a classification-only validation field | Open callback-contract issue [NNx #189](https://github.com/thekaveh/NNx/issues/189); unchanged through `v0.2.3` | `tabular_regression-diabetes-mlp-pytorch` guidance | Use `monitor="val_edp.loss"`; the notebook does not instantiate the callback, so no local code follow-up |
| F5 | Training completion output embeds an absolute run path | Open portability issue [NNx #190](https://github.com/thekaveh/NNx/issues/190); unchanged through `v0.2.3` | Any notebook that captures NNx training output | Verifier rule `E13.stale_active_notebook_path` rejects leaked paths; no local follow-up before an upstream release |

### 9.1.1.1 `NNDataset` default `batch_size` packs the whole train set into one batch

**Upstream disposition:** [NNx #188](https://github.com/thekaveh/NNx/issues/188)
tracks prominent public documentation of the default and its training
consequence without silently changing the backward-compatible runtime contract.

**Release evidence:** NNx `v0.2.0` through `v0.2.3` and current `main` declare
`batch_sizes=(None, None, None)` and resolve each `None` to the corresponding
complete split size. Source comments and tests state the semantic, while the
generated public class entry exposes only the signature.

**Affected notebooks:** `diffusion-mnist-ddpm-pytorch`,
`moe-fmnist-mixture-of-experts-pytorch`, and
`self_supervised-fmnist-jepa-pytorch`. `text_generation-tinyshakespeare-transformer-pytorch`
has the same small-batch training need, but it uses an intentional custom
sequence-window dataset rather than bypassing `NNDataset`.

`NNDataset(ds_class=thv.datasets.MNIST, ...)`'s `train_loader` defaults to `batch_size=54000` (the whole 60k train set minus the val carve-off). For full-batch SGD on classifiers this is fine; for **diffusion / MoE / transformer / JEPA / any task that needs many noise- or routing-level samples per epoch**, one batch per epoch is far too few — the train step runs ~1 time per epoch and the loss barely budges.

Before Issue #69, each affected notebook worked around this with:

```python
from torch.utils.data import DataLoader
train_loader = DataLoader(ds.train_loader.dataset, batch_size=128, shuffle=True)
```

**Local workaround/status:** Issue #69 resolved the loader bypass.
`nnx.NNDataset` accepts a
`batch_sizes: tuple[Optional[int], Optional[int], Optional[int]] = (None, None, None)`
constructor argument (`nnx/src/nnx/nn/dataset/nn_dataset.py:24`). Diffusion and MoE now pass
`batch_sizes=(128, None, None)`; JEPA passes `batch_sizes=(128, 128, None)` because its linear-probe
path also consumes the validation loader in 128-sample batches. All three notebooks use the
wrapper-owned loaders directly. This removes the internal `.dataset` bypass while preserving
batch size, shuffle, worker, split, seed, batch-count, and recorded-output semantics. The
TinyShakespeare notebook intentionally remains on its custom sequence-window dataset.

**Remaining work:** None in ml-eng-lab. The supported public `batch_sizes=`
contract fully resolves the local workaround. NNx #188 owns discoverability of
the unchanged upstream default for new callers.

### 9.1.1.2 `nnx.deepen` is function-preserving only for `Activations.RELU`

**Upstream disposition:** Accepted, documented design constraint; this is not an
open upstream bug. Net2DeeperNet's identity insertion is exactly
function-preserving only when the inserted activation is idempotent on the
preceding activation's range. A different constant bias cannot make arbitrary
sigmoid, tanh, or GELU compositions equal to the original function.

**Release evidence:** NNx `v0.2.0` and every later release through `v0.2.3`
document the ReLU equation in `src/nnx/surgery/deepen.py`, state the restriction
in the public docstring, and raise a targeted `ValueError` for unsupported
activations. Current `main` retains the same contract.

**Affected notebooks:** `model_surgery-mnist-ffnn-pytorch`.

`nnx.deepen(net, after_layer_name=...)` inserts an identity-init `Linear` after a target Linear. The identity init only preserves the forward output when the *activation between* the original Linear and the new Linear is ReLU (since `ReLU(I x) == ReLU(x)` for any `x`; sigmoid/tanh/GELU pass non-negative *and* negative values through differently).

On any non-ReLU activation the surgery raises `ValueError: deepen: activation is 'leaky_relu', but identity-init insertion is function-preserving only for ReLU.` at construction.

**Local workaround/status:** The notebook deliberately constructs its baseline
with `Activations.RELU`, asserts post-surgery forward equivalence, and documents
the ReLU-only boundary in its README and task specification.

**Remaining work:** None. A future distinct non-function-preserving deepening
operation would be a new NNx feature, not a fix for this contract.

### 9.1.1.3 `NNTabularDataset` coerces targets to `torch.long` (classification-only)

**Upstream disposition:** Resolved by
[NNx #81](https://github.com/thekaveh/NNx/issues/81).

**Release evidence:** NNx `v0.2.2` added
`NNTabularDataset(target_dtype=torch.float32)`, skips classification label
validation in regression mode, sets `output_dim=1`, and yields targets shaped
`(batch, 1)`. The `v0.2.2` changelog and released tests cover the contract.

**Affected notebooks:** `tabular_regression-diabetes-mlp-pytorch`.

The retained NNx 0.2.0 contract's `NNTabularDataset(..., target_col=...)` hard-codes `y = torch.tensor(..., dtype=torch.long)` in `__post_init__`. This is correct for classification but breaks regression: `Losses.MEAN_SQUARED_ERROR` expects `float32` targets of shape `(N, 1)`.

Regression notebooks must build the DataLoaders manually:

```python
DataLoader(
    TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(y).float().unsqueeze(-1),
    ),
    ...,
)
```

The NNx 0.2.0 `NNTabularDataset` docstring says *"For regression, prefer to
construct the DataLoaders yourself"*, accurately documenting the retained
runtime but not providing the newer released API.

**Local workaround/status:** Issue #61 proved the 0.2.2 surface in an isolated
canonical-wheel trial, then retained 0.2.0 because the default Atlas JupyterHub
runtime remained on 0.2.0. The notebook therefore keeps its manual seeded split
and float-target loader.

**Remaining work:** [ml-eng-lab #146](https://github.com/thekaveh/ml-eng-lab/issues/146)
owns a coordinated root/Atlas NNx upgrade and the subsequent migration to
`target_dtype=torch.float32`. Do not remove the workaround before that runtime
qualification passes.

### 9.1.1.4 `EarlyStopping(monitor=...)` default is `"val_edp.error"`, doesn't exist for regression EDPs

**Upstream disposition:** [NNx #189](https://github.com/thekaveh/NNx/issues/189)
tracks an explicit regression-safe default/fallback contract.

**Release evidence:** NNx `v0.2.0` through `v0.2.3` and current `main` default
to `monitor="val_edp.error"`. When the selected field is absent or `None`,
`_lookup_monitored()` returns `None` and `on_epoch_end()` returns without
advancing the best value or patience counter.

**Affected notebooks:** `tabular_regression-diabetes-mlp-pytorch` documents the
contract in §6 but does not instantiate `EarlyStopping`.

`EarlyStopping`'s default works for classification (lower error = better). For
regression the evaluation data point has `loss` but may have no `error` value,
so `monitor="val_edp.loss"` must be passed explicitly. The default does not
raise a clear error; it silently skips the stopping update.

**Local workaround/status:** All regression guidance requires the explicit
`monitor="val_edp.loss"` form. No active notebook relies on the unsafe default.

**Remaining work:** None in current ml-eng-lab code. NNx #189 owns the upstream
callback behavior; future regression notebooks must keep the explicit monitor
until a fixed released contract is adopted.

### 9.1.1.5 training completion messages expose absolute run paths

**Upstream disposition:** [NNx #190](https://github.com/thekaveh/NNx/issues/190)
tracks one portable display-path contract shared by both training entry points.

**Release evidence:** In NNx `v0.2.0` through `v0.2.3` and current `main`,
`NNModel.train()` and `Trainer.train()` compute the human-facing message with
`os.path.join(os.getcwd(), "runs", run.id)`. `NNRun.save()` persists the run but
does not print this message.

**Affected notebooks:** Any notebook that captures NNx training output can
record the executing worktree, JupyterHub mount, container path, or host-local
directory.

The finding was surfaced by historical active notebook outputs carrying baked-in local paths
such as maintainer worktrees, JupyterHub mounts, removed in-repo source trees,
and host-local Python environments. The 2026-07-04 maintenance pass normalized
the remaining active-notebook artifacts and added verifier rule
`E13.stale_active_notebook_path`, so `python scripts/verify_repo.py --check
execution --fast` now rejects stale path artifacts in active notebooks.

The two training entry points emit a confirmation string with the absolute
filesystem path of the saved run directory. Two related issues follow:

1. **Execution-environment path leak**: any committed notebook output can carry the path from whatever machine, container, or worktree last executed it. This is reproducibility noise because the path is meaningless to readers outside that runtime.
2. **CI normalization is not sufficient**: CI now writes Tier-A re-execution
   results to a temporary artifact tree and deliberately leaves committed
   notebook snapshots untouched. It therefore cannot (and should not) be used
   to normalize machine-local paths in tracked output.

NNx #190 proposes a path relative to the effective runs root or current working
directory while leaving persistence and semantically required machine-local
metadata unchanged.

**Local workaround/status:** Keep active notebook outputs free of stale
machine-local paths through `E13.stale_active_notebook_path`, and avoid
claiming that a CI re-run alone makes these outputs portable. Once nnx prints a
relative run path, a maintainer can deliberately refresh the committed snapshot
with `make run-tier-a` without reintroducing environment-specific paths.

**Remaining work:** None before an upstream release. The verifier remains
load-bearing while NNx #190 is open and until ml-eng-lab adopts the released
change.
