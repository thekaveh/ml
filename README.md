# ml-eng-lab — personal ML lab

A multi-project repository of machine-learning task demonstrations, organized as a portfolio of self-contained ML experiments. Active experiments live under `notebooks/[task]-[dataset]-[model]-[framework]/`; each experiment directory contains its notebook(s) and README, with gitignored `data/` and `runs/` directories created as needed by runtime downloads and training/evaluation runs.

## 1. Overview

This repo serves three overlapping purposes:

- **Personal lab** — a place to prototype new ML tasks quickly.
- **Portfolio** — each task folder reads as a standalone demonstration of a technique.
- **Educational resource** — notebooks include narrative explanations alongside code.

**Paradigms covered** (see [§4.1](#41-active) for the per-task mapping): image classification (numpy from-scratch + PyTorch FFNN), tabular classification + regression, GNNs on graphs (`pytorch-geometric` GraphSAGE / GraphConv / GAT — node classification, link prediction, community detection), NLP (spaCy + NLTK pipelines, BPE tokenizer), transformer LM with sampling stack, diffusion (DDPM), preference alignment (DPO), self-supervised (I-JEPA), Mixture-of-Experts, PEFT (LoRA / DoRA), quantization (PTQ + QAT), pruning, knowledge distillation, model surgery (Net2Net), autoencoders, clustering.

A shared PyTorch toolkit (`nnx`, [`thekaveh-nnx`](https://pypi.org/project/thekaveh-nnx/) on PyPI) provides reusable training-loop, dataset, and visualization primitives that the notebooks consume. Library and tasks co-evolve: each new task lands its required `nnx` additions upstream first ([`thekaveh/NNx`](https://github.com/thekaveh/NNx)), then ml-eng-lab bumps the pinned version here. YAGNI applies — no speculative abstractions in `nnx`.

## 2. Repository layout

```
ml-eng-lab/
├── README.md                                  (this file)
├── CONTRIBUTING.md                            (workflow + conventions)
├── CHANGELOG.md                               (release notes)
├── Makefile                                   (papermill tier targets)
├── docs/                                      (env/runtime docs, dependency contracts, findings, maintenance log)
├── requirements.txt + torch-*.txt             (pip deps; thekaveh-nnx[lm]==0.2.0)
├── infra/                                     (Atlas git submodule; pinned infrastructure)
├── atlas.consumer.yml                         (ml-eng Atlas consumer contract)
├── compose/                                   (parent-owned Atlas compose overlays)
├── scripts/                                   (Atlas lifecycle, verifier, notebook edit/import helpers)
├── tests/                                     (pytest: nnx_surface contract + verifier + helpers)
└── notebooks/                                 (21 active task folders plus notebooks/archive/)
```

See [CHANGELOG.md](CHANGELOG.md) for release history; per-task folders are linked from [§4.1 Active](#41-active), and secondary docs are linked from [§10 Other documentation](#10-other-documentation).

## 3. Quick start

Four ways to run these notebooks, ordered from managed runtime to local execution.

### 3.1. Atlas JupyterHub + local VS Code (recommended)

Atlas is the direct successor to the previous infrastructure seam. This repository
pins it as the `infra/` submodule at `61c7c5103660e2226bf107c115dae42bf46f8374` and starts the
`ml-eng` track. The default workflow keeps notebooks and VS Code on the host while execution uses
the running Atlas JupyterHub kernel. The NumPy MNIST task's `mounted-workspace` mode is
`mounted-required`, so run it from Browser JupyterLab or VS Code attached to the JupyterHub
container at `/home/jovyan/work/ml-eng-lab` rather than relying on a host-local notebook's
remote-kernel CWD.

```bash
git submodule update --init --recursive  # one-time after clone or a pin update
make atlas-setup

# In a separate terminal, only if the host-native daemon is not already running:
ollama serve

# Return to this terminal after the daemon is ready.
make atlas-up
make atlas-connect
```

`make atlas-connect` prints a token-bearing local URL only to an interactive
terminal. In VS Code: open a local notebook, run **Jupyter: Specify Jupyter Server for
Connections**, choose **Existing Jupyter Server**, paste that URL, then choose the remote kernel.
Treat the URL as a password; do not commit or paste it into documentation, and reconnect after
Atlas or JupyterHub restarts instead of relying on a saved URL.

Atlas is intentionally configured with `LLM_PROVIDER_SOURCE=ollama-localhost`: use the native
host Ollama daemon, never an Ollama Docker container. ComfyUI is off for this `ml-eng` consumer
until a task has an approved need for a host-native source. See
[docs/jupyterhub-integration.md](docs/jupyterhub-integration.md) and
[docs/vscode-remote-access.md](docs/vscode-remote-access.md) for the lifecycle, persistence,
and fallback details.

### 3.2. Local Docker

```bash
docker build -t ml-eng-lab .
docker run -p 8888:8888 -v "$(pwd):/home/jovyan/work" --shm-size=4g ml-eng-lab
```

`--shm-size=4g` is the minimum for the GNN notebooks; see [docs/env-setup.md](docs/env-setup.md) §2 for more.

### 3.3. Local venv

```bash
python -m venv .venv && source .venv/bin/activate
make install-torch-stack
pip install -r requirements.txt   # pulls thekaveh-nnx[lm]==0.2.0 from PyPI
make nlp-assets  # one-time spaCy + NLTK assets used by the 2 NLP Tier-A notebooks
jupyter lab
```

See [docs/env-setup.md](docs/env-setup.md) for environment details.

### 3.4. GitHub Codespaces (zero-click cloud dev)

Click **Code → Codespaces → Create codespace on main** on [github.com/thekaveh/ml-eng-lab](https://github.com/thekaveh/ml-eng-lab). After ~2-3 minutes of one-time dep install you have a browser-based VS Code (or JupyterLab — see below) with the 21 active task folders available and 28 of 29 active notebooks runnable under the pinned environment.

**Why this path was added.** The §3.1 / §3.2 / §3.3 paths each require local services or
dependency setup. Codespaces avoids that setup: the `.devcontainer/devcontainer.json`
declaratively bakes the install recipe (so the dep set is synchronized to `requirements.txt`,
`torch-core-requirements.txt`, and `torch-requirements.txt` during Codespace creation via
`postCreateCommand`), and the repo is auto-cloned into `/workspaces/ml-eng-lab` inside the
container.

**Scenarios this supports**:
- Onboarding a new contributor — they click "Create codespace" and have a working env in ~2-3 minutes, no local install at all.
- Running a notebook on a larger host without local install (the smallest Codespace machine is 2-core / 8 GB RAM — comparable to a low-end laptop, sufficient for every Tier-A notebook; bump to 4-core / 16 GB if any Tier-B sweep feels slow).
- Short exploratory run without polluting the local Python env.
- The `notebooks/image_classification-mnist-ffnn-numpy/notebook.ipynb` edge case (it imports sibling `.py` modules from its own folder) works natively because Codespaces clones the repo into `/workspaces/ml-eng-lab`.

**Scenarios this does NOT support**:
- GPU workloads — GitHub deprecated GPU Codespaces 2025-08-29 (Azure NCv3 retirement). The few GPU-benefiting notebooks (heaviest is `self_supervised-fmnist-jepa-pytorch`) still run on CPU here, just slowly; for real GPU you want a separate path (Modal `function.spawn`, a self-hosted GPU box behind Jupyter Enterprise Gateway, or Vertex AI Workbench / Colab Enterprise).
- Data persistence across Codespace deletions — anything written to `./data/` or `./runs/` is gone when the Codespace is deleted (Codespaces are intended to be cheap and disposable). Commit any results you want to keep, or use Codespaces' "prebuild" feature if dep install time becomes a bottleneck.
- The quantization-mnist-ffnn-pytorch notebook remains manual-only: the local/CI Torch 2.4.1 stack cannot import the required torchao path, while the Atlas package surface has not yet received a full notebook smoke. See [docs/dependency-contracts.md](docs/dependency-contracts.md).

**How to use**:

1. On [github.com/thekaveh/ml-eng-lab](https://github.com/thekaveh/ml-eng-lab) → green **Code** button → **Codespaces** tab → **Create codespace on main**.
2. Wait ~2-3 min for `postCreateCommand` to run `make codespace-setup` (= Torch-first dependency install + `make nlp-assets`). Progress is visible in the terminal panel.
3. Open any notebook. You can either:
   - **Stay in VS Code (browser)** — the Jupyter / Python extensions are preinstalled per the devcontainer config; works for the 28 tier-covered active notebooks. The quantization notebook is manual-only under `torch>=2.5`.
   - **Switch to JupyterLab** — click the dropdown next to "Open" on github.com → choose JupyterLab. To make JupyterLab the single-click default for all your codespaces, go to [github.com/settings/codespaces → Editor preference → JupyterLab](https://github.com/settings/codespaces).

See [`.devcontainer/devcontainer.json`](.devcontainer/devcontainer.json) for the exact image + extension set, and [`Makefile`](Makefile) `codespace-setup` target for the Codespaces/venv install recipe. The §3.2 Docker path bakes the same Torch-first dependency order into [`Dockerfile`](Dockerfile). GitHub currently includes 120 Codespaces compute hours per month for Free accounts and 180 for Pro; on the default two-core machine, that corresponds to 60 or 90 machine-hours.

## 4. Tasks

### 4.1. Active

| Folder | Task | Dataset | Model | Framework |
|---|---|---|---|---|
| [notebooks/image_classification-mnist-ffnn-numpy/](notebooks/image_classification-mnist-ffnn-numpy/) | Image classification | MNIST | Feed-forward NN (from scratch) | NumPy |
| [notebooks/image_classification-mnist-ffnn-pytorch/](notebooks/image_classification-mnist-ffnn-pytorch/) | Image classification | MNIST | Feed-forward NN | PyTorch (via nnx) |
| [notebooks/node_classification-reddit-gnn-pyg/](notebooks/node_classification-reddit-gnn-pyg/) | Node classification | Reddit2 | GNN (GraphConv, GraphSAGE, GAT) | PyTorch Geometric (via nnx) |
| [notebooks/tabular_classification-iris-mlp-pytorch/](notebooks/tabular_classification-iris-mlp-pytorch/) | Tabular classification | Iris | Feed-forward NN | PyTorch (via nnx) |
| [notebooks/model_surgery-mnist-ffnn-pytorch/](notebooks/model_surgery-mnist-ffnn-pytorch/) | Model surgery (Net2Net) | MNIST | Feed-forward NN | PyTorch (via nnx) |
| [notebooks/quantization-mnist-ffnn-pytorch/](notebooks/quantization-mnist-ffnn-pytorch/) | Quantization (PTQ + QAT) | MNIST | Feed-forward NN | PyTorch (via nnx) + torchao |
| [notebooks/pruning-mnist-ffnn-pytorch/](notebooks/pruning-mnist-ffnn-pytorch/) | Pruning (magnitude sparsity sweep) | MNIST | Feed-forward NN | PyTorch (via nnx) |
| [notebooks/knowledge_distillation-mnist-ffnn-pytorch/](notebooks/knowledge_distillation-mnist-ffnn-pytorch/) | Knowledge distillation (born-again) | MNIST | Feed-forward NN | PyTorch (via nnx) |
| [notebooks/text_generation-tinyshakespeare-transformer-pytorch/](notebooks/text_generation-tinyshakespeare-transformer-pytorch/) | Text generation (autoregressive LM) | TinyShakespeare (embedded) | Decoder-only transformer | PyTorch (via nnx) |
| [notebooks/peft-mnist-to-fmnist-dora-vs-lora-pytorch/](notebooks/peft-mnist-to-fmnist-dora-vs-lora-pytorch/) | PEFT cross-task adaptation (LoRA vs DoRA) | MNIST → Fashion-MNIST | Feed-forward NN + LoRA / DoRA adapters | PyTorch (via nnx) |
| [notebooks/dim_reduction-iris-autoencoder-pytorch/](notebooks/dim_reduction-iris-autoencoder-pytorch/) | Dimensionality reduction (PCA vs autoencoder) | Iris | Autoencoder (FFN with input_dim==output_dim) | PyTorch (via nnx) + sklearn |
| [notebooks/tabular_regression-diabetes-mlp-pytorch/](notebooks/tabular_regression-diabetes-mlp-pytorch/) | Tabular regression | Diabetes | Feed-forward MLP + sklearn baselines | PyTorch (via nnx) + sklearn |
| [notebooks/diffusion-mnist-ddpm-pytorch/](notebooks/diffusion-mnist-ddpm-pytorch/) | Generative (DDPM diffusion) | MNIST | DiffusionMLP denoiser (no U-Net) | PyTorch (via nnx) |
| [notebooks/moe-fmnist-mixture-of-experts-pytorch/](notebooks/moe-fmnist-mixture-of-experts-pytorch/) | Mixture-of-Experts classification | Fashion-MNIST | FeedFwdNN + MoELinear (4 experts, top-2 routing) | PyTorch (via nnx) |
| [notebooks/clustering-iris-kmeans-vs-ae-pytorch/](notebooks/clustering-iris-kmeans-vs-ae-pytorch/) | Unsupervised clustering | Iris | KMeans on raw features vs on AE latent | PyTorch (via nnx) + sklearn |
| [notebooks/link_prediction-karate-graphsage-pyg/](notebooks/link_prediction-karate-graphsage-pyg/) | Link prediction (GNN encoder) | Zachary Karate Club | GraphSAGE + dot-product scorer | PyTorch Geometric |
| [notebooks/community_detection-karate-louvain-vs-gnn-pyg/](notebooks/community_detection-karate-louvain-vs-gnn-pyg/) | Community detection (classical vs GNN) | Zachary Karate Club | Louvain vs GraphSAGE+KMeans | PyTorch Geometric + python-louvain |
| [notebooks/text_classification-agnews-spacy-mlp-pytorch/](notebooks/text_classification-agnews-spacy-mlp-pytorch/) | Text classification (4-topic) | Embedded AG-News-style corpus | spaCy + bag-of-words + MLP | PyTorch (via nnx) + spaCy + sklearn |
| [notebooks/sentiment_classification-vader-mlp-pytorch/](notebooks/sentiment_classification-vader-mlp-pytorch/) | Sentiment classification (rule vs neural) | Embedded review corpus | VADER (lexicon) vs MLP | PyTorch (via nnx) + nltk + spaCy + sklearn |
| [notebooks/preference_alignment-toy-dpo-pytorch/](notebooks/preference_alignment-toy-dpo-pytorch/) | Preference alignment (DPO) | Embedded 16-triplet preference corpus | Tiny TransformerNN (ref + policy) | PyTorch (via nnx) |
| [notebooks/self_supervised-fmnist-jepa-pytorch/](notebooks/self_supervised-fmnist-jepa-pytorch/) | Self-supervised (I-JEPA) + linear probe | Fashion-MNIST | ViT + EMA target + JEPA predictor | PyTorch (via nnx) |

> **Tip:** GitHub may show "Unable to render code block" on output cells with large matplotlib PNGs. [Browse this repo on nbviewer](https://nbviewer.org/github/thekaveh/ml-eng-lab/tree/main/) for full rendering of any notebook.

### 4.2. Archived

| Folder | Task | Dataset | Model | Framework |
|---|---|---|---|---|
| [notebooks/archive/codexglue_summarization/](notebooks/archive/codexglue_summarization/) | Code summarization (22 experiments) | CodeXGLUE | Transformers | HuggingFace |

### 4.3. Planned

See [§8 Roadmap](#8-roadmap).

## 5. Notebook re-execution policy

Notebooks are tiered by execution cost:

| Tier | What it is | Re-run policy |
|---|---|---|
| **A** | Cheap (<5 min) | `make run-tier-a` deliberately refreshes committed snapshots. CI uses non-mutating `make smoke-tier-a`, which writes fresh artifacts to `/tmp/ml-tier-a`. Tier-A notebooks also accept a `SMOKE_TEST` papermill parameter (default `0` = full run). |
| **B** | Moderate (model-selection sweeps) | Original outputs preserved. `make smoke-tier-b` runs `SMOKE_TEST=1` and writes to `/tmp/`: the parameterized `image_classification-mnist-ffnn-pytorch` notebook shrinks its sweep, and the 4 phase2 reddit notebooks run smoke-truncated epochs/subsets (notebook4 also reduces fanout). |
| **C** | Expensive (main GPU training) | Historical Aug-2023 GPU training-run outputs preserved as artifact. `make smoke-tier-c` runs CPU with `SMOKE_TEST=1` to validate the pipeline without overwriting outputs. |

Tier-A CI writes executed notebook copies under `/tmp/ml-tier-a`; Tier-B/C smoke targets write under `/tmp/ml-smoke`. Papermill intentionally runs each notebook from its own task directory so relative paths behave like an interactive run. Training and evaluation may therefore create ignored task-local `./data/` or `./runs/` artifacts even when source notebook outputs are preserved; committed output text such as `Run saved to ./runs/...` describes that notebook-local runtime location, not files guaranteed to exist in a clean checkout.

See [docs/env-setup.md](docs/env-setup.md) for the tier mapping.

## 6. NNx library

Throughout this README, `NNx` refers to the [GitHub project](https://github.com/thekaveh/NNx); the importable Python package is lowercase `nnx`; the PyPI distribution is [`thekaveh-nnx`](https://pypi.org/project/thekaveh-nnx/).

The library is consumed via PyPI — `thekaveh-nnx[lm]==0.2.0` is pinned in `requirements.txt` (since 2026-06-14, replacing the prior git-submodule editable install). The `[lm]` extra pulls the BPE tokenizer + datasets backbone for the two notebooks that call `train_bpe`/`NNTokenizerParams` (`notebooks/text_generation-tinyshakespeare-transformer-pytorch/notebook.ipynb` and `notebooks/preference_alignment-toy-dpo-pytorch/notebook.ipynb`); without it both `ImportError` (issue #12). Notebooks import via `from nnx.X import Y` exactly as before — only the distribution name and install mechanism changed.

To extend `nnx` for a new task:

1. Open a PR against [`thekaveh/NNx`](https://github.com/thekaveh/NNx) with the new feature + a smoke test.
2. After merge, wait for the next NNx release cut (or, for editable iteration during the design phase: clone `thekaveh/NNx` outside the ml-eng-lab tree and `pip install -e <path-to-clone>[lm]` into your venv).
3. Bump the pinned version in `requirements.txt` here (e.g. `thekaveh-nnx[lm]==0.2.1`); open a PR. Tier-A papermill CI re-runs the Tier-A list against the new version; run `make smoke-tier-b`, `make smoke-tier-c`, and manual quantization validation when the NNx change touches those surfaces — same validation discipline as the prior submodule-pointer-bump workflow.

## 7. Repository conventions

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow. Key points:

- Each active task is a self-contained directory under `notebooks/` using the `[task]-[dataset]-[model]-[framework]` naming convention. No `tasks/` subdirectory.
- Shared library code lives in `nnx` (the PyPI-installed `thekaveh-nnx` package), not a local `common/`.
- Notebooks are saved with executed cells (outputs included) for active tasks.
- Tier-C notebooks have their Aug-2023 outputs preserved; never re-execute them in place.
- `notebooks/archive/` is read-only.

## 8. Roadmap

The `tabular_classification-iris-mlp-pytorch` task added in 2026-05-28 seeds the `tabular_classification-titanic-xgboost-sklearn` roadmap entry below.

Future tasks planned (each will become a new `notebooks/<task>/` directory):

- [ ] `image_classification-cifar10-resnet-pytorch`
- [ ] `tabular_classification-titanic-xgboost-sklearn`
- [ ] `text_classification-imdb-distilbert-hf` — distinct from the shipped `notebooks/text_classification-agnews-spacy-mlp-pytorch/` (pre-transformer baseline); this entry is specifically the DistilBERT fine-tune / PEFT continuation.
- [ ] `link_prediction-citation-graphsage-pyg` — distinct from the shipped `notebooks/link_prediction-karate-graphsage-pyg/` (small-graph smoke); this entry is on a real citation network.
- [ ] `time_series_forecasting-electricity-tft-pytorch`
- [ ] `anomaly_detection-creditcard-autoencoder-pytorch`
- [ ] `recommendation-movielens-mf-pytorch`
- [ ] `generative-mnist-vae-pytorch` — distinct from the shipped `notebooks/diffusion-mnist-ddpm-pytorch/`; VAEs and diffusion are different generative families.
- [ ] `reinforcement_learning-cartpole-dqn-pytorch`
- [x] `diffusion-mnist-ddpm-pytorch` — shipped 2026-05-29 in PR #4.

Adding a new task: see the "Adding a new task folder" section in [CONTRIBUTING.md](CONTRIBUTING.md).

## 9. License

MIT. See [LICENSE](LICENSE).

## 10. Other documentation

The README is the entry point; the items below are the hub's index of secondary documentation.

### 10.1. Workflow + history

- [CONTRIBUTING.md](CONTRIBUTING.md) — workflow, conventions, "Adding a new task folder" recipe, verifier+pytest gates.
- [CHANGELOG.md](CHANGELOG.md) — Keep-a-Changelog release notes.

### 10.2. Environment + runtimes

- [docs/env-setup.md](docs/env-setup.md) — the four setup paths (Atlas / Docker / venv / Codespaces), GPU notes, Tier mapping.
- [docs/jupyterhub-integration.md](docs/jupyterhub-integration.md) — Atlas JupyterHub lifecycle and ownership boundary.
- [docs/vscode-remote-access.md](docs/vscode-remote-access.md) — local VS Code remote-kernel path and browser fallback.
- [docs/atlas-pin-bump-runbook.md](docs/atlas-pin-bump-runbook.md) — reviewed Atlas pin-bump and future-service admission runbook.
- [docs/dependency-contracts.md](docs/dependency-contracts.md) — dependency audit ledger, local/CI Torch contract, Atlas runtime evidence, and manual-only quantization contract.
- [docs/architecture.md](docs/architecture.md) — system/context view for the notebook lab, verifier, CI, runtime environments, and documentation delivery.
- [docs/diagrams/README.md](docs/diagrams/README.md) — provenance and regeneration contract for embedded architecture diagrams.
- [docs/maintenance/overnight-2026-07-04.md](docs/maintenance/overnight-2026-07-04.md) — current overnight maintenance pass log and issue tracker.
- [docs/maintenance/overnight-2026-07-02.md](docs/maintenance/overnight-2026-07-02.md) — historical overnight maintenance run that reached its hard cap.
- [docs/maintenance/notebooks-reorganization-design.md](docs/maintenance/notebooks-reorganization-design.md) — completed design record for the `notebooks/<task>/` layout and archive move.
- [docs/maintenance/notebooks-reorganization-implementation.md](docs/maintenance/notebooks-reorganization-implementation.md) — completed implementation record for the notebook/archive reorganization and runtime-path contract.

### 10.3. Issue sinks for external code

- [docs/FINDINGS-NNX.md](docs/FINDINGS-NNX.md) — issue log for the `thekaveh-nnx` library (append findings here; do not edit nnx directly via this repo — fixes land upstream at [`thekaveh/NNx`](https://github.com/thekaveh/NNx)).
- [docs/FINDINGS-ATLAS.md](docs/FINDINGS-ATLAS.md) — Atlas-consumer findings; fixes to Atlas itself belong upstream.

### 10.4. Archive

- [notebooks/archive/README.md](notebooks/archive/README.md) — preserved Aug-2023 codexglue summarization experiments (22 runs); read-only.
