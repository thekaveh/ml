# Environment setup

Four paths are supported. Atlas is the default because it gives local VS Code a reproducible,
running JupyterHub kernel while retaining the checked-out notebooks on the host.

## 1. Atlas JupyterHub + local VS Code (recommended)

Atlas is the direct successor to the previous infrastructure seam. This repository
pins Atlas as `infra/` at `61c7c5103660e2226bf107c115dae42bf46f8374`; the lifecycle wrapper
`scripts/atlas-up.sh` supplies `--track ml-eng`, while `atlas.consumer.yml` declares
`BASE_PORT=auto`, a container JupyterHub, and native host Ollama. The parent-owned overlay
bind-mounts this checkout at `/home/jovyan/work/ml-eng-lab`. That mounted path is available to
Browser JupyterLab or an attached JupyterHub container; it is not assumed by every host-local
notebook paired with a remote kernel.

### 1.1. Start the runtime

```bash
git submodule update --init --recursive
make atlas-setup

# In a separate terminal, only if the host daemon is not already running:
ollama serve

# Return here after the daemon is ready.
make atlas-up
make atlas-connect
```

`make atlas-setup` creates ignored, machine-local environment files without replacing existing
ones. `make atlas-up` validates the consumer contract and starts Atlas in detached mode. It refuses
to proceed unless native Ollama answers on the loopback address. `make atlas-connect` prints a
short-lived token URL only to an interactive terminal; follow its VS Code instructions rather than
recording the URL in a shell history, issue, or document.

### 1.2. Source and persistence policy

- **Ollama:** `LLM_PROVIDER_SOURCE=ollama-localhost` is mandatory. Use the host-native Ollama
  daemon; never run an Ollama Docker container for this consumer.
- **ComfyUI:** the `ml-eng` track leaves it disabled. A future approved task can request only a
  host-native source (`localhost` or managed Apple-silicon MPS); automatic and container sources
  are prohibited.
- **Ports:** `BASE_PORT=auto` avoids collisions with sibling Atlas consumers. Do not hard-code a
  published port in a notebook or docs; use `make atlas-connect` for the current endpoint.
- **Artifacts:** the default task contract keeps remote-workspace artifacts on the Atlas Jupyter
  volume. The NumPy MNIST task is explicitly `mounted-workspace` / `mounted-required`: use Browser
  JupyterLab or VS Code attached to the JupyterHub container from `/home/jovyan/work/ml-eng-lab`,
  so its ignored `data/` and `runs/` directories remain in this checkout.

### 1.3. Stop and clean up

```bash
make atlas-down          # normal shutdown; preserves Atlas volumes
COLD=1 make atlas-down   # deliberate destructive teardown of Atlas volumes
```

The normal stop is the expected development command. Use the cold form only when intentionally
discarding persisted Atlas data. Never edit files inside `infra/`; consumer configuration belongs
in the parent repository. See [atlas-pin-bump-runbook.md](atlas-pin-bump-runbook.md) for pin
updates and future service admission.

## 2. Local Docker

```bash
docker build -t ml-eng-lab .
docker run -p 8888:8888 -v "$(pwd):/home/jovyan/work" --shm-size=4g ml-eng-lab
```

Open the token URL printed at startup. This is a self-contained notebook image, not the Atlas
runtime; use it when you specifically need an isolated local image. `--shm-size=4g` is the minimum
for the GNN notebooks; serious GNN training may need 16–50 GiB.

## 3. Local Python venv

```bash
python -m venv .venv && source .venv/bin/activate
make install-torch-stack
pip install -r requirements.txt
make nlp-assets
jupyter lab
```

The local/CI package contract remains intentionally separate from Atlas. In particular,
`torch-core-requirements.txt` pins Torch 2.4.1 and the corresponding PyG wheels; see
[dependency-contracts.md](dependency-contracts.md) before changing it. `make nlp-assets` obtains
the spaCy `en_core_web_sm` model and NLTK `vader_lexicon` required by two Tier-A notebooks.

## 4. GitHub Codespaces

Create a codespace on `main` from GitHub. `.devcontainer/devcontainer.json` runs
`make codespace-setup`, which installs the local/CI dependency contract and NLP assets in the
auto-cloned `/workspaces/ml-eng-lab` checkout. Use browser VS Code or JupyterLab as appropriate.

Codespaces is CPU-only and disposable: `data/` and `runs/` are lost when a codespace is deleted.
The quantization notebook is still manual-only because the Codespaces/local package contract uses
Torch 2.4.1; Atlas has a newer observed package surface but has not yet received a full
quantization-notebook smoke.

## 5. GPU notes

The current checked-in local Docker and Codespaces paths are CPU-oriented. For a new GPU run, use
an environment with a matching CUDA Torch/PyG stack and record the compatibility result. Do not
silently repoint the Atlas consumer to a containerized Ollama or ComfyUI service to obtain GPU
access; service admission is a separate reviewed change.

## 6. Tier mapping

The authoritative lists are `Makefile` (`TIER_A`, `TIER_B`, and `TIER_C`) and
`scripts/verify_repo_config.yaml` (`tier_a_notebooks`).

- **Tier A:** `make run-tier-a` executes the cheap notebooks in place; CI checks the resulting
  tracked outputs.
- **Tier B:** `make smoke-tier-b` writes parameterized smoke outputs to `/tmp/ml-smoke`.
- **Tier C:** `make smoke-tier-c` runs the expensive Reddit training pipelines in smoke mode;
  preserved code cells remain guarded by the baseline verifier.
- **Manual-only:** `notebooks/quantization-mnist-ffnn-pytorch/notebook.ipynb` stays out of the
  automated tiers. The local/CI 2.4.1 stack cannot import its torchao path; package presence in
  Atlas is not sufficient evidence to reclassify it.
