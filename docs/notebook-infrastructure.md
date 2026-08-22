# 4.4 Notebook infrastructure

Every notebook environment is installed from the target selected by
`requirements/lock-policy.toml`: Darwin arm64, Linux x86_64, or Linux aarch64. The committed
hash-required root lock includes notebook tooling and the exact spaCy model wheel; `make nlp-assets`
adds only the official size- and SHA-256-verified VADER ZIP, and `make verify-nlp-assets` verifies
the installed asset offline before a workload. These locks make a run reproducible for the
qualified platform lock. Issue #64 owns the completed VADER integrity contract, the completed
retained Atlas pin defines the remote runtime boundary, and Issue #66 restores the complete
quantization notebook to Tier B with explicit checkpoint and output semantics.

Atlas tasks use a remote JupyterHub kernel from VS Code by default. Open the
repository in VS Code, connect to the Atlas JupyterHub server, and select the
remote kernel for the task. This keeps the compute environment remote while
the editor remains local. The runtime is the pinned `infra/` Atlas submodule
on the `ml-eng` track, launched through `make atlas-up`; `make atlas-connect`
is the sole source of the token-bearing VS Code URL. The consumer requires
host-native Ollama and does not allow a containerized Ollama or ComfyUI source.

Most tasks use `remote` workspace access and keep notebooks, checkpoints, and
other run artifacts on the Atlas Jupyter volume. The NumPy MNIST fallback is
the exception: it imports sibling Python modules and therefore requires a
mounted checkout. Its `default_mode` is `mounted-workspace`; run it from
Browser JupyterLab or VS Code attached to the JupyterHub container at
`/home/jovyan/work/ml-eng-lab`; its task-local ignored paths hold its artifacts.

Every contract explicitly declares `required_env`. Each entry has exactly a
`name` and `service`: the uppercase environment-variable name and the required
service that injects it. The current JupyterHub-only tasks use
`required_env: []`. A future non-JupyterHub service is invalid without at least
one binding, and a binding cannot reference a service absent from
`required_services`. The contract must not contain environment values,
endpoints, credentials, tokens, or host paths; those remain runtime-owned and
the matrix renders names only.

Atlas track defaults are not notebook authorization: availability does not
authorize notebook use. A task must declare the service and its environment bindings, but an injected variable alone does
not prove that the service is enabled or healthy. Admission also requires a
central consumer source, successful doctor validation, and a targeted Atlas
JupyterHub smoke. Do not copy artifacts from Atlas volumes into the repository
unless a task explicitly documents that policy. The full admission sequence is
[atlas-pin-bump-runbook.md](atlas-pin-bump-runbook.md).

## 4.4.1 Active task contracts

<!-- atlas-task-contracts:start -->
| Task | Tier | Default mode | Workspace access | Required Atlas services | Required environment | Artifact policy | Constraints |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tabular_classification-iris-mlp-pytorch | A | vscode-remote | remote | jupyterhub | — | atlas-jupyter-volume | — |
| tabular_regression-diabetes-mlp-pytorch | A | vscode-remote | remote | jupyterhub | — | atlas-jupyter-volume | — |
| image_classification-mnist-ffnn-numpy | A | mounted-workspace | mounted-required | jupyterhub | — | task-local-ignored-paths | Browser JupyterLab or VS Code attached to the JupyterHub container is required from /home/jovyan/work/ml-eng-lab because sibling Python modules need the mounted checkout. |
| image_classification-mnist-ffnn-pytorch | B | vscode-remote | remote | jupyterhub | — | atlas-jupyter-volume | — |
| model_surgery-mnist-ffnn-pytorch | A | vscode-remote | remote | jupyterhub | — | atlas-jupyter-volume | — |
| knowledge_distillation-mnist-ffnn-pytorch | A | vscode-remote | remote | jupyterhub | — | atlas-jupyter-volume | — |
| pruning-mnist-ffnn-pytorch | A | vscode-remote | remote | jupyterhub | — | atlas-jupyter-volume | — |
| quantization-mnist-ffnn-pytorch | B | vscode-remote | remote | jupyterhub | — | atlas-jupyter-volume | Tier-B smoke is deterministic and bounded to one epoch for FP32 plus one epoch for QAT.<br>QAT acceptance reconstructs the saved FP-shadow checkpoint with exact state/metadata parity and separately proves final torchao conversion. |
| moe-fmnist-mixture-of-experts-pytorch | A | vscode-remote | remote | jupyterhub | — | atlas-jupyter-volume | — |
| diffusion-mnist-ddpm-pytorch | A | vscode-remote | remote | jupyterhub | — | atlas-jupyter-volume | — |
| self_supervised-fmnist-jepa-pytorch | A | vscode-remote | remote | jupyterhub | — | atlas-jupyter-volume | — |
| peft-mnist-to-fmnist-dora-vs-lora-pytorch | A | vscode-remote | remote | jupyterhub | — | atlas-jupyter-volume | — |
| node_classification-reddit-gnn-pyg | B/C | vscode-remote | remote | jupyterhub | — | atlas-jupyter-volume | Issue #62 requires preferred pyg-lib sampling and forced torch-sparse fallback on the repository Torch 2.11 CPU stack; the retained Atlas Torch 2.13 runtime executes through pyg-lib and intentionally has no legacy torch-sparse wheel. |
| link_prediction-karate-graphsage-pyg | A | vscode-remote | remote | jupyterhub | — | atlas-jupyter-volume | — |
| community_detection-karate-louvain-vs-gnn-pyg | A | vscode-remote | remote | jupyterhub | — | atlas-jupyter-volume | — |
| text_generation-tinyshakespeare-transformer-pytorch | A | vscode-remote | remote | jupyterhub | — | atlas-jupyter-volume | — |
| text_classification-agnews-spacy-mlp-pytorch | A | vscode-remote | remote | jupyterhub | — | atlas-jupyter-volume | — |
| sentiment_classification-vader-mlp-pytorch | A | vscode-remote | remote | jupyterhub | — | atlas-jupyter-volume | — |
| preference_alignment-toy-dpo-pytorch | A | vscode-remote | remote | jupyterhub | — | atlas-jupyter-volume | — |
| dim_reduction-iris-autoencoder-pytorch | A | vscode-remote | remote | jupyterhub | — | atlas-jupyter-volume | — |
| clustering-iris-kmeans-vs-ae-pytorch | A | vscode-remote | remote | jupyterhub | — | atlas-jupyter-volume | — |
<!-- atlas-task-contracts:end -->

## 4.4.2 Execution freshness stamping

Each `run-tier-a`, `smoke-tier-a`, `smoke-tier-b`, and `smoke-tier-c` Papermill execution invokes
`scripts/stamp_notebook_source_hashes.py` only after success. Failed, partial, or error-output
notebooks are not stamped; stamper failure fails the Make target. The normal targets therefore
keep execution and source-freshness evidence together without rewriting source notebooks for
smoke runs. The exact `metadata.source_hash` algorithm, E8 scope, and repair command are
canonicalized in [the conventions](conventions.md#524-notebook-output-freshness).
