# 4.3 Notebook infrastructure

Atlas tasks use a remote JupyterHub kernel from VS Code by default. Open the
repository in VS Code, connect to the Atlas JupyterHub server, and select the
remote kernel for the task. This keeps the compute environment remote while
the editor remains local.

Most tasks use `remote` workspace access and keep notebooks, checkpoints, and
other run artifacts on the Atlas Jupyter volume. The NumPy MNIST fallback is
the exception: it imports sibling Python modules and therefore requires a
mounted checkout; its task-local ignored paths hold its artifacts.

Every contract currently requires JupyterHub. Additional Atlas services stay
inactive until a task declares them in its contract and their package and
service validation has passed. Do not copy artifacts from Atlas volumes into
the repository unless a task explicitly documents that policy.

## 1. Active task contracts

<!-- atlas-task-contracts:start -->
| Task | Tier | Default mode | Workspace access | Required Atlas services | Artifact policy | Constraints |
| --- | --- | --- | --- | --- | --- | --- |
| tabular_classification-iris-mlp-pytorch | A | vscode-remote | remote | jupyterhub | atlas-jupyter-volume | — |
| tabular_regression-diabetes-mlp-pytorch | A | vscode-remote | remote | jupyterhub | atlas-jupyter-volume | — |
| image_classification-mnist-ffnn-numpy | A | vscode-remote | mounted-required | jupyterhub | task-local-ignored-paths | Sibling Python modules require the mounted checkout. |
| image_classification-mnist-ffnn-pytorch | B | vscode-remote | remote | jupyterhub | atlas-jupyter-volume | — |
| model_surgery-mnist-ffnn-pytorch | A | vscode-remote | remote | jupyterhub | atlas-jupyter-volume | — |
| knowledge_distillation-mnist-ffnn-pytorch | A | vscode-remote | remote | jupyterhub | atlas-jupyter-volume | — |
| pruning-mnist-ffnn-pytorch | A | vscode-remote | remote | jupyterhub | atlas-jupyter-volume | — |
| quantization-mnist-ffnn-pytorch | manual | vscode-remote | remote | jupyterhub | atlas-jupyter-volume | Manual-only until Atlas Jupyter package validation passes. |
| moe-fmnist-mixture-of-experts-pytorch | A | vscode-remote | remote | jupyterhub | atlas-jupyter-volume | — |
| diffusion-mnist-ddpm-pytorch | A | vscode-remote | remote | jupyterhub | atlas-jupyter-volume | — |
| self_supervised-fmnist-jepa-pytorch | A | vscode-remote | remote | jupyterhub | atlas-jupyter-volume | — |
| peft-mnist-to-fmnist-dora-vs-lora-pytorch | A | vscode-remote | remote | jupyterhub | atlas-jupyter-volume | — |
| node_classification-reddit-gnn-pyg | B/C | vscode-remote | remote | jupyterhub | atlas-jupyter-volume | — |
| link_prediction-karate-graphsage-pyg | A | vscode-remote | remote | jupyterhub | atlas-jupyter-volume | — |
| community_detection-karate-louvain-vs-gnn-pyg | A | vscode-remote | remote | jupyterhub | atlas-jupyter-volume | — |
| text_generation-tinyshakespeare-transformer-pytorch | A | vscode-remote | remote | jupyterhub | atlas-jupyter-volume | — |
| text_classification-agnews-spacy-mlp-pytorch | A | vscode-remote | remote | jupyterhub | atlas-jupyter-volume | — |
| sentiment_classification-vader-mlp-pytorch | A | vscode-remote | remote | jupyterhub | atlas-jupyter-volume | — |
| preference_alignment-toy-dpo-pytorch | A | vscode-remote | remote | jupyterhub | atlas-jupyter-volume | — |
| dim_reduction-iris-autoencoder-pytorch | A | vscode-remote | remote | jupyterhub | atlas-jupyter-volume | — |
| clustering-iris-kmeans-vs-ae-pytorch | A | vscode-remote | remote | jupyterhub | atlas-jupyter-volume | — |
<!-- atlas-task-contracts:end -->
