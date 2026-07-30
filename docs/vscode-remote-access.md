# VS Code remote access to Atlas JupyterHub

The default notebook experience is deliberately split: VS Code and `.ipynb` files stay local; the
kernel runs in the active Atlas JupyterHub service. That preserves familiar local editing while
making the Atlas dependency contract explicit.

## 1. Default mode: connect a local notebook to the remote kernel

1. Start the runtime from the repository root:

   ```bash
   make atlas-up
   make atlas-connect
   ```

2. Open a local notebook in VS Code.
3. Run **Jupyter: Specify Jupyter Server for Connections** from the Command Palette.
4. Choose **Existing Jupyter Server**, then paste only the URL printed by `make atlas-connect`.
5. Use **Select Kernel** to choose the remote Atlas kernel and run a small cell before beginning
   the notebook.

The connection helper refuses non-interactive output because its URL contains a token. Treat that
URL like a password: do not save it in settings, commit it, attach it to an issue, or paste it into
an untrusted application. `BASE_PORT=auto` means the visible port is intentionally not a stable
contract; the helper is the source of truth for each running instance.

## 2. Workspace and artifact semantics

The local editor owns the notebook file. The kernel sees the repository through the Atlas mount at
`/home/jovyan/work/ml-eng-lab`. This lets the NumPy MNIST task import its sibling modules and lets
mounted-required tasks write ignored task-local artifacts. Most task contracts use a remote
workspace and place runtime artifacts on the Atlas Jupyter volume instead. The per-task policy is
documented in [notebook-infrastructure.md](notebook-infrastructure.md).

Select the remote kernel after opening the local file, rather than opening the same path twice in
both host and browser clients. It prevents accidental disagreement about which copy owns notebook
metadata and outputs.

## 3. Fallback modes

### 3.1. Browser JupyterLab

Use the token URL from `make atlas-connect` in a browser for a quick investigation or notebook
session. Navigate to `/home/jovyan/work/ml-eng-lab` when you need the mounted checkout. Browser
mode is a fallback; local VS Code remains the primary authoring surface.

### 3.2. Attach VS Code to the running JupyterHub container

Use VS Code's Dev Containers support when a task needs an integrated shell in the JupyterHub
container. Open `/home/jovyan/work/ml-eng-lab` after attaching. This uses the same JupyterHub
service and does not change the Atlas consumer or source policy.

## 4. Troubleshooting

- **No remote server option:** ensure `make atlas-up` completed and use `make atlas-connect` in an
  interactive terminal.
- **Kernel cannot import a sibling module:** confirm the kernel is Atlas JupyterHub and its working
  checkout is `/home/jovyan/work/ml-eng-lab`; the NumPy MNIST task requires that mount.
- **Token rejected after a restart:** run `make atlas-connect` again. Connection URLs are
  short-lived and should not be reused from notes or editor history.
- **Ollama startup error:** Atlas requires the host-native daemon. Start or repair it locally and
  retry; never substitute an Ollama container.
- **Need a Jupyter shell:** use the attached-container fallback rather than changing the default
  workflow or patching `infra/`.
