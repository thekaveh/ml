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

This default applies to tasks with `workspace_access: remote`. The NumPy MNIST task instead uses
`default_mode: mounted-workspace` with `workspace_access: mounted-required`: use Browser JupyterLab
or VS Code attached to the JupyterHub container from `/home/jovyan/work/ml-eng-lab`, not a
host-local notebook paired with the remote kernel.

## 2. Workspace and artifact semantics

The local editor owns the notebook file. The Atlas container has the repository mounted at
`/home/jovyan/work/ml-eng-lab`, but a host-local notebook paired with a remote kernel does not
guarantee that working directory. Most task contracts use a remote workspace and place runtime
artifacts on the Atlas Jupyter volume. The NumPy MNIST task's `mounted-workspace` mode needs its
mounted checkout for sibling modules and ignored task-local artifacts, so run it through Browser
JupyterLab or VS Code attached to the JupyterHub container. The per-task policy is documented in
[notebook-infrastructure.md](notebook-infrastructure.md).

Select the remote kernel after opening the local file, rather than opening the same path twice in
both host and browser clients. It prevents accidental disagreement about which copy owns notebook
metadata and outputs.

## 3. Fallback modes

### 3.1. Browser JupyterLab

Use the token URL from `make atlas-connect` in a browser for a quick investigation or notebook
session. Navigate to `/home/jovyan/work/ml-eng-lab` when you need the mounted checkout. Browser
mode implements the NumPy MNIST `mounted-workspace` default unless using the attached-container
alternative; local VS Code remains the primary authoring surface for ordinary remote-workspace tasks.

### 3.2. Attach VS Code to the running JupyterHub container

Use VS Code's Dev Containers support when a task needs an integrated shell in the JupyterHub
container. Open `/home/jovyan/work/ml-eng-lab` after attaching. This is the other implementation
of the NumPy MNIST `mounted-workspace` default. It uses the same JupyterHub service and does not
change the Atlas consumer or source policy.

## 4. Troubleshooting

- **No remote server option:** ensure `make atlas-up` completed and use `make atlas-connect` in an
  interactive terminal.
- **Kernel cannot import a sibling module:** the NumPy MNIST task must use Browser JupyterLab or
  VS Code attached to the JupyterHub container with `/home/jovyan/work/ml-eng-lab` open. Do not
  rely on a local notebook plus remote kernel to establish that mounted workspace.
- **Token rejected after a restart:** run `make atlas-connect` again. Connection URLs are
  short-lived and should not be reused from notes or editor history.
- **Ollama startup error:** Atlas requires the host-native daemon. Start or repair it locally and
  retry; never substitute an Ollama container.
- **Need a Jupyter shell:** use the attached-container fallback rather than changing the default
  workflow or patching `infra/`.
