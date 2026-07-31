# 4.2 JupyterHub integration

Atlas supplies the notebook infrastructure for this repository. It is consumed as the pinned
`infra/` submodule, not as a vendored application tree. The active consumer uses the `ml-eng`
track and a parent-owned compose overlay to mount this checkout into JupyterHub.

## 4.2.1 Default path: local VS Code, remote Atlas kernel

The notebook file remains open on the host in VS Code; computation runs in the Atlas JupyterHub
kernel. This is the primary path for tasks with `workspace_access: remote`. The NumPy MNIST task
uses `default_mode: mounted-workspace` with `workspace_access: mounted-required`: use Browser
JupyterLab or VS Code attached to the JupyterHub container from `/home/jovyan/work/ml-eng-lab`.
A local notebook connected to a remote kernel does not guarantee that mounted working directory,
which its sibling Python modules and task-local `data/` and `runs/` paths require.

```bash
git submodule update --init --recursive
make atlas-setup
make atlas-up
make atlas-connect
```

Use the connection URL that the last command prints in the VS Code Jupyter server selector. The
URL contains a credential. It must stay out of the repository, tickets, and docs. Reconnect after
Atlas or JupyterHub restarts rather than assuming that a saved token is still valid.
See [vscode-remote-access.md](vscode-remote-access.md) for the exact editor flow.

## 4.2.2 Consumer contract and ownership

`atlas.consumer.yml` is the committed source-policy contract. The lifecycle wrapper
`scripts/atlas-up.sh` supplies `--track ml-eng`; the manifest deliberately has no `track` key:

- It sets `JUPYTERHUB_SOURCE=container`.
- It delegates port selection to Atlas with `BASE_PORT=auto`.
- It requires `LLM_PROVIDER_SOURCE=ollama-localhost`.
- `compose/ml-eng-lab-atlas.yml` is the only parent-owned compose overlay and mounts
  `ML_ENG_LAB_REPO_PATH` at `/home/jovyan/work/ml-eng-lab`.
- `atlas.env.user` is ignored and contains the absolute checkout path, plus an optional native
  Ollama port override.

Do not patch `infra/`, generated Atlas configuration, or Atlas service definitions from this
repository. Consumer behavior belongs in the files above; changes to Atlas itself belong in the
upstream project. The exact gitlink and bump procedure are recorded in
[atlas-pin-bump-runbook.md](atlas-pin-bump-runbook.md).

## 4.2.3 Native AI-service policy

Before `make atlas-up`, the host-native Ollama daemon must answer on loopback. The lifecycle
wrapper checks `http://127.0.0.1:<port>/api/version` and refuses a different source.

```bash
# Start only when the native daemon is not already managed by the host.
ollama serve
ollama list
make atlas-up
```

Never launch an Ollama Docker container for this Atlas consumer: it is slow and memory-heavy on
the target workstation. The wrapper clears ambient source variables and accepts only the committed
native source. ComfyUI is not enabled by the `ml-eng` configuration. If a future task genuinely
needs it, it must first have an approved consumer specification, an explicit host-native source
(`localhost` or managed MPS), an in-network configuration, a targeted runtime smoke, and matching
documentation. Automatic and containerized ComfyUI sources are rejected.

## 4.2.4 Artifact and workspace behavior

The notebook-contract table in [notebook-infrastructure.md](notebook-infrastructure.md) is
authoritative per task. The normal remote workflow stores runtime artifacts on the Atlas Jupyter
volume; task source remains local and version controlled. The NumPy MNIST task is explicitly
`mounted-workspace` / `mounted-required`, so use Browser JupyterLab or VS Code attached to the
JupyterHub container from the mounted checkout; its ignored artifacts are written through that
checkout mount instead.

Do not copy volume artifacts into the repository without a task-level policy. A task that needs a
new Atlas service must declare that service in its contract before enabling it; it must not infer
availability from other services that happen to be in the `ml-eng` track.

## 4.2.5 Browser and container-attached workspace mode

Browser JupyterLab and VS Code's container-attach mode are alternatives for normal
remote-workspace tasks. They implement the required `mounted-workspace` mode for NumPy MNIST:
open `/home/jovyan/work/ml-eng-lab` before running it. They use the same JupyterHub service and
mount; they do not authorize changing the track, modifying `infra/`, or running containerized
Ollama.

## 4.2.6 Lifecycle troubleshooting

- **Submodule missing:** run `git submodule update --init --recursive` from the repository root.
- **Native Ollama check failed:** start or repair the host-native daemon (`ollama serve` is one
  option), then re-run `make atlas-up`. Do not substitute a Dockerized daemon.
- **Connection URL missing:** Atlas must be running, and `make atlas-connect` must be invoked in
  an interactive terminal so a token is not written to automation logs.
- **Wrong notebook paths:** for normal tasks, use the remote kernel after opening the local
  repository in VS Code. For NumPy MNIST, use Browser JupyterLab or VS Code attached to the
  JupyterHub container and open `/home/jovyan/work/ml-eng-lab`; selecting a remote kernel for the
  local file is not a substitute for its mounted-workspace requirement.
- **Need a clean service reset:** use `make atlas-down` first. `COLD=1 make atlas-down` destroys
  persisted volumes and is deliberately not the normal reset command.
