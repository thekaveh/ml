# Atlas Infrastructure Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the legacy `vendor/genai-vanilla` JupyterHub consumption seam with a pinned Atlas `infra/` submodule, make each active notebook's Atlas runtime contract machine-readable, and document local VS Code-to-Atlas-JupyterLab as the primary notebook workflow.

**Architecture:** `ml-eng-lab` owns a root `atlas.consumer.yml`, a narrow JupyterHub workspace-mount overlay, local lifecycle wrappers, notebook runtime metadata, and its documentation. Atlas owns the immutable `infra/` implementation and local runtime state. The default workflow keeps notebooks in the local checkout and connects VS Code to a running Atlas Jupyter server; a writable mounted checkout at `/home/jovyan/work/ml-eng-lab` is the explicit fallback for sibling-module imports and host-visible artifacts.

**Tech Stack:** Git submodule; Atlas `61c7c5103660e2226bf107c115dae42bf46f8374`; Docker Compose through Atlas; Bash with ShellCheck; Python 3.11; PyYAML; pytest; MkDocs/site/wiki documentation pipeline; GitHub Actions.

## Global constraints

- Pin `infra/` to the detached Atlas SHA `61c7c5103660e2226bf107c115dae42bf46f8374`. `.gitmodules` contains only the Atlas URL and `infra` path—never a moving `branch` entry.
- Treat `infra/` as read-only implementation owned by Atlas. Do not add consumer symlinks under it, alter its compose files, patch its Dockerfiles, or write parent-specific scripts inside it.
- The consumer manifest is the committed source of infrastructure intent. It uses `profile: dev`, `BASE_PORT: auto`, `JUPYTERHUB_SOURCE: container`, one parent-owned compose overlay, and no `track:` key. Only the start wrapper selects `--track ml-eng`.
- Pin `LLM_PROVIDER_SOURCE: ollama-localhost` in the committed manifest. Do not use `auto` because it can resolve to `ollama-container-cpu`; this consumer must never launch containerized Ollama. A normal `atlas-up` must fail fast if the operator-owned daemon is unavailable at `127.0.0.1:${OLLAMA_LOCALHOST_PORT:-11434}` and must not attempt to start it or fall back to a container.
- Keep ComfyUI disabled for the current Jupyter-only scope. A future task may explicitly opt into only `COMFYUI_SOURCE=localhost` (or the documented native managed-MPS mode), never `container-cpu` or `container-gpu`.
- The initial runtime is JupyterHub only. Do not enable Spark, Ray, MinIO, MLflow, Label Studio, Airflow, Iceberg, Trino, Redpanda, endpoint exports, domain bootstrap, or data-eng-lab profile overrides.
- Treat `atlas.env.user` and `infra/.env` as required local runtime inputs, not optional conveniences. On a fresh clone the wrappers/CI must create them from secret-free templates without overwriting an existing local file. Neither file nor a Jupyter token may be committed.
- Do not hard-code a Jupyter or future service port in source. `atlas-connect` reads the current Jupyter port/token from Atlas-managed local state and prints it only interactively; it must not use the stable endpoint-export contract, write a token-bearing file, or `source infra/.env`.
- Notebook metadata must be driven by the 21 active tasks listed in `scripts/verify_repo_config.yaml`, ordered by `docs/manifest.yaml`, rather than filesystem glob order. It is not a replacement for existing task content metadata.
- Preserve all three documentation surfaces: edit canonical docs and the manifest only; let existing generators produce `generated/site`, `generated/wiki`, and `mkdocs.yml`. Never hand-edit generated outputs.
- Do not call the migration complete, remove the legacy gitlink, or change the manual-only quantization status before the static contract, live Atlas package probe, remote VS Code connection, and mounted NumPy import gates have succeeded.

## Migration phases

1. Build and test the parent-owned Atlas contract while the old gitlink remains only as an unmodified rollback reference.
2. Validate the real Atlas runtime at the reviewed pin. Resolve the documented package-compatibility boundary from evidence, not image-version assumptions.
3. In one final replacement change, retarget all verifier/docs/CI seams and remove the legacy submodule, override, launcher, and active instructions.

---

## Task 1: Add the reusable notebook-infrastructure contract parser and renderer

**Files:**

- Create: `scripts/docs/notebook_infrastructure.py`
- Create: `tests/test_notebook_infrastructure.py`
- Modify: `scripts/docs/check_docs.py`
- Modify: `tests/test_check_docs.py`
- Modify: `scripts/verify_repo.py`
- Modify: `tests/test_verify_repo.py`

**Interfaces to establish:**

```python
@dataclass(frozen=True)
class AtlasTaskContract:
    task: str
    title: str
    tier: str
    executor: str
    default_mode: str
    required_services: tuple[str, ...]
    workspace_access: str
    artifact_policy: str
    constraints: tuple[str, ...]

class NotebookInfrastructureError(ValueError): ...

def load_atlas_task_contracts(repo_root: Path, manifest: Manifest) -> list[AtlasTaskContract]: ...
def render_atlas_task_table(contracts: Sequence[AtlasTaskContract]) -> str: ...
def verify_atlas_task_table(doc_path: Path, expected_table: str) -> list[str]: ...
def write_atlas_task_table(doc_path: Path, expected_table: str) -> None: ...
```

- [ ] Add tests first. Use minimal temporary manifests/specs plus the real repository fixture to prove that loading follows `manifest.notebooks` order and that the manifest task set exactly equals `active_task_dirs` in `scripts/verify_repo_config.yaml`.
- [ ] Add failing tests for each owned `atlas` field: a mapping is required; `executor == "jupyterhub"`; `default_mode == "vscode-remote"`; `required_services` is a non-empty, unique service-ID list containing `jupyterhub`; `workspace_access` is `remote` or `mounted-required`; `artifact_policy` is `atlas-jupyter-volume` or `task-local-ignored-paths`; and `constraints` is a list of non-empty strings.
- [ ] Keep future service IDs forward-compatible: validate their syntax and uniqueness, not a closed set of today’s services. Require `jupyterhub` only because the current executor is JupyterHub.
- [ ] Implement the module under `scripts/docs/`, using the existing `scripts.docs.manifest.Manifest` and PyYAML already available to both runtime and docs requirements. Do not put task-spec parsing into `scripts/docs/manifest.py`; that module continues to own only the documentation manifest.
- [ ] Make the renderer produce exactly one deterministic Markdown table with columns `Task | Tier | Default mode | Workspace access | Required Atlas services | Artifact policy | Constraints`, use `—` for no constraints, and join multiple constraints with `<br>`.
- [ ] Add the canonical marker protocol `<!-- atlas-task-contracts:start -->` / `<!-- atlas-task-contracts:end -->`. Make missing, duplicated, reversed, or drifted markers/table content an error. `--write` is an explicit maintainer action; ordinary docs builds and CI use `--check` and never rewrite canonical Markdown.
- [ ] Add a thin `check_notebook_infrastructure(manifest, repo_root)` adapter in `scripts/docs/check_docs.py`, and invoke it from `check()` before generated-surface validation. Add a no-duplication adapter in `scripts/verify_repo.py::check_docs`; it may no-op only for existing synthetic fixtures that do not contain `docs/manifest.yaml`.
- [ ] Cover the adapters with tests that surface schema/table drift as normal docs/verifier findings rather than tracebacks.

Run:

```bash
uv run pytest tests/test_notebook_infrastructure.py tests/test_check_docs.py tests/test_verify_repo.py -q
uv run ruff check scripts/docs/notebook_infrastructure.py scripts/docs/check_docs.py scripts/verify_repo.py
```

Expected: the new contract tests pass, invalid fixtures identify the precise field or marker defect, and existing verifier fixtures remain valid.

- [ ] Commit the coherent unit-test and implementation slice.

```bash
git add scripts/docs/notebook_infrastructure.py scripts/docs/check_docs.py scripts/verify_repo.py tests/test_notebook_infrastructure.py tests/test_check_docs.py tests/test_verify_repo.py
git commit -m "feat: validate notebook Atlas runtime contracts"
```

## Task 2: Declare every active task’s Atlas contract and publish the canonical table

**Files:**

- Modify: all `notebooks/*/docs/spec.yaml` paths listed in `docs/manifest.yaml`
- Create: `docs/notebook-infrastructure.md`
- Modify: `docs/manifest.yaml`
- Modify: `scripts/docs/build_docs.py`
- Modify: `tests/test_build_docs.py`
- Modify: `tests/test_transforms.py`
- Modify: `tests/test_wiki.py`
- Modify: `scripts/verify_repo.py`
- Modify: `Makefile`

- [ ] Add a failing real-repository test that calls `load_atlas_task_contracts()` and asserts 21 validated contracts in documentation-manifest order.
- [ ] Add the following exact block to every active task spec, preserving existing task content:

```yaml
atlas:
  executor: jupyterhub
  default_mode: vscode-remote
  required_services: [jupyterhub]
  workspace_access: remote
  artifact_policy: atlas-jupyter-volume
  constraints: []
```

- [ ] Replace that default only for `notebooks/image_classification-mnist-ffnn-numpy/docs/spec.yaml`:

```yaml
atlas:
  executor: jupyterhub
  default_mode: vscode-remote
  required_services: [jupyterhub]
  workspace_access: mounted-required
  artifact_policy: task-local-ignored-paths
  constraints:
    - Sibling Python modules require the mounted checkout.
```

- [ ] Replace the default only for `notebooks/quantization-mnist-ffnn-pytorch/docs/spec.yaml`:

```yaml
atlas:
  executor: jupyterhub
  default_mode: vscode-remote
  required_services: [jupyterhub]
  workspace_access: remote
  artifact_policy: atlas-jupyter-volume
  constraints:
    - Manual-only until Atlas Jupyter package validation passes.
```

- [ ] Create `docs/notebook-infrastructure.md` with H1 `# 4.3 Notebook infrastructure`, an explanation of the default remote-kernel and mounted-workspace modes, artifact locations, future-service activation rules, and the required generated marker block. Generate the table with the new `--write` command and commit the rendered table.
- [ ] Expose an explicit Make target such as `docs-sync-notebook-infrastructure` that runs `python -m scripts.docs.notebook_infrastructure --write`; add the matching check to `docs-check`/the verifier, never an implicit canonical-doc rewrite.
- [ ] Register existing runtime docs and the new page under the existing Environment section in `docs/manifest.yaml`, without renumbering the current `8.x` notebook pages:

```yaml
- id: environment
  number: "4"
  title: Environment & runtimes
  source: docs/env-setup.md
  children:
    - id: jupyterhub-integration
      number: "4.1"
      title: JupyterHub integration
      source: docs/jupyterhub-integration.md
    - id: vscode-remote-access
      number: "4.2"
      title: VS Code remote access
      source: docs/vscode-remote-access.md
    - id: notebook-infrastructure
      number: "4.3"
      title: Notebook infrastructure
      source: docs/notebook-infrastructure.md
```

- [ ] Fix `scripts/docs/build_docs.py::_nav_lines()` so a manifest section with both `source` and `children` renders a nested navigation mapping: the section’s source becomes the first leaf and each child remains visible. Do not use the current `if source / elif children` shape that hides children.
- [ ] Confirm site source maps, wiki output, and sidebar output contain both the parent Environment page and its three children. Add unit tests for the nav map, transformed paths, and wiki sidebar.
- [ ] Add `docs/notebook-infrastructure.md` to `scripts/verify_repo.py::_iter_numbered_doc_files` so its baked H1/H2 numbering is checked.

Run:

```bash
python -m scripts.docs.notebook_infrastructure --write
python -m scripts.docs.notebook_infrastructure --check
uv run pytest tests/test_notebook_infrastructure.py tests/test_build_docs.py tests/test_transforms.py tests/test_wiki.py tests/test_check_docs.py -q
make docs-check
```

Expected: exactly 21 rows, two documented exceptions, visible nested navigation on site/wiki, and no generated-output changes staged.

- [ ] Commit the metadata/table/navigation slice.

```bash
git add notebooks docs/notebook-infrastructure.md docs/manifest.yaml scripts/docs/build_docs.py scripts/verify_repo.py Makefile tests
git commit -m "docs: declare notebook Atlas runtime contracts"
```

## Task 3: Add the pinned Atlas consumer seam and its static configuration contract

**Files:**

- Modify: `.gitmodules`
- Add gitlink: `infra/` at `61c7c5103660e2226bf107c115dae42bf46f8374`
- Create: `atlas.consumer.yml`
- Create: `atlas.env.user.example`
- Create: `compose/ml-eng-lab-atlas.yml`
- Modify: `.gitignore`
- Create: `tests/test_atlas_consumer_contract.py`
- Modify: `pyproject.toml`

- [ ] Start with tests that parse the new manifest/overlay and reject a moving branch, a `track:` manifest key, a non-`auto` base port, non-container JupyterHub source, a missing/non-`ollama-localhost` LLM source, any containerized/`auto` Ollama source, a containerized ComfyUI source, any SSH mount, a relative workspace path, or an overlay that mounts over `/home/jovyan/work` itself.
- [ ] Add the Atlas submodule without deleting `vendor/genai-vanilla` yet; it remains a rollback reference only until Task 6’s live acceptance gate. Use the exact upstream and detached revision:

```bash
git submodule add https://github.com/thekaveh/atlas.git infra
git -C infra checkout --detach 61c7c5103660e2226bf107c115dae42bf46f8374
git submodule status -- infra
```

- [ ] Make the final `.gitmodules` entry structurally equivalent to this and verify it contains no `branch =` line:

```ini
[submodule "infra"]
	path = infra
	url = https://github.com/thekaveh/atlas.git
```

- [ ] Add this committed consumer contract exactly; Atlas’s `ml-eng` selection belongs to the wrapper command, not this YAML:

```yaml
name: ml-eng-lab
project_name: ml-eng-lab
profile: dev

brand:
  name: ML Eng Lab

env:
  file: ./atlas.env.user
  values:
    BASE_PORT: auto
    JUPYTERHUB_SOURCE: container
    LLM_PROVIDER_SOURCE: ollama-localhost

compose_overlays:
  - ./compose/ml-eng-lab-atlas.yml
```

- [ ] Add the local-only template and the sole parent overlay:

```dotenv
# Copy/create this as atlas.env.user; it is ignored and machine-local.
ML_ENG_LAB_REPO_PATH=/absolute/path/to/ml-eng-lab
# Optional: only when the native Ollama daemon uses a non-default port.
# OLLAMA_LOCALHOST_PORT=11434
```

```yaml
services:
  jupyterhub:
    volumes:
      - ${ML_ENG_LAB_REPO_PATH}:/home/jovyan/work/ml-eng-lab
```

- [ ] Ignore `atlas.env.user` and only Atlas-generated parent-local connection/probe artifacts if implementation introduces them. Keep the already-correct exception that allows reviewed `docs/superpowers/specs/*.md` and `docs/superpowers/plans/*.md` through the broad planning ignore rule.
- [ ] Add `infra/` beside the existing `vendor/` pytest/ruff exclusions in `pyproject.toml` so the transitional checkout cannot recurse into either submodule. Task 8 removes the now-obsolete `vendor/` exclusions after the legacy gitlink is gone.
- [ ] Implement the tests with a temporary copy of the manifest/overlay, then run them against the actual files. They must demonstrate Atlas input correctness without starting Docker.

Run:

```bash
git submodule status --recursive
uv run pytest tests/test_atlas_consumer_contract.py -q
git -C infra status --short
```

Expected: the new gitlink prints the exact detached SHA, static contract tests pass, and `infra/` is clean.

- [ ] Commit the new consumer seam and configuration only. Do not delete the legacy gitlink in this commit.

```bash
git add .gitmodules .gitignore atlas.consumer.yml atlas.env.user.example compose/ml-eng-lab-atlas.yml infra pyproject.toml tests/test_atlas_consumer_contract.py
git commit -m "feat: add pinned Atlas consumer configuration"
```

## Task 4: Implement safe Atlas lifecycle wrappers and the default VS Code connection helper

**Files:**

- Create: `scripts/atlas-up.sh`
- Create: `scripts/atlas-down.sh`
- Create: `scripts/atlas-connect.sh`
- Modify: `Makefile`
- Create: `tests/test_atlas_lifecycle.py`
- Modify: `tests/test_makefile_contract.py`

- [ ] Write dry-run tests before the scripts. Require `set -euo pipefail`, absolute root/manifest resolution, actionable uninitialized-submodule errors, and no `eval` or secret-bearing artifact writes. Add normal-start tests that require the materialized source to be exactly `ollama-localhost`, probe only `127.0.0.1:<validated-port>/api/version`, fail clearly when it is unavailable, and never execute that probe in `--prepare`, `--validate`, or `--dry-run` mode.
- [ ] Add `make atlas-setup`, `make atlas-up`, `make atlas-down`, `make atlas-connect`, and `make atlas-contract` help/`.PHONY` entries. `atlas-setup` must initialize the exact submodule and then invoke `atlas-up.sh --prepare`; ordinary `atlas-up` also performs preparation idempotently so a fresh clone cannot reach a confusing missing-env validation failure.

```make
atlas-setup:
	git submodule update --init --recursive infra
	./scripts/atlas-up.sh --prepare

atlas-up:
	./scripts/atlas-up.sh

atlas-down:
	./scripts/atlas-down.sh $(if $(COLD),--cold,)

atlas-connect:
	./scripts/atlas-connect.sh

atlas-contract:
	./scripts/atlas-up.sh --validate
```

- [ ] In `atlas-up.sh`, accept only `--prepare`, `--validate`, and `--dry-run`. Resolve `REPO_ROOT`, `INFRA`, and an absolute `MANIFEST`. When absent, copy `infra/.env.example` to `infra/.env` and create `atlas.env.user` from the template with the resolved absolute root; never overwrite either existing file. Validate the resulting `ML_ENG_LAB_REPO_PATH` is an absolute path matching the checkout. This is mandatory because `env backfill` is additive and does not create an absent `.env`, while Atlas requires `env.file` for non-live validation.
- [ ] For a normal start, execute and dry-run-print this exact ordered sequence from `infra/`:

```bash
./start.sh env backfill
./start.sh --consumer "$MANIFEST" compose validate
./start.sh --consumer "$MANIFEST" doctor --format json
./start.sh --consumer "$MANIFEST" --track ml-eng --no-tui --detach
```

  `--validate` first calls the same idempotent preparation routine, then runs the first three commands and exits without starting services; `--prepare` creates/verifies only local state. After the doctor succeeds during a normal start, read the materialized `.env` using the shared non-evaluating dotenv parser: reject any `LLM_PROVIDER_SOURCE` other than `ollama-localhost`, validate `OLLAMA_LOCALHOST_PORT` (default `11434`), and require `curl --fail --silent --show-error --max-time 2 http://127.0.0.1:<port>/api/version` to succeed before detach. On failure, tell the operator to start native Ollama; do not run `ollama serve`, pass a source flag, or fallback to a container. After a normal final command, fail clearly if `git -C "$INFRA" status --porcelain` reports a non-ignored change. Do not add a `start` subcommand, `endpoints export`, a Docker command, or a service assertion to this wrapper.
- [ ] In `atlas-down.sh`, permit only `--cold` and `--dry-run`, construct an argument array, and run `(cd "$INFRA" && ./stop.sh "${stop_args[@]}")`. Default shutdown must preserve volumes; `--cold` is opt-in and its help text says it destroys persisted volumes.
- [ ] In `atlas-connect.sh`, provide an interactive-only read-only adapter for the Atlas pin: parse only needed dotenv values from `infra/.env` without sourcing it, use `JUPYTERHUB_PORT` and `PROJECT_NAME`, prefer a configured `JUPYTERHUB_TOKEN` if present, otherwise obtain the one-time token from the running `${PROJECT_NAME}-jupyterhub` container log. Validate every parsed value, avoid shell evaluation, never write a URL, and print the VS Code command-palette steps plus the current `http://localhost:<resolved-port>/?token=<resolved-token>` connection URL to the terminal.
- [ ] Test the precise dry-run command order; that `--prepare` and `--validate` do not start services or probe host Ollama; that a normal start rejects a non-localhost resolved source and an unavailable/malformed host port before detach; that `--cold` never appears unless requested; that a missing local env produces a preparation path rather than validation; and that the connection helper neither relies on `ATLAS_*JUPYTER*ENDPOINT` nor writes token files.

Run:

```bash
uv run pytest tests/test_atlas_lifecycle.py tests/test_makefile_contract.py -q
bash -n scripts/atlas-up.sh scripts/atlas-down.sh scripts/atlas-connect.sh
make -n atlas-setup atlas-up atlas-down atlas-connect
```

Expected: all static wrapper tests pass; no command starts Docker during dry-run; Make exposes the four documented entry points.

- [ ] Commit lifecycle behavior separately from live-runtime evidence.

```bash
git add Makefile scripts/atlas-up.sh scripts/atlas-down.sh scripts/atlas-connect.sh tests/test_atlas_lifecycle.py tests/test_makefile_contract.py
git commit -m "feat: add Atlas notebook lifecycle wrappers"
```

## Task 5: Retarget repository verification and add a non-live Atlas CI contract

**Files:**

- Modify: `scripts/verify_repo.py`
- Modify: `tests/test_verify_repo.py`
- Create: `.github/workflows/atlas-contract.yml`
- Modify: `.github/workflows/docs.yml`
- Modify: `Makefile`

- [ ] Add failing verifier tests first for a missing/malformed `atlas.consumer.yml`, illegal `track`, missing overlay/template/lifecycle script, an Ollama source other than `ollama-localhost`, containerized ComfyUI, invalid active-task metadata, hard-coded executable `localhost:<port>` or `127.0.0.1:<port>` Atlas service URLs, and a dirty `infra` worktree.
- [ ] Retarget `_required_submodule_paths()` from `vendor/genai-vanilla` to `infra`, preserving existing E6 clean-gitlink/worktree behavior. Retarget the three legacy D10 dependency-ledger fixtures and git index lookup to `git ls-files --stage -- infra`; retain the stable `D10.dependency_ledger_submodule_sha` finding identity.
- [ ] Remove Atlas implementation files from `_required_shellcheck_targets()`: the parent owns only its local wrappers. Add explicit required/executable coverage for `scripts/atlas-up.sh`, `scripts/atlas-down.sh`, and `scripts/atlas-connect.sh` so their absence cannot be hidden by glob discovery.
- [ ] Add focused `E15.atlas_manifest`, `E16.atlas_task_metadata`, and `E17.atlas_hardcoded_endpoint` findings. Reuse the shared notebook-infrastructure parser rather than copying its schema. Exclude documentation, tests, and historical records from the endpoint literal scan; scope it to executable integration code and notebook code cells.
- [ ] Add `make atlas-contract` as the local deterministic wrapper check. Keep the existing `verify-repo` job’s recursive checkout in `.github/workflows/ci.yml`. The dedicated GitHub workflow deliberately uses the raw Atlas preflight below as its single command source, so CI tests can assert it never starts services.
- [ ] Create `.github/workflows/atlas-contract.yml` with recursive checkout and path filters for `.gitmodules`, `infra`, `atlas.consumer.yml`, `atlas.env.user.example`, `compose/ml-eng-lab-atlas.yml`, `scripts/atlas-*.sh`, `scripts/docs/notebook_infrastructure.py`, `docs/notebook-infrastructure.md`, `docs/atlas-pin-bump-runbook.md`, `notebooks/**/docs/spec.yaml`, verifier files, and workflow files.
- [ ] In that job, provision only disposable non-secret state, then run the exact non-live preflight. It must not start containers, detach services, call `endpoints assert`, or inspect a live endpoint:

```bash
cp infra/.env.example infra/.env
printf 'ML_ENG_LAB_REPO_PATH=%s\n' "$GITHUB_WORKSPACE" > atlas.env.user
(
  cd infra
  ./start.sh env backfill
  ./start.sh --consumer ../atlas.consumer.yml compose validate
  ./start.sh --consumer ../atlas.consumer.yml doctor --format json
)
```

  End with `git -C infra status --porcelain --untracked-files=all --ignored=no` and fail on non-ignored changes.
- [ ] Extend `.github/workflows/docs.yml` path filters and explicit pytest command to include the new parser/tests, `docs/manifest.yaml`, and task specs so metadata table drift cannot bypass docs CI.

Run:

```bash
uv run pytest tests/test_verify_repo.py tests/test_atlas_consumer_contract.py tests/test_atlas_lifecycle.py -q
make atlas-contract
git -C infra status --short
```

Expected: the verifier is now Atlas-aware, static contract CI is provably non-live, and all submodule checks target only the parent-owned `infra` seam.

- [ ] Commit verifier and CI migration once the static commands are green.

```bash
git add scripts/verify_repo.py tests/test_verify_repo.py Makefile .github/workflows/atlas-contract.yml .github/workflows/docs.yml
git commit -m "ci: validate Atlas consumer contract"
```

## Task 6: Capture Atlas runtime compatibility evidence before changing dependency claims

**Files:**

- Create: `scripts/atlas_runtime_probe.py`
- Create: `tests/test_atlas_runtime_probe.py`
- Modify: `docs/dependency-contracts.md`
- Modify as evidence requires: `requirements.txt`, `torch-core-requirements.txt`, `torch-requirements.txt`, `Dockerfile`, `.devcontainer/devcontainer.json`, `Makefile`, and their tests/docs

- [ ] Write pure unit tests for a JSON probe result evaluator before implementing the probe. It must distinguish a missing module/asset from a version mismatch and must never serialize connection tokens, environment dumps, home paths, or credentials.
- [ ] Implement an in-container probe that reports Python, distribution/version, and import evidence for `thekaveh-nnx[lm]`, `nnx`, Torch, TorchVision, TorchAudio, TorchAO, PyTorch Geometric, `python-louvain`, spaCy `en_core_web_sm`, and NLTK VADER. Include a deterministic check of the direct imports used by active notebook code.
- [ ] Start the real pin locally with `make atlas-setup`, `make atlas-up`, and `make atlas-connect`. Use the printed URL in local VS Code’s **Jupyter: Specify Jupyter Server for Connections** command and execute a cheap remote-mode notebook while the `.ipynb` remains in the host checkout.
- [ ] In the mounted workspace mode, open a Jupyter terminal in the running JupyterLab (or use a verified `docker exec` command for the running Jupyter container), then run the probe and verify every NumPy MNIST sibling module is importable from `/home/jovyan/work/ml-eng-lab/notebooks/image_classification-mnist-ffnn-numpy`:

```bash
cd /home/jovyan/work/ml-eng-lab
python scripts/atlas_runtime_probe.py --json /tmp/ml-eng-lab-atlas-probe.json
cd notebooks/image_classification-mnist-ffnn-numpy
python -c "import consts, feed_fwd_nn, funcs, iteration_data_point, linear_layer, relu_layer, softmax_cross_entropy_layer, utils"
```

- [ ] Record the observed package matrix in `docs/dependency-contracts.md`. The expected Atlas image lineage at the reviewed pin includes Torch 2.11.0 / TorchVision 0.26.0 / TorchAudio 2.11.0 / PyG 2.6.1, but treat those as a hypothesis until the probe proves the built image. Do not retain a statement that `torch==2.4.1` is required for genai-vanilla image parity.
- [ ] Reconcile each local dependency manifest deliberately. If its independent venv/Docker/Codespaces stack is retained at another version, label it as a separately tested local contract rather than Atlas parity. If it is moved to the observed Atlas stack, update Torch/TorchVision/TorchAudio/PyG pins together and rerun the matching CPU/Linux tests. Do not install packages at Jupyter session time.
- [ ] Keep quantization manual-only unless an explicit Atlas Jupyter quantization smoke imports TorchAO and completes the notebook’s relevant surface. A newer Torch version alone is insufficient evidence.
- [ ] If the probe fails a Tier-A import, stop before legacy removal; retain the evidence, correct the parent dependency/documentation contract or obtain a reviewed Atlas pin, then repeat this task. Do not patch `infra/` to make the test pass.

Run:

```bash
uv run pytest tests/test_atlas_runtime_probe.py -q
make test-nnx-surface
make test
make atlas-down
git -C infra status --short
```

Expected: package evidence is captured without secrets, the default remote connection and mounted NumPy fallback work, normal shutdown preserves volumes, and `infra/` remains clean.

- [ ] Commit the evidence-driven dependency reconciliation and probe only after the live gate passes.

```bash
git add scripts/atlas_runtime_probe.py tests/test_atlas_runtime_probe.py docs/dependency-contracts.md requirements.txt torch-core-requirements.txt torch-requirements.txt Dockerfile .devcontainer/devcontainer.json Makefile
git commit -m "test: establish Atlas Jupyter runtime compatibility"
```

## Task 7: Rewrite the user-facing documentation, architecture, and Atlas pin-bump runbook

**Files:**

- Modify: `README.md`
- Modify: `CONTRIBUTING.md`
- Modify: `docs/env-setup.md`
- Modify: `docs/jupyterhub-integration.md`
- Modify: `docs/vscode-remote-access.md`
- Modify: `docs/architecture.md`
- Modify: `docs/diagrams/ml-eng-lab-system.html`
- Modify: `docs/diagrams/ml-eng-lab-runtime-flow.html`
- Modify: `docs/dependency-contracts.md`
- Create: `docs/atlas-pin-bump-runbook.md`
- Modify: `docs/manifest.yaml`
- Modify: `CHANGELOG.md`

- [ ] Register `docs/atlas-pin-bump-runbook.md` as child `6.1` of the existing Dependency contracts manifest section, preserving current notebook numbers. Update the doc’s H1/H2 numbering and add it to the numbered-doc verifier list.
- [ ] Rewrite the README quick start and layout to make Atlas local VS Code remote-kernel mode the recommended path. Show `git submodule update --init --recursive`, `make atlas-setup`, `make atlas-up`, `make atlas-connect`, and the VS Code server-selection step. Explain the mounted fallback for NumPy MNIST and host-visible `data/`/`runs/`; retain Docker, venv, and Codespaces as supported non-Atlas modes.
- [ ] Make `docs/jupyterhub-integration.md` the authoritative lifecycle guide: parent/Atlas ownership boundary, prepare/backfill/validate/doctor/start ordering, `BASE_PORT: auto`, required native Ollama (`LLM_PROVIDER_SOURCE=ollama-localhost`) and its fail-fast preflight, no-container guarantee, local state, normal versus `COLD=1` stop, token safety, troubleshooting, persistence semantics, and the `infra` clean-worktree check.
- [ ] Make `docs/vscode-remote-access.md` lead with remote VS Code to running JupyterLab. Include the exact visual workflow, a warning that notebook-relative writes in this mode land in the Atlas Jupyter volume, and the mounted-workspace fallback for sibling imports/host artifacts. Do not document an invented exported Jupyter endpoint or a fixed port.
- [ ] Update `CONTRIBUTING.md` with the notebook `atlas:` metadata requirement, generator/check command, the future-service activation sequence (spec first, central enablement, injected in-network variable, targeted runtime smoke, docs), and the rule against changes inside `infra/`.
- [ ] State in the future-service guide that an injected variable alone is not availability evidence: for example, `SPARK_REMOTE` may have a default value while Spark remains disabled. A task may rely on a future service only after its manifest source is explicitly enabled, doctor confirms the consumer configuration, and its targeted runtime smoke succeeds. For future image generation, the only permissible sources are host `localhost` or native managed-MPS; containerized ComfyUI is prohibited for this consumer.
- [ ] Update environment/dependency docs to distinguish Atlas’s observed Jupyter package contract from separate local installs, describe the manual-only quantization evidence, and remove all instructions to invoke `vendor/genai-vanilla` or `scripts/start-jupyterhub.sh`.
- [ ] Update the system and runtime-flow diagram source to show: local VS Code → Atlas Jupyter kernel as default; browser/container-attached mode → mounted checkout as fallback; parent manifest/overlay/wrappers/specs around a pinned `infra` gitlink; future services disabled until a task declares them. Regenerate PNG/SVG only through the existing diagram tooling.
- [ ] Write the pin-bump runbook with the immutable procedure: choose/review target SHA; update only the gitlink; provision local state; run backfill/compose validate/doctor; execute static suite; run `make atlas-up`, package probe, remote connection, mounted NumPy smoke, normal stop, and `infra` clean check; then update ledger/docs/changelog and stage only the reviewed gitlink and parent-owned changes. Explain that Atlas automatically adds `--build` after a committed Atlas SHA change through `.atlas-build-state`; an ordinary pin bump does not require `--cold`. Require an explicit rebuild only for documented uncommitted Dockerfile/context changes, and reserve `--cold` for the separate destructive volume-reset path.
- [ ] Add a Changelog entry that identifies Atlas as the direct successor to the former genai-vanilla seam and states the exact pin.

Run:

```bash
make docs-build
make docs-check
rg -n -i "genai-vanilla|vendor/genai-vanilla|start-jupyterhub" README.md CONTRIBUTING.md docs Makefile scripts .github pyproject.toml
```

Expected: all active guidance points to Atlas, documentation surfaces include all runtime pages, and remaining legacy matches are only explicitly historical records awaiting Task 8 removal/annotation.

- [ ] Commit the documentation and diagram migration.

```bash
git add README.md CONTRIBUTING.md CHANGELOG.md docs Makefile scripts/verify_repo.py
git commit -m "docs: make Atlas the primary notebook runtime"
```

## Task 8: Complete the direct replacement and remove the legacy consumer seam

**Files:**

- Remove gitlink: `vendor/genai-vanilla/`
- Remove: `deploy/genai-vanilla-jupyterhub.override.yml`
- Remove: `scripts/start-jupyterhub.sh`
- Replace: `docs/FINDINGS-VENDOR.md` with `docs/FINDINGS-ATLAS.md`
- Modify: `.gitmodules`, `.gitignore`, `pyproject.toml`, `scripts/verify_repo.py`, tests, docs, workflows, and all active references identified by the final search

- [ ] Re-run the Task 6 live acceptance evidence immediately before deletion. Do not remove the legacy path if the normal Atlas start, VS Code connection, mounted NumPy import, or dependency probe regressed.
- [ ] Retire the old gitlink precisely, then verify no submodule config remains for it:

```bash
git submodule deinit -f -- vendor/genai-vanilla
git rm -f vendor/genai-vanilla
git add .gitmodules
```

  Do not recursively delete arbitrary `.git/modules` state; Git’s retained local metadata is harmless and recoverable.
- [ ] Delete the legacy parent compose override and launcher. Ensure no target, verifier, workflow, README, or active environment document references their old paths.
- [ ] Replace the old vendor findings ledger with `docs/FINDINGS-ATLAS.md`. It must state that Atlas findings are reported upstream without modifying the pinned submodule, link to the Atlas dependency-contract ledger, and contain only current observations rather than copied historical genai-vanilla comments.
- [ ] Finish the `pyproject.toml` migration so `infra/` is excluded from pytest/ruff traversal and `vendor/` is no longer treated as a consumed repository path.
- [ ] Use a scoped search to eliminate active legacy instructions. Preserve dated maintenance records only when they plainly describe historical state; do not rewrite audit history to pretend Atlas existed then.

Run:

```bash
git submodule status --recursive
test ! -e vendor/genai-vanilla
test ! -e deploy/genai-vanilla-jupyterhub.override.yml
test ! -e scripts/start-jupyterhub.sh
rg -n -i "vendor/genai-vanilla|scripts/start-jupyterhub\.sh|genai-vanilla" README.md CONTRIBUTING.md docs Makefile scripts .github pyproject.toml
git -C infra status --short
```

Expected: `infra/` is the sole infrastructure gitlink, legacy launch/config files are gone, active docs are clean, and deliberately historical matches are clearly dated.

- [ ] Commit the irreversible-in-practice replacement only after the gate above is green.

```bash
git add -A .gitmodules pyproject.toml docs scripts Makefile .github tests infra vendor deploy
git commit -m "refactor: replace genai-vanilla submodule with Atlas"
```

## Task 9: Run the complete acceptance suite and record release evidence

**Files:**

- Modify if required by verified results only: `CHANGELOG.md`, `docs/dependency-contracts.md`, `docs/atlas-pin-bump-runbook.md`
- No generated files should be committed.

- [ ] Start from a clean clone or a fresh worktree. Initialize recursive submodules and verify the target gitlink is the reviewed detached SHA.
- [ ] Run all static gates after the legacy deletion:

```bash
git submodule update --init --recursive
make atlas-setup
make atlas-contract
make verify
make test
make test-nnx-surface
make lint
make docs-check
git diff --check
```

- [ ] Run the manual live release smoke in a Docker-capable environment with a running native Ollama daemon: `make atlas-up`; assert `docker compose ps` has neither `ollama` nor `ollama-pull` for the `ml-eng-lab` project; `make atlas-connect`; attach local VS Code to the returned Jupyter server; execute a representative remote-mode notebook; open a running-Jupyter terminal for the mounted-mode NumPy sibling-import command and runtime probe; run `make atlas-down`; verify `git -C infra status --short` is empty.
- [ ] Verify `COLD=1 make atlas-down` is documented but is not exercised in routine validation because it removes persistent volumes. If it must be tested, use disposable state and record that scope in the runbook evidence.
- [ ] Confirm the GitHub static Atlas job starts no services by reviewing its logged command sequence, and schedule/dispatch the opt-in live smoke only after the static workflow is green. The live smoke must cover Jupyter health, package probe, connection guidance, mounted NumPy import, and clean `infra` state without becoming a default PR job.
- [ ] Update the ledger/runbook/changelog only with facts proven by these commands. Confirm no token URL, local absolute path, `atlas.env.user`, `infra/.env`, `.atlas-build-state`, data, runs, or generated docs entered Git.

Run final audit:

```bash
git status --short
git diff --cached --check
git submodule status --recursive
git -C infra status --short
rg -n "JUPYTERHUB_TOKEN|token=" README.md CONTRIBUTING.md CHANGELOG.md docs Makefile scripts .github atlas.consumer.yml atlas.env.user.example compose --glob '!docs/maintenance/**' --glob '!docs/superpowers/**'
```

Expected: only intentional parent-owned changes are staged, the Atlas gitlink is clean and pinned, all documented workflows have evidence, and no secret-bearing runtime artifact is tracked.

- [ ] Create the final implementation commit(s) and open review with the static and live evidence attached. Do not advance the Atlas SHA as part of review cleanup; pin bumps follow the new runbook.

## Task 10: Enforce the native-host AI service policy

**Files:**

- Modify: `atlas.consumer.yml`
- Modify: `atlas.env.user.example`
- Modify: `scripts/atlas-up.sh`
- Modify: `scripts/lib/atlas-dotenv.sh` only if the existing parser lacks a safe last-value lookup
- Modify: `scripts/verify_repo.py`
- Modify: `tests/test_atlas_consumer_contract.py`
- Modify: `tests/test_atlas_lifecycle.py`
- Modify: `tests/test_verify_repo.py`

**Interfaces:**

- The manifest must materialize `LLM_PROVIDER_SOURCE=ollama-localhost` before every Atlas start.
- `scripts/atlas-up.sh` may call a single internal host preflight only for an ordinary start. It clears inherited `LLM_PROVIDER_SOURCE`, `COMFYUI_SOURCE`, and `OLLAMA_LOCALHOST_PORT` values, rejects an explicitly declared `atlas.env.user` ComfyUI source unless it is host-native (or disabled), then probes `http://127.0.0.1:<OLLAMA_LOCALHOST_PORT>/api/version` with proxy and curlrc configuration disabled; `<OLLAMA_LOCALHOST_PORT>` defaults to `11434` and must be an integer from `1` through `65535`.
- `--prepare`, `--validate`, and `--dry-run` remain non-live and must not require, start, or contact a host Ollama daemon.

- [ ] Add failing manifest/verification tests that reject an omitted or non-`ollama-localhost` `LLM_PROVIDER_SOURCE`, including `auto`, `ollama-container-cpu`, and `ollama-container-gpu`. Reject `COMFYUI_SOURCE=container-cpu` and `COMFYUI_SOURCE=container-gpu`; do not require a ComfyUI source while the ml-eng track keeps it disabled.
- [ ] Update the exact committed manifest contract:

```yaml
env:
  file: ./atlas.env.user
  values:
    BASE_PORT: auto
    JUPYTERHUB_SOURCE: container
    LLM_PROVIDER_SOURCE: ollama-localhost
```

  Keep `COMFYUI_SOURCE` absent. Add only a commented `OLLAMA_LOCALHOST_PORT=11434` example to `atlas.env.user.example`, explaining that an operator sets it only for a non-default native-daemon port.
- [ ] Add lifecycle tests with fake `start.sh` and `curl` executables. The ordinary-start test must record the ordered Atlas calls, show that `compose validate` materializes `LLM_PROVIDER_SOURCE=ollama-localhost`, and require the preflight before the detach call. Add failures for a changed source, a malformed/out-of-range port, missing `curl`, and a failed `/api/version` response. Exercise `atlas.env.user` values that attempt `COMFYUI_SOURCE=auto`, `container-cpu`, or `container-gpu`, and conflicting parent-shell exports. Assert that the diagnostics tell the operator to start native Ollama and that none invokes `ollama serve`, `docker`, or a container-source flag.
- [ ] Implement the smallest non-evaluating helper needed to read the final materialized `.env` values and any explicit consumer-local ComfyUI override. In `atlas-up.sh`, clear inherited policy variables and run the host check only after `env backfill`, consumer compose validation, and doctor pass, and only in ordinary start mode. When `atlas.env.user` explicitly declares `COMFYUI_SOURCE`, permit only `disabled`, `localhost`, or `managed-localhost-mps`; reject `auto` and every containerized source. When it is absent, leave the pinned default for the final `ml-eng` track to disable. Do not add a source flag to Atlas’s command line. Use a loopback URL plus `curl --disable --noproxy '*'` for the host-side check; Atlas itself routes its containers through `host.docker.internal`.
- [ ] Extend repository verification with a stable Atlas-manifest finding that enforces the same source policy without requiring Docker, curl, a host Ollama installation, or local runtime state.

Run:

```bash
uv run pytest tests/test_atlas_consumer_contract.py tests/test_atlas_lifecycle.py tests/test_verify_repo.py -q
bash -n scripts/atlas-up.sh scripts/lib/atlas-dotenv.sh
make atlas-contract
git diff --check
```

Expected: static CI remains non-live; a normal local start cannot reach detach unless the manifest resolves to native host Ollama and the daemon answers on loopback; no consumer source can select a containerized Ollama or ComfyUI.

- [ ] Commit the policy enforcement slice.

```bash
git add atlas.consumer.yml atlas.env.user.example scripts/atlas-up.sh scripts/lib/atlas-dotenv.sh scripts/verify_repo.py tests/test_atlas_consumer_contract.py tests/test_atlas_lifecycle.py tests/test_verify_repo.py
git commit -m "feat: require native Ollama for Atlas"
```
