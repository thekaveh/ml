# 12.22 Issue 62 Coordinated Torch Stack Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the repository's Torch 2.4 runtime with one exact, platform-qualified Torch 2.11, PyG, Lightning, TorchMetrics, and torchao contract without changing Atlas, NNx 0.2.0, notebook tier membership, or committed notebook outputs.

**Architecture:** Canonical requirement manifests define one matrix, a platform-aware Python installer applies it, and a fail-closed runtime verifier proves metadata, wheel provenance, CPU selection, compiled PyG operators, sampling, spline convolution, and NNx ownership. CI, Docker, Codespaces, advisory policy, and all three documentation surfaces consume those same contracts; no consumer may carry an independent installation algorithm.

**Tech Stack:** Python 3.11, pip, GNU Make, PyTorch 2.11.0, TorchVision 0.26.0, TorchAudio 2.11.0, PyTorch Lightning 2.6.1, TorchMetrics 1.9.0, torchao 0.18.0, PyG 2.8.0.post1, compiled PyG extensions, NNx 0.2.0, pytest, Ruff, Docker, GitHub Actions, papermill, pip-audit, MkDocs.

**Spec:** `docs/superpowers/specs/2026-08-14-issue-62-torch-stack-upgrade-design.md`

## 12.22.1 Global Constraints

- Use Python 3.11 and exact pins: Torch 2.11.0, TorchVision 0.26.0, TorchAudio 2.11.0, PyTorch Lightning 2.6.1, TorchMetrics 1.9.0, torchao 0.18.0, torch-geometric 2.8.0.post1, pyg-lib 0.8.0, torch-scatter 2.1.2, torch-sparse 0.6.18, torch-cluster 1.6.3, and torch-spline-conv 1.2.2.
- Keep `thekaveh-nnx[lm]==0.2.0`, `.gitmodules`, the `infra` gitlink, Atlas image requirements, Atlas Compose, and the `ml-eng` track unchanged; Issue #65 owns Atlas changes.
- Support and qualify Linux x86_64 CPU and Darwin arm64. Qualify native Linux arm64 through Docker on Apple Silicon. Do not claim CUDA, ROCm, XPU, Windows, direct-host Linux arm64, or source-built Torch support.
- Linux must install the Torch trio from `https://download.pytorch.org/whl/cpu`; Darwin must use native PyPI wheels. Never permit a Linux CUDA runtime package.
- Install pyg-lib, torch-scatter, torch-sparse, and torch-cluster as wheels from `https://data.pyg.org/whl/torch-2.11.0+cpu.html`; build only torch-spline-conv from source with `--no-build-isolation` and prove it with a `SplineConv` forward pass.
- Keep the quantization notebook manual-only and outside Tier A/B/C. Issue #66 owns notebook execution, output refresh, thresholds, and tier promotion.
- Do not edit Reddit notebook code, notebook outputs, execution counts, or notebook metadata for dependency accommodation. Any public-contract incompatibility requires a separate TDD correction wave and review.
- Do not initialize or start Atlas, JupyterHub, Docker Compose, Ollama, ComfyUI, or any repository service. Docker use is limited to the root image build and its non-service runtime probes. Containerized Ollama is prohibited.
- Keep historical Issue #59/#60/#61 records, released changelog entries, maintenance logs, Atlas evidence, and committed notebook results historical. Update only current operational truth and add durable Issue #62 history.
- Generate site/wiki derivatives from canonical sources; never commit `generated/`, `site/`, or root `mkdocs.yml`.
- Treat any resolver, ABI, sampler, quantization, security, Docker, notebook, or documentation failure as a stop condition. Do not drop a package, loosen a pin, add an exclusion, add a skip, suppress a command, or continue the matrix after a mandatory failure.

---

## 12.22.2 File Structure

### 12.22.2.1 New files

- `torch-ecosystem-requirements.txt` — exact PyPI-only Lightning, TorchMetrics, and torchao pins.
- `scripts/install_torch_stack.py` — the sole platform-aware command planner/executor for the ordered Torch/PyG installation.
- `scripts/verify_torch_stack.py` — the reusable fail-closed installed-stack oracle and runtime canaries.
- `scripts/verify_smoke_outputs.py` — the exact Tier A/B/C generated-notebook artifact oracle.
- `tests/test_install_torch_stack.py` — unit and mutation tests for platform selection, command order, wheel/source policy, and safe failures.
- `tests/test_verify_torch_stack.py` — unit and mutation tests for manifest parsing, distribution versions, local wheel versions/tags, CPU selection, architecture, imports, operators, sampling, spline convolution, NNx delegation, and redacted diagnostics.
- `tests/test_verify_smoke_outputs.py` — unit tests for tier inventory, artifact count, nonempty files, code-cell execution counts, error outputs, and safe diagnostics.

### 12.22.2.2 Existing dependency and execution files

- `torch-core-requirements.txt` — exact Torch/TorchVision/TorchAudio trio only.
- `torch-requirements.txt` — ecosystem include, Torch 2.11 PyG selector, pyg-lib, four retained extensions, and PyG.
- `torch-audit-requirements.txt` — selector-free core/ecosystem/PyG resolver projection.
- `pyg-extension-audit-requirements.txt` — exact pre-resolved PyPI extension supplement; pyg-lib remains excluded because it is external-index-only.
- `requirements.txt` — remove the duplicate open-ended torchao constraint; retain exact NNx 0.2.0.
- `Makefile` — publish installer/verifier targets and route Codespaces through them.
- `Dockerfile` — reuse the Make installer and require pip-check, stack verification, and sampler proof.
- `.devcontainer/devcontainer.json` — correct the current stack and quantization comments; keep `make codespace-setup` as its single command.

### 12.22.2.3 Existing CI, tests, policy, and documentation

- `.github/workflows/ci.yml` — cache every manifest, verify the stack after final install, and preserve all existing job identities/triggers.
- `tests/test_makefile_contract.py` — lock target names, selected interpreter, exact recipes, and failure propagation.
- `tests/test_verify_repo.py` — lock CI/Docker/cache/order/no-service/no-late-install contracts and exact matrix manifests.
- `tests/nnx_surface/test_node_classification_reddit_gnn_pyg.py` — make canonical sampler execution fail closed.
- `tests/nnx_surface/test_quantization_mnist_ffnn_pytorch.py` — make PTQ/QAT import and tiny execution fail closed without promoting the notebook.
- `scripts/advisory_baseline.py`, `tests/test_advisory_baseline.py`, and `security/accepted-advisories.json` — reconcile the new four-surface dependency observation without weakening exact projection/policy checks.
- `docs/architecture.md`, `docs/dependency-contracts.md`, `docs/env-setup.md`, `docs/FINDINGS-ATLAS.md`, `docs/notebooks/pruning-mnist-ffnn-pytorch.md`, `docs/notebooks/quantization-mnist-ffnn-pytorch.md`, `notebooks/quantization-mnist-ffnn-pytorch/README.md`, `notebooks/quantization-mnist-ffnn-pytorch/docs/spec.yaml`, `README.md`, `CONTRIBUTING.md`, and `CHANGELOG.md` — state current supported paths, residual risk, manual-only boundary, rollback, and qualification evidence.
- `tests/test_check_docs.py` — reject stale current-version, platform, audit, quantization, Atlas-ownership, and historical-rewrite claims.
- `docs/manifest.yaml` — project this plan as section 12.22 immediately after the design.

---

## 12.22.3 Plan Publication Precondition

- [x] Add this plan to `docs/manifest.yaml` as the sole section 12.22 source immediately after design 12.21.
- [x] Run `python -m scripts.docs.check_docs`, focused manifest/docs tests, `make docs-check`, `make docs-wiki`, and `git diff --check` before implementation.
- [x] Stage only `docs/manifest.yaml` and this plan and commit them with `docs: plan coordinated Torch stack upgrade`; Task 1 begins only from that clean committed state.

---

## 12.22.4 Task 1: Define the exact matrix and platform-aware installer

**Files:**
- Create: `torch-ecosystem-requirements.txt`
- Create: `scripts/install_torch_stack.py`
- Create: `tests/test_install_torch_stack.py`
- Modify: `torch-core-requirements.txt`
- Modify: `torch-requirements.txt`
- Modify: `torch-audit-requirements.txt`
- Modify: `pyg-extension-audit-requirements.txt`
- Modify: `requirements.txt`
- Modify: `Makefile`
- Modify: `tests/test_makefile_contract.py`
- Modify: `tests/test_advisory_baseline.py`

**Interfaces:**
- Consumes: Python 3.11, the exact versions in Global Constraints, and `sys.platform`/`platform.machine()`.
- Produces: `InstallStage(StrEnum)` with `UPGRADE_PIP`, `CORE`, `RUNTIME`, `ROOT`; `InstallCommand(stage: InstallStage, argv: tuple[str, ...])`; `CommandResult(Protocol)` exposing integer `returncode`; `CommandRunner(Protocol).__call__(argv: Sequence[str], *, check: bool) -> CommandResult`; `build_install_commands(python: str, system: str, machine: str) -> tuple[InstallCommand, ...]`; `install_torch_stack(commands: Sequence[InstallCommand], runner: CommandRunner = subprocess.run) -> None`; CLI `python -m scripts.install_torch_stack`; Make target `install-torch-stack`.

- [x] **Step 1: Add strict RED manifest and installer tests**

  In `tests/test_install_torch_stack.py`, define the complete canonical manifest bytes and assert the command plans:

  ```python
  UPGRADE_PIP = (
      sys.executable, "-m", "pip", "install", "--upgrade", "pip", "wheel",
  )
  LINUX_CORE = (
      sys.executable, "-m", "pip", "install", "--index-url",
      "https://download.pytorch.org/whl/cpu", "-r", "torch-core-requirements.txt",
  )
  RUNTIME = (
      sys.executable, "-m", "pip", "install",
      "--only-binary=pyg-lib,torch-scatter,torch-sparse,torch-cluster",
      "--no-binary=torch-spline-conv", "--no-build-isolation",
      "-r", "torch-requirements.txt",
  )
  ROOT = (
      sys.executable, "-m", "pip", "install",
      "--only-binary=thekaveh-nnx", "-r", "requirements.txt",
  )

  def test_linux_and_darwin_command_plans_are_exact():
      linux = build_install_commands(sys.executable, "Linux", "x86_64")
      darwin = build_install_commands(sys.executable, "Darwin", "arm64")
      assert linux[0].argv == darwin[0].argv == UPGRADE_PIP
      assert linux[1].argv == LINUX_CORE
      assert darwin[1].argv == tuple(x for x in LINUX_CORE if x not in (
          "--index-url", "https://download.pytorch.org/whl/cpu"
      ))
      assert linux[2].argv == darwin[2].argv == RUNTIME
      assert linux[3].stage is InstallStage.ROOT
      assert linux[3].argv == darwin[3].argv == ROOT
  ```

  The bootstrap command is exactly `python -m pip install --upgrade pip wheel`; wheel supplies the
  source-build tooling needed when the runtime stage disables build isolation. Add mutations for
  omitting `wheel` from all three supported system/machine pairs, unsupported
  systems/architectures,
  reordered/missing/duplicate stages, missing CPU index, broad `--only-binary=:all:`, omitted wheel
  names, adding spline to `--only-binary`, omitting `--no-binary`/`--no-build-isolation`, changing
  the PyG URL, omitting binary-only NNx, adding a fifth installer, using shell strings, ignoring a
  nonzero return, and leaking runner output in a raised error. In `tests/test_makefile_contract.py`,
  require `install-torch-stack` to contain exactly `$(PYTHON) -m scripts.install_torch_stack` and
  require `codespace-setup` to perform no later pip install. In
  `tests/test_advisory_baseline.py`, change the fixtures to the new exact
  core/ecosystem/runtime/audit split and add synchronized-deletion/version/option/duplicate
  mutations.

- [x] **Step 2: Run the RED tests**

  Run:

  ```bash
  pytest -p no:cacheprovider tests/test_install_torch_stack.py tests/test_makefile_contract.py tests/test_advisory_baseline.py -q
  ```

  Expected: failures because the ecosystem manifest/module do not exist and the existing pins/Make recipe still describe Torch 2.4.1.

- [x] **Step 3: Write the exact manifests**

  Use these semantic contents, retaining only explanatory comments around them:

  ```text
  # torch-core-requirements.txt
  torch==2.11.0
  torchvision==0.26.0
  torchaudio==2.11.0

  # torch-ecosystem-requirements.txt
  pytorch-lightning==2.6.1
  torchmetrics==1.9.0
  torchao==0.18.0

  # torch-requirements.txt
  -r torch-ecosystem-requirements.txt
  --find-links https://data.pyg.org/whl/torch-2.11.0+cpu.html
  pyg-lib==0.8.0
  torch-scatter==2.1.2
  torch-sparse==0.6.18
  torch-cluster==1.6.3
  torch-spline-conv==1.2.2
  torch_geometric==2.8.0.post1

  # torch-audit-requirements.txt
  -r torch-core-requirements.txt
  -r torch-ecosystem-requirements.txt
  torch_geometric==2.8.0.post1
  ```

  Keep the four PyPI extension pins in `pyg-extension-audit-requirements.txt`; do not add pyg-lib there. Remove only the root `torchao>=0.17` line and its now-stale direct-dependency comment from `requirements.txt`.

- [x] **Step 4: Implement the installer and Make seam**

  Implement immutable argv plans with these exact stage values:

  ```python
  class InstallStage(StrEnum):
      UPGRADE_PIP = "upgrade-pip"
      CORE = "core"
      RUNTIME = "runtime"
      ROOT = "root"

  @dataclass(frozen=True)
  class InstallCommand:
      stage: InstallStage
      argv: tuple[str, ...]
  ```

  Accept only `("Linux", "x86_64"|"aarch64")` and `("Darwin", "arm64")`; execute all four stages with `check=False`; raise `TorchStackInstallError(f"torch stack installation failed: {command.stage}")` on nonzero without echoing argv/stdout/stderr. The root stage is exactly binary-only for `thekaveh-nnx` and last, so its dependency solve sees the already-satisfied exact matrix. Replace the Make recipe with:

  ```make
  install-torch-stack:
	$(PYTHON) -m scripts.install_torch_stack
  ```

  Change `codespace-setup` to depend on `install-torch-stack` and run only `$(MAKE) nlp-assets`; it must not install root requirements a second time. Update the current quantization explanation in the Makefile header to Torch 2.11/torchao 0.18/manual-only Issue #66 truth.

- [x] **Step 5: Prove GREEN and mutation resistance**

  Run:

  ```bash
  pytest -p no:cacheprovider tests/test_install_torch_stack.py tests/test_makefile_contract.py tests/test_advisory_baseline.py -q
  ruff check scripts/install_torch_stack.py tests/test_install_torch_stack.py tests/test_makefile_contract.py tests/test_advisory_baseline.py
  git diff --check
  ```

  Then mutate each selected version, include, selector, binary/source flag, platform, and return code in an isolated temporary copy; each mutation must make at least one named test fail while an unmodified control passes.

- [x] **Step 6: Commit Task 1**

  ```bash
  git add torch-core-requirements.txt torch-ecosystem-requirements.txt torch-requirements.txt torch-audit-requirements.txt pyg-extension-audit-requirements.txt requirements.txt Makefile scripts/install_torch_stack.py tests/test_install_torch_stack.py tests/test_makefile_contract.py tests/test_advisory_baseline.py docs/superpowers/plans/2026-08-14-issue-62-torch-stack-upgrade-implementation-plan.md
  git commit -m "deps: define Torch 2.11 compatibility matrix"
  ```

---

## 12.22.5 Task 2: Add the canonical installed-stack verifier

**Files:**
- Create: `scripts/verify_torch_stack.py`
- Create: `tests/test_verify_torch_stack.py`
- Create: `tests/test_verify_torch_stack_platform.py`
- Modify: `Makefile`
- Modify: `tests/test_makefile_contract.py`

**Interfaces:**
- Consumes: the five root stack manifests, installed distribution metadata, `packaging.version.Version`, imported modules, `scripts.verify_nnx_install.verify_nnx_install`, and tiny CPU tensors/graphs.
- Produces: `StackPin(distribution: str, import_name: str, public_version: Version)`; `StackContract(pins: tuple[StackPin, ...], system: str, machine: str)`; `StackEvidence(system: str, machine: str, torch_version: str, backend: str)`; `DistributionView(Protocol)` exposing `version: str`, `files: Sequence[PackagePath] | None`, and `read_text(filename: str) -> str | None`; `CanaryHooks(scatter, sparse, cluster, sampler, spline)` where every callable is `Callable[[Mapping[str, ModuleType]], None]`; `VerificationHooks(distribution: Callable[[str], DistributionView], installed_names: Callable[[], Iterable[str]], import_module: Callable[[str], ModuleType], system: Callable[[], str], machine: Callable[[], str], nnx_verify: Callable[[], object], canaries: CanaryHooks)`; `load_stack_contract(repo: Path, system: str, machine: str) -> StackContract`; `verify_torch_stack(repo: Path = REPO_ROOT, hooks: VerificationHooks = DEFAULT_HOOKS) -> StackEvidence`; CLI `python -m scripts.verify_torch_stack`; Make target `verify-torch-stack`.

- [x] **Step 1: Write parser, metadata, provenance, and safe-error RED tests**

  Define the exact distribution/import map in the test and production module:

  ```python
  IMPORTS = {
      "torch": "torch", "torchvision": "torchvision", "torchaudio": "torchaudio",
      "pytorch-lightning": "pytorch_lightning", "torchmetrics": "torchmetrics",
      "torchao": "torchao", "torch-geometric": "torch_geometric",
      "pyg-lib": "pyg_lib", "torch-scatter": "torch_scatter",
      "torch-sparse": "torch_sparse", "torch-cluster": "torch_cluster",
      "torch-spline-conv": "torch_spline_conv",
  }
  ```

  Add exact-manifest fixtures and one `FakeDistribution` implementing the declared protocol. Tests must reject missing/duplicate/ranged pins, includes in the wrong manifest, public-version drift, incompatible local versions, absent WHEEL/RECORD ownership, an imported module path absent from RECORD, x86-only wheels on Darwin arm64, non-CPU Linux Torch, any installed distribution whose normalized name begins `nvidia-`, and messages containing injected credentials, URLs, paths, installer output, or tracebacks. Linux PyG-index local versions require the exact `pt211cpu` tag. Darwin arm64 accepts the exact `pt211` local tag or an absent local tag. An absent local tag is accepted only after independent WHEEL/RECORD, Python ABI/platform architecture, and mandatory runtime-canary checks pass.

  ```python
  def test_linux_local_pyg_versions_require_pt211cpu_and_compatible_wheel_tags(fake_stack):
      fake_stack.version("pyg-lib", "0.8.0+pt211cpu")
      fake_stack.wheel("pyg-lib", "Tag: cp311-cp311-manylinux_2_28_x86_64\n")
      evidence = verify_torch_stack(repo=fake_stack.repo, hooks=fake_stack.hooks)
      assert evidence.backend == "pyg-lib"

  def test_darwin_local_pyg_versions_accept_pt211_with_native_abi3_wheel(fake_stack):
      fake_stack.version("pyg-lib", "0.8.0+pt211")
      fake_stack.wheel("pyg-lib", "Tag: cp310-abi3-macosx_11_0_arm64\n")
      evidence = verify_torch_stack(repo=fake_stack.repo, hooks=fake_stack.hooks)
      assert evidence.backend == "pyg-lib"
  ```

- [x] **Step 2: Write runtime-canary RED tests**

  Build `CanaryHooks` fakes that append their stable names to a list and assert exact order `("scatter", "sparse", "cluster", "sampler", "spline")`, once each. Add a failure mutation for every canary and for skipped NNx delegation. Assert no `pytest.skip`, environment bypass, warning-only result, or `try/except: pass` can satisfy the verifier.

- [x] **Step 3: Run RED**

  ```bash
  pytest -p no:cacheprovider tests/test_verify_torch_stack.py tests/test_verify_torch_stack_platform.py tests/test_makefile_contract.py -q
  ```

  Expected: collection failure because `scripts.verify_torch_stack` does not exist.

- [x] **Step 4: Implement the fail-closed verifier**

  Implement the orchestration with this exact dependency flow; helper functions raise `TorchStackVerificationError(component, category)` rather than raw third-party exceptions:

  ```python
  def verify_torch_stack(
      repo: Path = REPO_ROOT,
      hooks: VerificationHooks = DEFAULT_HOOKS,
  ) -> StackEvidence:
      contract = load_stack_contract(repo, hooks.system(), hooks.machine())
      modules: dict[str, ModuleType] = {}
      installed = tuple(normalize_name(name) for name in hooks.installed_names())
      _verify_no_cuda_distributions(contract, installed)
      for pin in contract.pins:
          distribution = hooks.distribution(pin.distribution)
          _verify_distribution(pin, distribution, contract)
          module = hooks.import_module(pin.import_name)
          _verify_record_ownership(pin, distribution, module)
          modules[pin.distribution] = module
      _verify_cpu_runtime(contract, modules["torch"])
      for canary in dataclasses.astuple(hooks.canaries):
          canary(modules)
      hooks.nnx_verify()
      return StackEvidence(contract.system, contract.machine, modules["torch"].__version__, "pyg-lib")
  ```

  Compare public versions with `Version(distribution.version).public`, derive a present PyG local tag from the selected Torch major/minor plus the platform CPU suffix, use `distribution.read_text("WHEEL")` for tags, and confirm the resolved module file is present in `distribution.files`/RECORD. A missing local tag never bypasses the independent WHEEL/RECORD, Python ABI/platform architecture, or runtime-canary gates. Require `torch.version.cuda is None` and no CUDA distributions on Linux. Default canaries execute `torch_scatter.scatter`, `torch_sparse.SparseTensor.matmul`, `torch_cluster.knn`, one real `NeighborLoader` batch with positive seed/edge counts, and a two-node `SplineConv` with output shape `(2, 2)`; `hooks.nnx_verify()` runs last. Normalize failures to:

  ```text
  torch stack verification failed: <component>: <manifest|metadata|wheel|platform|cpu|abi|operator|sampler|nnx>
  ```

  The CLI prints the stable message to stderr and exits 1; success prints one version/platform summary and exits 0.

- [x] **Step 5: Publish and test the Make target**

  Add `verify-torch-stack` once to `.PHONY` and `help`, with the exact recipe:

  ```make
  verify-torch-stack:
	$(PYTHON) -m scripts.verify_torch_stack
  ```

  Lock the recipe, selected interpreter, one-command inventory, and failure propagation in `tests/test_makefile_contract.py`.

- [x] **Step 6: Prove GREEN and commit**

  ```bash
  pytest -p no:cacheprovider tests/test_verify_torch_stack.py tests/test_verify_torch_stack_platform.py tests/test_makefile_contract.py -q
  ruff check scripts/verify_torch_stack.py tests/test_verify_torch_stack.py tests/test_verify_torch_stack_platform.py tests/test_makefile_contract.py
  python -m py_compile scripts/verify_torch_stack.py
  git diff --check
  git add scripts/verify_torch_stack.py tests/test_verify_torch_stack.py tests/test_verify_torch_stack_platform.py Makefile tests/test_makefile_contract.py docs/superpowers/plans/2026-08-14-issue-62-torch-stack-upgrade-implementation-plan.md
  git commit -m "test: add canonical Torch stack verifier"
  ```

---

## 12.22.6 Task 3: Make PyG and quantization consumer gates fail closed

**Files:**
- Modify: `tests/nnx_surface/test_node_classification_reddit_gnn_pyg.py`
- Modify: `tests/nnx_surface/test_quantization_mnist_ffnn_pytorch.py`
- Modify: `tests/nnx_surface/conftest.py`
- Test: `tests/test_verify_torch_stack.py`
- Create: `scripts/verify_smoke_outputs.py`
- Create: `tests/test_verify_smoke_outputs.py`
- Modify: `Makefile`
- Modify: `tests/test_makefile_contract.py`

**Interfaces:**
- Consumes: the verified canonical stack from Task 2, the existing tiny graph/image fixtures, NNx 0.2.0's public quantization facade, and torchao 0.18.0.
- Produces: mandatory SAGE/GraphConv/GAT sampled training; a tiny PTQ prediction; a tiny QAT prepare/train/convert/inference contract; `Tier = Literal["a", "b", "c"]`; `InventoryLoader(Protocol).__call__(tier: Tier) -> Sequence[str]`; `NotebookArtifact(source: str, output: Path, code_cells: int)`; `verify_smoke_outputs(tier: Tier, root: Path, inventory_loader: InventoryLoader = load_make_inventory) -> tuple[NotebookArtifact, ...]`; CLI `python -m scripts.verify_smoke_outputs --tier <a|b|c> --root <absolute-path>`; no notebook-tier change.

- [ ] **Step 1: Capture graph and quantization RED against the pre-upgrade environment**

  Remove neither skip yet. First add meta-contract tests that reject `pytest.skip`, `pytest.importorskip`, `_HAS_PYG_SAMPLER`, `@pytest.mark.skipif`, and the Torch `int1` guard in these two modules. Add mutations that restore each bypass and require failure. Run:

  ```bash
  pytest -p no:cacheprovider tests/nnx_surface/test_node_classification_reddit_gnn_pyg.py tests/nnx_surface/test_quantization_mnist_ffnn_pytorch.py tests/test_verify_torch_stack.py -q
  ```

  Expected under the old active environment: the new no-bypass contracts fail, proving the upgrade is required rather than silently skipped.

- [ ] **Step 2: Remove optional-backend paths**

  Delete `_has_pyg_sampler`, `_HAS_PYG_SAMPLER`, skip reasons/decorators, and `_import_torchao_or_skip`. Import required backends normally inside the tests. Keep the pure `NNParams.state()` test. In `tests/nnx_surface/conftest.py`, call `verify_torch_stack()` once before `verify_nnx_install()` and before importing NNx; do not add a second session invocation.

- [ ] **Step 3: Add exact PyG runtime assertions**

  For every NeighborLoader test, consume one batch and assert `batch.batch_size > 0`, `batch.edge_index.numel() > 0`, and successful one-epoch NNx training. Add explicit import/use coverage for pyg-lib and torch-sparse so the preferred backend and fallback are both ABI-proven in the canonical environment:

  ```python
  def test_canonical_sampler_backends_and_batch_are_executable(tiny_graph_data):
      import pyg_lib
      import torch_sparse
      from torch_geometric.loader import NeighborLoader

      batch = next(iter(NeighborLoader(
          tiny_graph_data.data, num_neighbors=[2, 2], batch_size=2,
          input_nodes=tiny_graph_data.data.train_mask, shuffle=False, num_workers=0,
      )))
      assert pyg_lib is not None and torch_sparse is not None
      assert batch.batch_size > 0
      assert batch.edge_index.numel() > 0
  ```

- [ ] **Step 4: Add the tiny PTQ/QAT contract**

  Preserve facade-signature tests. Replace the guarded PTQ test with mandatory execution. Build a two-batch 8da4w QAT smoke using the notebook's public call shape:

  ```python
  callback = nnx.QATLifecycleCallback(qat_config="8da4w")
  train_step = nnx.qat_train_step_factory(qat_config="8da4w")
  run = model.train(
      params=NNTrainParams(n_epochs=1)
          .with_train_loader(value=tiny_image_batch.train_loader)
          .with_val_loader(value=tiny_image_batch.val_loader),
      callbacks=[callback],
      train_step_fn=train_step,
  )
  logits, classes = model.predict(X=tiny_image_batch.X)
  assert run is not None
  assert callback.is_prepared and callback.is_converted
  assert logits.shape == (4, 10) and classes.shape == (4,)
  assert np.issubdtype(classes.dtype, np.integer)
  ```

  Do not execute or edit the notebook.

- [ ] **Step 5: Prove GREEN in a clean candidate environment**

  Create and activate a disposable Python 3.11 environment and invoke the one authoritative installer exactly once:

  ```bash
  FOCUS_ROOT=$(mktemp -d /private/tmp/ml-eng-lab-issue62-focus.XXXXXX)
  python3.11 -m venv "$FOCUS_ROOT/venv"
  export PATH="$FOCUS_ROOT/venv/bin:$PATH"
  make install-torch-stack
  python -m pip check
  make verify-torch-stack
  make verify-nnx-install
  pytest -p no:cacheprovider tests/nnx_surface/test_node_classification_reddit_gnn_pyg.py tests/nnx_surface/test_quantization_mnist_ffnn_pytorch.py tests/test_verify_torch_stack.py -q
  ruff check tests/nnx_surface/test_node_classification_reddit_gnn_pyg.py tests/nnx_surface/test_quantization_mnist_ffnn_pytorch.py tests/nnx_surface/conftest.py tests/test_verify_torch_stack.py
  git diff --check
  ```

  Expected: zero skips and zero failures in the focused modules.

- [ ] **Step 6: Add the smoke-output artifact oracle with RED-GREEN tests**

  Add Make targets `print-tier-a`, `print-tier-b`, and `print-tier-c` that print exactly one source notebook path per line from the existing variables. Implement `load_make_inventory(tier)` with `subprocess.run(("make", "--no-print-directory", "-s", f"print-tier-{tier}"), check=False, capture_output=True, text=True)` and return stripped nonempty stdout lines only on exit 0. `verify_smoke_outputs` calls its injected `inventory_loader(tier)`, maps Tier A to `<root>/<source path>` and Tier B/C to `<root>/<basename>`, and rejects duplicate mapped paths. For every expected notebook require: regular nonempty file, valid nbformat JSON, every code cell has a non-null execution count, and no output has `output_type == "error"`:

  ```python
  def verify_smoke_outputs(tier, root, inventory_loader=load_make_inventory):
      sources = tuple(inventory_loader(tier))
      outputs = tuple(root / (source if tier == "a" else Path(source).name) for source in sources)
      _require_unique_inventory(tier, sources, outputs)
      artifacts = tuple(_validate_notebook(tier, source, output) for source, output in zip(sources, outputs))
      return artifacts
  ```

  Emit only `smoke output verification failed: <tier>: <inventory|missing|empty|invalid|unexecuted|error>`.

  RED tests must cover missing/extra/duplicate inventory, empty/invalid notebooks, one unexecuted code cell, one error output, a runner failure, and injected path/traceback redaction. GREEN:

  ```bash
  pytest -p no:cacheprovider tests/test_verify_smoke_outputs.py tests/test_makefile_contract.py -q
  ruff check scripts/verify_smoke_outputs.py tests/test_verify_smoke_outputs.py
  ```

- [ ] **Step 7: Commit Task 3**

  ```bash
  git add tests/nnx_surface/test_node_classification_reddit_gnn_pyg.py tests/nnx_surface/test_quantization_mnist_ffnn_pytorch.py tests/nnx_surface/conftest.py tests/test_verify_torch_stack.py scripts/verify_smoke_outputs.py tests/test_verify_smoke_outputs.py Makefile tests/test_makefile_contract.py docs/superpowers/plans/2026-08-14-issue-62-torch-stack-upgrade-implementation-plan.md
  git commit -m "test: require Torch 2.11 graph and quantization surfaces"
  ```

---

## 12.22.7 Task 4: Route CI, Docker, and Codespaces through the canonical contract

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `Dockerfile`
- Modify: `.devcontainer/devcontainer.json`
- Modify: `Makefile`
- Modify: `tests/test_verify_repo.py`
- Modify: `tests/test_makefile_contract.py`

**Interfaces:**
- Consumes: `make install-torch-stack`, `python -m pip check`, `make verify-torch-stack`, `make verify-nnx-install`.
- Produces: unchanged CI job identities/triggers; exact manifest cache coverage; verifier-before-workload ordering; root Docker build with sampler proof; unchanged Codespaces `postCreateCommand`.

- [ ] **Step 1: Add RED workflow and Docker contracts**

  Extend the existing exact job dictionaries and mutation helpers. Every stack-consuming job must cache `requirements.txt`, `torch-core-requirements.txt`, `torch-ecosystem-requirements.txt`, and `torch-requirements.txt`, and install the complete root environment only through `make install-torch-stack`. In repository, NNx-surface, verify-repo, Tier B, and Tier C jobs, require `python -m pip check`, `make verify-torch-stack`, and `make verify-nnx-install` immediately after that installer and before the workload. In Tier A, require the existing spaCy model wheel install and NLTK data download first, then the three verification commands, then papermill. The dependency-audit job must additionally cache both new/changed audit inputs.

  Reject missing/reordered verification, any later pip/uv/conda/`python -m spacy download` command, alternate Torch indexes, CUDA packages, service/container declarations, `docker compose`, Atlas/Ollama/ComfyUI commands, masking `if` on install/verification/workload steps, `continue-on-error`, timeout weakening, and extra steps. Preserve and assert byte-exact the existing Tier B/Tier C job-level activation conditions and Tier A artifact-upload `if: always()`. Docker contracts must require the Make installer, NLP assets, pip-check, verifier, and sampler probe in that order, with no separate root installer.

- [ ] **Step 2: Run RED**

  ```bash
  pytest -p no:cacheprovider tests/test_verify_repo.py tests/test_makefile_contract.py -q -k 'torch_stack or repository_test_contract or nnx_surface_job or tier_nnx_provenance or dependency_audit or docker'
  ```

  Expected: failures for missing ecosystem cache entries, stack verifier steps, and Docker runtime checks.

- [ ] **Step 3: Update CI without adding a new context**

  Preserve all workflow triggers, permissions, runner labels, timeouts, job keys, tier conditions, and artifact paths. Remove the duplicated `python -m pip install --only-binary=thekaveh-nnx -r requirements.txt` steps because Task 1's root stage owns that exact policy. Add the new cache path everywhere the stack is installed/audited. In jobs without NLP setup, add the two commands immediately after `make install-torch-stack`; in Tier A, add them immediately after both NLP setup steps:

  ```yaml
  - name: Verify installed dependency graph
    run: python -m pip check
  - name: Verify canonical Torch stack
    run: make verify-torch-stack
  ```

  Keep `make verify-nnx-install` after the stack verifier and before the workload.

- [ ] **Step 4: Update Docker and Codespaces**

  Replace Docker's duplicate core/runtime/root install commands with `RUN make install-torch-stack`, retain NLP assets, then add `RUN python -m pip check && make verify-torch-stack && make verify-nnx-install`. The stack verifier's mandatory `NeighborLoader` canary is the image's tiny sampler probe; the Docker contract test must reject a verifier implementation or invocation that bypasses that canary. Do not add an entrypoint, daemon, compose file, health service, or exposed Ollama port. Keep devcontainer `postCreateCommand` exactly `make codespace-setup`; update only its stale comments. Replace the current Torch 2.4/torchao-range comments in the tier artifact section with the implemented Torch 2.11/manual-only Issue #66 boundary.

- [ ] **Step 5: Prove GREEN and mutation resistance**

  ```bash
  pytest -p no:cacheprovider tests/test_verify_repo.py tests/test_makefile_contract.py -q
  python -c 'import pathlib, yaml; yaml.safe_load(pathlib.Path(".github/workflows/ci.yml").read_text())'
  ruff check tests/test_verify_repo.py tests/test_makefile_contract.py
  git diff --check
  ```

  Run isolated workflow/Docker mutations for every cache entry, command, ordering edge, service/container path, CUDA token, failure mask, and late installer. The unmodified control must pass; every mutation must fail a named test.

- [ ] **Step 6: Commit Task 4**

  ```bash
  git add .github/workflows/ci.yml Dockerfile .devcontainer/devcontainer.json Makefile tests/test_verify_repo.py tests/test_makefile_contract.py docs/superpowers/plans/2026-08-14-issue-62-torch-stack-upgrade-implementation-plan.md
  git commit -m "ci: enforce coordinated Torch stack contract"
  ```

---

## 12.22.8 Task 5: Refresh the four-surface advisory policy and human ledger

**Files:**
- Modify: `scripts/advisory_baseline.py`
- Modify: `scripts/verify_repo.py`
- Modify: `tests/test_advisory_baseline.py`
- Modify: `security/accepted-advisories.json`
- Modify: `docs/dependency-contracts.md`
- Modify: `tests/test_verify_repo.py`
- Modify: `tests/test_check_docs.py`

**Interfaces:**
- Consumes: exact Task 1 manifests, `pip-audit==2.10.0`, the existing four logical surfaces (`combined-runtime`, `torch`, `documentation`, `atlas-contract`), and Task 2's local pyg-lib provenance evidence.
- Produces: canonical accepted-advisory JSON; current dated ledger snapshot; strict policy/ledger parity; preserved historical snapshot.

- [ ] **Step 1: Update projection RED tests before policy or prose**

  Require runtime semantic lines to equal the ecosystem include, Torch 2.11 PyG selector, pyg-lib, four extensions, and PyG; require audit lines to equal core include, ecosystem include, and PyG; require the supplement to equal the four PyPI extensions. Add tests proving pyg-lib is deliberately absent from pip-audit inputs but exact in runtime and named in provenance limitations. Retain malformed UTF-8, continuation, comment, duplicate, synchronized-drift, fenced-Markdown, and safe-diagnostic mutations.

- [ ] **Step 2: Run projection RED and implement it**

  ```bash
  pytest -p no:cacheprovider tests/test_advisory_baseline.py -q -k 'projection or surface or command or diagnostics'
  ```

  Expected: failures against the Torch 2.4 constants. Update only the fixed semantic tuples/URLs and any parser branch needed for the new ecosystem include; do not loosen exact equality or safe errors.

- [ ] **Step 3: Capture fresh raw evidence with exact commands**

  In a clean, network-capable Python 3.11 audit environment, retain all six physical JSON outputs and command metadata under the ignored `.superpowers/sdd/issue62-advisory/` directory. Run:

  ```bash
  mkdir -p .superpowers/sdd/issue62-advisory
  python -m pip_audit -r requirements.txt -r torch-audit-requirements.txt --strict --vulnerability-service pypi --format json --aliases on --desc off --progress-spinner off --output .superpowers/sdd/issue62-advisory/combined-runtime.json
  python -m pip_audit --disable-pip --no-deps -r pyg-extension-audit-requirements.txt --strict --vulnerability-service pypi --format json --aliases on --desc off --progress-spinner off --output .superpowers/sdd/issue62-advisory/combined-runtime-pyg-extensions.json
  python -m pip_audit -r torch-audit-requirements.txt --strict --vulnerability-service pypi --format json --aliases on --desc off --progress-spinner off --output .superpowers/sdd/issue62-advisory/torch.json
  python -m pip_audit --disable-pip --no-deps -r pyg-extension-audit-requirements.txt --strict --vulnerability-service pypi --format json --aliases on --desc off --progress-spinner off --output .superpowers/sdd/issue62-advisory/torch-pyg-extensions.json
  python -m pip_audit --disable-pip -r docs-requirements.txt --strict --vulnerability-service pypi --format json --aliases on --desc off --progress-spinner off --output .superpowers/sdd/issue62-advisory/documentation.json
  python -m pip_audit --disable-pip -r atlas-contract-requirements.txt --strict --vulnerability-service pypi --format json --aliases on --desc off --progress-spinner off --output .superpowers/sdd/issue62-advisory/atlas-contract.json
  ```

  Merge `combined-runtime-pyg-extensions.json` only into `combined-runtime.json` and `torch-pyg-extensions.json` only into `torch.json`, matching the executable's six-command/four-logical-surface contract. Write `.superpowers/sdd/issue62-advisory/commands.txt` with Python/pip-audit versions and the six redacted command templates. Reconcile aliases into primary identities using the existing normalizer; record SHA-256 for every input/JSON, resolved counts/versions, raw records, unique identities, retained/removed/re-keyed/new findings, fix versions, risk rationale, and revisit triggers. Do not describe a missing feed record as remediation.

- [ ] **Step 4: Capture policy/ledger RED then update both atomically**

  Run `make audit-advisories` before edits and preserve the expected accepted-version/new/removed diagnostics. Update `security/accepted-advisories.json` to the exact observed identities/surfaces and canonical ordering. Archive the old current ledger as historical Issue #59/#61 evidence with an explicit non-remediation disclaimer; add one new current section with exact counts, hashes, versions, Lightning 2.6.1 rationale, residual Torch risk, and pyg-lib's external-index limitation.

- [ ] **Step 5: Harden D10 and documentation contracts**

  Update the existing `verify_repo._dependency_ledger_findings` parser only where the current heading/table/version contract changes. Keep code masking before parsing and compute one `current_section` slice from the single semantic current heading to the next same/higher-level heading. Parse exactly one Result line, one summary table, and one advisory table from that slice, then compare both directions with `load_baseline()` and the recorded hashes:

  ```python
  current_section = _dependency_current_section(masked_markdown)
  result = _require_single_dependency_result(current_section)
  summary_rows = _parse_dependency_summary_table(current_section)
  advisory_rows = _parse_dependency_advisory_table(current_section)
  findings.extend(_compare_dependency_policy_and_ledger(policy, result, summary_rows, advisory_rows))
  findings.extend(_compare_recorded_dependency_hashes(repo, current_section))
  ```

  Add RED fixtures that duplicate the current heading/Result/row, remove or alter each header/separator, hide the only structures in fences/raw HTML, add an advisory-only package, mutate one selected version/hash/count, and claim zero vulnerabilities. Each mutation must yield a named D10 finding; the real ledger and an extra historical table must pass.

- [ ] **Step 6: Prove GREEN and commit**

  ```bash
  make audit-advisories
  pytest -p no:cacheprovider tests/test_advisory_baseline.py tests/test_verify_repo.py tests/test_check_docs.py -q -k 'advisory or dependency or d10 or issue_62'
  python scripts/verify_repo.py --check docs --fast
  ruff check scripts/advisory_baseline.py scripts/verify_repo.py tests/test_advisory_baseline.py tests/test_verify_repo.py tests/test_check_docs.py
  git diff --check
  git add scripts/advisory_baseline.py scripts/verify_repo.py security/accepted-advisories.json docs/dependency-contracts.md tests/test_advisory_baseline.py tests/test_verify_repo.py tests/test_check_docs.py docs/superpowers/plans/2026-08-14-issue-62-torch-stack-upgrade-implementation-plan.md
  git commit -m "security: reconcile Torch 2.11 advisory policy"
  ```

---

## 12.22.9 Task 6: Synchronize current operational documentation across all surfaces

**Files:**
- Modify: `README.md`
- Modify: `CONTRIBUTING.md`
- Modify: `CHANGELOG.md`
- Modify: `docs/env-setup.md`
- Modify: `docs/architecture.md`
- Modify: `docs/FINDINGS-ATLAS.md`
- Modify: `docs/notebooks/pruning-mnist-ffnn-pytorch.md`
- Modify: `docs/dependency-contracts.md`
- Modify: `docs/notebooks/quantization-mnist-ffnn-pytorch.md`
- Modify: `notebooks/quantization-mnist-ffnn-pytorch/README.md`
- Modify: `notebooks/quantization-mnist-ffnn-pytorch/docs/spec.yaml`
- Modify: `tests/test_check_docs.py`
- Modify: `docs/superpowers/plans/2026-08-14-issue-62-torch-stack-upgrade-implementation-plan.md`

**Interfaces:**
- Consumes: the implemented matrix, verifier, CI/Docker contract, fresh advisory evidence, and unchanged Atlas ownership/tier map.
- Produces: one current operational story in repository Markdown, generated site, and generated wiki; durable Unreleased history; no historical rewrite.

- [ ] **Step 1: Write strict current-vs-historical RED documentation tests**

  Scope assertions to current README/contributor/architecture/env/dependency/pruning/quantization/spec sections, current Makefile/CI/devcontainer comments, and the Unreleased changelog prefix. Require every selected version, supported platform, exact install/verifier command, CPU-only Linux rule, source-built spline rule, manual-only quantization boundary, Issue #66 ownership, Atlas Issue #65 ownership, and atomic rollback instruction. Reject current claims that Torch is 2.4.1, torchao is ranged, PyG backends are unavailable on Darwin arm64, quantization imports are blocked, the notebook is tier-covered, Atlas was upgraded, or the upgrade has passed before Task 7 evidence exists.

- [ ] **Step 2: Run RED**

  ```bash
  pytest -p no:cacheprovider tests/test_check_docs.py -q -k 'torch or quantization or dependency or atlas or issue_62'
  ```

  Expected: current pages still describe Torch 2.4.1 and backend-skippable quantization.

- [ ] **Step 3: Update current operational prose only**

  Document the selected matrix, `make install-torch-stack`, `python -m pip check`, `make verify-torch-stack`, Linux/Darwin/Docker architecture rules, PyG wheel/source split, qualified PTQ/QAT surface, manual-only notebook, residual advisory risk, and atomic fresh-environment rollback. Keep Atlas as the default remote notebook runtime and an independently owned image; explicitly route its future pin work to #65. Add one Unreleased changelog entry without editing released sections.

  Before Task 7 completes, use the exact status sentence: `The dependency and focused runtime contracts are implemented; complete Tier A/B/C and container acceptance evidence is pending.` Do not claim full acceptance yet.

- [ ] **Step 4: Generate and inspect all three documentation surfaces**

  ```bash
  make docs-check
  make docs-wiki
  NO_MKDOCS_2_WARNING=1 mkdocs build --strict
  python scripts/verify_repo.py --check docs --fast
  ```

  Compare the changed canonical pages with `generated/site/` and `generated/wiki/`; verify selected versions, commands, manual-only boundary, Atlas ownership, residual-risk language, and pending-acceptance sentence on each applicable surface. Confirm generated trees remain ignored/untracked.

- [ ] **Step 5: Prove GREEN, historical immutability, and commit**

  ```bash
  pytest -p no:cacheprovider tests/test_check_docs.py tests/test_manifest.py tests/test_transforms.py tests/test_build_docs.py tests/test_wiki.py -q
  ruff check tests/test_check_docs.py
  git diff --check
  git diff --name-only df0652b48554b1c3562177ce9e3ee28084be88cf -- docs/superpowers/specs docs/superpowers/plans docs/maintenance
  git add README.md CONTRIBUTING.md CHANGELOG.md docs/architecture.md docs/env-setup.md docs/FINDINGS-ATLAS.md docs/dependency-contracts.md docs/notebooks/pruning-mnist-ffnn-pytorch.md docs/notebooks/quantization-mnist-ffnn-pytorch.md notebooks/quantization-mnist-ffnn-pytorch/README.md notebooks/quantization-mnist-ffnn-pytorch/docs/spec.yaml tests/test_check_docs.py docs/superpowers/plans/2026-08-14-issue-62-torch-stack-upgrade-implementation-plan.md
  git commit -m "docs: document coordinated Torch 2.11 runtime"
  ```

  Inspect the final staged diff rather than treating the grep as the sole immutability proof. Any historical file change outside the Issue #62 plan/design must be justified by a current broken link; otherwise revert it.

---

## 12.22.10 Task 7: Qualify the exact feature SHA, integrate through GitFlow, and clean up

**Files:**
- Modify: `docs/superpowers/plans/2026-08-14-issue-62-torch-stack-upgrade-implementation-plan.md`
- Write ignored evidence: `.superpowers/sdd/issue62-qualification-report.md`
- No notebook source/output, Atlas, or generated documentation file may change.

**Interfaces:**
- Consumes: Tasks 1-6, a reviewed prequalification candidate, and then one immutable final feature SHA.
- Produces: clean Darwin arm64, Linux arm64 Docker, Linux x86_64 CI, advisory, full-test, Tier A/B/C, docs, and GitFlow evidence attached externally/ignored after the final freeze; merged develop/main; closed issue/project item; cleaned branches/PRs/worktrees/services.

- [ ] **Step 1: Create the clean prequalification worktree, venv, and kernel**

  Commit all Tasks 1-6 changes and record the prequalification SHA. From the main worktree run these exact commands, replacing no path with a network URL:

  ```bash
  ISSUE62_ROOT=$(mktemp -d /private/tmp/ml-eng-lab-issue62-prequal.XXXXXX)
  ISSUE62_SHA=$(git rev-parse HEAD)
  git worktree add --detach "$ISSUE62_ROOT/worktree" "$ISSUE62_SHA"
  git -C "$ISSUE62_ROOT/worktree" -c submodule.infra.url=/Users/kaveh/repos/ml-eng-lab/infra submodule update --init --recursive infra
  python3.11 -m venv "$ISSUE62_ROOT/venv"
  export PATH="$ISSUE62_ROOT/venv/bin:$PATH"
  export JUPYTER_PATH="$ISSUE62_ROOT/jupyter/share/jupyter"
  cd "$ISSUE62_ROOT/worktree"
  git status --porcelain=v1
  git diff --check
  git submodule status infra
  find notebooks -type d \( -name runs -o -name checkpoints \) -print
  ```

  Require empty status/diff/find output and a leading-space exact `61c7c5103660e2226bf107c115dae42bf46f8374` submodule status. Install no editable package and use no global kernelspec path.

- [ ] **Step 2: Run the complete local dependency and repository gate**

  With the venv activated through `PATH` rather than `make PYTHON=...`, run in order:

  ```bash
  make install-torch-stack
  python -m pip install -r docs-requirements.txt
  make nlp-assets
  python -m ipykernel install --prefix "$ISSUE62_ROOT/jupyter" --name python3 --display-name "Issue 62 Python 3"
  python -m jupyter kernelspec list --json
  python -m pip check
  make verify-torch-stack
  make verify-nnx-install
  make audit-advisories
  make test-nnx-surface
  make test
  make lint
  make verify
  make docs-check
  make docs-wiki
  git diff --check
  ```

  Require `python3` kernelspec argv equal to `$ISSUE62_ROOT/venv/bin/python`. Record exact versions, WHEEL/RECORD/direct-url state, architecture, pass/skip/warning counts, durations, and hashes. Any compiled-PyG or quantization skip is a failure.

- [ ] **Step 3: Qualify native Linux arm64 Docker on Apple Silicon**

  Build without `--platform` emulation, inspect the image architecture as `arm64`, and run only non-service probes:

  ```bash
  docker build --no-cache -t ml-eng-lab:issue62-arm64 .
  docker image inspect ml-eng-lab:issue62-arm64 --format '{{.Architecture}}'
  docker run --rm ml-eng-lab:issue62-arm64 python -m pip check
  docker run --rm ml-eng-lab:issue62-arm64 python -m scripts.verify_torch_stack
  ```

  Require the Dockerfile's internal sampler probe to have passed during build. Do not start Jupyter, Compose, or any daemon.

- [ ] **Step 4: Run the full notebook matrix from clean source directories**

  Use unique absolute output roots and the isolated kernelspec. Apply `JUPYTER_PATH` to every papermill command and run the executable artifact oracle after every tier:

  ```bash
  JUPYTER_PATH="$JUPYTER_PATH" TIER_A_OUT="$ISSUE62_ROOT/tier-a" make smoke-tier-a
  TIER_A_OUT="$ISSUE62_ROOT/tier-a" make check-tier-a-artifacts
  python -m scripts.verify_smoke_outputs --tier a --root "$ISSUE62_ROOT/tier-a"
  make check-tier-a-clean
  JUPYTER_PATH="$JUPYTER_PATH" SMOKE_OUT="$ISSUE62_ROOT/tier-b" make smoke-tier-b
  python -m scripts.verify_smoke_outputs --tier b --root "$ISSUE62_ROOT/tier-b"
  JUPYTER_PATH="$JUPYTER_PATH" SMOKE_OUT="$ISSUE62_ROOT/tier-c" make smoke-tier-c
  python -m scripts.verify_smoke_outputs --tier c --root "$ISSUE62_ROOT/tier-c"
  git status --porcelain=v1
  git diff --check
  git submodule status infra
  ```

  Require Tier A 18/18, Tier B 6/6, Tier C 4/4, exactly 28 fresh nonempty outputs, complete code-cell execution counts, zero error outputs, unchanged source notebooks, and unchanged tracked files/infra gitlink. Stop at the first failure; diagnose and fix only through a separately reviewed TDD correction commit, then restart the complete affected matrix from a fresh worktree/environment.

- [ ] **Step 5: Commit prequalification evidence and finish broad review before the final freeze**

  After Steps 1-4 are fully green, label that evidence `prequalification` rather than final acceptance. Replace the Task 6 pending sentence with exact candidate SHA/platform/test/tier/Docker results and the durable rule: `Merge acceptance additionally requires an immutable final-SHA rerun attached to Issue #62; this tracked candidate record is not a substitute for that external evidence.` This sentence remains true before and after merge and avoids a stale tracked “pending” flag. Append the prequalification evidence to this plan, regenerate site/wiki, rerun `tests/test_check_docs.py`, `make docs-check`, `make docs-wiki`, `make verify`, Ruff, and diff checks, then commit:

  ```bash
  git add README.md CONTRIBUTING.md CHANGELOG.md docs/env-setup.md docs/FINDINGS-ATLAS.md docs/dependency-contracts.md docs/notebooks/quantization-mnist-ffnn-pytorch.md notebooks/quantization-mnist-ffnn-pytorch/README.md tests/test_check_docs.py docs/superpowers/plans/2026-08-14-issue-62-torch-stack-upgrade-implementation-plan.md
  git commit -m "docs: record Torch 2.11 qualification evidence"
  ```

  Review the complete branch against Issue #62, the design, this plan, every commit, current/historical documentation, manifest source authority, safe diagnostics, mutation resistance, platform claims, advisory parity, Docker, notebook cleanliness, Atlas non-diff, and rollback atomicity. Resolve every Critical/Important/Minor finding with RED-GREEN evidence and separate commits. Repeat until the reviewer reports zero findings.

- [ ] **Step 6: Freeze and qualify the immutable final feature SHA**

  Only after broad review reports zero findings, commit any last tracked correction and freeze `FINAL_SHA=$(git rev-parse HEAD)`. Create a second fresh worktree/venv/kernel and repeat the complete qualification without changing tracked files afterward:

  ```bash
  FINAL_ROOT=$(mktemp -d /private/tmp/ml-eng-lab-issue62-final.XXXXXX)
  FINAL_SHA=$(git rev-parse HEAD)
  git worktree add --detach "$FINAL_ROOT/worktree" "$FINAL_SHA"
  git -C "$FINAL_ROOT/worktree" -c submodule.infra.url=/Users/kaveh/repos/ml-eng-lab/infra submodule update --init --recursive infra
  python3.11 -m venv "$FINAL_ROOT/venv"
  export PATH="$FINAL_ROOT/venv/bin:$PATH"
  export JUPYTER_PATH="$FINAL_ROOT/jupyter/share/jupyter"
  cd "$FINAL_ROOT/worktree"
  make install-torch-stack
  python -m pip install -r docs-requirements.txt
  make nlp-assets
  python -m ipykernel install --prefix "$FINAL_ROOT/jupyter" --name python3 --display-name "Issue 62 Final Python 3"
  python -m jupyter kernelspec list --json
  python -m pip check
  make verify-torch-stack
  make verify-nnx-install
  make audit-advisories
  make test-nnx-surface
  make test
  make lint
  make verify
  make docs-check
  make docs-wiki
  docker build --no-cache -t ml-eng-lab:issue62-final-arm64 .
  test "$(docker image inspect ml-eng-lab:issue62-final-arm64 --format '{{.Architecture}}')" = arm64
  docker run --rm ml-eng-lab:issue62-final-arm64 python -m pip check
  docker run --rm ml-eng-lab:issue62-final-arm64 python -m scripts.verify_torch_stack
  JUPYTER_PATH="$JUPYTER_PATH" TIER_A_OUT="$FINAL_ROOT/tier-a" make smoke-tier-a
  TIER_A_OUT="$FINAL_ROOT/tier-a" make check-tier-a-artifacts
  python -m scripts.verify_smoke_outputs --tier a --root "$FINAL_ROOT/tier-a"
  make check-tier-a-clean
  JUPYTER_PATH="$JUPYTER_PATH" SMOKE_OUT="$FINAL_ROOT/tier-b" make smoke-tier-b
  python -m scripts.verify_smoke_outputs --tier b --root "$FINAL_ROOT/tier-b"
  JUPYTER_PATH="$JUPYTER_PATH" SMOKE_OUT="$FINAL_ROOT/tier-c" make smoke-tier-c
  python -m scripts.verify_smoke_outputs --tier c --root "$FINAL_ROOT/tier-c"
  test "$(git rev-parse HEAD)" = "$FINAL_SHA"
  git status --porcelain=v1
  git diff --check
  git submodule status infra
  ```

  Record final evidence only in `.superpowers/sdd/issue62-qualification-report.md` and the GitHub issue/PR; do not make another tracked commit. Any final failure or later tracked commit invalidates the freeze: diagnose through a separate TDD/review commit, return to this step with a new fresh environment, and rerun every command above.

- [ ] **Step 7: Open and qualify the feature-to-develop PR**

  Push the final feature SHA and open a ready PR to `develop` that closes #62 only after release integration. Add the `tier-b-smoke` label. Wait for every required/diagnostic job, including Linux x86_64 Docker, to pass. Manually dispatch Tier C on the exact feature SHA and attach Tier A/B/C, Docker, advisory, verifier, test, and review evidence to Issue #62. Do not merge on pending, skipped, neutral, cancelled, stale-SHA, or rerun-masking results.

- [ ] **Step 8: Merge through develop and main**

  Merge the feature PR into `develop` via GitHub. Open a separate `develop` to `main` PR, require all checks and content review, and merge it. Verify published Pages and wiki current pages return HTTP 200 and contain the Torch 2.11 matrix/current evidence. Compare `main` and `develop` trees; if merge commits leave content drift, use a protected `main` to `develop` sync PR and reverify equality.

- [ ] **Step 9: Close Issue #62 and clean all state**

  Close Issue #62 with exact merged PRs/SHAs/evidence, move its project item to Done, and confirm #65/#66 remain separately open. Delete the local and remote feature branch, remove temporary worktrees/environments/images/artifacts, close obsolete PRs, prune stale remote refs, and verify no repository-owned container/service is running. Do not touch unrelated containers. End with clean `main` and `develop`, no dangling Issue #62 branches/worktrees/PRs, exact Atlas gitlink, and no active Issue #62 workflow runs.

---

## 12.22.11 Plan Self-Review Gate

- [x] Map every design section 12.21.2-12.21.16 to at least one task and record the mapping in the Task 7 report.
- [x] Search this plan for unfinished-work markers, vague deferred instructions, placeholder ellipses, undefined interfaces, stale versions, and claims of completed qualification; the search must return no plan defect.
- [x] Verify every function/type/target/file introduced in one task has the identical name in later tasks.
- [x] Verify generated documentation paths are absent from every `git add` command.
- [x] Verify no task edits `infra`, `.gitmodules`, Reddit notebook code, quantization notebook outputs, tier lists, Atlas services, or protected-branch rules.
- [x] Verify the rollback boundary includes manifests, installer, verifier, CI/Docker, advisory policy/ledger, current docs, and a fresh environment/image while preserving durable history.
