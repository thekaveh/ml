# 12.22 Issue 62 Coordinated Torch Stack Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the repository's Torch 2.4 runtime with one exact, platform-qualified Torch 2.11, PyG, Lightning, TorchMetrics, and torchao contract while retaining only the compiled graph packages used by active consumers.

**Architecture:** Five canonical manifests define one matrix. A four-stage installer applies it, a fail-closed verifier proves ten selected distributions plus NNx 0.2.0, and CI, Docker, Codespaces, advisory policy, notebook gates, and documentation consume those same contracts. The supported compiled PyG boundary is exactly pyg-lib, torch-scatter, and torch-sparse; torch-cluster and torch-spline-conv are removed from current runtime, audit, verifier, and documentation surfaces.

**Tech Stack:** Python 3.11, pip, GNU Make, PyTorch 2.11.0, TorchVision 0.26.0, TorchAudio 2.11.0, PyTorch Lightning 2.6.1, TorchMetrics 1.9.0, torchao 0.18.0, torch-geometric 2.8.0.post1, pyg-lib 0.8.0, torch-scatter 2.1.2, torch-sparse 0.6.18, thekaveh-nnx 0.2.0, pytest, Ruff, Docker, GitHub Actions, papermill, pip-audit, and MkDocs.

**Spec:** `docs/superpowers/specs/2026-08-14-issue-62-torch-stack-upgrade-design.md`

## 12.22.1 Global Constraints

- Use Python 3.11 and exact pins: Torch 2.11.0, TorchVision 0.26.0, TorchAudio 2.11.0, PyTorch Lightning 2.6.1, TorchMetrics 1.9.0, torchao 0.18.0, torch-geometric 2.8.0.post1, pyg-lib 0.8.0, torch-scatter 2.1.2, and torch-sparse 0.6.18.
- Remove torch-cluster and torch-spline-conv from every current runtime manifest, audit supplement, installer flag, verifier map, canary set, CI/Docker contract, architecture claim, and operational document. Their historical r1-r3 evidence remains immutable.
- Keep `thekaveh-nnx[lm]==0.2.0`, `.gitmodules`, the `infra` gitlink at `61c7c5103660e2226bf107c115dae42bf46f8374`, Atlas requirements/images/Compose, and the `ml-eng` track unchanged. Issue #65 owns Atlas changes.
- Support Linux x86_64 CPU and Darwin arm64. Qualify native Linux arm64 through Docker on Apple Silicon. Do not claim CUDA, ROCm, XPU, Windows, direct-host Linux arm64, or source-built Torch support.
- Linux installs the Torch trio from `https://download.pytorch.org/whl/cpu`; Darwin uses native PyPI wheels. Linux must have `torch.version.cuda is None` and no normalized installed distribution beginning `nvidia-`.
- The runtime resolver must use `--only-binary=pyg-lib,torch-scatter,torch-sparse`. Do not bootstrap wheel, use `--no-binary`, use `--no-build-isolation`, or build a selected graph package from source.
- Keep `notebooks/quantization-mnist-ffnn-pytorch/notebook.ipynb` manual-only and outside Tier A/B/C. Issue #66 owns its execution, output refresh, thresholds, and tier promotion.
- Do not edit Reddit notebook code, notebook outputs, execution counts, metadata, or tier membership for dependency accommodation.
- Do not initialize or start Atlas, JupyterHub, Docker Compose, Ollama, ComfyUI, or any repository service. Containerized Ollama is prohibited. Docker use is limited to the root image build and non-service probes in Task 7.
- Keep historical Issue #59/#60/#61 records, released changelog entries, maintenance logs, Atlas evidence, and committed notebook results immutable. Add current Issue #62 truth without rewriting history.
- Generate site/wiki derivatives from canonical sources; never edit or commit `generated/`, `site/`, or root `mkdocs.yml`.
- Any resolver, wheel, ABI, sampler, quantization, advisory, Docker, notebook, documentation, or protected-branch failure is a stop condition. Correct it through a reviewed RED-GREEN commit and rerun the affected clean matrix.

---

## 12.22.2 Final File Map and Interfaces

### 12.22.2.1 Dependency and installation authority

- `torch-core-requirements.txt`: exact Torch/TorchVision/TorchAudio trio only.
- `torch-ecosystem-requirements.txt`: exact Lightning/TorchMetrics/torchao trio only.
- `torch-requirements.txt`: ecosystem include, exact Torch 2.11 CPU PyG selector, pyg-lib/scatter/sparse, and torch-geometric.
- `torch-audit-requirements.txt`: selector-free core, ecosystem, and torch-geometric resolver projection.
- `pyg-extension-audit-requirements.txt`: pre-resolved PyPI supplement containing only scatter and sparse; pyg-lib remains external-index provenance evidence.
- `requirements.txt`: unchanged NNx 0.2.0 root contract and no second torchao constraint.
- `scripts/install_torch_stack.py`: sole four-stage platform-aware installer.
- `tests/test_install_torch_stack.py`: exact manifest, argv, mutation, platform, and safe-failure contract.
- `scripts/advisory_baseline.py` and `tests/test_advisory_baseline.py`: exact manifest projections and six-command/four-surface audit model.
- `Makefile`: `install-torch-stack` is exactly `$(PYTHON) -m scripts.install_torch_stack`; Codespaces adds only NLP assets afterward.

### 12.22.2.2 Installed-runtime authority

- `scripts/verify_torch_stack.py`: exact ten-component metadata/import/wheel/provenance/CPU verifier, three mandatory canaries, and final NNx delegation.
- `tests/test_verify_torch_stack_platform.py`: supported-platform and wheel-tag contract.
- `tests/test_verify_repo.py`: runtime availability plus CI/Docker/order/cache/service contracts.
- `tests/test_verify_torch_stack.py`: existing Task 3 work-in-progress file; Task 3 reconciles it with the final ten-component verifier and owns its commit.
- `tests/nnx_surface/conftest.py`: one stack-verifier call before NNx verification/import.
- `tests/nnx_surface/test_node_classification_reddit_gnn_pyg.py`: mandatory pyg-lib/sparse NeighborLoader and SAGE/GraphConv/GAT execution.
- `tests/nnx_surface/test_quantization_mnist_ffnn_pytorch.py`: mandatory tiny PTQ/QAT execution.
- `scripts/verify_smoke_outputs.py` and `tests/test_verify_smoke_outputs.py`: exact Tier A/B/C output oracle.
- `scripts/verify_junit.py` and `tests/test_verify_junit.py`: reusable positive-test, zero-failure/error/skip JUnit gate used by focused, CI, and final NNx acceptance.

The final public verifier interfaces are:

```python
@dataclass(frozen=True)
class StackPin:
    distribution: str
    import_name: str
    public_version: Version

@dataclass(frozen=True)
class StackContract:
    pins: tuple[StackPin, ...]
    system: str
    machine: str

@dataclass(frozen=True)
class StackEvidence:
    system: str
    machine: str
    torch_version: str
    backend: str

Canary = Callable[[Mapping[str, ModuleType]], None]

class DistributionView(Protocol):
    version: str
    files: Sequence[PackagePath] | None

    def read_text(self, filename: str) -> str | None:
        raise NotImplementedError

@dataclass(frozen=True)
class CanaryHooks:
    scatter: Canary
    sparse: Canary
    sampler: Canary

@dataclass(frozen=True)
class VerificationHooks:
    distribution: Callable[[str], DistributionView]
    installed_names: Callable[[], Iterable[str]]
    import_module: Callable[[str], ModuleType]
    system: Callable[[], str]
    machine: Callable[[], str]
    nnx_verify: Callable[[], object]
    canaries: CanaryHooks
```

The final smoke-output interfaces are:

```python
Tier = Literal["a", "b", "c"]

class InventoryLoader(Protocol):
    def __call__(self, tier: Tier) -> Sequence[str]:
        raise NotImplementedError

@dataclass(frozen=True)
class NotebookArtifact:
    source: str
    output: Path
    code_cells: int

def verify_smoke_outputs(
    tier: Tier,
    root: Path,
    inventory_loader: InventoryLoader = load_make_inventory,
) -> tuple[NotebookArtifact, ...]:
    raise NotImplementedError
```

### 12.22.2.3 Integration, audit, documentation, and evidence

- `.github/workflows/ci.yml`, `Dockerfile`, `.devcontainer/devcontainer.json`, `Makefile`, and `tests/test_verify_repo.py`: one install algorithm, complete cache inputs, no late install, no service startup.
- `security/accepted-advisories.json`, `docs/dependency-contracts.md`, `scripts/verify_repo.py`, `tests/test_advisory_baseline.py`, `tests/test_verify_repo.py`, and `tests/test_check_docs.py`: current audit policy, ledger, D10 enforcement, and the Issue #61 requirements-hash correction.
- `README.md`, `CONTRIBUTING.md`, `SECURITY.md`, `CHANGELOG.md`, `docs/env-setup.md`, `docs/architecture.md`, `docs/FINDINGS-ATLAS.md`, `docs/dependency-contracts.md`, `docs/notebook-infrastructure.md`, `docs/notebooks/pruning-mnist-ffnn-pytorch.md`, `docs/notebooks/quantization-mnist-ffnn-pytorch.md`, `notebooks/node_classification-reddit-gnn-pyg/README.md`, `notebooks/quantization-mnist-ffnn-pytorch/README.md`, `notebooks/quantization-mnist-ffnn-pytorch/docs/spec.yaml`, `docs/assets/badges/pytorch.svg`, and current Make/CI/Docker/devcontainer comments: one operational story.
- `.superpowers/sdd/issue62-advisory/`: ignored six-command audit evidence.
- `.superpowers/sdd/issue62-qualification-report.md`: ignored immutable-final-SHA evidence.

---

## 12.22.3 Historical Execution State and Restart Rule

The branch preserves implementation history through `1ca1bd8ffeede71151eba71a597cb9e9bfc144ae`. Commits `cefb8e0`, `abe4b6d`, `45b58dc`, `dcea2e9`, `78e23d2`, `162d403`, `ea0c25b`, `f5cc681`, `9a03b7a`, and `018e214` remain evidence of the former five-extension/source-build design. They are not final Task 1 or Task 2 completion under the approved design at `1ca1bd8`.

Task 1 and Task 2 are reopened below. Their checkboxes are unchecked because final-state code and tests still contain wheel bootstrap, source-build flags, two legacy pins, twelve verifier components, and five canaries.

Task 3 has five legitimate uncommitted files. Before Task 1 begins, record and require these exact SHA-256 values:

```text
d0dadde0c1de06227a88e5151a3f74c91b791c93e2b4fc1200cbc9424c34778b  tests/nnx_surface/conftest.py
19394ee43f2c1ba5e7e5d836436803dbd317008e6643730d8185304bf7ba70a2  tests/nnx_surface/test_node_classification_reddit_gnn_pyg.py
d5ee4ae82159cb2a576d8fb2421f68dc8db3107832399d81e985eff1396d9a11  tests/nnx_surface/test_quantization_mnist_ffnn_pytorch.py
d94958579ca5b920b1e50222dc3a1ee09a8d6ba25fcebd7ce23237462a337dda  tests/test_makefile_contract.py
8d723345898d9af461dd78f6ecfe3dcf0b7ef7a332d80046953f3850d3a343ad  tests/test_verify_torch_stack.py
```

Task 1 and Task 2 must not edit, stage, or commit those files. Their focused gates use other test files. Task 3 adapts and commits all five only after the new clean matrix passes. At each Task 1/2 commit gate run:

```bash
shasum -a 256 tests/nnx_surface/conftest.py tests/nnx_surface/test_node_classification_reddit_gnn_pyg.py tests/nnx_surface/test_quantization_mnist_ffnn_pytorch.py tests/test_makefile_contract.py tests/test_verify_torch_stack.py
git diff --cached --name-only
```

Expected: the five hashes equal the block above and none of those paths appears in the staged list.

---

## 12.22.4 Task 1: Reopen the manifests and installer at the supported PyG boundary

**Files:**
- Modify: `torch-requirements.txt`
- Modify: `pyg-extension-audit-requirements.txt`
- Modify: `scripts/install_torch_stack.py`
- Modify: `scripts/advisory_baseline.py`
- Modify: `Makefile`
- Modify: `tests/test_install_torch_stack.py`
- Modify: `tests/test_advisory_baseline.py`
- Verify unchanged: `torch-core-requirements.txt`
- Verify unchanged: `torch-ecosystem-requirements.txt`
- Verify unchanged: `torch-audit-requirements.txt`
- Verify unchanged: `requirements.txt`

**Interfaces:**
- Consumes: exact versions in Global Constraints and supported host pairs `("Linux", "x86_64")`, `("Linux", "aarch64")`, and `("Darwin", "arm64")`.
- Produces: unchanged `InstallStage`, `InstallCommand`, `CommandResult`, `CommandRunner`, `build_install_commands`, `install_torch_stack`, and CLI names; exact final manifest bytes; exact four-stage argv; exact advisory projection constants.
- Commit ownership: only files listed as Modify in this task. The five Task 3 work-in-progress files remain byte-identical and unstaged.

- [ ] **Step 1: Write the final exact-manifest RED oracle**

  Replace the manifest constants in `tests/test_install_torch_stack.py` with:

  ```python
  CORE = "torch==2.11.0\ntorchvision==0.26.0\ntorchaudio==2.11.0\n"
  ECOSYSTEM = (
      "pytorch-lightning==2.6.1\n"
      "torchmetrics==1.9.0\n"
      "torchao==0.18.0\n"
  )
  RUNTIME_MANIFEST = (
      "-r torch-ecosystem-requirements.txt\n"
      "--find-links https://data.pyg.org/whl/torch-2.11.0+cpu.html\n"
      "pyg-lib==0.8.0\n"
      "torch-scatter==2.1.2\n"
      "torch-sparse==0.6.18\n"
      "torch_geometric==2.8.0.post1\n"
  )
  AUDIT = (
      "-r torch-core-requirements.txt\n"
      "-r torch-ecosystem-requirements.txt\n"
      "torch_geometric==2.8.0.post1\n"
  )
  SUPPLEMENT = (
      "# Pre-resolved compiled PyG extension supplement for the strict audit.\n"
      "# Runtime source: torch-requirements.txt retains the approved PyG wheel selector.\n"
      "torch-scatter==2.1.2\n"
      "torch-sparse==0.6.18\n"
  )
  ```

  Assert exact bytes for all five manifests and assert `requirements.txt` contains `thekaveh-nnx[lm]==0.2.0` exactly once and contains no line whose normalized package name is torchao.

- [ ] **Step 2: Write the final installer argv RED oracle**

  Define these independent expectations in `tests/test_install_torch_stack.py`:

  ```python
  UPGRADE_PIP = (sys.executable, "-m", "pip", "install", "--upgrade", "pip")
  LINUX_CORE = (
      sys.executable, "-m", "pip", "install", "--index-url",
      "https://download.pytorch.org/whl/cpu", "-r", "torch-core-requirements.txt",
  )
  DARWIN_CORE = (
      sys.executable, "-m", "pip", "install", "-r", "torch-core-requirements.txt",
  )
  RUNTIME = (
      sys.executable, "-m", "pip", "install",
      "--only-binary=pyg-lib,torch-scatter,torch-sparse",
      "-r", "torch-requirements.txt",
  )
  ROOT = (
      sys.executable, "-m", "pip", "install",
      "--only-binary=thekaveh-nnx", "-r", "requirements.txt",
  )

  def test_final_four_stage_plans_are_exact():
      for system, machine, core in (
          ("Linux", "x86_64", LINUX_CORE),
          ("Linux", "aarch64", LINUX_CORE),
          ("Darwin", "arm64", DARWIN_CORE),
      ):
          commands = build_install_commands(sys.executable, system, machine)
          assert tuple(item.stage for item in commands) == tuple(InstallStage)
          assert tuple(item.argv for item in commands) == (
              UPGRADE_PIP, core, RUNTIME, ROOT,
          )
  ```

  Add a Makefile assertion in this same test module that parses the `install-torch-stack` recipe and requires exactly `$(PYTHON) -m scripts.install_torch_stack`; require `codespace-setup` to depend on it and have exactly one recipe, `$(MAKE) nlp-assets`.

- [ ] **Step 3: Add non-vacuous legacy-removal and anti-reintroduction mutations**

  In isolated temporary copies, make each mutation and require the named exact contract test to fail:

  ```python
  LEGACY_MANIFEST_MUTATIONS = (
      ("torch_geometric==2.8.0.post1\n", "torch-cluster==1.6.3\ntorch_geometric==2.8.0.post1\n"),
      ("torch_geometric==2.8.0.post1\n", "torch-spline-conv==1.2.2\ntorch_geometric==2.8.0.post1\n"),
  )
  LEGACY_ARG_MUTATIONS = (
      ("--only-binary=pyg-lib,torch-scatter,torch-sparse", "--only-binary=pyg-lib,torch-scatter,torch-sparse,torch-cluster"),
      ("-r\", \"torch-requirements.txt", "--no-binary=torch-spline-conv\", \"--no-build-isolation\", \"-r\", \"torch-requirements.txt"),
      (("--upgrade", "pip"), ("--upgrade", "pip", "wheel")),
  )
  ```

  Also mutate away one of pyg-lib/scatter/sparse, add `:all:`, reorder stages, move ROOT earlier, remove NNx binary-only, add a fifth command, accept an unsupported platform, or ignore a nonzero runner return. First assert every source mutation changed the source; then import the temporary module and prove the expected tuple or failure category differs.

- [ ] **Step 4: Run RED**

  ```bash
  pytest -p no:cacheprovider tests/test_install_torch_stack.py tests/test_advisory_baseline.py -q -k 'manifest or four_stage or legacy or runtime or makefile or projection'
  ```

  Expected: failures show current cluster/spline pins, wheel bootstrap, and source-build flags remain. No network command runs because runner hooks and temporary files isolate all installer tests.

- [ ] **Step 5: Write the final manifests and installer**

  Make `torch-requirements.txt` and `pyg-extension-audit-requirements.txt` byte-equal to the Step 1 constants. Keep the other three stack manifests and `requirements.txt` byte-identical.

  In `scripts/install_torch_stack.py`, retain the public types and replace only the command tuples:

  ```python
  return (
      InstallCommand(InstallStage.UPGRADE_PIP, pip + ("--upgrade", "pip")),
      InstallCommand(InstallStage.CORE, core),
      InstallCommand(
          InstallStage.RUNTIME,
          pip + (
              "--only-binary=pyg-lib,torch-scatter,torch-sparse",
              "-r",
              "torch-requirements.txt",
          ),
      ),
      InstallCommand(
          InstallStage.ROOT,
          pip + ("--only-binary=thekaveh-nnx", "-r", "requirements.txt"),
      ),
  )
  ```

  Keep `check=False`, stop on the first nonzero return, and emit only `torch stack installation failed: <stage>`.

- [ ] **Step 6: Update advisory projection constants and the Make comment**

  In `scripts/advisory_baseline.py`, use:

  ```python
  TORCH_RUNTIME_LINES = (
      "-r torch-ecosystem-requirements.txt",
      PYG_FIND_LINKS,
      "pyg-lib==0.8.0",
      "torch-scatter==2.1.2",
      "torch-sparse==0.6.18",
      "torch_geometric==2.8.0.post1",
  )
  PYG_EXTENSION_AUDIT_LINES = (
      "torch-scatter==2.1.2",
      "torch-sparse==0.6.18",
  )
  ```

  Update `tests/test_advisory_baseline.py` fixtures and mutations to the same tuple sizes. Update the current Make comment to say the runtime uses three binary PyG wheels and has no source-build fallback. Do not change the stale Issue #61 `requirements.txt` hash assertion here; Task 5 owns that ledger-evidence correction.

- [ ] **Step 7: Prove GREEN and static quality**

  ```bash
  pytest -p no:cacheprovider tests/test_install_torch_stack.py tests/test_advisory_baseline.py -q
  ruff check scripts/install_torch_stack.py scripts/advisory_baseline.py tests/test_install_torch_stack.py tests/test_advisory_baseline.py
  python -m py_compile scripts/install_torch_stack.py scripts/advisory_baseline.py tests/test_install_torch_stack.py tests/test_advisory_baseline.py
  git diff --check
  ```

  Expected: all selected tests pass; Ruff prints `All checks passed!`; compilation and whitespace checks exit 0.

- [ ] **Step 8: Commit only Task 1 ownership**

  ```bash
  git add torch-requirements.txt pyg-extension-audit-requirements.txt scripts/install_torch_stack.py scripts/advisory_baseline.py Makefile tests/test_install_torch_stack.py tests/test_advisory_baseline.py
  git diff --cached --name-only
  git commit -m "deps: narrow Torch 2.11 PyG runtime boundary"
  ```

  Expected staged paths are exactly the seven paths in `git add`. Re-run the five Task 3 hashes from section 12.22.3 before committing.

---

## 12.22.5 Task 2: Reopen the installed-stack verifier at ten selected components

**Files:**
- Modify: `scripts/verify_torch_stack.py`
- Modify: `tests/test_verify_torch_stack_platform.py`
- Modify: `tests/test_verify_repo.py`
- Do not modify or stage: `tests/test_verify_torch_stack.py` until Task 3

**Interfaces:**
- Consumes: Task 1's five final manifests and exact supported platform set.
- Produces: a ten-pin `StackContract`; `CanaryHooks(scatter, sparse, sampler)`; mandatory public/local version, WHEEL/RECORD, import ownership, Python ABI, platform architecture, Linux CPU/NVIDIA, warning-free canary, and NNx-last gates; unchanged CLI and stable error types.
- Commit ownership: the three modified files above only.

- [ ] **Step 1: Write the ten-component RED map in the platform test**

  In `tests/test_verify_torch_stack_platform.py`, define:

  ```python
  EXPECTED_IMPORTS = {
      "torch": "torch",
      "torchvision": "torchvision",
      "torchaudio": "torchaudio",
      "pytorch-lightning": "pytorch_lightning",
      "torchmetrics": "torchmetrics",
      "torchao": "torchao",
      "torch-geometric": "torch_geometric",
      "pyg-lib": "pyg_lib",
      "torch-scatter": "torch_scatter",
      "torch-sparse": "torch_sparse",
  }
  EXPECTED_CANARIES = ("scatter", "sparse", "sampler")

  def test_selected_component_and_canary_boundaries_are_exact():
      assert IMPORTS == EXPECTED_IMPORTS
      assert tuple(CanaryHooks.__dataclass_fields__) == EXPECTED_CANARIES
  ```

  Construct fake distributions for exactly those ten names. The fake WHEEL contains one compatible tag, its file inventory contains exactly one WHEEL and one RECORD in the same `.dist-info`, the RECORD contains the imported module path, and each fake module's `__file__` resolves to that owned path.

- [ ] **Step 2: Add mandatory provenance and ordering RED tests**

  Parameterize mutations that remove WHEEL, remove RECORD, split them across dist-info directories, remove imported-module ownership, change a public version, use a wrong PyG local tag, use `cp310-cp310`, use `py3-none-any` for a binary distribution, use Darwin x86_64, use Linux wheel architecture opposite the host, set `torch.version.cuda`, add `nvidia-cublas-cu12`, raise or warn in each canary, warn in NNx, or return noncanonical NNx evidence. For each of pyg-lib, scatter, and sparse, Linux `Version("<public>")` with no local tag must fail `abi`; Linux accepts only `+pt211cpu`. Darwin arm64 accepts `+pt211` or no local tag, but both cases must still run WHEEL/RECORD, Python ABI/platform, ownership, import, and every canary before NNx.

  Add these explicit local-tag tests and a non-vacuous production-source mutation:

  ```python
  @pytest.mark.parametrize("distribution", ("pyg-lib", "torch-scatter", "torch-sparse"))
  def test_linux_pyg_index_wheels_require_pt211cpu(tmp_path, distribution):
      stack = PlatformStack(tmp_path, "Linux", "x86_64")
      stack.version(distribution, EXPECTED_VERSIONS[distribution])
      with pytest.raises(
          TorchStackVerificationError,
          match=rf"^torch stack verification failed: {distribution}: abi$",
      ):
          verify_torch_stack(repo=REPO_ROOT, hooks=stack.hooks)

  def test_linux_absent_local_rejection_cannot_be_removed(tmp_path):
      source = (REPO_ROOT / "scripts/verify_torch_stack.py").read_text(encoding="utf-8")
      mutated = source.replace(
          'if contract.system == "Linux" and local != expected:',
          'if contract.system == "Linux" and local not in (None, expected):',
          1,
      )
      assert mutated != source
      module = _import_mutated_verifier(tmp_path, mutated)
      stack = PlatformStack(tmp_path, "Linux", "x86_64")
      stack.version("pyg-lib", EXPECTED_VERSIONS["pyg-lib"])
      with pytest.raises(module.TorchStackVerificationError, match=r"pyg-lib: abi$"):
          module.verify_torch_stack(repo=REPO_ROOT, hooks=stack.hooks)
  ```

  Require exact call order:

  ```python
  evidence = verify_torch_stack(repo=REPO_ROOT, hooks=stack.hooks)
  assert stack.calls == ["scatter", "sparse", "sampler", "nnx"]
  assert evidence.backend == "pyg-lib"
  ```

  Add source mutations that delete each canary invocation or move NNx earlier. Require every mutation to fail a named test. After `nnx_verify` succeeds, the only executable statement must be `return evidence`.

- [ ] **Step 3: Add runtime-availability RED coverage to `tests/test_verify_repo.py`**

  Replace `scripts/verify_repo.py`'s legacy canary tuple with this explicit execution-availability
  boundary:

  ```python
  _RUNTIME_AVAILABLE_IMPORTS = (
      "torch",
      "torch_geometric",
      "pyg_lib",
      "torch_scatter",
      "torch_sparse",
  )

  def _runtime_available() -> bool:
      return all(
          importlib.util.find_spec(import_name) is not None
          for import_name in _RUNTIME_AVAILABLE_IMPORTS
      )
  ```

  Add a helper that parses the current manifests and verifier source, then assert:

  ```python
  def _torch_runtime_import_names(repo: Path) -> set[str]:
      tree = ast.parse(
          (repo / "scripts/verify_torch_stack.py").read_text(encoding="utf-8")
      )
      assignment = next(
          node for node in tree.body
          if isinstance(node, ast.Assign)
          and any(isinstance(target, ast.Name) and target.id == "IMPORTS" for target in node.targets)
      )
      imports = ast.literal_eval(assignment.value)
      assert isinstance(imports, dict)
      return set(imports.values())

  def test_issue62_runtime_availability_uses_only_supported_graph_modules():
      required = {"pyg_lib", "torch_scatter", "torch_sparse", "torch_geometric"}
      forbidden = {"torch_cluster", "torch_spline_conv"}
      assert required <= _torch_runtime_import_names(REPO_ROOT)
      assert forbidden.isdisjoint(_torch_runtime_import_names(REPO_ROOT))
      assert verify_repo._RUNTIME_AVAILABLE_IMPORTS == (
          "torch", "torch_geometric", "pyg_lib", "torch_scatter", "torch_sparse",
      )
  ```

  Add mutations that reinsert either forbidden import into the verifier or CI/Docker availability list and require `D10.torch_runtime_contract`. Do not require either legacy module to be importable for repository test collection.

- [ ] **Step 4: Run RED**

  ```bash
  pytest -p no:cacheprovider tests/test_verify_torch_stack_platform.py tests/test_verify_repo.py -q -k 'selected_component or provenance or runtime_canary or nnx or runtime_availability or torch_runtime_contract'
  ```

  Expected: failures report the current twelve-name map, five-canary dataclass, and legacy runtime requirements.

- [ ] **Step 5: Implement the ten-component manifest contract**

  In `scripts/verify_torch_stack.py`, use:

  ```python
  IMPORTS = {
      "torch": "torch",
      "torchvision": "torchvision",
      "torchaudio": "torchaudio",
      "pytorch-lightning": "pytorch_lightning",
      "torchmetrics": "torchmetrics",
      "torchao": "torchao",
      "torch-geometric": "torch_geometric",
      "pyg-lib": "pyg_lib",
      "torch-scatter": "torch_scatter",
      "torch-sparse": "torch_sparse",
  }
  _RUNTIME_NAMES = frozenset(
      ("pyg-lib", "torch-scatter", "torch-sparse", "torch-geometric")
  )
  _EXTENSION_NAMES = frozenset(("torch-scatter", "torch-sparse"))
  _PYG_INDEX_DISTRIBUTIONS = frozenset(
      ("pyg-lib", "torch-scatter", "torch-sparse")
  )
  _COMPILED_DISTRIBUTIONS = _PYG_INDEX_DISTRIBUTIONS

  def _verify_local_version(pin: StackPin, version: Version, contract: StackContract) -> None:
      local = version.local
      if pin.distribution in _PYG_INDEX_DISTRIBUTIONS:
          expected = "pt211cpu" if contract.system == "Linux" else "pt211"
          if contract.system == "Linux" and local != expected:
              raise TorchStackVerificationError(pin.distribution, "abi")
          if contract.system == "Darwin" and local not in (None, expected):
              raise TorchStackVerificationError(pin.distribution, "abi")
          return
      if local is None:
          return
      if pin.distribution in _CORE_NAMES and contract.system == "Linux" and local == "cpu":
          return
      raise TorchStackVerificationError(pin.distribution, "metadata")
  ```

  Keep audit equality bidirectional: core has three pins, ecosystem has three, runtime has four, audit includes core/ecosystem and pins PyG, supplement has scatter/sparse, and authoritative pins equal `IMPORTS` exactly.

- [ ] **Step 6: Implement three warning-free canaries and preserve mandatory metadata gates**

  Delete cluster/spline canaries and error allow-list entries. Define:

  ```python
  @dataclass(frozen=True)
  class CanaryHooks:
      scatter: Canary
      sparse: Canary
      sampler: Canary

  names_and_categories = (
      ("scatter", "operator"),
      ("sparse", "operator"),
      ("sampler", "sampler"),
  )

  def _sampler_canary(modules: Mapping[str, ModuleType]) -> None:
      torch = modules["torch"]
      geometric = modules["torch-geometric"]
      pyg_lib = modules["pyg-lib"]
      torch_sparse = modules["torch-sparse"]
      neighbor_sampler = importlib.import_module("torch_geometric.sampler.neighbor_sampler")
      pyg_calls = 0
      sparse_calls = 0
      original_pyg = pyg_lib.sampler.neighbor_sample
      original_sparse = torch_sparse.neighbor_sample
      original_with_pyg = neighbor_sampler.WITH_PYG_LIB

      def counted_pyg(*args, **kwargs):
          nonlocal pyg_calls
          pyg_calls += 1
          return original_pyg(*args, **kwargs)

      def counted_sparse(*args, **kwargs):
          nonlocal sparse_calls
          sparse_calls += 1
          return original_sparse(*args, **kwargs)

      data = geometric.data.Data(
          x=torch.tensor([[1.0], [2.0], [3.0]]),
          edge_index=torch.tensor([[0, 1, 2, 1], [1, 0, 1, 2]]),
      )
      try:
          pyg_lib.sampler.neighbor_sample = counted_pyg
          torch_sparse.neighbor_sample = counted_sparse
          neighbor_sampler.WITH_PYG_LIB = True
          preferred = next(iter(geometric.loader.NeighborLoader(
              data, num_neighbors=[-1], input_nodes=torch.tensor([0]),
              batch_size=1, shuffle=False, num_workers=0,
          )))
          if pyg_calls != 1 or sparse_calls != 0:
              raise RuntimeError
          neighbor_sampler.WITH_PYG_LIB = False
          fallback = next(iter(geometric.loader.NeighborLoader(
              data, num_neighbors=[-1], input_nodes=torch.tensor([0]),
              batch_size=1, shuffle=False, num_workers=0,
          )))
          if pyg_calls != 1 or sparse_calls != 1:
              raise RuntimeError
          if min(int(preferred.batch_size), int(fallback.batch_size)) <= 0:
              raise RuntimeError
          if min(int(preferred.num_edges), int(fallback.num_edges)) <= 0:
              raise RuntimeError
      finally:
          pyg_lib.sampler.neighbor_sample = original_pyg
          torch_sparse.neighbor_sample = original_sparse
          neighbor_sampler.WITH_PYG_LIB = original_with_pyg
  ```

  Add direct-body tests whose spies prove `pyg_lib.sampler.neighbor_sample` and `torch_sparse.neighbor_sample` are each called exactly once. Mutate away the preferred block and fallback block independently; each mutation must fail its named direct-body test. Keep these gates mandatory for every selected distribution: exact public version; strict platform local-version policy; one owned WHEEL plus RECORD; compatible Python ABI and platform tag; direct import; imported `__file__` owned by RECORD. Keep Linux CPU and normalized `nvidia-` rejection. Run `_run_warning_free` around each canary and canonical NNx verification. Build immutable `StackEvidence` before NNx, call NNx last, and return it immediately.

- [ ] **Step 7: Prove GREEN, mutations, and safe CLI behavior**

  ```bash
  pytest -p no:cacheprovider tests/test_verify_torch_stack_platform.py tests/test_verify_repo.py -q -k 'selected_component or provenance or wheel or record or ownership or cpu or nvidia or runtime_canary or warning or nnx or runtime_availability or torch_runtime_contract'
  ruff check scripts/verify_torch_stack.py tests/test_verify_torch_stack_platform.py tests/test_verify_repo.py
  python -m py_compile scripts/verify_torch_stack.py tests/test_verify_torch_stack_platform.py tests/test_verify_repo.py
  git diff --check
  ```

  Expected: all selected tests pass, all mutations are killed, Ruff reports success, and the CLI tests show one allow-listed line with no warning payload, URL, filesystem path, captured output, or traceback.

- [ ] **Step 8: Commit only Task 2 ownership**

  ```bash
  git add scripts/verify_torch_stack.py tests/test_verify_torch_stack_platform.py tests/test_verify_repo.py
  git diff --cached --name-only
  git commit -m "test: narrow Torch stack verifier boundary"
  ```

  Expected staged paths are exactly those three. Re-run the five Task 3 hashes from section 12.22.3; they must still match and remain unstaged.

---

## 12.22.6 Task 3: Finish fail-closed graph, quantization, and notebook artifact gates

**Files:**
- Modify and commit existing work in progress: `tests/nnx_surface/conftest.py`
- Modify and commit existing work in progress: `tests/nnx_surface/test_node_classification_reddit_gnn_pyg.py`
- Modify and commit existing work in progress: `tests/nnx_surface/test_quantization_mnist_ffnn_pytorch.py`
- Modify and commit existing work in progress: `tests/test_makefile_contract.py`
- Modify and commit existing work in progress: `tests/test_verify_torch_stack.py`
- Create: `scripts/verify_smoke_outputs.py`
- Create: `tests/test_verify_smoke_outputs.py`
- Create: `scripts/verify_junit.py`
- Create: `tests/test_verify_junit.py`
- Modify: `Makefile`

**Interfaces:**
- Consumes: Task 1 installer, Task 2 verifier, existing tiny graph/image fixtures, NNx 0.2.0 public quantization facade, torchao 0.18.0, and current Tier A/B/C Make inventories.
- Produces: mandatory SAGE/GraphConv/GAT sampled training; tiny PTQ and QAT prepare/train/convert/inference; exact one-time conftest ordering; `Tier`, `InventoryLoader`, `NotebookArtifact`, `load_make_inventory`, `verify_smoke_outputs`, `verify_junit`, both CLIs, and Make print/check seams.
- Commit ownership: all ten files above. This is the first task allowed to stage the five preserved work-in-progress files.

- [ ] **Step 1: Reconcile the preserved verifier tests with Task 2's final boundary**

  In `tests/test_verify_torch_stack.py`, retain the existing AST-backed consumer bypass and conftest-order tests. Replace the twelve-name/five-canary fixtures with the ten-name/three-canary constants from Task 2. Delete assertions requiring cluster, spline, source builds, or their modules. Add mutations that reinsert `torch_cluster`, `torch_spline_conv`, `cluster`, or `spline` and require the exact-boundary tests to fail.

  Run:

  ```bash
  pytest -p no:cacheprovider tests/test_verify_torch_stack.py -q -k 'public_interfaces or manifest_contract or canary or consumer_gate or nnx_delegation'
  ```

  Expected: all selected tests pass against Task 2; no legacy module is required.

- [ ] **Step 2: Complete the one-time session ordering contract**

  Delete the entire redundant `_verify_nnx_installation_contract` autouse session fixture from `tests/nnx_surface/conftest.py`. Keep exactly one module-level call to each verifier:

  ```python
  from scripts.verify_nnx_install import verify_nnx_install
  from scripts.verify_torch_stack import verify_torch_stack

  verify_torch_stack()
  verify_nnx_install()
  import nnx  # noqa: E402  # both provenance gates precede collection imports
  ```

  Replace `_assert_nnx_install_fixture_contract` with an AST helper requiring one import of each verifier, zero functions named `_verify_nnx_installation_contract`, exactly one top-level zero-argument call to each verifier, at least one NNx import, and strict order `verify_torch_stack()` then `verify_nnx_install()` then the first NNx import:

  ```python
  def _is_nnx_import(node: ast.stmt) -> bool:
      if isinstance(node, ast.Import):
          return any(alias.name == "nnx" or alias.name.startswith("nnx.") for alias in node.names)
      return (
          isinstance(node, ast.ImportFrom)
          and node.module is not None
          and (node.module == "nnx" or node.module.startswith("nnx."))
      )

  def _assert_nnx_collection_verifier_contract(source: str) -> None:
      tree = ast.parse(source)
      assert not [
          node for node in tree.body
          if isinstance(node, ast.FunctionDef)
          and node.name == "_verify_nnx_installation_contract"
      ]
      calls = {
          name: tuple(
              node for node in tree.body
              if isinstance(node, ast.Expr)
              and isinstance(node.value, ast.Call)
              and isinstance(node.value.func, ast.Name)
              and node.value.func.id == name
              and not node.value.args
              and not node.value.keywords
          )
          for name in ("verify_torch_stack", "verify_nnx_install")
      }
      assert len(calls["verify_torch_stack"]) == 1
      assert len(calls["verify_nnx_install"]) == 1
      nnx_imports = tuple(node for node in tree.body if _is_nnx_import(node))
      assert nnx_imports
      assert tree.body.index(calls["verify_torch_stack"][0]) < tree.body.index(calls["verify_nnx_install"][0])
      assert tree.body.index(calls["verify_nnx_install"][0]) < tree.body.index(nnx_imports[0])
  ```

  Mutations delete or duplicate either call, reverse calls, move either after NNx, put either inside a function/fixture, wrap either in `try`, add an environment bypass, or restore the deleted autouse fixture; each must fail the helper.

- [ ] **Step 3: Finish mandatory graph consumer tests**

  Preserve the partial removal of `pytest.skip`, `pytest.importorskip`, `_HAS_PYG_SAMPLER`, `_has_pyg_sampler`, and skip decorators. Keep only pyg-lib and sparse backend imports:

  ```python
  def test_canonical_sampler_backends_and_batch_are_executable(tiny_graph_data):
      import pyg_lib
      import torch_sparse
      from torch_geometric.loader import NeighborLoader

      batch = next(iter(NeighborLoader(
          tiny_graph_data.data,
          num_neighbors=[2, 2],
          batch_size=2,
          input_nodes=tiny_graph_data.data.train_mask,
          shuffle=False,
          num_workers=0,
      )))
      assert pyg_lib is not None
      assert torch_sparse is not None
      assert int(batch.batch_size) > 0
      assert int(batch.edge_index.numel()) > 0
  ```

  Require SAGE, GraphConv, and GAT tests each to consume a real positive sampled batch and finish one NNx training epoch. Do not import or canary cluster/spline.

- [ ] **Step 4: Finish mandatory PTQ and QAT consumer tests**

  Preserve the partial removal of `_import_torchao_or_skip`, `torch.int1` guards, and skip paths. Keep the public-facade signature tests and this tiny QAT execution:

  ```python
  callback = nnx.QATLifecycleCallback(qat_config="8da4w")
  train_step = nnx.qat_train_step_factory(qat_config="8da4w")
  run = model.train(
      params=(
          NNTrainParams(n_epochs=1)
          .with_train_loader(value=tiny_image_batch.train_loader)
          .with_val_loader(value=tiny_image_batch.val_loader)
      ),
      callbacks=[callback],
      train_step_fn=train_step,
  )
  logits, classes = model.predict(X=tiny_image_batch.X)
  assert run is not None
  assert callback.is_prepared
  assert callback.is_converted
  assert logits.shape == (4, 10)
  assert classes.shape == (4,)
  assert np.issubdtype(classes.dtype, np.integer)
  ```

  PTQ must import torchao normally, quantize through the NNx public facade, and predict. Do not execute or edit the complete quantization notebook.

- [ ] **Step 5: Prove graph and quantization GREEN in a brand-new environment**

  Discard every r1-r3 environment. From a clean Task 2 HEAD create one new disposable environment and invoke the installer once:

  ```bash
  FOCUS_ROOT=$(mktemp -d /private/tmp/ml-eng-lab-issue62-focus-r4.XXXXXX)
  python3.11 -m venv "$FOCUS_ROOT/venv"
  export PATH="$FOCUS_ROOT/venv/bin:$PATH"
  export PIP_CACHE_DIR="$FOCUS_ROOT/pip-cache"
  export MPLCONFIGDIR="$FOCUS_ROOT/matplotlib"
  make install-torch-stack
  python -m pip check
  make verify-torch-stack
  make verify-nnx-install
  pytest -p no:cacheprovider -W error --junitxml="$FOCUS_ROOT/focused.xml" tests/nnx_surface/test_node_classification_reddit_gnn_pyg.py tests/nnx_surface/test_quantization_mnist_ffnn_pytorch.py tests/test_verify_torch_stack.py tests/test_verify_torch_stack_platform.py tests/test_makefile_contract.py -q
  python -m scripts.verify_junit "$FOCUS_ROOT/focused.xml"
  ```

  Expected: installer completes without wheel bootstrap or source build; verifier reports pyg-lib; pytest treats warnings as errors; `verify_junit` reports a positive test count with failures=0, errors=0, skipped=0. Any skip, warning, import failure, or ABI failure stops Task 3.

- [ ] **Step 6: Write smoke-output oracle RED tests**

  In `tests/test_verify_smoke_outputs.py`, use an injected inventory loader and temporary notebooks. Cover exact counts 18/6/4, missing and extra inventory items, duplicate sources, duplicate mapped outputs, an extra `.ipynb` anywhere below the output root, inventory command failure, missing/empty/invalid JSON, non-notebook JSON, zero code cells, a null code-cell execution count, an error output, and safe redaction. Mutate away the recursive output-set equality and the nonempty-code-cell gate independently; each mutation must fail its named test. A valid fixture is:

  ```python
  def write_executed_notebook(path: Path) -> None:
      path.parent.mkdir(parents=True, exist_ok=True)
      path.write_text(json.dumps({
          "cells": [{
              "cell_type": "code",
              "execution_count": 1,
              "metadata": {},
              "outputs": [{"output_type": "stream", "name": "stdout", "text": ["ok\n"]}],
              "source": ["print('ok')\n"],
          }],
          "metadata": {},
          "nbformat": 4,
          "nbformat_minor": 5,
      }), encoding="utf-8")
  ```

  Expected stable failures are only `smoke output verification failed: <a|b|c>: <inventory|missing|empty|invalid|unexecuted|error>`.

- [ ] **Step 7: Implement Make inventory seams and the oracle**

  Add `.PHONY` entries and exact recipes:

  ```make
  print-tier-a:
	@printf '%s\n' $(TIER_A)

  print-tier-b:
	@printf '%s\n' $(TIER_B)

  print-tier-c:
	@printf '%s\n' $(TIER_C)
  ```

  Implement:

  ```python
  EXPECTED_COUNTS: Mapping[Tier, int] = {"a": 18, "b": 6, "c": 4}

  class SmokeOutputError(RuntimeError):
      def __init__(self, tier: Tier, category: str) -> None:
          super().__init__(f"smoke output verification failed: {tier}: {category}")

  def _validate_notebook(tier: Tier, source: str, output: Path) -> NotebookArtifact:
      if not output.is_file():
          raise SmokeOutputError(tier, "missing")
      if output.stat().st_size == 0:
          raise SmokeOutputError(tier, "empty")
      try:
          document = json.loads(output.read_text(encoding="utf-8"))
          cells = document["cells"]
          if document["nbformat"] != 4 or not isinstance(cells, list):
              raise ValueError
      except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
          raise SmokeOutputError(tier, "invalid") from None
      code_cells = tuple(cell for cell in cells if cell.get("cell_type") == "code")
      if not code_cells:
          raise SmokeOutputError(tier, "invalid")
      if any(cell.get("execution_count") is None for cell in code_cells):
          raise SmokeOutputError(tier, "unexecuted")
      if any(
          output_item.get("output_type") == "error"
          for cell in code_cells
          for output_item in cell.get("outputs", ())
      ):
          raise SmokeOutputError(tier, "error")
      return NotebookArtifact(source, output, len(code_cells))

  def load_make_inventory(tier: Tier) -> Sequence[str]:
      result = subprocess.run(
          ("make", "--no-print-directory", "-s", f"print-tier-{tier}"),
          check=False,
          capture_output=True,
          text=True,
      )
      if result.returncode != 0:
          raise SmokeOutputError(tier, "inventory")
      return tuple(line.strip() for line in result.stdout.splitlines() if line.strip())

  def verify_smoke_outputs(tier, root, inventory_loader=load_make_inventory):
      sources = tuple(inventory_loader(tier))
      if len(sources) != EXPECTED_COUNTS[tier] or len(set(sources)) != len(sources):
          raise SmokeOutputError(tier, "inventory")
      outputs = tuple(
          root / (source if tier == "a" else Path(source).name)
          for source in sources
      )
      if len(set(outputs)) != len(outputs):
          raise SmokeOutputError(tier, "inventory")
      expected_outputs = {output.resolve() for output in outputs}
      actual_outputs = {output.resolve() for output in root.rglob("*.ipynb")}
      if actual_outputs != expected_outputs:
          raise SmokeOutputError(tier, "inventory")
      return tuple(
          _validate_notebook(tier, source, output)
          for source, output in zip(sources, outputs, strict=True)
      )
  ```

  Tier A maps to `root / source`, preserving nested `notebooks/<task>/...` paths. Tier B/C map to `root / Path(source).name`; duplicate basenames fail before validation. The recursive equality gate compares the exact resolved mapped set with every notebook below `root`, so an extra file fails on all tiers. The CLI accepts only `--tier a|b|c` and an absolute `--root`, prints no third-party payload on failure, and returns 1.

  Implement `scripts/verify_junit.py` so warnings-as-errors plus JUnit totals provide the reusable zero-skip gate:

  ```python
  class JUnitVerificationError(RuntimeError):
      """The JUnit report is missing, malformed, empty, failing, or skipping."""

  def verify_junit(path: Path) -> tuple[int, int, int, int]:
      try:
          root = ElementTree.parse(path).getroot()
      except FileNotFoundError:
          raise JUnitVerificationError("junit verification failed: missing") from None
      except (OSError, ElementTree.ParseError):
          raise JUnitVerificationError("junit verification failed: invalid") from None
      suites = (root,) if root.tag == "testsuite" else tuple(root.findall("testsuite"))
      if not suites:
          raise JUnitVerificationError("junit verification failed: schema")
      try:
          totals = tuple(
              sum(int(suite.attrib.get(field, "0")) for suite in suites)
              for field in ("tests", "failures", "errors", "skipped")
          )
      except ValueError:
          raise JUnitVerificationError("junit verification failed: schema") from None
      tests, failures, errors, skipped = totals
      if tests <= 0 or failures or errors or skipped:
          raise JUnitVerificationError("junit verification failed: outcome")
      return totals
  ```

  Its CLI accepts one XML path, prints only `junit verification ok: tests=<N> failures=0 errors=0 skipped=0`, and returns 1 with a stable category for missing, invalid XML, schema, or outcome. `tests/test_verify_junit.py` covers each category and positive multi-suite aggregation.

- [ ] **Step 8: Prove smoke oracle GREEN and mutation resistance**

  ```bash
  pytest -p no:cacheprovider tests/test_verify_smoke_outputs.py tests/test_verify_junit.py tests/test_makefile_contract.py -q
  ruff check scripts/verify_smoke_outputs.py scripts/verify_junit.py tests/test_verify_smoke_outputs.py tests/test_verify_junit.py tests/nnx_surface/conftest.py tests/nnx_surface/test_node_classification_reddit_gnn_pyg.py tests/nnx_surface/test_quantization_mnist_ffnn_pytorch.py tests/test_makefile_contract.py tests/test_verify_torch_stack.py
  python -m py_compile scripts/verify_smoke_outputs.py scripts/verify_junit.py tests/test_verify_smoke_outputs.py tests/test_verify_junit.py
  git diff --check
  ```

  Expected: all tests pass; isolated mutations for each 18/6/4 count, validation category, and Make inventory recipe fail a named test.

- [ ] **Step 9: Commit Task 3 only after the clean matrix**

  ```bash
  git add tests/nnx_surface/conftest.py tests/nnx_surface/test_node_classification_reddit_gnn_pyg.py tests/nnx_surface/test_quantization_mnist_ffnn_pytorch.py tests/test_makefile_contract.py tests/test_verify_torch_stack.py scripts/verify_smoke_outputs.py tests/test_verify_smoke_outputs.py scripts/verify_junit.py tests/test_verify_junit.py Makefile
  git diff --cached --name-only
  git commit -m "test: require supported Torch graph and quantization surfaces"
  ```

  Expected: exactly those ten paths are staged; the formerly partial five-file state is now one reviewed Task 3 commit.

---

## 12.22.7 Task 4: Make CI, Docker, and Codespaces consume the final install contract

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `Dockerfile`
- Modify: `.devcontainer/devcontainer.json`
- Modify: `Makefile`
- Modify: `tests/test_makefile_contract.py`
- Modify: `tests/test_verify_repo.py`
- Modify: `tests/test_check_docs.py`

**Interfaces:**
- Consumes: Task 1 installer, Task 2 verifier, Task 3 smoke oracle, and all five canonical stack manifests.
- Produces: final-install ordering for repository, NNx-surface, Tier A/B/C, audit, Docker, and Codespaces paths; no late package changes; no services.

- [ ] **Step 1: Write CI/cache/order RED contracts**

  Require every job that calls `make install-torch-stack` to cache:

  ```text
  requirements.txt
  torch-core-requirements.txt
  torch-ecosystem-requirements.txt
  torch-requirements.txt
  torch-audit-requirements.txt
  pyg-extension-audit-requirements.txt
  ```

  For runtime jobs require the ordered subsequence `make install-torch-stack`, last job-specific package/data install, `python -m pip check`, `make verify-torch-stack`, `make verify-nnx-install`, workload. Reject `pip install`, `python -m pip install`, `uv pip`, `conda install`, `apt install`, or `apt-get install` from the start of `pip check` through the workload, including between stack and NNx verification. The installer already installs root requirements; delete every duplicate NNx/root install.

  The `dependency-audit` cache is required even though that job does not install the runtime. Its `cache-dependency-path` is exactly:

  ```text
  vulnerability-audit-requirements.txt
  requirements.txt
  torch-core-requirements.txt
  torch-ecosystem-requirements.txt
  torch-requirements.txt
  torch-audit-requirements.txt
  pyg-extension-audit-requirements.txt
  docs-requirements.txt
  atlas-contract-requirements.txt
  ```

  Mutate out each of the six runtime/audit manifests (`requirements.txt`, core, ecosystem, runtime, audit, supplement) independently and require the dependency-audit cache contract to fail.

  Use an assertion that treats the final verifier as the install/workload boundary:

  ```python
  PACKAGE_INSTALL_MARKERS = (
      "pip install",
      "uv pip",
      "conda install",
      "apt install",
      "apt-get install",
  )

  def _package_install_commands(source: str) -> tuple[str, ...]:
      return tuple(
          line.strip() for line in source.splitlines()
          if any(marker in line for marker in PACKAGE_INSTALL_MARKERS)
      )

  def _assert_final_install_order(commands: tuple[str, ...], workload: str) -> None:
      joined = "\n".join(commands)
      install = joined.index("make install-torch-stack")
      pip_check = joined.index("python -m pip check", install)
      stack = joined.index("make verify-torch-stack", pip_check)
      nnx = joined.index("make verify-nnx-install", stack)
      work = joined.index(workload, nnx)
      assert not _package_install_commands(joined[pip_check:work])
  ```

- [ ] **Step 2: Write Docker and Codespaces RED contracts**

  Require Docker to copy repository files, invoke `make install-torch-stack`, install NLP assets, then run `python -m pip check`, `python -m scripts.verify_torch_stack`, and `python -m scripts.verify_nnx_install` as its final build gates. Require `.devcontainer/devcontainer.json` to keep exactly `"postCreateCommand": "make codespace-setup"`. Require `codespace-setup` to finish with pip-check, stack verification, and NNx verification after `nlp-assets`. Reject direct pip algorithms, any later package change, `docker compose`, Jupyter startup, Ollama, ComfyUI, and Atlas initialization.

  ```python
  def _target_recipe(makefile: str, target: str) -> tuple[str, ...]:
      lines = makefile.splitlines()
      start = next(index for index, line in enumerate(lines) if line.startswith(f"{target}:"))
      recipes: list[str] = []
      for line in lines[start + 1:]:
          if line.startswith("\t"):
              recipes.append(line.removeprefix("\t"))
              continue
          if line and not line.startswith((" ", "#")):
              break
      return tuple(recipes)

  def test_docker_and_codespaces_verify_after_the_last_package_change():
      docker = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
      makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
      assert docker.index("make nlp-assets") < docker.index("python -m pip check")
      assert docker.index("python -m pip check") < docker.index("python -m scripts.verify_torch_stack")
      assert _target_recipe(makefile, "codespace-setup") == (
          "$(MAKE) nlp-assets",
          "$(PYTHON) -m pip check",
          "$(MAKE) verify-torch-stack",
          "$(MAKE) verify-nnx-install",
      )
  ```

- [ ] **Step 3: Run RED**

  ```bash
  pytest -p no:cacheprovider tests/test_makefile_contract.py tests/test_verify_repo.py tests/test_check_docs.py -q -k 'torch_stack or cache_manifest or install_order or docker or codespace or service or late_install'
  ```

  Expected: failures identify duplicate root installs, missing cache manifests, missing pip-check/stack-verifier steps, and Docker's independent source-build algorithm.

- [ ] **Step 4: Update CI jobs with the final ordered block**

  In each runtime job use one install step. Add job-specific documentation or NLP installs in that step before the final verification step. The NNx-surface job uses warnings-as-errors and the Task 3 JUnit gate:

  ```yaml
  - name: Install dependencies
    run: |
      make install-torch-stack
      make nlp-assets

  - name: Check and verify canonical Torch and NNx stack
    run: |
      python -m pip check
      make verify-torch-stack
      make verify-nnx-install

  - name: Run NNx-surface tests
    run: |
      pytest -p no:cacheprovider -W error --junitxml=/tmp/nnx-surface.xml tests/nnx_surface -v
      python -m scripts.verify_junit /tmp/nnx-surface.xml
  ```

  Omit `make nlp-assets` from jobs that do not consume those assets. Install `docs-requirements.txt` in the install step for jobs that need it. Keep every package/data setup before pip-check and verification, and allow no install afterward. After each notebook workload invoke the exact artifact command before upload/cleanup:

  ```yaml
  # tier-a-papermill, after `make smoke-tier-a`
  - run: python -m scripts.verify_smoke_outputs --tier a --root /tmp/ml-tier-a

  # smoke-tier-b, after `make smoke-tier-b`
  - run: python -m scripts.verify_smoke_outputs --tier b --root /tmp/ml-smoke

  # smoke-tier-c, after `make smoke-tier-c`
  - run: python -m scripts.verify_smoke_outputs --tier c --root /tmp/ml-smoke
  ```

  Add order/mutation tests requiring one oracle call after its matching workload, the exact tier/root pair above, and no second workload or install between them. Keep existing job identities, triggers, Tier B label condition, Tier C dispatch/schedule condition, permissions, and protected-branch contexts.

- [ ] **Step 5: Update Docker and current comments**

  Replace Docker's pip chain with:

  ```dockerfile
  RUN make install-torch-stack \
    && make nlp-assets \
    && python -m pip check \
    && python -m scripts.verify_torch_stack \
    && python -m scripts.verify_nnx_install
  ```

  Remove the later duplicate NLP download layer. Extend `codespace-setup` to the exact four-recipe sequence in Step 2. Update CI, Docker, and devcontainer current comments to state Torch 2.11, three supported PyG wheels, importable torchao 0.18, and manual-only Issue #66. Do not add a service command.

- [ ] **Step 6: Prove GREEN and commit**

  ```bash
  pytest -p no:cacheprovider tests/test_makefile_contract.py tests/test_verify_repo.py tests/test_check_docs.py -q -k 'torch_stack or cache_manifest or install_order or docker or codespace or service or late_install'
  ruff check tests/test_makefile_contract.py tests/test_verify_repo.py tests/test_check_docs.py
  git diff --check
  git add .github/workflows/ci.yml Dockerfile .devcontainer/devcontainer.json Makefile tests/test_makefile_contract.py tests/test_verify_repo.py tests/test_check_docs.py
  git commit -m "ci: enforce final Torch stack installation order"
  ```

  Expected: selected tests pass, no package-changing command occurs after stack verification, and the staged diff contains exactly seven paths.

---

## 12.22.8 Task 5: Reconcile four logical advisory surfaces from six physical commands

**Files:**
- Modify: `scripts/advisory_baseline.py`
- Modify: `tests/test_advisory_baseline.py`
- Modify: `security/accepted-advisories.json`
- Modify: `docs/dependency-contracts.md`
- Modify: `scripts/verify_repo.py`
- Modify: `tests/test_verify_repo.py`
- Modify: `tests/test_check_docs.py`
- Write ignored evidence: `.superpowers/sdd/issue62-advisory/commands.txt`
- Write ignored evidence: `.superpowers/sdd/issue62-advisory/combined-runtime-resolver.json`
- Write ignored evidence: `.superpowers/sdd/issue62-advisory/combined-runtime-pyg-extensions.json`
- Write ignored evidence: `.superpowers/sdd/issue62-advisory/torch-resolver.json`
- Write ignored evidence: `.superpowers/sdd/issue62-advisory/torch-pyg-extensions.json`
- Write ignored evidence: `.superpowers/sdd/issue62-advisory/documentation.json`
- Write ignored evidence: `.superpowers/sdd/issue62-advisory/atlas-contract.json`

**Interfaces:**
- Consumes: Task 1 final manifests, the complete clean environment, pip-audit JSON schema, current accepted policy, and existing D10 parser.
- Produces: exactly four logical `Observation` values from exactly six physical commands; current policy and ledger parity; pyg-lib external-index limitation; corrected current requirements hash.

- [ ] **Step 1: Lock the six-command/four-surface model with RED tests**

  Require this exact `AUDIT_SURFACES` shape:

  ```python
  AUDIT_SURFACES = (
      AuditSurface("combined-runtime", ("requirements.txt", "torch-audit-requirements.txt"), output_name="combined-runtime-resolver"),
      AuditSurface("combined-runtime", ("pyg-extension-audit-requirements.txt",), True, True, "combined-runtime-pyg-extensions"),
      AuditSurface("torch", ("torch-audit-requirements.txt",), output_name="torch-resolver"),
      AuditSurface("torch", ("pyg-extension-audit-requirements.txt",), True, True, "torch-pyg-extensions"),
      AuditSurface("documentation", ("docs-requirements.txt",), disable_pip=True),
      AuditSurface("atlas-contract", ("atlas-contract-requirements.txt",)),
  )
  ```

  Assert the supplement bytes contain exactly scatter/sparse, each supplement observation contains only those packages, and merging affects only combined-runtime and torch. Assert pyg-lib appears in no pip-audit supplement and is documented as external-index provenance verified by `verify_torch_stack`.

- [ ] **Step 2: Run the structural RED suite**

  ```bash
  pytest -p no:cacheprovider tests/test_advisory_baseline.py tests/test_verify_repo.py tests/test_check_docs.py -q -k 'audit_surface or extension or dependency_ledger or advisory or requirements_hash'
  ```

  Expected: the stale four-package supplement fixtures and the Issue #61 requirements hash fail.

- [ ] **Step 3: Capture the six physical audit commands in the ignored directory**

  In the clean Task 3 environment, install the pinned audit tool before the final environment checks, then re-run the complete installed-runtime gates:

  ```bash
  python -m pip install -r vulnerability-audit-requirements.txt
  python -m pip check
  make verify-torch-stack
  make verify-nnx-install
  ```

  Expected: the pinned `pip-audit` tool is present before the final pip-check/verifier boundary and all three commands exit 0.

  Create `.superpowers/sdd/issue62-advisory/` and use the production `AUDIT_SURFACES` and `_audit_command` directly so recorded argv cannot drift. This produces exactly: resolver commands without `--disable-pip` for combined-runtime, Torch, and Atlas; `--disable-pip --no-deps` for both supplements; and `--disable-pip` only for documentation.

  ```bash
  python - <<'PY'
  from __future__ import annotations

  import hashlib
  import json
  import platform
  import subprocess
  import sys
  from importlib.metadata import version
  from pathlib import Path

  from scripts.advisory_baseline import (
      AUDIT_SURFACES,
      AdvisoryBaselineError,
      AuditSurfaceError,
      _audit_command,
      _classify_missing_output,
      _load_pip_audit_output,
      normalize_pip_audit,
  )

  root = Path.cwd()
  out = root / ".superpowers/sdd/issue62-advisory"
  out.mkdir(parents=True, exist_ok=True)
  input_paths = (
      "vulnerability-audit-requirements.txt",
      "requirements.txt",
      "torch-core-requirements.txt",
      "torch-ecosystem-requirements.txt",
      "torch-requirements.txt",
      "torch-audit-requirements.txt",
      "pyg-extension-audit-requirements.txt",
      "docs-requirements.txt",
      "atlas-contract-requirements.txt",
      "security/accepted-advisories.json",
  )
  records: list[dict[str, object]] = []
  for surface in AUDIT_SURFACES:
      output = out / f"{surface.output_name or surface.name}.json"
      argv = _audit_command(surface, output)
      result = subprocess.run(
          argv, cwd=root, check=False, capture_output=True, text=True,
      )
      if result.returncode not in (0, 1):
          raise AuditSurfaceError(surface.name, "unexpected-exit")
      if not output.is_file() or output.stat().st_size == 0:
          raise AuditSurfaceError(
              surface.name,
              _classify_missing_output(result.returncode, result.stderr),
          )
      try:
          payload = _load_pip_audit_output(output)
      except FileNotFoundError as error:
          raise AuditSurfaceError(
              surface.name,
              _classify_missing_output(result.returncode, result.stderr),
          ) from error
      except OSError as error:
          raise AuditSurfaceError(surface.name, "unavailable-output") from error
      except (UnicodeError, json.JSONDecodeError) as error:
          raise AuditSurfaceError(surface.name, "invalid-json") from error
      except AdvisoryBaselineError as error:
          raise AuditSurfaceError(surface.name, "invalid-schema") from error
      try:
          observation = normalize_pip_audit(surface.name, payload)
      except AdvisoryBaselineError as error:
          raise AuditSurfaceError(surface.name, "invalid-schema") from error
      records.append({
          "logical_surface": surface.name,
          "output": output.relative_to(root).as_posix(),
          "argv": argv,
          "returncode": result.returncode,
          "outcome": "clean" if result.returncode == 0 else "advisories",
          "resolved_count": len(observation.resolved_versions),
          "advisory_count": len(observation.advisories),
      })
  metadata = {
      "python": platform.python_version(),
      "python_executable": sys.executable,
      "pip_audit": version("pip-audit"),
      "commands": records,
      "input_sha256": {
          path: hashlib.sha256((root / path).read_bytes()).hexdigest()
          for path in input_paths
      },
      "output_sha256": {
          record["output"]: hashlib.sha256((root / str(record["output"])).read_bytes()).hexdigest()
          for record in records
      },
  }
  (out / "commands.txt").write_text(
      json.dumps(metadata, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  PY
  git check-ignore -v .superpowers/sdd/issue62-advisory/commands.txt .superpowers/sdd/issue62-advisory/*.json
  ```

  Expected: every physical command returns only 0 or 1, every output is nonempty valid normalized JSON, `commands.txt` records exact argv/exit/outcome/counts/versions/hashes, and all seven evidence paths resolve to `.superpowers/sdd/.gitignore:1:*`. Any unexpected exit, missing output, bootstrap/resolution/service category, invalid JSON, or invalid schema stops the task.

- [ ] **Step 4: Reconcile observations and write exact metadata**

  Merge the two paired supplements into their resolver observations and keep documentation/Atlas unchanged, yielding exactly four logical surfaces. Record exact argv in ignored `commands.txt`; put redacted command templates in the published current ledger. Both record Python/pip-audit versions, SHA-256 of the ten tracked inputs in `input_paths` and all six JSON files, resolved package counts and exact selected versions, raw feed records and aliases, unique identities, retained/disappeared/re-keyed/new findings, fix versions, risk rationale, revisit triggers, and pyg-lib's external-index/PyPI-audit limitation. Feed disappearance is never called remediation.

- [ ] **Step 5: Update policy and current ledger atomically**

  Run `make audit-advisories` before editing and preserve its exact accepted-version/new/removed diagnostics. Update `security/accepted-advisories.json` to the observed identity/version/surface tuples in canonical key and item order. Archive the prior current ledger subsection as historical Issue #59/#61 evidence with a non-remediation disclaimer, then add exactly one `### 6.1.1.2 Current Issue #62 four-surface audit` section containing one Result line, one summary table with the existing `_DEPENDENCY_SUMMARY_HEADER`, one advisory table with the existing `_DEPENDENCY_ADVISORY_HEADER`, and one input table headed exactly `| Input | SHA-256 |` / `| --- | --- |`. The input rows follow `_DEPENDENCY_HASH_INPUTS` order. Include exact hashes/counts, Lightning 2.6.1 supply-chain rationale, residual Torch risk, and pyg-lib provenance.

- [ ] **Step 6: Correct the stale Issue #61 requirements hash only in Task 5 evidence**

  The final Task 1 `requirements.txt` SHA-256 is already `6e86caa5a287e9566e15bdffbb6628249397307783dee3b6e98e728ef06275b9`. In `tests/test_check_docs.py`, rename the test to current ledger semantics and replace both old `3f35f04f95bd1e293c844b41a2dcf96f7978b8c61ccd436e4813a604d9e528a7` assertions with `6e86caa5a287e9566e15bdffbb6628249397307783dee3b6e98e728ef06275b9`. Put the same exact hash in the new current ledger input table. Do not alter the archived Issue #61 historical snapshot text.

- [ ] **Step 7: Harden D10 parsing and mutations**

  Keep the existing `_dependency_table_rows()` and
  `_dependency_advisory_baseline_findings()` implementations; they already return
  `None`/`list[Finding]` with stable D10 errors and compare the Markdown advisory identities,
  accepted versions, and canonical surface order to `security/accepted-advisories.json`.
  Replace the current-section regex and add the hash parser below. The regex deliberately
  names only the Issue #62 heading, so the archived Issue #59/#61 tables remain historical
  evidence and cannot satisfy current enforcement.

  ```python
  _DEPENDENCY_CURRENT_SNAPSHOT_RE = re.compile(
      r"^###[ \t]+6[.]1[.]1[.]2[ \t]+Current[ \t]+Issue[ \t]+#62[ \t]+"
      r"four-surface[ \t]+audit[ \t]*\r?$"
      r"(?P<body>.*?)(?=^#{1,3}[ \t]|\Z)",
      re.MULTILINE | re.DOTALL,
  )
  _DEPENDENCY_HASH_HEADER = "| Input | SHA-256 |"
  _DEPENDENCY_HASH_SEPARATOR = "| --- | --- |"
  _DEPENDENCY_HASH_ROW_RE = re.compile(
      r"\| `(?P<path>[^`]+)` \| `(?P<sha256>[0-9a-f]{64})` \|"
  )
  _DEPENDENCY_HASH_INPUTS = (
      "vulnerability-audit-requirements.txt",
      "requirements.txt",
      "torch-core-requirements.txt",
      "torch-ecosystem-requirements.txt",
      "torch-requirements.txt",
      "torch-audit-requirements.txt",
      "pyg-extension-audit-requirements.txt",
      "docs-requirements.txt",
      "atlas-contract-requirements.txt",
      "security/accepted-advisories.json",
  )
  _DEPENDENCY_RAW_HTML_BLOCK_RE = re.compile(
      r"(?ims)^<(?P<tag>address|article|aside|blockquote|details|dialog|div|fieldset|"
      r"figure|footer|form|header|main|nav|pre|script|section|style|table)\b[^>]*>"
      r".*?</(?P=tag)\s*>[ \t]*(?:\n|\Z)"
  )

  def _mask_dependency_raw_html(text: str) -> str:
      return _DEPENDENCY_RAW_HTML_BLOCK_RE.sub(
          lambda match: "".join(
              "\n" if character == "\n" else " " for character in match.group(0)
          ),
          text,
      )

  def _dependency_input_hash_findings(repo: Path, body: str) -> list[Finding]:
      location = "docs/dependency-contracts.md"
      lines = _dependency_table_rows(
          body,
          header=_DEPENDENCY_HASH_HEADER,
          separator=_DEPENDENCY_HASH_SEPARATOR,
      )
      if lines is None:
          return [Finding(
              id="D10.dependency_input_hash",
              check="docs",
              severity="error",
              location=location,
              message="current Issue #62 input-hash table is missing or malformed",
          )]
      rows = [_DEPENDENCY_HASH_ROW_RE.fullmatch(line) for line in lines]
      if not all(rows):
          return [Finding(
              id="D10.dependency_input_hash",
              check="docs",
              severity="error",
              location=location,
              message="current Issue #62 input-hash row is malformed",
          )]
      parsed = [(row["path"], row["sha256"]) for row in rows if row is not None]
      names = [name for name, _ in parsed]
      if len(names) != len(set(names)):
          return [Finding(
              id="D10.dependency_input_hash",
              check="docs",
              severity="error",
              location=location,
              message="current Issue #62 input-hash table has duplicate paths",
          )]
      if tuple(names) != _DEPENDENCY_HASH_INPUTS:
          return [Finding(
              id="D10.dependency_input_hash",
              check="docs",
              severity="error",
              location=location,
              message="current Issue #62 input-hash paths or order drifted",
              detail={"expected": list(_DEPENDENCY_HASH_INPUTS), "actual": names},
          )]
      findings: list[Finding] = []
      for relative_path, recorded in parsed:
          source = repo / relative_path
          if not source.is_file():
              findings.append(Finding(
                  id="D10.dependency_input_hash",
                  check="docs",
                  severity="error",
                  location=relative_path,
                  message="current Issue #62 hashed input is missing",
              ))
              continue
          actual = hashlib.sha256(source.read_bytes()).hexdigest()
          if actual != recorded:
              findings.append(Finding(
                  id="D10.dependency_input_hash",
                  check="docs",
                  severity="error",
                  location=relative_path,
                  message="current Issue #62 recorded input hash is stale",
                  detail={"expected": recorded, "actual": actual},
              ))
      return findings
  ```

  Add `import hashlib` to both `scripts/verify_repo.py` and `tests/test_verify_repo.py`. In the existing single-match branch of
  `_dependency_ledger_findings()`, retain all current summary/result/count logic and integrate the
  two comparisons at these exact points—there is no new parser or comparison symbol:

  ```python
  body = snapshot_matches[0].group("body")
  summary_lines = _dependency_table_rows(
      body,
      header=_DEPENDENCY_SUMMARY_HEADER,
      separator=_DEPENDENCY_SUMMARY_SEPARATOR,
  )
  advisory_lines = _dependency_table_rows(
      body,
      header=_DEPENDENCY_ADVISORY_HEADER,
      separator=_DEPENDENCY_ADVISORY_SEPARATOR,
  )
  findings.extend(_dependency_advisory_baseline_findings(repo, advisory_lines))
  findings.extend(_dependency_input_hash_findings(repo, body))
  ```

  Change the existing publication-mask assignment to
  `published_text = _mask_dependency_raw_html(_strip_markdown_code(text, strip_inline=False))`
  before calculating `snapshot_matches`; HTML comments remain masked by `_strip_markdown_code`,
  and the new helper masks complete CommonMark container blocks while preserving newlines.
  Insert those two `findings.extend` calls immediately before the existing `package_rows = (`
  statement; retain that statement and the complete existing summary/advisory/result logic below
  it byte-for-byte except where the new Issue #62 heading changes expected fixture text.

  In `tests/test_verify_repo.py`, build valid current ledger text from real policy identities and
  current file digests; do not hard-code future audit results. Add tests named
  `test_dependency_ledger_rejects_missing_or_duplicate_current_issue62_section`,
  `test_dependency_ledger_rejects_malformed_result_summary_and_advisory_tables`,
  `test_dependency_ledger_ignores_complete_historical_audit_tables`,
  `test_dependency_ledger_rejects_missing_duplicate_reordered_and_stale_input_hashes`,
  `test_dependency_ledger_couples_advisory_identity_version_and_surfaces_to_policy`,
  `test_dependency_ledger_rejects_advisory_only_package_and_count_drift`,
  `test_dependency_ledger_rejects_zero_vulnerability_and_legacy_extension_claims`, and
  `test_dependency_ledger_requires_pyg_lib_external_index_limitation`.

  Use the existing `_advisory_baseline_repo`, `_load_verify_module`, and
  `_write_canonical_baseline` test helpers with these executable foundations:

  ```python
  def _issue62_ledger_repo(tmp_path: Path) -> Path:
      repo = _advisory_baseline_repo(tmp_path)
      module = _load_verify_module()
      for relative in module._DEPENDENCY_HASH_INPUTS:
          source = REPO / relative
          target = repo / relative
          target.parent.mkdir(parents=True, exist_ok=True)
          shutil.copyfile(source, target)
      return repo

  def _d10_ids(repo: Path) -> set[str]:
      return {
          finding.id
          for finding in _load_verify_module()._dependency_ledger_findings(repo)
          if finding.id.startswith("D10.dependency_")
      }

  def test_dependency_ledger_rejects_missing_or_duplicate_current_issue62_section(tmp_path):
      repo = _issue62_ledger_repo(tmp_path)
      ledger = repo / "docs/dependency-contracts.md"
      marker = "### 6.1.1.2 Current Issue #62 four-surface audit"
      original = ledger.read_text(encoding="utf-8")
      ledger.write_text(original.replace(marker, "### 6.1.1.2 Archived audit", 1), encoding="utf-8")
      assert "D10.dependency_ledger_count" in _d10_ids(repo)
      ledger.write_text(original + "\n" + marker + "\n", encoding="utf-8")
      assert "D10.dependency_ledger_count" in _d10_ids(repo)

  @pytest.mark.parametrize(
      ("needle", "replacement"),
      (
          ("| Package | Manifest Constraint | Audited Resolved Version | Finding Count | Current Disposition |",
           "| Broken summary header |"),
          ("| --- | --- | ---: | ---: | --- |", "| --- |"),
          ("Result: ", "Result malformed: "),
      ),
  )
  def test_dependency_ledger_rejects_malformed_result_summary_and_advisory_tables(
      tmp_path, needle, replacement,
  ):
      repo = _issue62_ledger_repo(tmp_path)
      ledger = repo / "docs/dependency-contracts.md"
      original = ledger.read_text(encoding="utf-8")
      mutated = original.replace(needle, replacement, 1)
      assert mutated != original
      ledger.write_text(mutated, encoding="utf-8")
      assert "D10.dependency_ledger_count" in _d10_ids(repo)

  def test_dependency_ledger_ignores_complete_historical_audit_tables(tmp_path):
      repo = _issue62_ledger_repo(tmp_path)
      assert _d10_ids(repo) == set()
      ledger = repo / "docs/dependency-contracts.md"
      original = ledger.read_text(encoding="utf-8")
      historical = original.replace(
          "### 6.1.1.2 Current Issue #62 four-surface audit",
          "### 6.1.13.1 Archived Issue #61 audit",
          1,
      )
      ledger.write_text(original + "\n## 6.1.13 Archive\n\n" + historical, encoding="utf-8")
      assert _d10_ids(repo) == set()

  def test_dependency_ledger_rejects_missing_duplicate_reordered_and_stale_input_hashes(tmp_path):
      repo = _issue62_ledger_repo(tmp_path)
      ledger = repo / "docs/dependency-contracts.md"
      digest = hashlib.sha256((repo / "requirements.txt").read_bytes()).hexdigest()
      row = f"| `requirements.txt` | `{digest}` |"
      original = ledger.read_text(encoding="utf-8")
      assert row in original
      ledger.write_text(original.replace(row, f"| `requirements.txt` | `{'0' * 64}` |", 1), encoding="utf-8")
      assert "D10.dependency_input_hash" in _d10_ids(repo)

  def test_dependency_ledger_couples_advisory_identity_version_and_surfaces_to_policy(tmp_path):
      repo = _issue62_ledger_repo(tmp_path)
      policy = repo / "security/accepted-advisories.json"
      document = json.loads(policy.read_text(encoding="utf-8"))
      document["accepted_advisories"][0]["accepted_version"] = "0.0.0"
      _write_canonical_baseline(repo, document)
      assert "D10.dependency_advisory_baseline" in _d10_ids(repo)
  ```

  Parameterize the malformed-table test across a missing header, changed header, missing
  separator, malformed row, duplicate row, and structures placed only in a fenced block or raw
  HTML. Parameterize policy coupling by mutating one advisory ID, accepted version, and surface at
  a time in Markdown and then JSON; every mutation must contain a
  `D10.dependency_advisory_baseline` finding. Hash mutations must contain
  `D10.dependency_input_hash`; malformed/count/current-section mutations must contain
  `D10.dependency_ledger_count`. An extra fully populated historical section must produce no new
  finding. Mutating either supplement row to cluster/spline or deleting the pyg-lib provenance
  sentence must produce a named D10 finding through the existing current-contract assertions.

- [ ] **Step 8: Prove GREEN and commit**

  ```bash
  make audit-advisories
  pytest -p no:cacheprovider tests/test_advisory_baseline.py tests/test_verify_repo.py tests/test_check_docs.py -q -k 'advisory or dependency or d10 or requirements_hash or issue_62'
  python scripts/verify_repo.py --check docs --fast
  ruff check scripts/advisory_baseline.py scripts/verify_repo.py tests/test_advisory_baseline.py tests/test_verify_repo.py tests/test_check_docs.py
  git diff --check
  git add scripts/advisory_baseline.py scripts/verify_repo.py security/accepted-advisories.json docs/dependency-contracts.md tests/test_advisory_baseline.py tests/test_verify_repo.py tests/test_check_docs.py
  git commit -m "security: reconcile Torch 2.11 advisory policy"
  ```

  Expected: four logical surfaces compare cleanly, six evidence files remain ignored, and the staged diff contains exactly seven tracked paths.

---

## 12.22.9 Task 6: Synchronize current operational truth across all documentation surfaces

**Files:**
- Modify: `README.md`
- Modify: `CONTRIBUTING.md`
- Modify: `SECURITY.md`
- Modify: `CHANGELOG.md`
- Modify: `docs/env-setup.md`
- Modify: `docs/architecture.md`
- Modify: `docs/FINDINGS-ATLAS.md`
- Modify: `docs/dependency-contracts.md`
- Modify generated canonical page: `docs/notebook-infrastructure.md`
- Modify: `docs/notebooks/pruning-mnist-ffnn-pytorch.md`
- Modify: `docs/notebooks/quantization-mnist-ffnn-pytorch.md`
- Modify: `notebooks/node_classification-reddit-gnn-pyg/README.md`
- Modify: `notebooks/quantization-mnist-ffnn-pytorch/README.md`
- Modify: `notebooks/quantization-mnist-ffnn-pytorch/docs/spec.yaml`
- Modify: `docs/assets/badges/pytorch.svg`
- Modify current comments only: `Makefile`
- Modify current comments only: `.github/workflows/ci.yml`
- Modify current comments only: `Dockerfile`
- Modify current comments only: `.devcontainer/devcontainer.json`
- Modify: `tests/test_check_docs.py`

**Interfaces:**
- Consumes: implemented matrix, verifier, CI/Docker/Codespaces order, current advisory ledger, immutable history, unchanged Atlas ownership, and unchanged tier map.
- Produces: one self-contained operational story in repository Markdown, generated MkDocs input, and generated wiki; durable Unreleased history; rollback runbook; no premature final-acceptance claim.

- [ ] **Step 1: Write current-surface RED tests**

  Scope assertions to current README, contributor, security, architecture, environment, dependency, notebook-infrastructure, graph README, pruning, quantization doc/README/spec, Make, CI, Docker, devcontainer, badge, and Unreleased changelog sections. Require the exact matrix, `make install-torch-stack`, `python -m pip check`, `make verify-torch-stack`, Linux CPU-only rule, three wheel names, no source build, manual-only Issue #66, Atlas Issue #65, NNx 0.2.0, no containerized Ollama, residual advisory language, and fresh-environment/image rollback.

  Reject current text containing Torch 2.4.1, `torchao>=`, separate cluster/spline requirements, wheel bootstrap, source-build flags, five canaries, twelve components, unavailable Darwin graph backends, tier-covered quantization, upgraded Atlas, or completed final acceptance before Task 7.

  Add a shared current-surface assertion and mutate each required/forbidden phrase once:

  ```python
  def _assert_issue62_current_contract(documents: Mapping[str, str]) -> None:
      current = "\n".join(documents.values())
      for required in (
          "torch==2.11.0",
          "--only-binary=pyg-lib,torch-scatter,torch-sparse",
          "make verify-torch-stack",
          "Issue #65",
          "Issue #66",
          "thekaveh-nnx[lm]==0.2.0",
      ):
          assert required in current
      for forbidden in (
          "torch==2.4.1",
          "torch-cluster==",
          "torch-spline-conv==",
          "--no-build-isolation",
      ):
          assert forbidden not in current
  ```

  The `documents` fixture must contain only explicit current slices: README setup/runtime sections, CONTRIBUTING setup/rollback sections, SECURITY current supported-dependency section, the Unreleased changelog prefix, the single Task 5 current ledger section, and the complete current-only operational pages/comments/assets. Historical changelog, archived ledger, specs, plans, and evidence are excluded before this helper is called.

- [ ] **Step 2: Run RED**

  ```bash
  pytest -p no:cacheprovider tests/test_check_docs.py -q -k 'torch or quantization or dependency or atlas or issue_62 or rollback or security'
  ```

  Expected: current pages/comments/badge still contain at least one obsolete Torch/PyG/quantization claim.

- [ ] **Step 3: Update current operational prose and architecture assets**

  Document: selected ten-component matrix; three compiled PyG wheels; pyg-lib preferred sampling and sparse fallback; scatter/sparse/sampler canaries; Linux/Darwin/native Linux arm64 Docker scope; exact install/check/verifier commands; qualified tiny PTQ/QAT dependency surface; manual-only full quantization notebook; Issue #66 ownership; unchanged Atlas and Issue #65 ownership; NNx 0.2.0; residual advisory evidence; no containerized Ollama; atomic rollback of manifests/installer/verifier/CI/Docker/policy/ledger/docs in a new environment or rebuilt image.

  Update the PyTorch badge label/value to 2.11 without changing unrelated badge geometry. Add one Unreleased changelog entry. Before Task 7, use exactly: `The dependency and focused runtime contracts are implemented; complete Tier A/B/C and container acceptance evidence is pending.`

  Use these exact anchors and replacement strings; tests select these same bounded sections rather
  than searching historical text:

  | File and anchor | Exact current replacement/addition |
  | --- | --- |
  | `README.md`, replace the complete fenced shell block under `### 3.3. Local venv` | `python3.11 -m venv .venv && source .venv/bin/activate`<br>`make install-torch-stack`<br>`make nlp-assets`<br>`python -m pip check`<br>`make verify-torch-stack`<br>`make verify-nnx-install`<br>`jupyter lab`; then add: `The supported CPU matrix is torch==2.11.0, torchvision==0.26.0, torch_geometric==2.8.0.post1, pyg-lib==0.8.0+pt211, torch-scatter==2.1.2+pt211, torch-sparse==0.6.18+pt211, torchao==0.18.0, and thekaveh-nnx[lm]==0.2.0; Linux wheels use the +pt211cpu local tag.` |
  | `CONTRIBUTING.md`, append to `## 6. Verification` | `After the last package or data install, run \`python -m pip check\`, \`make verify-torch-stack\`, and \`make verify-nnx-install\`; never mutate the environment between those gates and the workload. Roll back manifests, installer, verifier, CI/Docker, advisory policy/ledger, and documentation atomically in a fresh environment or rebuilt image.` |
  | `SECURITY.md`, replace `## 13.6 Dependency advisories` current opening | `Issue #62 audits four logical surfaces through six physical commands. Resolver audits cover core plus ecosystem plus PyG; supplement audits cover only torch-scatter and torch-sparse. pyg-lib is external-index provenance verified by the stack verifier, not a PyPI supplement result. Feed disappearance is reconciliation evidence, never proof of remediation.` |
  | `CHANGELOG.md`, add first bullet under `[Unreleased]` → `### Changed` | `- Coordinated the supported CPU Torch stack at Torch 2.11/PyG 2.8.0.post1/torchao 0.18 with binary-only pyg-lib, torch-scatter, and torch-sparse wheels, NNx 0.2.0 verification, and manual-only Issue #66 quantization ownership.` |
  | `docs/env-setup.md`, replace the fenced shell block and both paragraphs in `## 4.1.3 Local Python venv`, stopping before `## 4.1.4` | the same seven-line README shell block, followed by: `Use Python 3.11 and make install-torch-stack; the installer ends with binary-only thekaveh-nnx[lm]==0.2.0. After the last asset install, package state is frozen through pip-check, Torch verification, NNx verification, and the workload. Linux is CPU-only; Darwin and native Linux arm64 Docker are locally qualified, and Linux x86_64 is qualified by the PR gates.` |
  | `docs/architecture.md`, replace the dependency paragraph in `## 2.1.3 Runtime entry paths` | `Every local, CI, Docker, and Codespaces runtime enters through the four-stage canonical installer, performs its last asset install, then freezes package state across pip-check, Torch verification, NNx verification, and workload. No repository container starts Jupyter, Atlas, Ollama, or ComfyUI as part of Issue #62.` |
  | `docs/FINDINGS-ATLAS.md`, append to `## 9.2.2 Atlas Jupyter runtime is distinct from local CI` | `Issue #62 does not upgrade Atlas: Atlas runtime ownership remains Issue #65. The host-native Ollama boundary is unchanged, and no containerized Ollama service is added.` |
  | `docs/dependency-contracts.md`, inside `## 6.1.2 Torch Stack Pin`, replace from `` `torch-core-requirements.txt` pins`` through `never by runtime installation.` | `torch-core-requirements.txt pins Torch 2.11.0, TorchVision 0.26.0, and TorchAudio 2.11.0. torch-ecosystem-requirements.txt pins Lightning 2.6.1, TorchMetrics 1.9.0, and torchao 0.18.0. The runtime contains ecosystem plus selector plus pyg-lib 0.8.0, torch-scatter 2.1.2, torch-sparse 0.6.18, and PyG 2.8.0.post1; the audit projection contains core plus ecosystem plus PyG; the supplement contains only torch-scatter and torch-sparse. Runtime wheel installation is --only-binary=pyg-lib,torch-scatter,torch-sparse; wheel bootstrap and source-build flags are forbidden.` |
  | `docs/notebooks/pruning-mnist-ffnn-pytorch.md`, insert after the H1 | `This Tier A notebook consumes the qualified Torch 2.11 CPU stack and is covered by the 18-output Tier A oracle.` |
  | `docs/notebooks/quantization-mnist-ffnn-pytorch.md`, insert after the H1 | `Issue #62 qualifies only the tiny PTQ/QAT dependency surface with torchao 0.18.0; the full notebook remains manual-only and belongs to Issue #66, so it is not a Tier A/B/C acceptance output.` |
  | `notebooks/node_classification-reddit-gnn-pyg/README.md`, replace the `torch` dependency bullet and the final availability sentence in `## 5. Dependencies` | `- torch==2.11.0 and torch_geometric==2.8.0.post1 with exactly three binary wheels: pyg-lib 0.8.0, torch-scatter 2.1.2, and torch-sparse 0.6.18. Sampling proves the preferred pyg-lib path and the torch-sparse fallback; no additional compiled extension package is supported.` and `Install through make install-torch-stack and prove it with make verify-torch-stack.` |
  | `notebooks/quantization-mnist-ffnn-pytorch/README.md`, replace the `torchao>=0.17` bullet and the final availability sentence in `## 5. Dependencies` | `- torchao==0.18.0 on torch==2.11.0 — mandatory for the tiny PTQ/QAT surface.` and `Install through make install-torch-stack. The complete notebook is manual-only under Issue #66 and excluded from Tier A/B/C.` |
  | `notebooks/quantization-mnist-ffnn-pytorch/docs/spec.yaml`, replace the sole `atlas.constraints` item and first `pitfalls` item exactly | Constraint: `- Manual-only under Issue #66; Issue #62 qualifies only the tiny PTQ/QAT dependency surface.` Pitfall: `- "MANUAL-ONLY: Issue #62 qualifies the tiny Torch 2.11.0 + torchao 0.18.0 PTQ/QAT surface; Issue #66 owns complete-notebook execution, which remains outside Tier A/B/C."`; keep `tier: manual` unchanged. |
  | `Makefile`, current Torch-stack comment | `# Issue #62 canonical CPU stack: Torch 2.11, binary pyg-lib/scatter/sparse, NNx 0.2.0 last.` |
  | `.github/workflows/ci.yml`, current Torch-stack comment | `# Issue #62: final install, pip-check, Torch/NNx verification, then workload; no late package mutation.` |
  | `Dockerfile`, current Torch-stack comment | `# Issue #62 CPU image: no service startup and no source-built PyG extension.` |
  | `.devcontainer/devcontainer.json`, current Torch-stack comment | `// Issue #62 setup delegates to make codespace-setup; it starts no service.` |
  | `docs/assets/badges/pytorch.svg`, existing `2.4.1` text node | replace only its text content with `2.11.0`. |

  Run `make docs-sync-notebook-infrastructure` after the two notebook README/spec edits. Require the
  generated row for both graph and quantization tasks to carry the same execution/dependency truth;
  do not hand-edit the table after generation.

- [ ] **Step 4: Preserve immutable records**

  Run:

  ```bash
  git diff --name-only df0652b48554b1c3562177ce9e3ee28084be88cf -- docs/superpowers/specs docs/superpowers/plans docs/maintenance
  git diff -- CHANGELOG.md docs/dependency-contracts.md
  ```

  Expected: only the Issue #62 design/plan changed under specs/plans; no Issue #59/#60/#61 spec, plan, snapshot, released changelog block, maintenance record, Atlas probe evidence, or notebook output is rewritten. The Task 5 archived ledger remains intact.

- [ ] **Step 5: Generate and compare all three documentation surfaces**

  ```bash
  make docs-check
  make docs-wiki
  NO_MKDOCS_2_WARNING=1 mkdocs build --strict
  python scripts/verify_repo.py --check docs --fast
  cmp docs/superpowers/specs/2026-08-14-issue-62-torch-stack-upgrade-design.md generated/site/superpowers/specs/2026-08-14-issue-62-torch-stack-upgrade-design.md
  cmp docs/superpowers/plans/2026-08-14-issue-62-torch-stack-upgrade-implementation-plan.md generated/site/superpowers/plans/2026-08-14-issue-62-torch-stack-upgrade-implementation-plan.md
  ```

  Compare every changed manifest-owned canonical page to its exact generated site and wiki
  transformation with this manifest-driven command (the image rewrites are part of the production
  generators and therefore part of parity):

  ```bash
  python - <<'PY'
  from pathlib import Path

  from scripts.docs.build_docs import _rewrite_images_site
  from scripts.docs.manifest import load_manifest
  from scripts.docs.transforms import build_source_map, rewrite_for_surface
  from scripts.docs.wiki import _rewrite_images_wiki

  root = Path.cwd()
  changed = (
      "docs/architecture.md",
      "docs/env-setup.md",
      "docs/FINDINGS-ATLAS.md",
      "docs/dependency-contracts.md",
      "docs/notebook-infrastructure.md",
      "docs/notebooks/pruning-mnist-ffnn-pytorch.md",
      "docs/notebooks/quantization-mnist-ffnn-pytorch.md",
      "docs/superpowers/plans/2026-08-14-issue-62-torch-stack-upgrade-implementation-plan.md",
      "SECURITY.md",
  )
  manifest = load_manifest(root / "docs/manifest.yaml", root)
  for surface, output_root, image_rewrite in (
      ("site", root / "generated/site", _rewrite_images_site),
      ("wiki", root / "generated/wiki", _rewrite_images_wiki),
  ):
      source_map = build_source_map(manifest, surface)
      for source in changed:
          expected = rewrite_for_surface(
              (root / source).read_text(encoding="utf-8"), surface, source_map,
          )
          expected = image_rewrite(expected)
          actual = (output_root / source_map[source]).read_text(encoding="utf-8")
          if actual != expected:
              raise SystemExit(f"{surface} parity drift: {source}")
  print("changed canonical documentation parity ok")
  PY
  ```

  Expected: strict build has zero warnings, wiki check succeeds, every changed canonical mapping
  compares exactly, and generated paths remain ignored/untracked.

- [ ] **Step 6: Prove GREEN and commit current documentation**

  ```bash
  make docs-sync-notebook-infrastructure
  pytest -p no:cacheprovider tests/test_check_docs.py tests/test_manifest.py tests/test_transforms.py tests/test_build_docs.py tests/test_wiki.py -q
  ruff check tests/test_check_docs.py
  git diff --check
  git add README.md CONTRIBUTING.md SECURITY.md CHANGELOG.md docs/env-setup.md docs/architecture.md docs/FINDINGS-ATLAS.md docs/dependency-contracts.md docs/notebook-infrastructure.md docs/notebooks/pruning-mnist-ffnn-pytorch.md docs/notebooks/quantization-mnist-ffnn-pytorch.md notebooks/node_classification-reddit-gnn-pyg/README.md notebooks/quantization-mnist-ffnn-pytorch/README.md notebooks/quantization-mnist-ffnn-pytorch/docs/spec.yaml docs/assets/badges/pytorch.svg Makefile .github/workflows/ci.yml Dockerfile .devcontainer/devcontainer.json tests/test_check_docs.py
  git commit -m "docs: document supported Torch 2.11 runtime"
  ```

  Expected: docs tests pass and no generated path is staged.

---

## 12.22.10 Task 7: Qualify one immutable final SHA, integrate through GitFlow, and clean up

**Files:**
- Modify before freeze only: current evidence paragraphs in `README.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, `docs/env-setup.md`, `docs/FINDINGS-ATLAS.md`, `docs/dependency-contracts.md`, `docs/notebooks/quantization-mnist-ffnn-pytorch.md`, `notebooks/quantization-mnist-ffnn-pytorch/README.md`, `tests/test_check_docs.py`, and this plan.
- Write after freeze only: ignored `.superpowers/sdd/issue62-qualification-report.md` and external GitHub issue/PR evidence.
- Never modify: notebook source/output, Atlas files/gitlink, generated documentation, or protected-branch rules.

**Interfaces:**
- Consumes: reviewed Tasks 1-6 and a clean candidate branch.
- Produces: clean Darwin arm64, native Linux arm64 Docker, Linux x86_64 PR, advisory, full tests, Tier A/B/C, documentation, immutable-SHA, GitFlow, publication, and cleanup evidence.

- [ ] **Step 1: Create and verify a clean prequalification worktree**

  ```bash
  PREQUAL_ROOT=$(mktemp -d /private/tmp/ml-eng-lab-issue62-prequal.XXXXXX)
  PREQUAL_SHA=$(git rev-parse HEAD)
  git worktree add --detach "$PREQUAL_ROOT/worktree" "$PREQUAL_SHA"
  git -C "$PREQUAL_ROOT/worktree" -c submodule.infra.url=/Users/kaveh/repos/ml-eng-lab/infra submodule update --init --recursive infra
  python3.11 -m venv "$PREQUAL_ROOT/venv"
  export PATH="$PREQUAL_ROOT/venv/bin:$PATH"
  export JUPYTER_PATH="$PREQUAL_ROOT/jupyter/share/jupyter"
  export PIP_CACHE_DIR="$PREQUAL_ROOT/pip-cache"
  export MPLCONFIGDIR="$PREQUAL_ROOT/matplotlib"
  cd "$PREQUAL_ROOT/worktree"
  test "$(uname -s)" = Darwin
  test "$(uname -m)" = arm64
  test "$(python3.11 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" = 3.11
  test "$(python -c 'import platform; print(platform.machine())')" = arm64
  test "$(python -c 'import sys; print(sys.prefix)')" = "$PREQUAL_ROOT/venv"
  test "$(python -c 'import sys; print(sys.executable)')" = "$PREQUAL_ROOT/venv/bin/python"
  git status --porcelain=v1
  git diff --check
  git submodule status infra
  find notebooks -type d \( -name runs -o -name checkpoints \) -print
  ```

  Expected: every Darwin/arm64/Python 3.11/`sys.prefix`/`sys.executable` assertion exits 0;
  status/diff/find are empty; and submodule status begins with a space followed by
  `61c7c5103660e2226bf107c115dae42bf46f8374`. Install no editable package and use no global
  kernelspec path.

- [ ] **Step 2: Run the complete Darwin dependency, advisory, repository, and docs gate**

  ```bash
  make install-torch-stack
  python -m pip install -r docs-requirements.txt
  python -m pip install -r vulnerability-audit-requirements.txt
  make nlp-assets
  python -m ipykernel install --prefix "$PREQUAL_ROOT/jupyter" --name python3 --display-name "Issue 62 Python 3"
  python - <<'PY'
  import json
  import subprocess
  import sys
  from pathlib import Path

  data = json.loads(subprocess.check_output(
      [sys.executable, "-m", "jupyter", "kernelspec", "list", "--json"],
      text=True,
  ))
  resource_dir = Path(data["kernelspecs"]["python3"]["resource_dir"])
  kernel = json.loads((resource_dir / "kernel.json").read_text(encoding="utf-8"))
  assert Path(kernel["argv"][0]).resolve() == Path(sys.executable).resolve()
  print(f"isolated kernelspec ok: {resource_dir} -> {sys.executable}")
  PY
  python -m pip check
  make verify-torch-stack
  make verify-nnx-install
  make audit-advisories
  pytest -p no:cacheprovider -W error --junitxml="$PREQUAL_ROOT/nnx-surface.xml" tests/nnx_surface -v
  python -m scripts.verify_junit "$PREQUAL_ROOT/nnx-surface.xml"
  make test
  make lint
  make verify
  make docs-check
  make docs-wiki
  git diff --check
  ```

  Expected: the parser proves kernelspec `argv[0]` resolves to `sys.executable`; every command exits
  0; the focused NNx suite treats warnings as errors and its JUnit totals have failures, errors,
  and skipped all equal to zero; exact versions, WHEEL/RECORD/local-version/platform/CPU/NVIDIA/
  import ownership, test counts, durations, and hashes are recorded. The audit-tool manifest is
  installed before the final pip-check/stack/NNx boundary.

- [ ] **Step 3: Qualify native Linux arm64 Docker without services**

  On Apple Silicon run without `--platform`:

  ```bash
  docker build --no-cache -t ml-eng-lab:issue62-prequal-arm64 .
  test "$(docker image inspect ml-eng-lab:issue62-prequal-arm64 --format '{{.Architecture}}')" = arm64
  docker run --rm ml-eng-lab:issue62-prequal-arm64 python -m pip check
  docker run --rm ml-eng-lab:issue62-prequal-arm64 python -m scripts.verify_torch_stack
  docker run --rm ml-eng-lab:issue62-prequal-arm64 python -m scripts.verify_nnx_install
  ```

  Expected: build-internal checks and all three external probes exit 0, architecture is arm64, and no Jupyter, Compose, Atlas, Ollama, ComfyUI, or daemon starts.

- [ ] **Step 4: Run complete Tier A/B/C and validate every output**

  ```bash
  JUPYTER_PATH="$JUPYTER_PATH" TIER_A_OUT="$PREQUAL_ROOT/tier-a" make smoke-tier-a
  TIER_A_OUT="$PREQUAL_ROOT/tier-a" make check-tier-a-artifacts
  python -m scripts.verify_smoke_outputs --tier a --root "$PREQUAL_ROOT/tier-a"
  make check-tier-a-clean
  JUPYTER_PATH="$JUPYTER_PATH" SMOKE_OUT="$PREQUAL_ROOT/tier-b" make smoke-tier-b
  python -m scripts.verify_smoke_outputs --tier b --root "$PREQUAL_ROOT/tier-b"
  JUPYTER_PATH="$JUPYTER_PATH" SMOKE_OUT="$PREQUAL_ROOT/tier-c" make smoke-tier-c
  python -m scripts.verify_smoke_outputs --tier c --root "$PREQUAL_ROOT/tier-c"
  git status --porcelain=v1
  git diff --check
  git submodule status infra
  ```

  Expected: Tier A 18/18, Tier B 6/6, Tier C 4/4; exactly 28 fresh nonempty outputs; every code cell executed; zero error outputs; tracked files, source notebooks, and Atlas gitlink unchanged. The quantization notebook is not among the 28.

- [ ] **Step 5: Record tracked prequalification, complete review, and commit before freeze**

  Replace Task 6's pending sentence with exact candidate SHA/platform/test/tier/Docker results plus: `Merge acceptance additionally requires an immutable final-SHA rerun attached to Issue #62; this tracked candidate record is not a substitute for that external evidence.` Regenerate site/wiki; run focused docs tests, `make docs-check`, `make docs-wiki`, `make verify`, Ruff, and diff checks; then commit only the listed current-evidence files.

  ```bash
  pytest -p no:cacheprovider tests/test_check_docs.py tests/test_manifest.py tests/test_transforms.py tests/test_build_docs.py tests/test_wiki.py -q
  make docs-check
  make docs-wiki
  make verify
  ruff check tests/test_check_docs.py
  git diff --check
  git add README.md CONTRIBUTING.md CHANGELOG.md docs/env-setup.md docs/FINDINGS-ATLAS.md docs/dependency-contracts.md docs/notebooks/quantization-mnist-ffnn-pytorch.md notebooks/quantization-mnist-ffnn-pytorch/README.md tests/test_check_docs.py docs/superpowers/plans/2026-08-14-issue-62-torch-stack-upgrade-implementation-plan.md
  git commit -m "docs: record Torch 2.11 prequalification evidence"
  ```

  Expected: tests, docs, verifier, Ruff, and diff checks pass; only the ten listed tracked paths enter the evidence commit; generated projections remain ignored.

  Review every branch commit and the complete diff for spec coverage, exact final manifests, mutation resistance, safe diagnostics, platform claims, advisory parity, Docker/CI ordering, notebook cleanliness, immutable history, Atlas non-diff, and rollback atomicity. Resolve each finding with a separate RED-GREEN commit and repeat review until zero findings remain.

- [ ] **Step 6: Freeze one final SHA and rerun every local acceptance gate from scratch**

  After the final tracked correction, run this complete sequence without reusing the prequalification environment:

  ```bash
  FINAL_ROOT=$(mktemp -d /private/tmp/ml-eng-lab-issue62-final.XXXXXX)
  FINAL_SHA=$(git rev-parse HEAD)
  git worktree add --detach "$FINAL_ROOT/worktree" "$FINAL_SHA"
  git -C "$FINAL_ROOT/worktree" -c submodule.infra.url=/Users/kaveh/repos/ml-eng-lab/infra submodule update --init --recursive infra
  python3.11 -m venv "$FINAL_ROOT/venv"
  export PATH="$FINAL_ROOT/venv/bin:$PATH"
  export JUPYTER_PATH="$FINAL_ROOT/jupyter/share/jupyter"
  export PIP_CACHE_DIR="$FINAL_ROOT/pip-cache"
  export MPLCONFIGDIR="$FINAL_ROOT/matplotlib"
  cd "$FINAL_ROOT/worktree"
  test "$(uname -s)" = Darwin
  test "$(uname -m)" = arm64
  test "$(python3.11 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" = 3.11
  test "$(python -c 'import platform; print(platform.machine())')" = arm64
  test "$(python -c 'import sys; print(sys.prefix)')" = "$FINAL_ROOT/venv"
  test "$(python -c 'import sys; print(sys.executable)')" = "$FINAL_ROOT/venv/bin/python"
  git status --porcelain=v1
  git diff --check
  git submodule status infra
  find notebooks -type d \( -name runs -o -name checkpoints \) -print
  make install-torch-stack
  python -m pip install -r docs-requirements.txt
  python -m pip install -r vulnerability-audit-requirements.txt
  make nlp-assets
  python -m ipykernel install --prefix "$FINAL_ROOT/jupyter" --name python3 --display-name "Issue 62 Final Python 3"
  python - <<'PY'
  import json
  import subprocess
  import sys
  from pathlib import Path

  data = json.loads(subprocess.check_output(
      [sys.executable, "-m", "jupyter", "kernelspec", "list", "--json"],
      text=True,
  ))
  resource_dir = Path(data["kernelspecs"]["python3"]["resource_dir"])
  kernel = json.loads((resource_dir / "kernel.json").read_text(encoding="utf-8"))
  assert Path(kernel["argv"][0]).resolve() == Path(sys.executable).resolve()
  print(f"isolated kernelspec ok: {resource_dir} -> {sys.executable}")
  PY
  python -m pip check
  make verify-torch-stack
  make verify-nnx-install
  make audit-advisories
  pytest -p no:cacheprovider -W error --junitxml="$FINAL_ROOT/nnx-surface.xml" tests/nnx_surface -v
  python -m scripts.verify_junit "$FINAL_ROOT/nnx-surface.xml"
  make test
  make lint
  make verify
  make docs-check
  make docs-wiki
  docker build --no-cache -t ml-eng-lab:issue62-final-arm64 .
  test "$(docker image inspect ml-eng-lab:issue62-final-arm64 --format '{{.Architecture}}')" = arm64
  docker run --rm ml-eng-lab:issue62-final-arm64 python -m pip check
  docker run --rm ml-eng-lab:issue62-final-arm64 python -m scripts.verify_torch_stack
  docker run --rm ml-eng-lab:issue62-final-arm64 python -m scripts.verify_nnx_install
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

  Expected: fresh preflight is clean; kernelspec uses `$FINAL_ROOT/venv/bin/python`; all dependency/advisory/test/lint/verifier/docs/Docker commands pass; Tier A/B/C report 18/6/4 with zero artifact errors; exact SHA matches; final status/diff are empty; Atlas gitlink is unchanged. Write evidence only to ignored `.superpowers/sdd/issue62-qualification-report.md` and GitHub. Any failure or later tracked commit invalidates the freeze and requires a new full run.

- [ ] **Step 7: Push the immutable feature SHA and qualify the feature-to-develop PR**

  `FINAL_SHA` becomes the immutable feature commit; ignored evidence is allowed after freeze, while
  any content-changing feature commit creates a new `FINAL_SHA` and restarts Steps 6-7. Push and
  open the ready PR without an auto-close keyword:

  ```bash
  REPO=thekaveh/ml-eng-lab
  FEATURE_REF=codex/issue-62-torch-stack-upgrade
  FEATURE_SHA="$FINAL_SHA"
  gh issue view 65 --repo "$REPO" --json state,title,body,labels,assignees,projectItems,updatedAt \
    > "$FINAL_ROOT/issue65-before.json"
  gh issue view 66 --repo "$REPO" --json state,title,body,labels,assignees,projectItems,updatedAt \
    > "$FINAL_ROOT/issue66-before.json"
  test "$(git rev-parse HEAD)" = "$FEATURE_SHA"
  git push --set-upstream origin "HEAD:refs/heads/$FEATURE_REF"
  test "$(git ls-remote origin "refs/heads/$FEATURE_REF" | cut -f1)" = "$FEATURE_SHA"
  FEATURE_PR_URL=$(gh pr create --repo "$REPO" --base develop --head "$FEATURE_REF" \
    --title "build: upgrade supported Torch stack to 2.11" \
    --body "Implements Issue #62 without closing it before release: supported binary pyg-lib/scatter/sparse boundary, ten-component verifier, advisory reconciliation, NNx 0.2.0, and Tier 18/6/4 evidence. Atlas Issue #65 and quantization Issue #66 remain out of scope; no service was started.")
  FEATURE_PR=$(gh pr view "$FEATURE_PR_URL" --repo "$REPO" --json number --jq .number)
  gh pr edit "$FEATURE_PR" --repo "$REPO" --add-label tier-b-smoke
  gh pr close "$FEATURE_PR" --repo "$REPO"
  gh pr reopen "$FEATURE_PR" --repo "$REPO"
  test "$(gh pr view "$FEATURE_PR" --repo "$REPO" --json headRefOid --jq .headRefOid)" = "$FEATURE_SHA"
  gh workflow run ci.yml --repo "$REPO" --ref "$FEATURE_REF"
  TIER_C_RUN=$(gh run list --repo "$REPO" --workflow CI --branch "$FEATURE_REF" \
    --event workflow_dispatch --limit 20 \
    --json databaseId,headSha,status,conclusion \
    --jq ".[] | select(.headSha == \"$FEATURE_SHA\") | .databaseId" | head -1)
  test -n "$TIER_C_RUN"
  gh run watch "$TIER_C_RUN" --repo "$REPO" --exit-status
  test "$(gh run view "$TIER_C_RUN" --repo "$REPO" --json headSha --jq .headSha)" = "$FEATURE_SHA"
  ```

  Fetch the PR test merge ref, distinguish it from the feature commit, and gate the SHA on which
  `pull_request` workflows actually ran:

  ```bash
  git fetch origin "+refs/pull/$FEATURE_PR/merge:refs/issue62/pr-$FEATURE_PR-merge"
  PR_MERGE_SHA=$(git rev-parse "refs/issue62/pr-$FEATURE_PR-merge")
  test "$PR_MERGE_SHA" != "$FEATURE_SHA"
  test "$(git rev-parse "$PR_MERGE_SHA^{tree}")" = "$(git rev-parse "$FEATURE_SHA^{tree}")"
  gh run list --repo "$REPO" --commit "$PR_MERGE_SHA" --limit 50 \
    --json databaseId,workflowName,event,headSha,status,conclusion,url \
    > "$FINAL_ROOT/pr-runs.json"
  python - "$FINAL_ROOT/pr-runs.json" "$PR_MERGE_SHA" <<'PY'
  import json
  import sys
  from pathlib import Path

  runs = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
  assert runs and all(run["headSha"] == sys.argv[2] for run in runs)
  assert all(run["status"] == "completed" for run in runs)
  assert all(run["conclusion"] == "success" for run in runs)
  assert {run["workflowName"] for run in runs} >= {"CI", "Docs gate"}
  PY
  gh pr checks "$FEATURE_PR" --repo "$REPO" --json name,state,bucket,link \
    > "$FINAL_ROOT/pr-checks.json"
  python - "$FINAL_ROOT/pr-checks.json" <<'PY'
  import json
  import sys
  from pathlib import Path

  checks = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
  expected = {
      "pytest-repository", "atlas-consumer-policy", "dependency-audit",
      "pytest-nnx-surface", "verify-repo", "docs-build", "docker-build",
      "tier-a-papermill", "smoke-tier-b",
  }
  by_name = {check["name"]: check for check in checks}
  assert expected <= by_name.keys()
  assert all(by_name[name]["bucket"] == "pass" for name in expected)
  assert all(
      check["bucket"] == "pass" or check["name"] == "smoke-tier-c"
      for check in checks
  )
  PY
  gh run view "$TIER_C_RUN" --repo "$REPO" --json jobs,url \
    --jq '{url, tier_c: [.jobs[] | select(.name == "smoke-tier-c") | {name,conclusion,url}]}' \
    > "$FINAL_ROOT/tier-c-run.json"
  test "$(jq -r '.tier_c | length' "$FINAL_ROOT/tier-c-run.json")" = 1
  test "$(jq -r '.tier_c[0].conclusion' "$FINAL_ROOT/tier-c-run.json")" = success
  ```

  Expected: every applicable PR check is green on the recorded synthetic `PR_MERGE_SHA`; the
  conditionally skipped PR-event `smoke-tier-c` job is not evidence and is replaced by the successful
  dispatch on exact `FEATURE_SHA`; no pending, skipped, neutral, cancelled, stale-SHA, or rerun-masked
  result is accepted as evidence. Attach the ignored Darwin/native-arm64/advisory/Tier 18/6/4 report and the Linux
  x86_64 run/check URLs to Issue #62 and the PR.

- [ ] **Step 8: Merge feature to develop and develop to main without changing rulesets**

  Read the protected ruleset without updating it, merge through GitHub, and preserve four distinct
  identities: feature commit, feature PR synthetic merge, develop merge, and release merge.

  ```bash
  gh api "repos/$REPO/rulesets/18620095" > "$FINAL_ROOT/ruleset.json"
  python - "$FINAL_ROOT/ruleset.json" <<'PY'
  import json
  import sys
  from pathlib import Path

  rule = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
  assert rule["name"] == "gitflow" and rule["enforcement"] == "active"
  assert set(rule["conditions"]["ref_name"]["include"]) == {
      "refs/heads/main", "refs/heads/develop",
  }
  status = next(item for item in rule["rules"] if item["type"] == "required_status_checks")
  assert {item["context"] for item in status["parameters"]["required_status_checks"]} == {
      "pytest-repository", "atlas-consumer-policy", "dependency-audit",
  }
  PY
  gh pr merge "$FEATURE_PR" --repo "$REPO" --merge --delete-branch
  DEVELOP_MERGE_SHA=$(gh pr view "$FEATURE_PR" --repo "$REPO" --json mergeCommit --jq .mergeCommit.oid)
  git fetch origin develop main
  test "$(git rev-parse origin/develop)" = "$DEVELOP_MERGE_SHA"
  test "$(git rev-parse "$DEVELOP_MERGE_SHA^{tree}")" = "$(git rev-parse "$FEATURE_SHA^{tree}")"

  RELEASE_PR_URL=$(gh pr create --repo "$REPO" --base main --head develop \
    --title "release: publish Issue 62 Torch 2.11 stack" \
    --body "Publishes the reviewed Issue #62 stack from develop to main. Issues #65 and #66 remain open. Closes #62 after successful release integration.")
  RELEASE_PR=$(gh pr view "$RELEASE_PR_URL" --repo "$REPO" --json number --jq .number)
  gh pr edit "$RELEASE_PR" --repo "$REPO" --add-label tier-b-smoke
  gh pr close "$RELEASE_PR" --repo "$REPO"
  gh pr reopen "$RELEASE_PR" --repo "$REPO"
  git fetch origin "+refs/pull/$RELEASE_PR/merge:refs/issue62/pr-$RELEASE_PR-merge"
  RELEASE_PR_MERGE_SHA=$(git rev-parse "refs/issue62/pr-$RELEASE_PR-merge")
  test "$(git rev-parse "$RELEASE_PR_MERGE_SHA^{tree}")" = "$(git rev-parse "$DEVELOP_MERGE_SHA^{tree}")"
  gh pr checks "$RELEASE_PR" --repo "$REPO" --watch
  test -z "$(gh pr checks "$RELEASE_PR" --repo "$REPO" --json name,bucket \
    --jq '.[] | select(.bucket != "pass" and .name != "smoke-tier-c") | .bucket')"
  gh run list --repo "$REPO" --commit "$RELEASE_PR_MERGE_SHA" --limit 50 \
    --json databaseId,workflowName,event,headSha,status,conclusion,url \
    > "$FINAL_ROOT/release-pr-runs.json"
  python - "$FINAL_ROOT/release-pr-runs.json" "$RELEASE_PR_MERGE_SHA" <<'PY'
  import json
  import sys
  from pathlib import Path

  runs = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
  assert runs and all(run["headSha"] == sys.argv[2] for run in runs)
  assert all(run["status"] == "completed" for run in runs)
  assert all(run["conclusion"] == "success" for run in runs)
  PY
  gh pr merge "$RELEASE_PR" --repo "$REPO" --merge
  RELEASE_MERGE_SHA=$(gh pr view "$RELEASE_PR" --repo "$REPO" --json mergeCommit --jq .mergeCommit.oid)
  git fetch origin main develop
  test "$(git rev-parse origin/main)" = "$RELEASE_MERGE_SHA"
  test "$(git rev-parse "$RELEASE_MERGE_SHA^{tree}")" = "$(git rev-parse "$DEVELOP_MERGE_SHA^{tree}")"
  ```

  If any tree comparison fails, stop: base drift changed content and requires review; a feature
  content correction also restarts local final qualification. If `main` is not an ancestor of
  `develop`, create the exact protected sync PR below; otherwise skip it. In either branch, final
  tree equality is mandatory.

  ```bash
  if ! git merge-base --is-ancestor origin/main origin/develop; then
    SYNC_PR_URL=$(gh pr create --repo "$REPO" --base develop --head main \
      --title "chore: synchronize Issue 62 release to develop" \
      --body "Content-neutral synchronization of the reviewed Issue #62 release merge.")
    SYNC_PR=$(gh pr view "$SYNC_PR_URL" --repo "$REPO" --json number --jq .number)
    gh pr checks "$SYNC_PR" --repo "$REPO" --watch --fail-fast
    gh pr merge "$SYNC_PR" --repo "$REPO" --merge
    git fetch origin main develop
  fi
  git diff --exit-code origin/main origin/develop
  git merge-base --is-ancestor origin/main origin/develop
  gh api "repos/$REPO/rulesets/18620095" > "$FINAL_ROOT/ruleset-after.json"
  cmp "$FINAL_ROOT/ruleset.json" "$FINAL_ROOT/ruleset-after.json"
  ```

- [ ] **Step 9: Verify publication, close Issue #62, and clean only Issue #62 state**

  Verify Pages/wiki and current-surface content, then update only Issue #62 bookkeeping:

  ```bash
  PAGES_URL=$(gh api "repos/$REPO/pages" --jq '.html_url | rtrimstr("/")')
  REPO_URL=$(gh repo view "$REPO" --json url --jq '.url | rtrimstr("/")')
  curl --fail --silent --show-error \
    "$PAGES_URL/dependency-contracts/" > "$FINAL_ROOT/pages.html"
  rg -q 'Torch 2\.11' "$FINAL_ROOT/pages.html"
  rg -q 'pyg-lib.*torch-scatter.*torch-sparse' "$FINAL_ROOT/pages.html"
  rg -q 'Issue #66' "$FINAL_ROOT/pages.html"
  curl --fail --silent --show-error \
    "$REPO_URL/wiki/6-1-Dependency-ledger" \
    > "$FINAL_ROOT/wiki.html"
  rg -q 'Torch 2\.11' "$FINAL_ROOT/wiki.html"
  rg -q 'pyg-lib.*torch-scatter.*torch-sparse' "$FINAL_ROOT/wiki.html"
  rg -q 'Issue #66' "$FINAL_ROOT/wiki.html"
  gh issue view 65 --repo "$REPO" --json state,title,body,labels,assignees,projectItems,updatedAt \
    > "$FINAL_ROOT/issue65-after.json"
  gh issue view 66 --repo "$REPO" --json state,title,body,labels,assignees,projectItems,updatedAt \
    > "$FINAL_ROOT/issue66-after.json"
  cmp "$FINAL_ROOT/issue65-before.json" "$FINAL_ROOT/issue65-after.json"
  cmp "$FINAL_ROOT/issue66-before.json" "$FINAL_ROOT/issue66-after.json"
  test "$(jq -r .state "$FINAL_ROOT/issue65-after.json")" = OPEN
  test "$(jq -r .state "$FINAL_ROOT/issue66-after.json")" = OPEN
  gh issue comment 62 --repo "$REPO" --body-file .superpowers/sdd/issue62-qualification-report.md
  gh issue close 62 --repo "$REPO" --reason completed \
    --comment "Released by feature PR #$FEATURE_PR at $DEVELOP_MERGE_SHA and release PR #$RELEASE_PR at $RELEASE_MERGE_SHA; immutable feature evidence is $FEATURE_SHA."
  gh issue comment 53 --repo "$REPO" \
    --body "Issue #62 completed via PR #$FEATURE_PR and release PR #$RELEASE_PR; Issues #65 and #66 remain open."
  gh api graphql -f query='query {
    repository(owner:"thekaveh", name:"ml-eng-lab") {
      issue(number:62) {
        projectItems(first:20) { nodes {
          id
          project { id number title }
          fieldValues(first:20) { nodes {
            ... on ProjectV2ItemFieldSingleSelectValue {
              field { ... on ProjectV2SingleSelectField { id name options { id name } } }
            }
          } }
        } }
      }
    }
  }' > "$FINAL_ROOT/project-item.json"
  test "$(jq -r '.data.repository.issue.projectItems.nodes | length' "$FINAL_ROOT/project-item.json")" = 1
  PROJECT_ID=$(jq -r '.data.repository.issue.projectItems.nodes[0].project.id' "$FINAL_ROOT/project-item.json")
  ITEM_ID=$(jq -r '.data.repository.issue.projectItems.nodes[0].id' "$FINAL_ROOT/project-item.json")
  STATUS_FIELD_ID=$(jq -r '.data.repository.issue.projectItems.nodes[0].fieldValues.nodes[] | select(.field.name == "Status") | .field.id' "$FINAL_ROOT/project-item.json")
  DONE_OPTION_ID=$(jq -r '.data.repository.issue.projectItems.nodes[0].fieldValues.nodes[] | select(.field.name == "Status") | .field.options[] | select(.name == "Done") | .id' "$FINAL_ROOT/project-item.json")
  test -n "$PROJECT_ID" && test -n "$ITEM_ID" && test -n "$STATUS_FIELD_ID" && test -n "$DONE_OPTION_ID"
  gh project item-edit --id "$ITEM_ID" --project-id "$PROJECT_ID" \
    --field-id "$STATUS_FIELD_ID" --single-select-option-id "$DONE_OPTION_ID"
  ```

  Clean only the exact validated Issue #62 targets. Do not use globs, delete unrelated containers,
  or remove any other worktree:

  ```bash
  cd /Users/kaveh/repos/ml-eng-lab
  case "$PREQUAL_ROOT" in /private/tmp/ml-eng-lab-issue62-prequal.*) ;; *) exit 1;; esac
  case "$FINAL_ROOT" in /private/tmp/ml-eng-lab-issue62-final.*) ;; *) exit 1;; esac
  git worktree list --porcelain | rg -F "worktree $PREQUAL_ROOT/worktree"
  git worktree list --porcelain | rg -F "worktree $FINAL_ROOT/worktree"
  git worktree remove "$PREQUAL_ROOT/worktree"
  git worktree remove "$FINAL_ROOT/worktree"
  rm -rf -- "$PREQUAL_ROOT"
  rm -rf -- "$FINAL_ROOT"
  test -z "$(docker ps --filter ancestor=ml-eng-lab:issue62-prequal-arm64 --format '{{.ID}}')"
  test -z "$(docker ps --filter ancestor=ml-eng-lab:issue62-final-arm64 --format '{{.ID}}')"
  docker image rm ml-eng-lab:issue62-prequal-arm64 ml-eng-lab:issue62-final-arm64
  git switch develop
  git merge --ff-only origin/develop
  git branch -d codex/issue-62-torch-stack-upgrade
  git switch main
  git merge --ff-only origin/main
  git switch develop
  test -z "$(git ls-remote origin refs/heads/codex/issue-62-torch-stack-upgrade)"
  git update-ref -d "refs/issue62/pr-$FEATURE_PR-merge"
  git update-ref -d "refs/issue62/pr-$RELEASE_PR-merge"
  test -z "$(gh pr list --repo "$REPO" --state open --head codex/issue-62-torch-stack-upgrade --json number --jq '.[].number')"
  test -z "$(gh run list --repo "$REPO" --branch codex/issue-62-torch-stack-upgrade \
    --status in_progress --json databaseId --jq '.[].databaseId')"
  test "$(git -C infra rev-parse HEAD)" = 61c7c5103660e2226bf107c115dae42bf46f8374
  git diff --exit-code origin/main origin/develop
  git status --porcelain=v1
  git worktree list
  ```

  Expected: Pages and wiki return HTTP 200 and publish the matrix, three-wheel boundary,
  manual-only Issue #66, and immutable evidence; #62 is closed and Done; #53 remains open; #65/#66
  remain open and unchanged; the two explicit worktrees/environments/images and feature refs are
  gone; no scoped workflow is in progress; `main`/`develop` trees match; and tracked status is clean.

---

## 12.22.11 Plan Self-Review Gate

- [x] **Spec coverage map:** 12.21.2-12.21.4 map to Task 1; 12.21.5-12.21.6 map to Task 2 and Task 7; 12.21.7-12.21.8 map to Task 3; 12.21.9 maps to Task 5; 12.21.10 maps to Task 4 and Task 7; 12.21.11 maps to Tasks 3 and 7; 12.21.12 maps to Task 6; 12.21.13 maps to Global Constraints and Tasks 4/6/7; 12.21.14 is preserved as design rationale; 12.21.15 maps to Global Constraints and Tasks 6/7; 12.21.16 maps to Task 7.
- [x] **Placeholder scan:** every code-changing step contains concrete code or exact replacement text; every test/run step has an exact command and expected result; no deferred marker or undefined neighboring interface remains.
- [x] **Type consistency:** `InstallStage`, `InstallCommand`, `StackPin`, `StackContract`, `StackEvidence`, `DistributionView`, `CanaryHooks`, `VerificationHooks`, `Tier`, `InventoryLoader`, and `NotebookArtifact` have one spelling and one signature throughout.
- [x] **Dependency order:** Task 1 produces manifests/installer; Task 2 consumes manifests and produces verifier; Task 3 consumes both and commits preserved WIP; Task 4 consumes installer/verifier/oracle; Task 5 consumes the clean final solve; Task 6 consumes implementation/audit truth; Task 7 consumes every tracked task.
- [x] **Final-SHA order:** all tracked evidence and review corrections precede `FINAL_SHA`; final qualification writes only ignored/external evidence; any later tracked commit invalidates and restarts the full final run.
- [x] **Boundary consistency:** current scope is pyg-lib/scatter/sparse, ten verifier components, three canaries, two supplement pins, four installer stages, stage-0 pip only, binary-only NNx last, Issue #65 Atlas ownership, Issue #66 quantization-notebook ownership, and no containerized Ollama.
- [x] **D10 executability:** every referenced parser/comparator is defined in the plan or already exists in `scripts/verify_repo.py`; current/historical slicing, raw-HTML masking, Result/summary/advisory validation, policy coupling, and ten-input hashes map failures to named `Finding` IDs.
- [x] **Audit cardinality:** `AUDIT_SURFACES` generates six physical commands and merges them into four logical observations; only both supplements and documentation use `--disable-pip`, only supplements use `--no-deps`, and all six require exit 0/1 plus valid nonempty JSON.
- [x] **Zero-skip and output gates:** focused, CI, prequalification, and final NNx runs use warnings-as-errors plus parsed JUnit totals; Tier A/B/C use recursive exact output sets with 18 nested, 6 basename, and 4 basename artifacts and no zero-code notebook.
- [x] **Immutable identities:** feature HEAD, feature PR synthetic merge, develop merge, release PR synthetic merge, and release merge are recorded separately; dispatch evidence is tied to the feature SHA, PR evidence to synthetic merge SHAs, and tree equality prevents content drift.
- [x] **Staging safety:** Task 1 and Task 2 exclude the five preserved Task 3 paths; Task 3 owns them after clean GREEN; generated docs and ignored evidence are absent from every `git add` command.
- [x] **Historical integrity:** r1-r3 and prior commits remain evidence, not final completion claims; Issue #59/#60/#61 records and released history remain immutable; the one stale Issue #61 requirements hash is corrected only in Task 5's current-ledger evidence.
