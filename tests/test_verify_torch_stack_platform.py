from __future__ import annotations

import dataclasses
import importlib.util
import sys
import uuid
import warnings
from importlib.metadata import PackagePath
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from packaging.version import Version

import scripts.verify_torch_stack as verifier
from scripts.verify_torch_stack import (
    IMPORTS,
    CanaryHooks,
    StackEvidence,
    TorchStackVerificationError,
    VerificationHooks,
    verify_torch_stack,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
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
EXPECTED_VERSIONS = {
    "torch": "2.11.0",
    "torchvision": "0.26.0",
    "torchaudio": "2.11.0",
    "pytorch-lightning": "2.6.1",
    "torchmetrics": "1.9.0",
    "torchao": "0.18.0",
    "torch-geometric": "2.8.0.post1",
    "pyg-lib": "0.8.0",
    "torch-scatter": "2.1.2",
    "torch-sparse": "0.6.18",
}
PYG_INDEX_DISTRIBUTIONS = {"pyg-lib", "torch-scatter", "torch-sparse"}
BINARY_DISTRIBUTIONS = PYG_INDEX_DISTRIBUTIONS | {"torch", "torchvision", "torchaudio"}


class PlatformDistribution:
    def __init__(self, name: str, version: str, root: Path, wheel_tag: str) -> None:
        self.version = version
        self.root = root
        self.wheel_tag = wheel_tag
        package = IMPORTS[name].replace(".", "/") + "/__init__.py"
        self.dist_info = name.replace("-", "_") + f"-{Version(version).public}.dist-info"
        self.files: list[PackagePath] | None = [
            self._path(f"{self.dist_info}/WHEEL"),
            self._path(f"{self.dist_info}/RECORD"),
            self._path(package),
        ]

    def _path(self, value: str) -> PackagePath:
        path = PackagePath(value)
        path.dist = self
        return path

    def locate_file(self, path: PackagePath) -> Path:
        return self.root / path

    def read_text(self, filename: str) -> str | None:
        return f"Wheel-Version: 1.0\nTag: {self.wheel_tag}\n" if filename == "WHEEL" else None


class PlatformStack:
    def __init__(self, tmp_path: Path, system: str, machine: str) -> None:
        self.system = system
        self.machine = machine
        self.calls: list[str] = []
        self.distributions: dict[str, PlatformDistribution] = {}
        self.modules: dict[str, ModuleType] = {}
        root = tmp_path / "site-packages"
        if system == "Darwin":
            binary_tag, pyg_local = "cp310-abi3-macosx_11_0_arm64", "pt211"
        else:
            binary_tag, pyg_local = f"cp311-cp311-manylinux_2_28_{machine}", "pt211cpu"
        for distribution, import_name in IMPORTS.items():
            module_path = root / import_name.replace(".", "/") / "__init__.py"
            module_path.parent.mkdir(parents=True, exist_ok=True)
            module_path.touch()
            version = EXPECTED_VERSIONS[distribution]
            if distribution in PYG_INDEX_DISTRIBUTIONS:
                version = f"{version}+{pyg_local}"
            if distribution == "torch" and system == "Linux":
                version = f"{version}+cpu"
            wheel_tag = binary_tag if distribution in BINARY_DISTRIBUTIONS else "py3-none-any"
            self.distributions[distribution] = PlatformDistribution(distribution, version, root, wheel_tag)
            module = ModuleType(import_name)
            module.__file__ = str(module_path)
            if distribution == "torch":
                module.__version__ = version
                module.version = SimpleNamespace(cuda=None)
            self.modules[import_name] = module
        self.hooks = VerificationHooks(
            distribution=self.distributions.__getitem__,
            installed_names=lambda: tuple(self.distributions),
            import_module=self.modules.__getitem__,
            system=lambda: self.system,
            machine=lambda: self.machine,
            nnx_verify=self._nnx,
            canaries=CanaryHooks(*(self._canary(name) for name in EXPECTED_CANARIES)),
        )

    def _canary(self, name: str):
        def canary(modules):
            assert tuple(modules) == tuple(IMPORTS)
            self.calls.append(name)

        return canary

    def _nnx(self) -> object:
        self.calls.append("nnx")
        return SimpleNamespace(mode="canonical-wheel", distribution="thekaveh-nnx", version="0.2.0")

    def version(self, distribution: str, version: str) -> None:
        self.distributions[distribution].version = version
        if distribution == "torch":
            self.modules["torch"].__version__ = version

    def wheel(self, distribution: str, wheel_tag: str) -> None:
        self.distributions[distribution].wheel_tag = wheel_tag

    def remove(self, distribution: str, suffix: str) -> None:
        selected = self.distributions[distribution]
        assert selected.files is not None
        selected.files = [path for path in selected.files if not str(path).endswith(suffix)]

    def split_record(self, distribution: str) -> None:
        selected = self.distributions[distribution]
        assert selected.files is not None
        selected.files = [
            selected._path(f"split-{selected.dist_info}/RECORD") if path.name == "RECORD" else path
            for path in selected.files
        ]


def test_selected_component_and_canary_boundaries_are_exact():
    assert IMPORTS == EXPECTED_IMPORTS
    assert tuple(CanaryHooks.__dataclass_fields__) == EXPECTED_CANARIES


def test_selected_stack_runs_three_canaries_then_nnx(tmp_path: Path) -> None:
    stack = PlatformStack(tmp_path, "Linux", "x86_64")

    evidence = verify_torch_stack(repo=REPO_ROOT, hooks=stack.hooks)

    assert evidence == StackEvidence("Linux", "x86_64", "2.11.0+cpu", "pyg-lib")
    assert stack.calls == ["scatter", "sparse", "sampler", "nnx"]
    assert evidence.backend == "pyg-lib"


@pytest.mark.parametrize("distribution", tuple(EXPECTED_IMPORTS))
@pytest.mark.parametrize("mutation", ("wheel", "record", "split", "ownership"))
def test_selected_component_provenance_is_mandatory(tmp_path: Path, distribution: str, mutation: str) -> None:
    stack = PlatformStack(tmp_path, "Linux", "x86_64")
    if mutation == "wheel":
        stack.remove(distribution, "WHEEL")
    elif mutation == "record":
        stack.remove(distribution, "RECORD")
    elif mutation == "split":
        stack.split_record(distribution)
    else:
        stack.remove(distribution, IMPORTS[distribution].replace(".", "/") + "/__init__.py")

    with pytest.raises(TorchStackVerificationError, match=rf"{distribution}: (wheel|metadata)$"):
        verify_torch_stack(repo=REPO_ROOT, hooks=stack.hooks)

    assert stack.calls == []


@pytest.mark.parametrize("distribution", tuple(EXPECTED_IMPORTS))
def test_selected_component_public_version_is_exact(tmp_path: Path, distribution: str) -> None:
    stack = PlatformStack(tmp_path, "Linux", "x86_64")
    stack.version(distribution, "9.9.9")

    with pytest.raises(TorchStackVerificationError, match=rf"{distribution}: metadata$"):
        verify_torch_stack(repo=REPO_ROOT, hooks=stack.hooks)


@pytest.mark.parametrize("distribution", tuple(PYG_INDEX_DISTRIBUTIONS))
def test_linux_pyg_index_wheels_require_pt211cpu(tmp_path: Path, distribution: str) -> None:
    stack = PlatformStack(tmp_path, "Linux", "x86_64")
    stack.version(distribution, EXPECTED_VERSIONS[distribution])
    with pytest.raises(TorchStackVerificationError, match=rf"^torch stack verification failed: {distribution}: abi$"):
        verify_torch_stack(repo=REPO_ROOT, hooks=stack.hooks)


@pytest.mark.parametrize("local", ("pt211", "pt211cu128", "cu128", "arbitrary"))
def test_linux_rejects_wrong_pyg_local_versions(tmp_path: Path, local: str) -> None:
    stack = PlatformStack(tmp_path, "Linux", "x86_64")
    stack.version("pyg-lib", f"0.8.0+{local}")
    with pytest.raises(TorchStackVerificationError, match=r"pyg-lib: abi$"):
        verify_torch_stack(repo=REPO_ROOT, hooks=stack.hooks)


@pytest.mark.parametrize("local", ("pt211", None))
def test_darwin_arm64_accepts_selected_or_absent_pyg_local_after_all_gates(tmp_path: Path, local: str | None) -> None:
    stack = PlatformStack(tmp_path, "Darwin", "arm64")
    stack.version("pyg-lib", "0.8.0" + (f"+{local}" if local else ""))

    evidence = verify_torch_stack(repo=REPO_ROOT, hooks=stack.hooks)

    assert evidence == StackEvidence("Darwin", "arm64", "2.11.0", "pyg-lib")
    assert stack.calls == ["scatter", "sparse", "sampler", "nnx"]


@pytest.mark.parametrize("local", ("pt211cpu", "pt211cu128", "arbitrary"))
def test_darwin_arm64_rejects_wrong_pyg_local_versions(tmp_path: Path, local: str) -> None:
    stack = PlatformStack(tmp_path, "Darwin", "arm64")
    stack.version("pyg-lib", f"0.8.0+{local}")
    with pytest.raises(TorchStackVerificationError, match=r"pyg-lib: abi$"):
        verify_torch_stack(repo=REPO_ROOT, hooks=stack.hooks)


@pytest.mark.parametrize(
    ("system", "machine", "distribution", "wheel_tag", "category"),
    (
        ("Darwin", "arm64", "pyg-lib", "cp310-abi3-macosx_11_0_x86_64", "platform"),
        ("Linux", "x86_64", "pyg-lib", "cp311-cp311-manylinux_2_28_aarch64", "platform"),
        ("Linux", "x86_64", "torch-scatter", "cp310-cp310-manylinux_2_28_x86_64", "abi"),
        ("Linux", "x86_64", "torch", "py3-none-any", "wheel"),
    ),
)
def test_platform_and_python_abi_provenance_is_mandatory(
    tmp_path: Path, system: str, machine: str, distribution: str, wheel_tag: str, category: str,
) -> None:
    stack = PlatformStack(tmp_path, system, machine)
    stack.wheel(distribution, wheel_tag)
    with pytest.raises(TorchStackVerificationError, match=rf"{distribution}: {category}$"):
        verify_torch_stack(repo=REPO_ROOT, hooks=stack.hooks)


def test_linux_cpu_and_nvidia_provenance_is_mandatory(tmp_path: Path) -> None:
    stack = PlatformStack(tmp_path, "Linux", "x86_64")
    stack.modules["torch"].version.cuda = "12.8"
    with pytest.raises(TorchStackVerificationError, match=r"torch: cpu$"):
        verify_torch_stack(repo=REPO_ROOT, hooks=stack.hooks)

    stack = PlatformStack(tmp_path, "Linux", "x86_64")
    stack.hooks = dataclasses.replace(stack.hooks, installed_names=lambda: ("nvidia-cublas-cu12",))
    with pytest.raises(TorchStackVerificationError, match=r"torch: cpu$"):
        verify_torch_stack(repo=REPO_ROOT, hooks=stack.hooks)


@pytest.mark.parametrize("name", EXPECTED_CANARIES)
@pytest.mark.parametrize("failure", ("raise", "warn"))
def test_runtime_canaries_fail_closed_on_exceptions_and_warnings(tmp_path: Path, name: str, failure: str) -> None:
    stack = PlatformStack(tmp_path, "Linux", "x86_64")

    def broken(modules):
        del modules
        if failure == "warn":
            warnings.warn("canary warning")
        else:
            raise RuntimeError("canary failed")

    stack.hooks = dataclasses.replace(stack.hooks, canaries=dataclasses.replace(stack.hooks.canaries, **{name: broken}))
    category = "sampler" if name == "sampler" else "operator"
    with pytest.raises(TorchStackVerificationError, match=rf"{name}: {category}$"):
        verify_torch_stack(repo=REPO_ROOT, hooks=stack.hooks)


def test_nnx_is_warning_free(tmp_path: Path) -> None:
    stack = PlatformStack(tmp_path, "Linux", "x86_64")

    def nnx():
        stack.calls.append("nnx")
        warnings.warn("nnx warning")
        return stack._nnx()

    stack.hooks = dataclasses.replace(stack.hooks, nnx_verify=nnx)
    with pytest.raises(TorchStackVerificationError, match=r"nnx: nnx$"):
        verify_torch_stack(repo=REPO_ROOT, hooks=stack.hooks)
    assert stack.calls[:3] == list(EXPECTED_CANARIES)


def test_default_nnx_hook_rejects_noncanonical_evidence(monkeypatch) -> None:
    monkeypatch.setattr(
        verifier,
        "verify_nnx_install",
        lambda *, environ: SimpleNamespace(
            mode="editable-development", distribution="thekaveh-nnx", version="0.2.0",
        ),
    )

    with pytest.raises(TorchStackVerificationError, match=r"nnx: nnx$"):
        verifier.DEFAULT_HOOKS.nnx_verify()


def _import_mutated_verifier(tmp_path: Path, source: str) -> ModuleType:
    module_name = f"_issue62_mutated_verifier_{uuid.uuid4().hex}"
    module_path = tmp_path / f"{module_name}.py"
    module_path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        assert sys.modules.pop(module_name, None) is module
    return module


def test_mutated_verifier_loader_is_unique_clean_and_fake_hook_compatible(tmp_path: Path) -> None:
    source = (REPO_ROOT / "scripts/verify_torch_stack.py").read_text(encoding="utf-8")
    first = _import_mutated_verifier(tmp_path, source)
    second = _import_mutated_verifier(tmp_path, source)
    assert first.__name__ != second.__name__
    assert first.__name__ not in sys.modules
    assert second.__name__ not in sys.modules
    stack = PlatformStack(tmp_path, "Linux", "x86_64")
    evidence = first.verify_torch_stack(repo=REPO_ROOT, hooks=stack.hooks)
    assert (evidence.system, evidence.machine, evidence.backend) == ("Linux", "x86_64", "pyg-lib")


def test_linux_absent_local_rejection_cannot_be_removed(tmp_path: Path) -> None:
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


@pytest.mark.parametrize("omitted", (*EXPECTED_CANARIES, "nnx"))
def test_source_mutations_cannot_omit_a_canary_or_nnx(tmp_path: Path, omitted: str) -> None:
    source = (REPO_ROOT / "scripts/verify_torch_stack.py").read_text(encoding="utf-8")
    original = '    _run_warning_free("nnx", "nnx", hooks.nnx_verify)' if omitted == "nnx" else "        _run_warning_free(name, category, canary, modules)"
    replacement = "    object()" if omitted == "nnx" else f'        if name != "{omitted}":\n            _run_warning_free(name, category, canary, modules)'
    mutated = source.replace(original, replacement, 1)
    assert mutated != source
    module = _import_mutated_verifier(tmp_path, mutated)
    stack = PlatformStack(tmp_path, "Linux", "x86_64")
    module.verify_torch_stack(repo=REPO_ROOT, hooks=stack.hooks)
    with pytest.raises(AssertionError):
        assert stack.calls == ["scatter", "sparse", "sampler", "nnx"]


def test_nnx_is_last_and_return_is_immediate(tmp_path: Path) -> None:
    source = (REPO_ROOT / "scripts/verify_torch_stack.py").read_text(encoding="utf-8")
    moved = source.replace(
        "    _verify_cpu_runtime(contract, modules[\"torch\"])\n",
        "    _verify_cpu_runtime(contract, modules[\"torch\"])\n"
        "    _run_warning_free(\"nnx\", \"nnx\", hooks.nnx_verify)\n",
        1,
    ).replace('    _run_warning_free("nnx", "nnx", hooks.nnx_verify)\n    return evidence', "    return evidence", 1)
    assert moved != source
    module = _import_mutated_verifier(tmp_path, moved)
    stack = PlatformStack(tmp_path, "Linux", "x86_64")
    module.verify_torch_stack(repo=REPO_ROOT, hooks=stack.hooks)
    with pytest.raises(AssertionError):
        assert stack.calls == ["scatter", "sparse", "sampler", "nnx"]


@dataclasses.dataclass
class _FakeBackendOperation:
    calls: int = 0

    def __call__(self, *args: object, **kwargs: object) -> tuple[object, ...]:
        del args, kwargs
        self.calls += 1
        return ()


def _sampler_test_rig(monkeypatch, module=verifier):
    typing = ModuleType("torch_geometric.typing")
    typing.WITH_PYG_LIB = True
    typing.WITH_TORCH_SPARSE = True
    pyg_operation = _FakeBackendOperation()
    sparse_operation = _FakeBackendOperation()
    torch = SimpleNamespace(tensor=lambda value: value, ops=SimpleNamespace(
        pyg=SimpleNamespace(neighbor_sample=pyg_operation),
        torch_sparse=SimpleNamespace(neighbor_sample=sparse_operation),
    ))

    class FakeData:
        def __init__(self, **values: object) -> None:
            self.__dict__.update(values)

    class FakeNeighborLoader:
        def __init__(self, data: object, **kwargs: object) -> None:
            del data
            assert kwargs == {"num_neighbors": [-1], "input_nodes": [0], "batch_size": 1, "shuffle": False, "num_workers": 0}

        def __iter__(self):
            if typing.WITH_PYG_LIB:
                torch.ops.pyg.neighbor_sample("colptr", "row", "seed", [-1], None, None, True, False, True, False, "uniform", True)
            else:
                torch.ops.torch_sparse.neighbor_sample("colptr", "row", "seed", [-1], False, True)
            yield SimpleNamespace(batch_size=1, edge_index=SimpleNamespace(numel=lambda: 2))

    geometric = SimpleNamespace(data=SimpleNamespace(Data=FakeData), loader=SimpleNamespace(NeighborLoader=FakeNeighborLoader))
    original_import = module.importlib.import_module
    monkeypatch.setattr(module.importlib, "import_module", lambda name: typing if name == "torch_geometric.typing" else original_import(name))
    return {"torch": torch, "torch-geometric": geometric}, typing, pyg_operation, sparse_operation


def test_sampler_canary_uses_pyg_then_forced_sparse_and_restores_state(monkeypatch) -> None:
    modules, typing, pyg_operation, sparse_operation = _sampler_test_rig(monkeypatch)
    original_flag = typing.WITH_PYG_LIB
    original_pyg = modules["torch"].ops.pyg.neighbor_sample
    original_sparse = modules["torch"].ops.torch_sparse.neighbor_sample
    verifier._sampler_canary(modules)
    assert (pyg_operation.calls, sparse_operation.calls) == (1, 1)
    assert typing.WITH_PYG_LIB is original_flag
    assert modules["torch"].ops.pyg.neighbor_sample is original_pyg
    assert modules["torch"].ops.torch_sparse.neighbor_sample is original_sparse


@pytest.mark.parametrize(
    ("before", "after"),
    (("typing.WITH_PYG_LIB = True", "typing.WITH_PYG_LIB = False"), ("typing.WITH_PYG_LIB = False", "typing.WITH_PYG_LIB = True")),
    ids=("delete-preferred-selection", "delete-sparse-fallback"),
)
def test_sampler_backend_selection_mutations_are_killed(tmp_path: Path, monkeypatch, before: str, after: str) -> None:
    source = (REPO_ROOT / "scripts/verify_torch_stack.py").read_text(encoding="utf-8")
    mutated = source.replace(before, after, 1)
    assert mutated != source
    module = _import_mutated_verifier(tmp_path, mutated)
    modules, _, _, _ = _sampler_test_rig(monkeypatch, module)
    with pytest.raises(RuntimeError, match=r"sampler"):
        module._sampler_canary(modules)
