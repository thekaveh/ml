from __future__ import annotations

import ast
import dataclasses
import importlib.util
import sys
import uuid
import warnings
from dataclasses import dataclass
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


@dataclass(frozen=True)
class ImportWarningSpec:
    category: type[Warning]
    message: str
    filename: Path
    lineno: int


class TorchScriptDeprecationSubclass(DeprecationWarning):
    pass


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
        if name == "torch":
            warning_path = root / "torch/jit/_script.py"
            warning_path.parent.mkdir(parents=True, exist_ok=True)
            warning_path.touch()
            assert self.files is not None
            self.files.append(self._path("torch/jit/_script.py"))

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
        self.tmp_path = tmp_path
        self.system = system
        self.machine = machine
        self.calls: list[str] = []
        self.import_calls: list[str] = []
        self.import_warnings: dict[str, tuple[ImportWarningSpec, ...]] = {}
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
            import_module=self._import_module,
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

    def warn_on_import(
        self,
        component: str,
        *records: ImportWarningSpec,
    ) -> None:
        self.import_warnings[IMPORTS[component]] = records

    def _import_module(self, import_name: str) -> ModuleType:
        self.import_calls.append(import_name)
        module = self.modules[import_name]
        for record in self.import_warnings.get(import_name, ()):
            warnings.warn_explicit(
                record.message,
                record.category,
                filename=str(record.filename),
                lineno=record.lineno,
            )
        return module

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


def _torch_warning_path(stack: PlatformStack) -> Path:
    distribution = stack.distributions["torch"]
    assert distribution.files is not None
    (owned,) = tuple(
        path for path in distribution.files
        if path.as_posix() == "torch/jit/_script.py"
    )
    return Path(owned.locate()).resolve(strict=True)


def _exact_warning(stack: PlatformStack, *, lineno: int = 73) -> ImportWarningSpec:
    return ImportWarningSpec(
        DeprecationWarning,
        "`torch.jit.script` is deprecated. Please switch to "
        "`torch.compile` or `torch.export`.",
        _torch_warning_path(stack),
        lineno,
    )


def _captured_warning(
    stack: PlatformStack,
    component: str,
    record: ImportWarningSpec,
) -> tuple[warnings.WarningMessage, ...]:
    stack.warn_on_import(component, record)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        stack._import_module(IMPORTS[component])
    return tuple(caught)


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
    rejected = False
    try:
        module.verify_torch_stack(repo=REPO_ROOT, hooks=stack.hooks)
    except module.TorchStackVerificationError as error:
        rejected = str(error).endswith("pyg-lib: abi")
    with pytest.raises(AssertionError):
        assert rejected


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


def test_import_warning_debt_constants_are_literal_and_exact() -> None:
    assert verifier._IMPORT_WARNING_DEBT_KEYS == frozenset({
        ("2.11.0", "torch-geometric", "2.8.0.post1"),
        ("2.11.0", "torch-sparse", "0.6.18"),
    })
    assert verifier._TORCH_SCRIPT_WARNING_MESSAGE == (
        "`torch.jit.script` is deprecated. Please switch to "
        "`torch.compile` or `torch.export`."
    )
    assert verifier._TORCH_SCRIPT_WARNING_PATH.as_posix() == "torch/jit/_script.py"


@pytest.mark.parametrize("component", ("torch-geometric", "torch-sparse"))
@pytest.mark.parametrize("count", (1, 2, 19))
def test_import_warning_debt_accepts_every_exact_record(
    tmp_path: Path,
    component: str,
    count: int,
) -> None:
    stack = PlatformStack(tmp_path, "Darwin", "arm64")
    stack.warn_on_import(component, *(_exact_warning(stack) for _ in range(count)))

    evidence = verify_torch_stack(repo=REPO_ROOT, hooks=stack.hooks)

    assert evidence == StackEvidence("Darwin", "arm64", "2.11.0", "pyg-lib")


@pytest.mark.parametrize("lineno", (1, 73, 10_000))
def test_exact_warning_line_number_is_not_part_of_the_predicate(
    tmp_path: Path,
    lineno: int,
) -> None:
    stack = PlatformStack(tmp_path, "Darwin", "arm64")
    stack.warn_on_import("torch-geometric", _exact_warning(stack, lineno=lineno))
    assert verify_torch_stack(repo=REPO_ROOT, hooks=stack.hooks).backend == "pyg-lib"


def test_production_warning_predicate_does_not_read_lineno() -> None:
    source = (REPO_ROOT / "scripts/verify_torch_stack.py").read_text(encoding="utf-8")
    validator = source.split("def _validate_import_warning_group(", 1)[1].split(
        "def _import_with_selected_warning_boundary(", 1,
    )[0]
    assert ".lineno" not in validator


@pytest.mark.parametrize(
    ("component", "mutate"),
    (
        ("torch-geometric", lambda stack, item: dataclasses.replace(item, category=UserWarning)),
        ("torch-geometric", lambda stack, item: dataclasses.replace(
            item, category=TorchScriptDeprecationSubclass,
        )),
        ("torch-geometric", lambda stack, item: dataclasses.replace(
            item, message="`torch.jit.script` is deprecated.",
        )),
        ("torch-geometric", lambda stack, item: dataclasses.replace(
            item, message=item.message.removesuffix("."),
        )),
        ("torch-geometric", lambda stack, item: dataclasses.replace(
            item, filename=stack.tmp_path / "outside/_script.py",
        )),
        ("torch-geometric", lambda stack, item: dataclasses.replace(
            item, filename=stack.tmp_path / "outside/torch/jit/_script.py",
        )),
        ("torch-sparse", lambda stack, item: dataclasses.replace(item, category=UserWarning)),
        ("torch-sparse", lambda stack, item: dataclasses.replace(
            item, message=item.message + " ",
        )),
    ),
    ids=(
        "wrong-category", "warning-subclass", "message-prefix", "punctuation",
        "same-basename-outsider", "matching-suffix-outsider",
        "sparse-wrong-category", "sparse-trailing-space",
    ),
)
def test_import_warning_near_misses_fail_closed(tmp_path, component, mutate):
    stack = PlatformStack(tmp_path, "Darwin", "arm64")
    item = _exact_warning(stack)
    mutated = mutate(stack, item)
    mutated.filename.parent.mkdir(parents=True, exist_ok=True)
    mutated.filename.touch(exist_ok=True)
    stack.warn_on_import(component, mutated)
    with pytest.raises(
        TorchStackVerificationError,
        match=rf"^torch stack verification failed: {component}: abi$",
    ):
        verify_torch_stack(repo=REPO_ROOT, hooks=stack.hooks)


@pytest.mark.parametrize("tail", ("message", "category", "extra"))
def test_import_warning_mixed_groups_fail_closed(tmp_path: Path, tail: str) -> None:
    stack = PlatformStack(tmp_path, "Darwin", "arm64")
    exact = _exact_warning(stack)
    if tail == "message":
        records = (exact, dataclasses.replace(exact, message=exact.message + " "))
    elif tail == "category":
        records = (exact, dataclasses.replace(exact, category=UserWarning))
    else:
        records = (exact, exact, dataclasses.replace(exact, message="extra warning"))
    stack.warn_on_import("torch-geometric", *records)

    with pytest.raises(
        TorchStackVerificationError,
        match=r"^torch stack verification failed: torch-geometric: abi$",
    ):
        verify_torch_stack(repo=REPO_ROOT, hooks=stack.hooks)


def test_exact_warning_from_wrong_outer_component_reaches_cli_outer_capture(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    stack = PlatformStack(tmp_path, "Darwin", "arm64")
    stack.warn_on_import("torchvision", _exact_warning(stack))
    monkeypatch.setattr(
        verifier,
        "verify_torch_stack",
        lambda: verify_torch_stack(repo=REPO_ROOT, hooks=stack.hooks),
    )

    assert verifier.main() == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "torch stack verification failed: verifier: metadata\n"


@pytest.mark.parametrize(
    ("outer_component", "changed_component", "version"),
    (
        ("torch-geometric", "torch", "2.11.1"),
        ("torch-geometric", "torch-geometric", "2.8.1"),
        ("torch-sparse", "torch-sparse", "0.6.19"),
    ),
)
def test_warning_debt_key_rejects_each_public_version_mutation(
    tmp_path: Path,
    outer_component: str,
    changed_component: str,
    version: str,
) -> None:
    stack = PlatformStack(tmp_path, "Darwin", "arm64")
    stack.distributions[changed_component].version = version
    caught = _captured_warning(stack, outer_component, _exact_warning(stack))

    with pytest.raises(
        TorchStackVerificationError,
        match=rf"^torch stack verification failed: {outer_component}: abi$",
    ):
        verifier._validate_import_warning_group(
            torch_distribution=stack.distributions["torch"],
            outer_component=outer_component,
            outer_distribution=stack.distributions[outer_component],
            caught=caught,
        )


@pytest.mark.parametrize(
    ("component", "version"),
    (
        ("torch", "2.11.0+cpu"),
        ("torch", "2.11.1"),
        ("torch-geometric", "2.8.0.post1+foreign"),
        ("torch-geometric", "2.8.1"),
        ("torch-sparse", "0.6.18+pt211cpu"),
        ("torch-sparse", "0.6.19+pt211"),
    ),
    ids=(
        "torch-foreign-local", "torch-wrong-public",
        "geometric-foreign-local", "geometric-wrong-public",
        "sparse-foreign-local", "sparse-wrong-public",
    ),
)
def test_full_verifier_rejects_foreign_version_before_debt_probe(
    tmp_path: Path,
    component: str,
    version: str,
) -> None:
    stack = PlatformStack(tmp_path, "Darwin", "arm64")
    stack.version(component, version)
    stack.warn_on_import(component, _exact_warning(stack))
    with pytest.raises(
        TorchStackVerificationError,
        match=rf"^torch stack verification failed: {component}: (metadata|abi)$",
    ):
        verify_torch_stack(repo=REPO_ROOT, hooks=stack.hooks)
    assert IMPORTS[component] not in stack.import_calls


def _torch_script_entries(stack: PlatformStack) -> list[PackagePath]:
    files = stack.distributions["torch"].files
    assert files is not None
    return [path for path in files if path.as_posix() == "torch/jit/_script.py"]


@pytest.mark.parametrize(
    "mutation",
    ("missing", "duplicate", "unlocatable", "non-file", "foreign-owner", "symlink"),
)
def test_warning_origin_inventory_fails_closed(tmp_path: Path, mutation: str) -> None:
    stack = PlatformStack(tmp_path, "Darwin", "arm64")
    exact = _exact_warning(stack)
    caught = _captured_warning(stack, "torch-geometric", exact)
    torch_distribution = stack.distributions["torch"]
    assert torch_distribution.files is not None
    if mutation == "missing":
        torch_distribution.files = [
            path for path in torch_distribution.files
            if path.as_posix() != "torch/jit/_script.py"
        ]
    elif mutation == "duplicate":
        torch_distribution.files.append(torch_distribution._path("torch/jit/_script.py"))
    elif mutation == "unlocatable":
        torch_distribution.locate_file = lambda path: (_ for _ in ()).throw(OSError("unlocatable"))
    elif mutation == "non-file":
        _torch_warning_path(stack).unlink()
    elif mutation == "foreign-owner":
        (entry,) = _torch_script_entries(stack)
        entry.dist = stack.distributions["torchvision"]
    else:
        warning_path = _torch_warning_path(stack)
        target = tmp_path / "foreign/_script.py"
        target.parent.mkdir(parents=True)
        target.touch()
        warning_path.unlink()
        warning_path.symlink_to(target)
    with pytest.raises(
        TorchStackVerificationError,
        match=r"^torch stack verification failed: torch-geometric: abi$",
    ):
        verifier._validate_import_warning_group(
            torch_distribution=torch_distribution,
            outer_component="torch-geometric",
            outer_distribution=stack.distributions["torch-geometric"],
            caught=caught,
        )


def test_warning_origin_ignores_unrelated_duplicate_basename(tmp_path: Path) -> None:
    stack = PlatformStack(tmp_path, "Darwin", "arm64")
    distribution = stack.distributions["torch"]
    unrelated = distribution.root / "unrelated/_script.py"
    unrelated.parent.mkdir(parents=True)
    unrelated.touch()
    assert distribution.files is not None
    distribution.files.append(distribution._path("unrelated/_script.py"))
    stack.warn_on_import("torch-geometric", _exact_warning(stack))

    assert verify_torch_stack(repo=REPO_ROOT, hooks=stack.hooks).backend == "pyg-lib"


@pytest.mark.parametrize(
    "error_factory",
    (
        lambda: RuntimeError("https://secret.invalid token traceback /private/path"),
        lambda: TorchStackVerificationError("inventory", "metadata"),
    ),
    ids=("ordinary-exception", "verification-error"),
)
@pytest.mark.parametrize("component", ("torchvision", "torch-geometric"))
def test_selected_import_exception_fails_with_stable_abi_error(
    tmp_path: Path,
    component: str,
    error_factory,
) -> None:
    stack = PlatformStack(tmp_path, "Darwin", "arm64")

    def import_module(import_name: str) -> ModuleType:
        if import_name == IMPORTS[component]:
            raise error_factory()
        return stack._import_module(import_name)

    stack.hooks = dataclasses.replace(stack.hooks, import_module=import_module)
    with pytest.raises(
        TorchStackVerificationError,
        match=rf"^torch stack verification failed: {component}: abi$",
    ):
        verify_torch_stack(repo=REPO_ROOT, hooks=stack.hooks)


def test_cached_repeated_import_zero_warning_is_normal_success(tmp_path: Path) -> None:
    stack = PlatformStack(tmp_path, "Darwin", "arm64")
    exact = _exact_warning(stack)
    warned = False

    def import_module(import_name: str) -> ModuleType:
        nonlocal warned
        module = stack._import_module(import_name)
        if import_name == "torch_geometric" and not warned:
            warned = True
            warnings.warn_explicit(
                exact.message,
                exact.category,
                filename=str(exact.filename),
                lineno=exact.lineno,
            )
        return module

    stack.hooks = dataclasses.replace(stack.hooks, import_module=import_module)
    first = verify_torch_stack(repo=REPO_ROOT, hooks=stack.hooks)
    second = verify_torch_stack(repo=REPO_ROOT, hooks=stack.hooks)
    assert first == second == StackEvidence("Darwin", "arm64", "2.11.0", "pyg-lib")
    assert stack.import_calls.count("torch_geometric") == 2


@pytest.mark.parametrize(
    "order",
    (("torch-geometric", "torch-sparse"), ("torch-sparse", "torch-geometric")),
)
def test_selected_import_order_accepts_one_exact_group_and_one_zero_group(
    tmp_path: Path,
    order: tuple[str, str],
) -> None:
    stack = PlatformStack(tmp_path, "Darwin", "arm64")
    stack.warn_on_import(order[0], _exact_warning(stack))
    for component in order:
        pin = verifier.StackPin(
            component,
            IMPORTS[component],
            Version(EXPECTED_VERSIONS[component]),
        )
        module = verifier._import_with_selected_warning_boundary(
            pin,
            stack.distributions[component],
            stack.distributions["torch"],
            stack.hooks,
        )
        assert module.__name__ == IMPORTS[component]
    assert stack.import_calls == [IMPORTS[name] for name in order]


@pytest.mark.parametrize(
    ("boundary", "expected_component", "expected_category"),
    (
        ("scatter", "scatter", "operator"),
        ("sparse", "sparse", "operator"),
        ("sampler", "sampler", "sampler"),
        ("nnx", "nnx", "nnx"),
    ),
)
def test_exact_debt_warning_outside_import_boundary_stays_fatal(
    tmp_path: Path,
    boundary: str,
    expected_component: str,
    expected_category: str,
) -> None:
    stack = PlatformStack(tmp_path, "Darwin", "arm64")
    exact = _exact_warning(stack)

    def warn(*args: object) -> object:
        del args
        warnings.warn_explicit(
            exact.message,
            exact.category,
            filename=str(exact.filename),
            lineno=exact.lineno,
        )
        return SimpleNamespace(mode="canonical-wheel", distribution="thekaveh-nnx", version="0.2.0")

    if boundary == "nnx":
        stack.hooks = dataclasses.replace(stack.hooks, nnx_verify=warn)
    else:
        stack.hooks = dataclasses.replace(
            stack.hooks,
            canaries=dataclasses.replace(stack.hooks.canaries, **{boundary: warn}),
        )
    with pytest.raises(
        TorchStackVerificationError,
        match=rf"^torch stack verification failed: {expected_component}: {expected_category}$",
    ):
        verify_torch_stack(repo=REPO_ROOT, hooks=stack.hooks)


def test_production_never_evicts_or_replaces_sys_modules() -> None:
    source = (REPO_ROOT / "scripts/verify_torch_stack.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    def is_sys_modules(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "sys"
            and node.attr == "modules"
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.Delete):
            assert not any(
                isinstance(target, ast.Subscript) and is_sys_modules(target.value)
                for target in node.targets
            )
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            assert not any(
                is_sys_modules(target)
                or (isinstance(target, ast.Subscript) and is_sys_modules(target.value))
                for target in targets
            )
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert not (is_sys_modules(node.func.value) and node.func.attr == "pop")


def _assert_warning_filters_are_local_always_captures_only(source: str) -> None:
    tree = ast.parse(source)
    allowed_functions = {"_capture_selected_import", "_run_warning_free", "main"}
    parent_function: dict[ast.AST, str | None] = {}
    parents: dict[ast.AST, ast.AST | None] = {}

    def record(
        node: ast.AST,
        function: str | None = None,
        parent: ast.AST | None = None,
    ) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function = node.name
        parent_function[node] = function
        parents[node] = parent
        for child in ast.iter_child_nodes(node):
            record(child, function, node)

    def is_inside_warning_capture(node: ast.AST) -> bool:
        parent = parents[node]
        while parent is not None:
            if isinstance(parent, ast.With):
                for item in parent.items:
                    context = item.context_expr
                    if (
                        isinstance(context, ast.Call)
                        and isinstance(context.func, ast.Attribute)
                        and isinstance(context.func.value, ast.Name)
                        and context.func.value.id == "warnings"
                        and context.func.attr == "catch_warnings"
                    ):
                        return True
            parent = parents[parent]
        return False

    record(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if isinstance(node.func.value, ast.Name) and node.func.value.id == "warnings":
            assert node.func.attr != "filterwarnings"
            if node.func.attr == "simplefilter":
                assert parent_function[node] in allowed_functions
                assert is_inside_warning_capture(node)
                assert isinstance(node.args[0], ast.Constant) and node.args[0].value == "always"


def test_production_warning_filters_are_local_always_captures_only() -> None:
    source = (REPO_ROOT / "scripts/verify_torch_stack.py").read_text(encoding="utf-8")
    _assert_warning_filters_are_local_always_captures_only(source)


def test_module_global_warning_filter_source_mutation_is_killed() -> None:
    source = (REPO_ROOT / "scripts/verify_torch_stack.py").read_text(encoding="utf-8")
    before = "import warnings\n"
    assert source.count(before) == 1
    mutated = source.replace(before, before + 'warnings.simplefilter("always")\n', 1)
    with pytest.raises(AssertionError):
        _assert_warning_filters_are_local_always_captures_only(mutated)


def test_warning_filter_moved_outside_local_capture_is_killed() -> None:
    source = (REPO_ROOT / "scripts/verify_torch_stack.py").read_text(encoding="utf-8")
    before = """    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module = hooks.import_module(import_name)
"""
    after = """    warnings.simplefilter("always")
    with warnings.catch_warnings(record=True) as caught:
        module = hooks.import_module(import_name)
"""
    assert source.count(before) == 1
    mutated = source.replace(before, after, 1)
    with pytest.raises(AssertionError):
        _assert_warning_filters_are_local_always_captures_only(mutated)


IMPORT_WARNING_SOURCE_MUTATIONS = (
    ("record.category is not DeprecationWarning", "not issubclass(record.category, DeprecationWarning)"),
    ("str(record.message) != _TORCH_SCRIPT_WARNING_MESSAGE", "not str(record.message).startswith('`torch.jit.script` is deprecated')"),
    ("resolved != expected_origin", "False"),
    ("if pin.distribution in _IMPORT_WARNING_OUTER_COMPONENTS:", "if True:"),
)


def _mutated_verifier_rejects_geometric_warning(module: ModuleType, stack: PlatformStack) -> bool:
    try:
        module.verify_torch_stack(repo=REPO_ROOT, hooks=stack.hooks)
    except module.TorchStackVerificationError as error:
        return str(error) == "torch stack verification failed: torch-geometric: abi"
    return False


@pytest.mark.parametrize(("before", "after"), IMPORT_WARNING_SOURCE_MUTATIONS)
def test_import_warning_source_mutations_are_killed(
    tmp_path: Path,
    before: str,
    after: str,
) -> None:
    source = (REPO_ROOT / "scripts/verify_torch_stack.py").read_text(encoding="utf-8")
    assert source.count(before) == 1
    mutated_source = source.replace(before, after, 1)
    module = _import_mutated_verifier(tmp_path, mutated_source)
    stack = PlatformStack(tmp_path, "Darwin", "arm64")
    exact = _exact_warning(stack)
    if before.startswith("record.category"):
        record = dataclasses.replace(exact, category=TorchScriptDeprecationSubclass)
    elif before.startswith("str(record.message)"):
        record = dataclasses.replace(exact, message="`torch.jit.script` is deprecated")
    elif before.startswith("resolved"):
        record = dataclasses.replace(exact, filename=stack.tmp_path / "foreign.py")
    else:
        stack.warn_on_import("torchvision", exact)
        with pytest.raises(
            module.TorchStackVerificationError,
            match=r"^torch stack verification failed: torchvision: abi$",
        ):
            module.verify_torch_stack(repo=REPO_ROOT, hooks=stack.hooks)
        return
    record.filename.parent.mkdir(parents=True, exist_ok=True)
    record.filename.touch(exist_ok=True)
    stack.warn_on_import("torch-geometric", record)

    with pytest.raises(AssertionError):
        assert _mutated_verifier_rejects_geometric_warning(module, stack)
def test_warning_origin_validation_call_cannot_be_deleted(tmp_path: Path) -> None:
    source = (REPO_ROOT / "scripts/verify_torch_stack.py").read_text(encoding="utf-8")
    before = """expected_origin = _torch_script_warning_origin(
            torch_distribution,
            component=outer_component,
        )"""
    assert source.count(before) == 1
    after = "expected_origin = Path(caught[0].filename).resolve(strict=True)"
    module = _import_mutated_verifier(tmp_path, source.replace(before, after, 1))
    stack = PlatformStack(tmp_path, "Darwin", "arm64")
    outsider = stack.tmp_path / "outside/_script.py"
    outsider.parent.mkdir(parents=True)
    outsider.touch()
    stack.warn_on_import(
        "torch-geometric",
        dataclasses.replace(_exact_warning(stack), filename=outsider),
    )
    with pytest.raises(AssertionError):
        assert _mutated_verifier_rejects_geometric_warning(module, stack)


def test_warning_validator_cannot_ignore_a_mixed_tail(tmp_path: Path) -> None:
    source = (REPO_ROOT / "scripts/verify_torch_stack.py").read_text(encoding="utf-8")
    before = "for record in caught:"
    assert source.count(before) == 1
    module = _import_mutated_verifier(tmp_path, source.replace(before, "for record in caught[:1]:", 1))
    stack = PlatformStack(tmp_path, "Darwin", "arm64")
    exact = _exact_warning(stack)
    stack.warn_on_import(
        "torch-geometric",
        exact,
        dataclasses.replace(exact, message="extra warning"),
    )
    with pytest.raises(AssertionError):
        assert _mutated_verifier_rejects_geometric_warning(module, stack)


def test_invalid_warning_group_cli_redacts_all_payloads(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    stack = PlatformStack(tmp_path, "Darwin", "arm64")
    secret_path = tmp_path / "secret/token/_script.py"
    secret_path.parent.mkdir(parents=True)
    secret_path.touch()
    payload = (
        "https://user:token@private.invalid/simple warning payload "
        "Traceback (most recent call last)"
    )
    stack.warn_on_import(
        "torch-geometric",
        ImportWarningSpec(UserWarning, payload, secret_path, 919),
    )
    monkeypatch.setattr(
        verifier,
        "verify_torch_stack",
        lambda: verify_torch_stack(repo=REPO_ROOT, hooks=stack.hooks),
    )

    assert verifier.main() == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "torch stack verification failed: torch-geometric: abi\n"
    assert payload not in captured.err
    assert str(secret_path) not in captured.err
