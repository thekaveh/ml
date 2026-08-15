"""Verify the repository's installed Torch stack and mandatory CPU operators."""

from __future__ import annotations

import importlib
import platform
import re
import sys
import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from importlib import metadata
from importlib.metadata import PackagePath
from io import StringIO
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Protocol

from packaging.tags import Tag, compatible_tags, cpython_tags, parse_tag
from packaging.version import Version

from scripts.verify_nnx_install import verify_nnx_install


REPO_ROOT = Path(__file__).resolve().parent.parent
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

_CORE_MANIFEST = "torch-core-requirements.txt"
_ECOSYSTEM_MANIFEST = "torch-ecosystem-requirements.txt"
_RUNTIME_MANIFEST = "torch-requirements.txt"
_AUDIT_MANIFEST = "torch-audit-requirements.txt"
_EXTENSION_AUDIT_MANIFEST = "pyg-extension-audit-requirements.txt"
_MANIFEST_NAMES = (
    _CORE_MANIFEST,
    _ECOSYSTEM_MANIFEST,
    _RUNTIME_MANIFEST,
    _AUDIT_MANIFEST,
    _EXTENSION_AUDIT_MANIFEST,
)
_ERROR_CATEGORIES = frozenset(
    ("manifest", "metadata", "wheel", "platform", "cpu", "abi", "operator", "sampler", "nnx")
)
_CORE_NAMES = frozenset(("torch", "torchvision", "torchaudio"))
_ECOSYSTEM_NAMES = frozenset(("pytorch-lightning", "torchmetrics", "torchao"))
_RUNTIME_NAMES = frozenset(
    ("pyg-lib", "torch-scatter", "torch-sparse", "torch-geometric")
)
_EXTENSION_NAMES = frozenset(("torch-scatter", "torch-sparse"))
_PYG_INDEX_DISTRIBUTIONS = frozenset(("pyg-lib", "torch-scatter", "torch-sparse"))
_COMPILED_DISTRIBUTIONS = _PYG_INDEX_DISTRIBUTIONS
_BINARY_DISTRIBUTIONS = _COMPILED_DISTRIBUTIONS | _CORE_NAMES
_SUPPORTED_PLATFORMS = frozenset(
    (("Linux", "x86_64"), ("Linux", "aarch64"), ("Darwin", "arm64"))
)
_PYTHON_311_ABIS = frozenset(
    (tag.interpreter, tag.abi)
    for tag in (
        *cpython_tags(python_version=(3, 11), platforms=("placeholder",)),
        *compatible_tags(
            python_version=(3, 11),
            interpreter="cp311",
            platforms=("placeholder",),
        ),
    )
)
_PIN_PATTERN = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)=="
    r"(?P<version>[A-Za-z0-9][A-Za-z0-9.!+_-]*)$"
)
_INCLUDE_PATTERN = re.compile(r"^-r\s+(?P<filename>[^\s]+)$")
_FIND_LINKS_PATTERN = re.compile(r"^--find-links\s+(?P<url>[^\s]+)$")


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


class DistributionView(Protocol):
    version: str
    files: Sequence[PackagePath] | None

    def locate_file(self, path: PackagePath) -> Path: ...

    def read_text(self, filename: str) -> str | None: ...


Canary = Callable[[Mapping[str, ModuleType]], None]


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


class TorchStackVerificationError(RuntimeError):
    """One stable component/category pair failed closed."""

    def __init__(self, component: str, category: str) -> None:
        self.component = component
        self.category = category
        super().__init__(f"torch stack verification failed: {component}: {category}")


_TORCH_SCRIPT_WARNING_MESSAGE = (
    "`torch.jit.script` is deprecated. Please switch to "
    "`torch.compile` or `torch.export`."
)
_TORCH_SCRIPT_WARNING_PATH = PurePosixPath("torch/jit/_script.py")
_IMPORT_WARNING_OUTER_COMPONENTS = frozenset(("torch-geometric", "torch-sparse"))
_IMPORT_WARNING_DEBT_KEYS = frozenset({
    ("2.11.0", "torch-geometric", "2.8.0.post1"),
    ("2.11.0", "torch-sparse", "0.6.18"),
})


@dataclass(frozen=True)
class ImportWarningEvidence:
    torch_public_version: str
    outer_component: str
    outer_public_version: str
    count: int
    message: str
    origin: Path


def _distribution_public_version(
    component: str,
    distribution: DistributionView,
) -> str:
    try:
        return Version(distribution.version).public
    except BaseException:
        raise TorchStackVerificationError(component, "abi") from None


def _torch_script_warning_origin(
    torch_distribution: DistributionView,
    *,
    component: str,
) -> Path:
    try:
        files = torch_distribution.files
        if files is None:
            raise TorchStackVerificationError(component, "abi")
        matches = tuple(
            path for path in files
            if isinstance(path, PackagePath)
            and path.as_posix() == _TORCH_SCRIPT_WARNING_PATH.as_posix()
        )
        if len(matches) != 1 or getattr(matches[0], "dist", None) is not torch_distribution:
            raise TorchStackVerificationError(component, "abi")
        located = Path(matches[0].locate())
        owned = Path(torch_distribution.locate_file(matches[0]))
        if located.is_symlink() or owned.is_symlink():
            raise TorchStackVerificationError(component, "abi")
        resolved = located.resolve(strict=True)
        if resolved != owned.resolve(strict=True) or not resolved.is_file():
            raise TorchStackVerificationError(component, "abi")
        return resolved
    except TorchStackVerificationError:
        raise
    except BaseException:
        raise TorchStackVerificationError(component, "abi") from None


def _capture_selected_import(
    import_name: str,
    hooks: VerificationHooks,
) -> tuple[ModuleType, tuple[warnings.WarningMessage, ...]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module = hooks.import_module(import_name)
    return module, tuple(caught)


def _validate_import_warning_group(
    *,
    torch_distribution: DistributionView,
    outer_component: str,
    outer_distribution: DistributionView,
    caught: Sequence[warnings.WarningMessage],
) -> ImportWarningEvidence:
    try:
        if not caught:
            raise TorchStackVerificationError(outer_component, "abi")
        torch_version = _distribution_public_version("torch", torch_distribution)
        outer_version = _distribution_public_version(outer_component, outer_distribution)
        if (torch_version, outer_component, outer_version) not in _IMPORT_WARNING_DEBT_KEYS:
            raise TorchStackVerificationError(outer_component, "abi")
        expected_origin = _torch_script_warning_origin(
            torch_distribution,
            component=outer_component,
        )
        for record in caught:
            resolved = Path(record.filename).resolve(strict=True)
            if (
                record.category is not DeprecationWarning
                or str(record.message) != _TORCH_SCRIPT_WARNING_MESSAGE
                or resolved != expected_origin
            ):
                raise TorchStackVerificationError(outer_component, "abi")
        return ImportWarningEvidence(
            torch_version,
            outer_component,
            outer_version,
            len(caught),
            _TORCH_SCRIPT_WARNING_MESSAGE,
            expected_origin,
        )
    except TorchStackVerificationError:
        raise
    except BaseException:
        raise TorchStackVerificationError(outer_component, "abi") from None


def _import_with_selected_warning_boundary(
    pin: StackPin,
    distribution: DistributionView,
    torch_distribution: DistributionView,
    hooks: VerificationHooks,
) -> ModuleType:
    try:
        module, caught = _capture_selected_import(pin.import_name, hooks)
    except BaseException:
        raise TorchStackVerificationError(pin.distribution, "abi") from None
    if not caught:
        return module
    _validate_import_warning_group(
        torch_distribution=torch_distribution,
        outer_component=pin.distribution,
        outer_distribution=distribution,
        caught=caught,
    )
    return module


@dataclass(frozen=True)
class _Manifest:
    includes: tuple[str, ...]
    find_links: tuple[str, ...]
    pins: Mapping[str, Version]


def normalize_name(value: str) -> str:
    """Return the PEP 503 normalized distribution name."""
    return re.sub(r"[-_.]+", "-", value).lower()


def _manifest_error(component: str) -> TorchStackVerificationError:
    return TorchStackVerificationError(component, "manifest")


def _parse_manifest(repo: Path, filename: str) -> _Manifest:
    try:
        text = (repo / filename).read_text(encoding="utf-8")
    except BaseException:
        raise _manifest_error(filename) from None

    includes: list[str] = []
    find_links: list[str] = []
    pins: dict[str, Version] = {}
    try:
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            line = re.split(r"\s+#", line, maxsplit=1)[0].strip()
            include = _INCLUDE_PATTERN.fullmatch(line)
            if include is not None:
                includes.append(include.group("filename"))
                continue
            link = _FIND_LINKS_PATTERN.fullmatch(line)
            if link is not None:
                find_links.append(link.group("url"))
                continue
            pin = _PIN_PATTERN.fullmatch(line)
            if pin is None:
                raise _manifest_error(filename)
            name = normalize_name(pin.group("name"))
            if name not in IMPORTS or name in pins:
                raise _manifest_error(filename)
            version = Version(pin.group("version"))
            if version.local is not None:
                raise _manifest_error(filename)
            pins[name] = version
    except TorchStackVerificationError:
        raise
    except BaseException:
        raise _manifest_error(filename) from None
    return _Manifest(tuple(includes), tuple(find_links), pins)


def _require_manifest_shape(
    filename: str,
    manifest: _Manifest,
    *,
    pins: frozenset[str],
    includes: tuple[str, ...] = (),
    find_links: tuple[str, ...] = (),
) -> None:
    if (
        frozenset(manifest.pins) != pins
        or manifest.includes != includes
        or manifest.find_links != find_links
    ):
        raise _manifest_error(filename)


def load_stack_contract(repo: Path, system: str, machine: str) -> StackContract:
    """Derive the selected stack from all five canonical root manifests."""
    if (system, machine) not in _SUPPORTED_PLATFORMS:
        raise TorchStackVerificationError("platform", "platform")
    manifests = {filename: _parse_manifest(repo, filename) for filename in _MANIFEST_NAMES}
    core = manifests[_CORE_MANIFEST]
    ecosystem = manifests[_ECOSYSTEM_MANIFEST]
    runtime = manifests[_RUNTIME_MANIFEST]
    audit = manifests[_AUDIT_MANIFEST]
    extension_audit = manifests[_EXTENSION_AUDIT_MANIFEST]

    _require_manifest_shape(_CORE_MANIFEST, core, pins=_CORE_NAMES)
    _require_manifest_shape(_ECOSYSTEM_MANIFEST, ecosystem, pins=_ECOSYSTEM_NAMES)
    torch_version = core.pins["torch"]
    selector = f"https://data.pyg.org/whl/torch-{torch_version.public}+cpu.html"
    _require_manifest_shape(
        _RUNTIME_MANIFEST,
        runtime,
        pins=_RUNTIME_NAMES,
        includes=(_ECOSYSTEM_MANIFEST,),
        find_links=(selector,),
    )
    _require_manifest_shape(
        _AUDIT_MANIFEST,
        audit,
        pins=frozenset(("torch-geometric",)),
        includes=(_CORE_MANIFEST, _ECOSYSTEM_MANIFEST),
    )
    _require_manifest_shape(_EXTENSION_AUDIT_MANIFEST, extension_audit, pins=_EXTENSION_NAMES)

    authoritative = dict(core.pins) | dict(ecosystem.pins) | dict(runtime.pins)
    if audit.pins["torch-geometric"] != authoritative["torch-geometric"]:
        raise _manifest_error(_AUDIT_MANIFEST)
    for name, version in extension_audit.pins.items():
        if authoritative.get(name) != version:
            raise _manifest_error(_EXTENSION_AUDIT_MANIFEST)
    if frozenset(authoritative) != frozenset(IMPORTS):
        raise _manifest_error(_RUNTIME_MANIFEST)
    return StackContract(
        pins=tuple(
            StackPin(name, import_name, authoritative[name])
            for name, import_name in IMPORTS.items()
        ),
        system=system,
        machine=machine,
    )


def _wheel_inventory(distribution_name: str, distribution: DistributionView) -> tuple[PackagePath, ...]:
    try:
        selected = distribution.files
        if selected is None:
            raise TorchStackVerificationError(distribution_name, "wheel")
        files = tuple(selected)
        wheels = tuple(path for path in files if path.name == "WHEEL" and path.parent.name.endswith(".dist-info"))
        records = tuple(path for path in files if path.name == "RECORD" and path.parent.name.endswith(".dist-info"))
        if len(wheels) != 1 or len(records) != 1 or wheels[0].parent != records[0].parent:
            raise TorchStackVerificationError(distribution_name, "wheel")
        return files
    except TorchStackVerificationError:
        raise
    except BaseException:
        raise TorchStackVerificationError(distribution_name, "wheel") from None


def _wheel_tags(distribution_name: str, distribution: DistributionView) -> tuple[Tag, ...]:
    try:
        wheel = distribution.read_text("WHEEL")
        if not isinstance(wheel, str):
            raise TorchStackVerificationError(distribution_name, "wheel")
        raw_tags = tuple(
            line.partition(":")[2].strip()
            for line in wheel.splitlines()
            if line.lower().startswith("tag:") and line.partition(":")[2].strip()
        )
        if not raw_tags:
            raise TorchStackVerificationError(distribution_name, "wheel")
        return tuple(tag for raw_tag in raw_tags for tag in parse_tag(raw_tag))
    except TorchStackVerificationError:
        raise
    except BaseException:
        raise TorchStackVerificationError(distribution_name, "wheel") from None


def _tag_matches_platform(tag: Tag, contract: StackContract) -> bool:
    platform_tag = tag.platform
    if platform_tag == "any":
        return True
    if contract.system == "Darwin":
        return "macosx_" in platform_tag and (
            platform_tag.endswith("_arm64") or platform_tag.endswith("_universal2")
        )
    if contract.machine == "aarch64":
        return ("manylinux" in platform_tag or "linux_" in platform_tag) and platform_tag.endswith("_aarch64")
    return ("manylinux" in platform_tag or "linux_" in platform_tag) and platform_tag.endswith("_x86_64")


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


def _verify_distribution(pin: StackPin, distribution: DistributionView, contract: StackContract) -> None:
    try:
        version = Version(distribution.version)
    except BaseException:
        raise TorchStackVerificationError(pin.distribution, "metadata") from None
    if version.public != pin.public_version.public:
        raise TorchStackVerificationError(pin.distribution, "metadata")
    _verify_local_version(pin, version, contract)
    _wheel_inventory(pin.distribution, distribution)
    tags = _wheel_tags(pin.distribution, distribution)
    platform_compatible = tuple(tag for tag in tags if _tag_matches_platform(tag, contract))
    if pin.distribution in _BINARY_DISTRIBUTIONS and all(tag.platform == "any" for tag in tags):
        raise TorchStackVerificationError(pin.distribution, "wheel")
    if not platform_compatible:
        raise TorchStackVerificationError(pin.distribution, "platform")
    if pin.distribution in _BINARY_DISTRIBUTIONS and not any(
        tag.platform != "any" and (tag.interpreter, tag.abi) in _PYTHON_311_ABIS
        for tag in platform_compatible
    ):
        raise TorchStackVerificationError(pin.distribution, "abi")


def _verify_record_ownership(pin: StackPin, distribution: DistributionView, module: ModuleType) -> None:
    try:
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str):
            raise TorchStackVerificationError(pin.distribution, "metadata")
        imported = Path(module_file).resolve()
        files = _wheel_inventory(pin.distribution, distribution)
        owned = tuple(Path(path.locate()).resolve() for path in files)
        if not imported.is_file() or imported not in owned:
            raise TorchStackVerificationError(pin.distribution, "metadata")
    except TorchStackVerificationError:
        raise
    except BaseException:
        raise TorchStackVerificationError(pin.distribution, "metadata") from None


def _verify_no_cuda_distributions(contract: StackContract, installed: tuple[str, ...]) -> None:
    if contract.system == "Linux" and any(name.startswith("nvidia-") for name in installed):
        raise TorchStackVerificationError("torch", "cpu")


def _verify_cpu_runtime(contract: StackContract, torch: ModuleType) -> None:
    if contract.system != "Linux":
        return
    try:
        if torch.version.cuda is not None:
            raise TorchStackVerificationError("torch", "cpu")
    except TorchStackVerificationError:
        raise
    except BaseException:
        raise TorchStackVerificationError("torch", "cpu") from None


def _scatter_canary(modules: Mapping[str, ModuleType]) -> None:
    torch = modules["torch"]
    scatter = modules["torch-scatter"].scatter
    result = scatter(
        torch.tensor([1.0, 2.0, 3.0]),
        torch.tensor([0, 1, 0]),
        dim=0,
        dim_size=2,
    )
    if tuple(result.shape) != (2,):
        raise RuntimeError


def _sparse_canary(modules: Mapping[str, ModuleType]) -> None:
    torch = modules["torch"]
    sparse = modules["torch-sparse"].SparseTensor(
        row=torch.tensor([0, 1]),
        col=torch.tensor([1, 0]),
        value=torch.tensor([1.0, 1.0]),
        sparse_sizes=(2, 2),
    )
    result = sparse.matmul(torch.tensor([[1.0], [2.0]]))
    if tuple(result.shape) != (2, 1):
        raise RuntimeError


@dataclass
class _OperationSpy:
    operation: Callable[..., object]
    calls: int = 0

    def __call__(self, *args: object, **kwargs: object) -> object:
        self.calls += 1
        return self.operation(*args, **kwargs)


@contextmanager
def _replace_operation(namespace: object, name: str) -> Iterator[_OperationSpy]:
    original = getattr(namespace, name)
    spy = _OperationSpy(original)
    setattr(namespace, name, spy)
    try:
        yield spy
    finally:
        setattr(namespace, name, original)


def _sampler_canary(modules: Mapping[str, ModuleType]) -> None:
    torch = modules["torch"]
    geometric = modules["torch-geometric"]
    typing = importlib.import_module("torch_geometric.typing")
    if not typing.WITH_PYG_LIB or not typing.WITH_TORCH_SPARSE:
        raise RuntimeError("both sampler backends must be available")
    original_with_pyg = typing.WITH_PYG_LIB
    data = geometric.data.Data(
        x=torch.tensor([[1.0], [2.0], [3.0]]),
        edge_index=torch.tensor([[0, 1, 2, 1], [1, 0, 1, 2]]),
    )
    try:
        with (
            _replace_operation(torch.ops.pyg, "neighbor_sample") as pyg_spy,
            _replace_operation(torch.ops.torch_sparse, "neighbor_sample") as sparse_spy,
        ):
            typing.WITH_PYG_LIB = True
            preferred = next(iter(geometric.loader.NeighborLoader(
                data, num_neighbors=[-1], input_nodes=torch.tensor([0]),
                batch_size=1, shuffle=False, num_workers=0,
            )))
            if pyg_spy.calls != 1 or sparse_spy.calls != 0:
                raise RuntimeError("preferred sampler did not use only pyg-lib")

            typing.WITH_PYG_LIB = False
            fallback = next(iter(geometric.loader.NeighborLoader(
                data, num_neighbors=[-1], input_nodes=torch.tensor([0]),
                batch_size=1, shuffle=False, num_workers=0,
            )))
            if pyg_spy.calls != 1 or sparse_spy.calls != 1:
                raise RuntimeError("fallback sampler did not use only torch-sparse")
            if min(int(preferred.batch_size), int(fallback.batch_size)) <= 0:
                raise RuntimeError("sampler returned an empty seed batch")
            if min(int(preferred.edge_index.numel()), int(fallback.edge_index.numel())) <= 0:
                raise RuntimeError("sampler returned no edges")
    finally:
        typing.WITH_PYG_LIB = original_with_pyg


def _installed_distribution_names() -> tuple[str, ...]:
    return tuple(str(distribution.metadata["Name"]) for distribution in metadata.distributions())


def _verify_canonical_nnx() -> object:
    try:
        evidence = verify_nnx_install(environ={})
        if (
            evidence.mode != "canonical-wheel"
            or evidence.distribution != "thekaveh-nnx"
            or evidence.version != "0.2.0"
        ):
            raise TorchStackVerificationError("nnx", "nnx")
        return evidence
    except TorchStackVerificationError:
        raise
    except BaseException:
        raise TorchStackVerificationError("nnx", "nnx") from None


DEFAULT_HOOKS = VerificationHooks(
    distribution=metadata.distribution,
    installed_names=_installed_distribution_names,
    import_module=importlib.import_module,
    system=platform.system,
    machine=platform.machine,
    nnx_verify=_verify_canonical_nnx,
    canaries=CanaryHooks(
        scatter=_scatter_canary,
        sparse=_sparse_canary,
        sampler=_sampler_canary,
    ),
)


def _run_warning_free(
    component: str,
    category: str,
    operation: Callable[..., object],
    *args: object,
) -> object:
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = operation(*args)
    except BaseException:
        raise TorchStackVerificationError(component, category) from None
    if caught:
        raise TorchStackVerificationError(component, category) from None
    return result


def verify_torch_stack(
    repo: Path = REPO_ROOT,
    hooks: VerificationHooks = DEFAULT_HOOKS,
) -> StackEvidence:
    """Fail closed unless metadata, provenance, operators, sampling, and NNx pass."""
    try:
        system = hooks.system()
        machine = hooks.machine()
    except BaseException:
        raise TorchStackVerificationError("platform", "platform") from None
    contract = load_stack_contract(repo, system, machine)
    modules: dict[str, ModuleType] = {}
    selected_distributions: dict[str, DistributionView] = {}
    try:
        installed = tuple(normalize_name(name) for name in hooks.installed_names())
    except BaseException:
        raise TorchStackVerificationError("inventory", "metadata") from None
    _verify_no_cuda_distributions(contract, installed)
    for pin in contract.pins:
        try:
            distribution = hooks.distribution(pin.distribution)
        except BaseException:
            raise TorchStackVerificationError(pin.distribution, "metadata") from None
        _verify_distribution(pin, distribution, contract)
        selected_distributions[pin.distribution] = distribution
        if pin.distribution in _IMPORT_WARNING_OUTER_COMPONENTS:
            try:
                torch_distribution = selected_distributions.get("torch")
                if torch_distribution is None:
                    raise TorchStackVerificationError(pin.distribution, "abi")
                module = _import_with_selected_warning_boundary(
                    pin,
                    distribution,
                    torch_distribution,
                    hooks,
                )
            except TorchStackVerificationError:
                raise
            except BaseException:
                raise TorchStackVerificationError(pin.distribution, "abi") from None
        else:
            try:
                module = hooks.import_module(pin.import_name)
            except BaseException:
                raise TorchStackVerificationError(pin.distribution, "abi") from None
        _verify_record_ownership(pin, distribution, module)
        modules[pin.distribution] = module
    _verify_cpu_runtime(contract, modules["torch"])
    names_and_categories = (
        ("scatter", "operator"),
        ("sparse", "operator"),
        ("sampler", "sampler"),
    )
    for name, category in names_and_categories:
        canary = getattr(hooks.canaries, name)
        _run_warning_free(name, category, canary, modules)
    try:
        torch_version = Version(str(modules["torch"].__version__))
        torch_pin = next(pin for pin in contract.pins if pin.distribution == "torch")
        if torch_version.public != torch_pin.public_version.public:
            raise TorchStackVerificationError("torch", "metadata")
        _verify_local_version(torch_pin, torch_version, contract)
        evidence = StackEvidence(
            contract.system,
            contract.machine,
            str(torch_version),
            "pyg-lib",
        )
    except TorchStackVerificationError:
        raise
    except BaseException:
        raise TorchStackVerificationError("torch", "metadata") from None
    _run_warning_free("nnx", "nnx", hooks.nnx_verify)
    return evidence


def _validated_cli_evidence(evidence: StackEvidence) -> StackEvidence:
    try:
        if (
            not isinstance(evidence, StackEvidence)
            or (evidence.system, evidence.machine) not in _SUPPORTED_PLATFORMS
            or evidence.backend != "pyg-lib"
        ):
            raise TorchStackVerificationError("verifier", "metadata")
        torch_version = Version(evidence.torch_version)
        if (
            torch_version.public != "2.11.0"
            or str(torch_version) != evidence.torch_version
            or torch_version.local not in (None, "cpu")
            or (evidence.system == "Darwin" and torch_version.local is not None)
        ):
            raise TorchStackVerificationError("verifier", "metadata")
        return evidence
    except TorchStackVerificationError:
        raise
    except BaseException:
        raise TorchStackVerificationError("verifier", "metadata") from None


def _validated_cli_error(error: TorchStackVerificationError) -> TorchStackVerificationError:
    try:
        allowed_components = frozenset(IMPORTS) | frozenset(_MANIFEST_NAMES) | {
            "platform",
            "inventory",
            "scatter",
            "sparse",
            "sampler",
            "nnx",
            "verifier",
        }
        if error.component not in allowed_components or error.category not in _ERROR_CATEGORIES:
            raise TorchStackVerificationError("verifier", "metadata")
        return TorchStackVerificationError(error.component, error.category)
    except TorchStackVerificationError:
        raise
    except BaseException:
        raise TorchStackVerificationError("verifier", "metadata") from None


def main() -> int:
    """Run the canonical verifier without exposing third-party diagnostics."""
    failure: TorchStackVerificationError | None = None
    evidence: StackEvidence | None = None
    with (
        redirect_stdout(StringIO()),
        redirect_stderr(StringIO()),
        warnings.catch_warnings(record=True) as caught_warnings,
    ):
        warnings.simplefilter("always")
        try:
            evidence = _validated_cli_evidence(verify_torch_stack())
        except TorchStackVerificationError as error:
            try:
                failure = _validated_cli_error(error)
            except TorchStackVerificationError as invalid_error:
                failure = invalid_error
        except BaseException:
            failure = TorchStackVerificationError("verifier", "metadata")
    if failure is None and caught_warnings:
        failure = TorchStackVerificationError("verifier", "metadata")
    if failure is not None or evidence is None:
        print(failure or TorchStackVerificationError("verifier", "metadata"), file=sys.stderr)
        return 1
    print(
        f"Torch stack verified: torch {evidence.torch_version}; "
        f"{evidence.system} {evidence.machine}; backend {evidence.backend}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
