from __future__ import annotations

import ast
import dataclasses
import importlib.util
import re
import sys
import traceback
import warnings
from importlib.metadata import PackagePath
from pathlib import Path
from types import ModuleType, SimpleNamespace
from warnings import warn

import pytest
from packaging.version import Version

import scripts.verify_torch_stack as verifier_module
from scripts.verify_nnx_install import NnxInstallEvidence
from scripts.verify_torch_stack import (
    IMPORTS,
    CanaryHooks,
    StackContract,
    StackEvidence,
    StackPin,
    TorchStackVerificationError,
    VerificationHooks,
    load_stack_contract,
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
CORE = "torch==2.11.0\ntorchvision==0.26.0\ntorchaudio==2.11.0\n"
ECOSYSTEM = "pytorch-lightning==2.6.1\ntorchmetrics==1.9.0\ntorchao==0.18.0\n"
RUNTIME = (
    "-r torch-ecosystem-requirements.txt\n"
    "--find-links https://data.pyg.org/whl/torch-2.11.0+cpu.html\n"
    "pyg-lib==0.8.0\ntorch-scatter==2.1.2\ntorch-sparse==0.6.18\n"
    "torch_geometric==2.8.0.post1\n"
)
AUDIT = "-r torch-core-requirements.txt\n-r torch-ecosystem-requirements.txt\ntorch_geometric==2.8.0.post1\n"
EXTENSIONS = (
    "# Pre-resolved compiled PyG extension supplement for the strict audit.\n"
    "# Runtime source: torch-requirements.txt retains the approved PyG wheel selector.\n"
    "torch-scatter==2.1.2\ntorch-sparse==0.6.18\n"
)
MANIFESTS = {
    "torch-core-requirements.txt": CORE,
    "torch-ecosystem-requirements.txt": ECOSYSTEM,
    "torch-requirements.txt": RUNTIME,
    "torch-audit-requirements.txt": AUDIT,
    "pyg-extension-audit-requirements.txt": EXTENSIONS,
}
COMPILED = {
    "pyg-lib",
    "torch-scatter",
    "torch-sparse",
}
BINARY_WHEELS = COMPILED | {"torch", "torchvision", "torchaudio"}
SENSITIVE = (
    "https://user:password@packages.invalid/private?token=secret "
    "/Users/example/private/installer.log Traceback RuntimeError installer-output"
)
WARNING_SENSITIVE = f"{SENSITIVE}\ncredential=second-secret"


def _write_manifests(repo: Path, replacements: dict[str, str] | None = None) -> None:
    selected = MANIFESTS | (replacements or {})
    for filename, text in selected.items():
        (repo / filename).write_text(text, encoding="utf-8")


@pytest.fixture
def manifest_repo(tmp_path: Path) -> Path:
    _write_manifests(tmp_path)
    return tmp_path


class FakeDistribution:
    def __init__(self, name: str, version: str, root: Path, wheel_text: str) -> None:
        self.name = name
        self.version = version
        self.root = root
        self.wheel_text = wheel_text
        package = EXPECTED_IMPORTS[name].replace(".", "/") + "/__init__.py"
        dist_info = name.replace("-", "_") + f"-{Version(version).public}.dist-info"
        self.files: list[PackagePath] | None = [
            self._path(f"{dist_info}/WHEEL"),
            self._path(f"{dist_info}/RECORD"),
            self._path(package),
        ]

    def _path(self, value: str) -> PackagePath:
        path = PackagePath(value)
        path.dist = self
        return path

    def locate_file(self, path: PackagePath) -> Path:
        return self.root / path

    def read_text(self, filename: str) -> str | None:
        return self.wheel_text if filename == "WHEEL" else None


def _platform_tag(system: str, machine: str) -> str:
    if system == "Darwin":
        return "cp311-cp311-macosx_14_0_arm64"
    if machine == "aarch64":
        return "cp311-cp311-manylinux_2_28_aarch64"
    return "cp311-cp311-manylinux_2_28_x86_64"


class FakeStack:
    def __init__(self, repo: Path, *, system: str = "Linux", machine: str = "x86_64") -> None:
        self.repo = repo
        self.system_name = system
        self.machine_name = machine
        self.calls: list[str] = []
        self.distributions: dict[str, FakeDistribution] = {}
        self.modules: dict[str, ModuleType] = {}
        self.names = list(EXPECTED_IMPORTS)
        platform_tag = _platform_tag(system, machine)
        for distribution, import_name in EXPECTED_IMPORTS.items():
            root = repo / "site-packages"
            module_path = root / import_name.replace(".", "/") / "__init__.py"
            module_path.parent.mkdir(parents=True, exist_ok=True)
            module_path.touch()
            wheel_tag = platform_tag if distribution in BINARY_WHEELS else "py3-none-any"
            version = EXPECTED_VERSIONS[distribution]
            if system == "Linux" and distribution in COMPILED:
                version = f"{version}+pt211cpu"
            self.distributions[distribution] = FakeDistribution(
                distribution,
                version,
                root,
                f"Wheel-Version: 1.0\nTag: {wheel_tag}\n",
            )
            module = ModuleType(import_name)
            module.__file__ = str(module_path)
            if distribution == "torch":
                module.__version__ = EXPECTED_VERSIONS[distribution]
                module.version = SimpleNamespace(cuda=None)
            self.modules[import_name] = module
        canaries = CanaryHooks(
            *(self._canary(name) for name in ("scatter", "sparse", "sampler"))
        )
        self.hooks = VerificationHooks(
            distribution=self._distribution,
            installed_names=lambda: tuple(self.names),
            import_module=self._import_module,
            system=lambda: self.system_name,
            machine=lambda: self.machine_name,
            nnx_verify=self._nnx,
            canaries=canaries,
        )

    def _distribution(self, name: str) -> FakeDistribution:
        return self.distributions[name]

    def _import_module(self, name: str) -> ModuleType:
        return self.modules[name]

    def _canary(self, name: str):
        def canary(modules):
            assert tuple(modules) == tuple(EXPECTED_IMPORTS)
            self.calls.append(name)

        return canary

    def _nnx(self):
        self.calls.append("nnx")
        return object()

    def version(self, distribution: str, version: str) -> None:
        self.distributions[distribution].version = version

    def wheel(self, distribution: str, text: str | None) -> None:
        selected = self.distributions[distribution]
        selected.wheel_text = text

    def remove_record_file(self, distribution: str, suffix: str) -> None:
        selected = self.distributions[distribution]
        assert selected.files is not None
        selected.files = [path for path in selected.files if not str(path).endswith(suffix)]


@pytest.fixture
def fake_stack(tmp_path: Path) -> FakeStack:
    _write_manifests(tmp_path)
    return FakeStack(tmp_path)


def test_public_interfaces_and_import_map_are_exact(manifest_repo: Path) -> None:
    contract = load_stack_contract(manifest_repo, "Linux", "x86_64")

    assert IMPORTS == EXPECTED_IMPORTS
    assert isinstance(contract, StackContract)
    assert contract == StackContract(
        pins=tuple(StackPin(name, import_name, Version(EXPECTED_VERSIONS[name])) for name, import_name in EXPECTED_IMPORTS.items()),
        system="Linux",
        machine="x86_64",
    )
    assert all(dataclasses.is_dataclass(value) for value in (StackPin, StackContract, StackEvidence))


def _assert_selected_import_and_canary_boundary(
    imports: dict[str, str], canaries: tuple[str, ...]
) -> None:
    assert imports == EXPECTED_IMPORTS
    assert canaries == ("scatter", "sparse", "sampler")


@pytest.mark.parametrize(
    ("imports", "canaries"),
    (
        (EXPECTED_IMPORTS | {"torch-cluster": "torch_cluster"}, ("scatter", "sparse", "sampler")),
        (
            EXPECTED_IMPORTS | {"torch-spline-conv": "torch_spline_conv"},
            ("scatter", "sparse", "sampler"),
        ),
        (EXPECTED_IMPORTS, ("scatter", "sparse", "cluster", "sampler")),
        (EXPECTED_IMPORTS, ("scatter", "sparse", "sampler", "spline")),
    ),
    ids=("torch-cluster", "torch-spline-conv", "cluster-canary", "spline-canary"),
)
def test_selected_boundary_rejects_legacy_reinsertions(
    imports: dict[str, str], canaries: tuple[str, ...]
) -> None:
    with pytest.raises(AssertionError):
        _assert_selected_import_and_canary_boundary(imports, canaries)


def test_current_five_manifests_produce_the_canonical_contract() -> None:
    contract = load_stack_contract(REPO_ROOT, "Darwin", "arm64")

    assert {pin.distribution: str(pin.public_version) for pin in contract.pins} == EXPECTED_VERSIONS
    assert (contract.system, contract.machine) == ("Darwin", "arm64")


@pytest.mark.parametrize(
    ("filename", "original", "replacement"),
    (
        ("torch-core-requirements.txt", "torch==2.11.0\n", ""),
        ("torch-core-requirements.txt", "torch==2.11.0\n", "torch==2.11.0\ntorch==2.11.0\n"),
        ("torch-core-requirements.txt", "torch==2.11.0", "torch>=2.11.0"),
        ("torch-requirements.txt", "-r torch-ecosystem-requirements.txt", "-r torch-core-requirements.txt"),
        ("torch-audit-requirements.txt", "-r torch-core-requirements.txt", "-r torch-requirements.txt"),
        ("torch-requirements.txt", "torch_geometric==2.8.0.post1", "torch_geometric~=2.8.0"),
        ("torch-requirements.txt", "torch-2.11.0+cpu", "torch-2.10.0+cpu"),
        ("pyg-extension-audit-requirements.txt", "torch-sparse==0.6.18", "torch-sparse==0.6.17"),
    ),
    ids=("missing", "duplicate", "range", "runtime-include", "audit-include", "runtime-range", "selector", "projection-drift"),
)
def test_manifest_contract_rejects_structural_and_projection_drift(
    tmp_path: Path, filename: str, original: str, replacement: str
) -> None:
    mutated = MANIFESTS[filename].replace(original, replacement, 1)
    assert mutated != MANIFESTS[filename]
    _write_manifests(tmp_path, {filename: mutated})

    with pytest.raises(TorchStackVerificationError, match=r"^torch stack verification failed: .+: manifest$"):
        load_stack_contract(tmp_path, "Linux", "x86_64")


@pytest.mark.parametrize(
    ("system", "machine"),
    (("Linux", "arm64"), ("Darwin", "x86_64"), ("Windows", "AMD64")),
)
def test_contract_rejects_unsupported_platforms(manifest_repo: Path, system: str, machine: str) -> None:
    with pytest.raises(
        TorchStackVerificationError,
        match=r"^torch stack verification failed: platform: platform$",
    ):
        load_stack_contract(manifest_repo, system, machine)


@pytest.mark.parametrize(
    ("system", "machine"),
    (("Linux", "x86_64"), ("Linux", "aarch64"), ("Darwin", "arm64")),
)
def test_supported_platform_wheel_contracts_pass(tmp_path: Path, system: str, machine: str) -> None:
    _write_manifests(tmp_path)
    stack = FakeStack(tmp_path, system=system, machine=machine)

    evidence = verify_torch_stack(repo=stack.repo, hooks=stack.hooks)

    assert evidence == StackEvidence(system, machine, "2.11.0", "pyg-lib")
    assert stack.calls == ["scatter", "sparse", "sampler", "nnx"]


def test_local_pyg_versions_require_pt211cpu_and_compatible_wheel_tags(fake_stack: FakeStack) -> None:
    fake_stack.version("pyg-lib", "0.8.0+pt211cpu")
    fake_stack.wheel("pyg-lib", "Tag: cp311-cp311-manylinux_2_28_x86_64\n")

    evidence = verify_torch_stack(repo=fake_stack.repo, hooks=fake_stack.hooks)

    assert evidence.backend == "pyg-lib"


@pytest.mark.parametrize("local", ("pt210cpu", "pt211cu128", "cu128", "arbitrary"))
def test_incompatible_local_pyg_versions_fail_closed(fake_stack: FakeStack, local: str) -> None:
    fake_stack.version("pyg-lib", f"0.8.0+{local}")

    with pytest.raises(
        TorchStackVerificationError,
        match=r"^torch stack verification failed: pyg-lib: abi$",
    ):
        verify_torch_stack(repo=fake_stack.repo, hooks=fake_stack.hooks)


def test_public_version_drift_fails_closed(fake_stack: FakeStack) -> None:
    fake_stack.version("torchmetrics", "1.8.2")

    with pytest.raises(
        TorchStackVerificationError,
        match=r"^torch stack verification failed: torchmetrics: metadata$",
    ):
        verify_torch_stack(repo=fake_stack.repo, hooks=fake_stack.hooks)


@pytest.mark.parametrize("suffix", ("/WHEEL", "/RECORD"))
def test_distribution_inventory_requires_wheel_and_record(fake_stack: FakeStack, suffix: str) -> None:
    fake_stack.remove_record_file("torch-scatter", suffix)

    with pytest.raises(
        TorchStackVerificationError,
        match=r"^torch stack verification failed: torch-scatter: wheel$",
    ):
        verify_torch_stack(repo=fake_stack.repo, hooks=fake_stack.hooks)


def test_distribution_requires_readable_wheel_metadata(fake_stack: FakeStack) -> None:
    fake_stack.wheel("torch-sparse", None)

    with pytest.raises(
        TorchStackVerificationError,
        match=r"^torch stack verification failed: torch-sparse: wheel$",
    ):
        verify_torch_stack(repo=fake_stack.repo, hooks=fake_stack.hooks)


def test_imported_module_must_be_owned_by_record(fake_stack: FakeStack) -> None:
    fake_stack.remove_record_file("torch-sparse", "torch_sparse/__init__.py")

    with pytest.raises(
        TorchStackVerificationError,
        match=r"^torch stack verification failed: torch-sparse: metadata$",
    ):
        verify_torch_stack(repo=fake_stack.repo, hooks=fake_stack.hooks)


def test_darwin_arm64_rejects_x86_only_extension_wheel(tmp_path: Path) -> None:
    _write_manifests(tmp_path)
    stack = FakeStack(tmp_path, system="Darwin", machine="arm64")
    stack.wheel("torch-scatter", "Tag: cp311-cp311-macosx_14_0_x86_64\n")

    with pytest.raises(
        TorchStackVerificationError,
        match=r"^torch stack verification failed: torch-scatter: platform$",
    ):
        verify_torch_stack(repo=stack.repo, hooks=stack.hooks)


def test_linux_rejects_non_cpu_torch_runtime(fake_stack: FakeStack) -> None:
    fake_stack.modules["torch"].version = SimpleNamespace(cuda="12.8")

    with pytest.raises(
        TorchStackVerificationError,
        match=r"^torch stack verification failed: torch: cpu$",
    ):
        verify_torch_stack(repo=fake_stack.repo, hooks=fake_stack.hooks)


@pytest.mark.parametrize("name", ("nvidia-cublas-cu12", "NVIDIA_CUDA_RUNTIME_CU12", "nvidia.cudnn.cu12"))
def test_linux_rejects_every_normalized_nvidia_distribution(fake_stack: FakeStack, name: str) -> None:
    fake_stack.names.append(name)

    with pytest.raises(
        TorchStackVerificationError,
        match=r"^torch stack verification failed: torch: cpu$",
    ):
        verify_torch_stack(repo=fake_stack.repo, hooks=fake_stack.hooks)


def test_canaries_run_once_in_stable_order_and_nnx_is_last(fake_stack: FakeStack) -> None:
    verify_torch_stack(repo=fake_stack.repo, hooks=fake_stack.hooks)

    assert fake_stack.calls == ["scatter", "sparse", "sampler", "nnx"]


@pytest.mark.parametrize(
    ("failed", "category"),
    (("scatter", "operator"), ("sparse", "operator"), ("sampler", "sampler")),
)
def test_each_runtime_canary_failure_is_fail_closed(fake_stack: FakeStack, failed: str, category: str) -> None:
    def explode(modules):
        raise RuntimeError(SENSITIVE)

    values = {field.name: getattr(fake_stack.hooks.canaries, field.name) for field in dataclasses.fields(CanaryHooks)}
    values[failed] = explode
    fake_stack.hooks = dataclasses.replace(fake_stack.hooks, canaries=CanaryHooks(**values))

    with pytest.raises(
        TorchStackVerificationError,
        match=rf"^torch stack verification failed: {failed}: {category}$",
    ) as caught:
        verify_torch_stack(repo=fake_stack.repo, hooks=fake_stack.hooks)

    assert "nnx" not in fake_stack.calls
    assert SENSITIVE not in "".join(traceback.format_exception(caught.type, caught.value, caught.tb))


def test_warning_only_canary_fails_at_its_direct_api_boundary(fake_stack: FakeStack) -> None:
    def warning_canary(modules):
        warn(WARNING_SENSITIVE)

    assert "https://" in WARNING_SENSITIVE
    assert "user:password" in WARNING_SENSITIVE
    assert "/Users/" in WARNING_SENSITIVE
    assert "\n" in WARNING_SENSITIVE
    fake_stack.hooks = dataclasses.replace(
        fake_stack.hooks,
        canaries=dataclasses.replace(fake_stack.hooks.canaries, sparse=warning_canary),
    )

    with warnings.catch_warnings(record=True) as escaped:
        warnings.simplefilter("always")
        with pytest.raises(
            TorchStackVerificationError,
            match=r"^torch stack verification failed: sparse: operator$",
        ) as caught:
            verify_torch_stack(repo=fake_stack.repo, hooks=fake_stack.hooks)

    diagnostic = "".join(traceback.format_exception(caught.type, caught.value, caught.tb))
    assert escaped == []
    assert fake_stack.calls == ["scatter"]
    for sensitive in ("user:password", "packages.invalid", "/Users/example/private", "second-secret"):
        assert sensitive not in diagnostic


def test_pytest_skip_from_a_canary_is_a_verification_failure(fake_stack: FakeStack) -> None:
    def skip(modules):
        pytest.skip(SENSITIVE)

    fake_stack.hooks = dataclasses.replace(
        fake_stack.hooks,
        canaries=dataclasses.replace(fake_stack.hooks.canaries, sampler=skip),
    )

    with pytest.raises(
        TorchStackVerificationError,
        match=r"^torch stack verification failed: sampler: sampler$",
    ):
        verify_torch_stack(repo=fake_stack.repo, hooks=fake_stack.hooks)


def test_nnx_delegation_cannot_skip_or_warn_and_continue(fake_stack: FakeStack) -> None:
    def skip_nnx():
        warn(SENSITIVE)
        pytest.skip(SENSITIVE)

    fake_stack.hooks = dataclasses.replace(fake_stack.hooks, nnx_verify=skip_nnx)

    with warnings.catch_warnings(record=True) as escaped:
        warnings.simplefilter("always")
        with pytest.raises(
            TorchStackVerificationError,
            match=r"^torch stack verification failed: nnx: nnx$",
        ) as caught:
            verify_torch_stack(repo=fake_stack.repo, hooks=fake_stack.hooks)

    assert escaped == []
    assert fake_stack.calls == ["scatter", "sparse", "sampler"]
    assert SENSITIVE not in "".join(traceback.format_exception(caught.type, caught.value, caught.tb))


def test_warning_only_nnx_fails_at_its_direct_api_boundary(fake_stack: FakeStack) -> None:
    def warning_nnx():
        warn(WARNING_SENSITIVE)
        fake_stack.calls.append("nnx")
        return NnxInstallEvidence("canonical-wheel", "thekaveh-nnx", "0.2.0")

    assert "https://" in WARNING_SENSITIVE
    assert "user:password" in WARNING_SENSITIVE
    assert "/Users/" in WARNING_SENSITIVE
    assert "\n" in WARNING_SENSITIVE
    fake_stack.hooks = dataclasses.replace(fake_stack.hooks, nnx_verify=warning_nnx)

    with warnings.catch_warnings(record=True) as escaped:
        warnings.simplefilter("always")
        with pytest.raises(
            TorchStackVerificationError,
            match=r"^torch stack verification failed: nnx: nnx$",
        ) as caught:
            verify_torch_stack(repo=fake_stack.repo, hooks=fake_stack.hooks)

    diagnostic = "".join(traceback.format_exception(caught.type, caught.value, caught.tb))
    assert escaped == []
    assert fake_stack.calls == ["scatter", "sparse", "sampler", "nnx"]
    for sensitive in ("user:password", "packages.invalid", "/Users/example/private", "second-secret"):
        assert sensitive not in diagnostic


def test_environment_variables_cannot_bypass_canaries_or_nnx(fake_stack: FakeStack, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("SKIP_TORCH_STACK_VERIFY", "TORCH_STACK_VERIFY", "CI", "NNX_ALLOW_EDITABLE"):
        monkeypatch.setenv(name, "0")

    verify_torch_stack(repo=fake_stack.repo, hooks=fake_stack.hooks)

    assert fake_stack.calls == ["scatter", "sparse", "sampler", "nnx"]


@pytest.mark.parametrize("boundary", ("platform", "inventory", "distribution", "import", "canary", "nnx"))
def test_all_external_failures_have_stable_redacted_messages(fake_stack: FakeStack, boundary: str) -> None:
    def explode(*args, **kwargs):
        raise RuntimeError(SENSITIVE)

    if boundary == "platform":
        fake_stack.hooks = dataclasses.replace(fake_stack.hooks, system=explode)
    elif boundary == "inventory":
        fake_stack.hooks = dataclasses.replace(fake_stack.hooks, installed_names=explode)
    elif boundary == "distribution":
        fake_stack.hooks = dataclasses.replace(fake_stack.hooks, distribution=explode)
    elif boundary == "import":
        fake_stack.hooks = dataclasses.replace(fake_stack.hooks, import_module=explode)
    elif boundary == "canary":
        fake_stack.hooks = dataclasses.replace(
            fake_stack.hooks,
            canaries=dataclasses.replace(fake_stack.hooks.canaries, scatter=explode),
        )
    else:
        fake_stack.hooks = dataclasses.replace(fake_stack.hooks, nnx_verify=explode)

    with pytest.raises(TorchStackVerificationError) as caught:
        verify_torch_stack(repo=fake_stack.repo, hooks=fake_stack.hooks)

    diagnostic = str(caught.value)
    formatted = "".join(traceback.format_exception(caught.type, caught.value, caught.tb))
    assert diagnostic.startswith("torch stack verification failed: ")
    for sensitive in (
        "user:password",
        "token=secret",
        "/Users/example/private",
        "installer-output",
        "RuntimeError",
    ):
        assert sensitive not in diagnostic
        assert sensitive not in formatted


def test_cli_prints_only_a_stable_failure(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(verifier_module, "verify_torch_stack", lambda: (_ for _ in ()).throw(RuntimeError(SENSITIVE)))

    assert verifier_module.main() == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "torch stack verification failed: verifier: metadata\n"
    assert SENSITIVE not in captured.err


def test_cli_prints_one_stable_success_summary(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(
        verifier_module,
        "verify_torch_stack",
        lambda: StackEvidence("Linux", "x86_64", "2.11.0+cpu", "pyg-lib"),
    )

    assert verifier_module.main() == 0

    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == "Torch stack verified: torch 2.11.0+cpu; Linux x86_64; backend pyg-lib\n"


def test_cli_suppresses_dependency_output_on_success(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def noisy_success():
        print(SENSITIVE)
        print(SENSITIVE, file=sys.stderr)
        return StackEvidence("Linux", "x86_64", "2.11.0+cpu", "pyg-lib")

    monkeypatch.setattr(verifier_module, "verify_torch_stack", noisy_success)

    assert verifier_module.main() == 0

    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == "Torch stack verified: torch 2.11.0+cpu; Linux x86_64; backend pyg-lib\n"
    assert SENSITIVE not in captured.out


def test_cli_converts_dependency_warning_to_one_redacted_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def warning_success():
        warnings.warn(SENSITIVE)
        return StackEvidence("Linux", "x86_64", "2.11.0+cpu", "pyg-lib")

    monkeypatch.setattr(verifier_module, "verify_torch_stack", warning_success)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = verifier_module.main()

    captured = capsys.readouterr()
    assert result == 1
    assert caught == []
    assert captured.out == ""
    assert captured.err == "torch stack verification failed: verifier: metadata\n"
    assert SENSITIVE not in captured.err


def test_cli_suppresses_dependency_noise_while_preserving_stable_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def noisy_failure():
        print(SENSITIVE)
        print(SENSITIVE, file=sys.stderr)
        warnings.warn(SENSITIVE)
        raise TorchStackVerificationError("sampler", "sampler")

    monkeypatch.setattr(verifier_module, "verify_torch_stack", noisy_failure)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = verifier_module.main()

    captured = capsys.readouterr()
    assert result == 1
    assert caught == []
    assert captured.out == ""
    assert captured.err == "torch stack verification failed: sampler: sampler\n"
    assert SENSITIVE not in captured.err


def test_cli_rejects_injected_verification_error_fields(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def injected_failure():
        raise TorchStackVerificationError(f"torch\n{SENSITIVE}", "metadata")

    monkeypatch.setattr(verifier_module, "verify_torch_stack", injected_failure)

    assert verifier_module.main() == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "torch stack verification failed: verifier: metadata\n"
    assert SENSITIVE not in captured.err


@pytest.mark.parametrize(
    "evidence",
    (
        StackEvidence("Linux\nhttps://token.invalid", "x86_64", "2.11.0+cpu", "pyg-lib"),
        StackEvidence("Linux", "x86_64\n/Users/private", "2.11.0+cpu", "pyg-lib"),
        StackEvidence("Linux", "x86_64", "2.11.0+cpu\ncredential=secret", "pyg-lib"),
        StackEvidence("Linux", "x86_64", "2.11.0+cpu", "pyg-lib\nTraceback"),
        StackEvidence("Darwin", "arm64", "2.11.0+cpu", "pyg-lib"),
    ),
    ids=("system", "machine", "version", "backend", "darwin-cpu-local"),
)
def test_cli_rejects_injected_success_evidence(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    evidence: StackEvidence,
) -> None:
    monkeypatch.setattr(verifier_module, "verify_torch_stack", lambda: evidence)

    assert verifier_module.main() == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "torch stack verification failed: verifier: metadata\n"
    assert "token.invalid" not in captured.err
    assert "/Users/private" not in captured.err
    assert "credential=secret" not in captured.err
    assert "Traceback" not in captured.err


def test_default_nnx_hook_ignores_editable_environment_and_requires_canonical_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[dict[str, str]] = []

    def fake_verify_nnx_install(*, environ):
        seen.append(environ)
        return NnxInstallEvidence(
            mode="canonical-wheel",
            distribution="thekaveh-nnx",
            version="0.2.0",
        )

    monkeypatch.setenv("NNX_ALLOW_EDITABLE", "1")
    monkeypatch.setattr(verifier_module, "verify_nnx_install", fake_verify_nnx_install)

    evidence = verifier_module.DEFAULT_HOOKS.nnx_verify()

    assert seen == [{}]
    assert evidence == NnxInstallEvidence("canonical-wheel", "thekaveh-nnx", "0.2.0")


@pytest.mark.parametrize(
    "evidence",
    (
        NnxInstallEvidence("editable-development", "thekaveh-nnx", "0.2.0"),
        NnxInstallEvidence("canonical-wheel", "other-nnx", "0.2.0"),
        NnxInstallEvidence("canonical-wheel", "thekaveh-nnx", "0.2.1"),
    ),
)
def test_default_nnx_hook_rejects_noncanonical_evidence(
    monkeypatch: pytest.MonkeyPatch, evidence: NnxInstallEvidence
) -> None:
    seen: list[dict[str, str]] = []

    def fake_verify_nnx_install(*, environ):
        seen.append(environ)
        return evidence

    monkeypatch.setattr(verifier_module, "verify_nnx_install", fake_verify_nnx_install)

    with pytest.raises(
        TorchStackVerificationError,
        match=r"^torch stack verification failed: nnx: nnx$",
    ):
        verifier_module.DEFAULT_HOOKS.nnx_verify()

    assert seen == [{}]


class _ExplodingTorchVersion:
    def __init__(self, module_file: str) -> None:
        self.__file__ = module_file
        self.version = SimpleNamespace(cuda=None)

    @property
    def __version__(self):
        raise RuntimeError(SENSITIVE)


def test_torch_version_evidence_is_validated_before_nnx(fake_stack: FakeStack) -> None:
    torch_file = fake_stack.modules["torch"].__file__
    assert torch_file is not None
    fake_stack.modules["torch"] = _ExplodingTorchVersion(torch_file)

    with pytest.raises(
        TorchStackVerificationError,
        match=r"^torch stack verification failed: torch: metadata$",
    ):
        verify_torch_stack(repo=fake_stack.repo, hooks=fake_stack.hooks)

    assert fake_stack.calls == ["scatter", "sparse", "sampler"]


def test_injected_torch_version_is_rejected_before_nnx(fake_stack: FakeStack) -> None:
    fake_stack.modules["torch"].__version__ = "2.11.0+cpu\nhttps://token.invalid/private"

    with pytest.raises(
        TorchStackVerificationError,
        match=r"^torch stack verification failed: torch: metadata$",
    ):
        verify_torch_stack(repo=fake_stack.repo, hooks=fake_stack.hooks)

    assert fake_stack.calls == ["scatter", "sparse", "sampler"]


@pytest.mark.parametrize("distribution", tuple(sorted(BINARY_WHEELS)))
@pytest.mark.parametrize(
    ("tag", "category"),
    (
        ("cp310-cp310-manylinux_2_28_x86_64", "abi"),
        ("cp311-cp310-manylinux_2_28_x86_64", "abi"),
        ("py3-none-any", "wheel"),
    ),
    ids=("python-310", "wrong-abi", "pure-any"),
)
def test_binary_wheels_require_python311_abi_and_platform(
    fake_stack: FakeStack, distribution: str, tag: str, category: str
) -> None:
    fake_stack.wheel(distribution, f"Tag: {tag}\n")

    with pytest.raises(
        TorchStackVerificationError,
        match=rf"^torch stack verification failed: {distribution}: {category}$",
    ):
        verify_torch_stack(repo=fake_stack.repo, hooks=fake_stack.hooks)


@pytest.mark.parametrize("distribution", tuple(sorted(BINARY_WHEELS)))
@pytest.mark.parametrize(
    "python_abi",
    ("cp311-cp311", "cp311-abi3", "cp311-none", "cp310-abi3", "py3-none"),
)
def test_binary_wheels_accept_python311_compatible_tags(
    fake_stack: FakeStack, distribution: str, python_abi: str
) -> None:
    fake_stack.wheel(distribution, f"Tag: {python_abi}-manylinux_2_28_x86_64\n")

    evidence = verify_torch_stack(repo=fake_stack.repo, hooks=fake_stack.hooks)

    assert evidence.torch_version == "2.11.0"


@pytest.mark.parametrize(
    "distribution",
    ("pytorch-lightning", "torchmetrics", "torchao", "torch-geometric"),
)
def test_pure_python_distributions_retain_any_wheels(fake_stack: FakeStack, distribution: str) -> None:
    fake_stack.wheel(distribution, "Tag: py3-none-any\n")

    assert verify_torch_stack(repo=fake_stack.repo, hooks=fake_stack.hooks).backend == "pyg-lib"


@dataclasses.dataclass
class _CanaryResult:
    shape: tuple[int, ...]
    elements: int = 1

    def numel(self) -> int:
        return self.elements


class _DefaultCanaryRig:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.scatter_result = _CanaryResult((2,))
        self.sparse_result = _CanaryResult((2, 1))
        rig = self

        class SparseTensor:
            def __init__(inner_self, **kwargs):
                rig.events.append("sparse-init")
                inner_self.kwargs = kwargs

            def matmul(inner_self, matrix):
                rig.events.append("sparse-matmul")
                return rig.sparse_result

        self.modules = {
            "torch": SimpleNamespace(tensor=lambda value: value),
            "torch-scatter": SimpleNamespace(scatter=self._scatter),
            "torch-sparse": SimpleNamespace(SparseTensor=SparseTensor),
        }

    def _scatter(self, *args, **kwargs):
        self.events.append("scatter")
        return self.scatter_result

def test_default_operator_canary_bodies_execute_required_operators() -> None:
    rig = _DefaultCanaryRig()

    verifier_module._scatter_canary(rig.modules)
    verifier_module._sparse_canary(rig.modules)

    assert rig.events == ["scatter", "sparse-init", "sparse-matmul"]


@pytest.mark.parametrize(
    ("canary", "mutation"),
    (
        ("scatter", lambda rig: setattr(rig, "scatter_result", _CanaryResult((1,)))),
        ("sparse", lambda rig: setattr(rig, "sparse_result", _CanaryResult((1, 1)))),
    ),
    ids=("scatter-shape", "sparse-shape"),
)
def test_default_operator_canary_bodies_reject_weakened_results(canary: str, mutation) -> None:
    rig = _DefaultCanaryRig()
    mutation(rig)

    with pytest.raises(RuntimeError):
        getattr(verifier_module, f"_{canary}_canary")(rig.modules)


def test_verification_hooks_are_explicit_and_frozen(fake_stack: FakeStack) -> None:
    assert tuple(field.name for field in dataclasses.fields(VerificationHooks)) == (
        "distribution",
        "installed_names",
        "import_module",
        "system",
        "machine",
        "nnx_verify",
        "canaries",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        fake_stack.hooks.system = lambda: "Darwin"


def test_verifier_module_has_no_environment_or_pytest_bypass() -> None:
    source = (REPO_ROOT / "scripts" / "verify_torch_stack.py").read_text(encoding="utf-8")

    assert "pytest" not in source
    assert "os.environ" not in source
    assert "getenv(" not in source
    assert "except: pass" not in source
    assert "except Exception: pass" not in source


@pytest.mark.parametrize("omitted", ("scatter", "sparse", "sampler", "nnx"))
def test_source_mutations_cannot_omit_a_canary_or_nnx(tmp_path: Path, omitted: str) -> None:
    source = (REPO_ROOT / "scripts" / "verify_torch_stack.py").read_text(encoding="utf-8")
    if omitted == "nnx":
        original = '    _run_warning_free("nnx", "nnx", hooks.nnx_verify)'
        replacement = "    object()"
    else:
        original = "        _run_warning_free(name, category, canary, modules)"
        replacement = (
            f'        if name != "{omitted}":\n'
            "            _run_warning_free(name, category, canary, modules)"
        )
    mutated = source.replace(original, replacement, 1)
    assert mutated != source
    module_path = tmp_path / f"mutated_verify_torch_stack_{omitted}.py"
    module_path.write_text(mutated, encoding="utf-8")
    module_name = f"mutated_verify_torch_stack_{omitted}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        repo = tmp_path / "repo"
        repo.mkdir()
        _write_manifests(repo)
        stack = FakeStack(repo)

        module.verify_torch_stack(repo=repo, hooks=stack.hooks)

        with pytest.raises(AssertionError):
            assert stack.calls == ["scatter", "sparse", "sampler", "nnx"]
    finally:
        sys.modules.pop(module_name, None)


def _assert_consumer_gates_fail_closed(graph_source: str, quantization_source: str) -> None:
    for source in (graph_source, quantization_source):
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert not (
                    isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "pytest"
                    and node.func.attr in {"skip", "importorskip"}
                )
                assert not (
                    isinstance(node.func.value, ast.Attribute)
                    and isinstance(node.func.value.value, ast.Name)
                    and node.func.value.value.id == "pytest"
                    and node.func.value.attr == "mark"
                    and node.func.attr == "skipif"
                )
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in {"_has_pyg_sampler", "_import_torchao_or_skip"}
            if isinstance(node, ast.Name):
                assert node.id != "_HAS_PYG_SAMPLER"
            if isinstance(node, ast.Attribute):
                assert not (
                    isinstance(node.value, ast.Name)
                    and node.value.id == "torch"
                    and node.attr == "int1"
                )
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == "hasattr" and len(node.args) >= 2:
                    assert not (
                        isinstance(node.args[0], ast.Name)
                        and node.args[0].id == "torch"
                        and isinstance(node.args[1], ast.Constant)
                        and node.args[1].value == "int1"
                    )


def test_graph_and_quantization_consumers_have_no_optional_backend_bypass() -> None:
    graph_source = (
        REPO_ROOT / "tests" / "nnx_surface" / "test_node_classification_reddit_gnn_pyg.py"
    ).read_text(encoding="utf-8")
    quantization_source = (
        REPO_ROOT / "tests" / "nnx_surface" / "test_quantization_mnist_ffnn_pytorch.py"
    ).read_text(encoding="utf-8")

    _assert_consumer_gates_fail_closed(graph_source, quantization_source)
    _assert_no_other_consumer_warning_capture(graph_source, quantization_source)


@pytest.mark.parametrize(
    ("consumer", "mutation"),
    (
        ("graph", "\ndef restored_skip():\n    pytest.skip('missing backend')\n"),
        ("graph", "\ndef restored_importorskip():\n    pytest.importorskip('pyg_lib')\n"),
        ("graph", "\n_HAS_PYG_SAMPLER = True\n"),
        ("graph", "\n@pytest.mark.skipif(True, reason='missing backend')\ndef restored_skipif():\n    pass\n"),
        ("graph", "\ndef restored_probe():\n    return _has_pyg_sampler()\n"),
        ("quantization", "\ndef restored_quantization_probe():\n    return _import_torchao_or_skip()\n"),
        ("quantization", "\ndef restored_int1_guard():\n    return hasattr(torch, 'int1')\n"),
    ),
    ids=(
        "pytest-skip",
        "pytest-importorskip",
        "sampler-flag",
        "skipif-decorator",
        "sampler-probe",
        "torchao-probe",
        "torch-int1-guard",
    ),
)
def test_consumer_gate_source_mutations_are_rejected(consumer: str, mutation: str) -> None:
    clean_graph = "def test_graph():\n    import pyg_lib\n    import torch_sparse\n"
    clean_quantization = "def test_quantization():\n    import torchao\n"
    graph_source = clean_graph + (mutation if consumer == "graph" else "")
    quantization_source = clean_quantization + (mutation if consumer == "quantization" else "")

    with pytest.raises(AssertionError):
        _assert_consumer_gates_fail_closed(graph_source, quantization_source)


def _warnings_call(node: ast.AST, attribute: str) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "warnings"
        and node.func.attr == attribute
    )


def _assert_qat_warning_capture_is_exact(source: str) -> None:
    tree = ast.parse(source)
    functions = tuple(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "test_qat_prepare_train_convert_and_inference"
    )
    assert len(functions) == 1
    function = functions[0]
    captures = tuple(
        node
        for node in function.body
        if isinstance(node, ast.With)
        and len(node.items) == 1
        and _warnings_call(node.items[0].context_expr, "catch_warnings")
    )
    assert len(captures) == 1
    capture = captures[0]
    context = capture.items[0]
    assert isinstance(context.optional_vars, ast.Name)
    assert context.optional_vars.id == "caught"
    assert len(context.context_expr.keywords) == 1
    keyword = context.context_expr.keywords[0]
    assert keyword.arg == "record"
    assert isinstance(keyword.value, ast.Constant) and keyword.value.value is True
    assert len(capture.body) == 2
    filter_statement, train_statement = capture.body
    assert isinstance(filter_statement, ast.Expr)
    assert _warnings_call(filter_statement.value, "simplefilter")
    assert len(filter_statement.value.args) == 1
    assert isinstance(filter_statement.value.args[0], ast.Constant)
    assert filter_statement.value.args[0].value == "always"
    assert not filter_statement.value.keywords
    assert isinstance(train_statement, ast.Assign)
    assert len(train_statement.targets) == 1
    assert isinstance(train_statement.targets[0], ast.Name)
    assert train_statement.targets[0].id == "run"
    assert isinstance(train_statement.value, ast.Call)
    assert isinstance(train_statement.value.func, ast.Attribute)
    assert isinstance(train_statement.value.func.value, ast.Name)
    assert train_statement.value.func.value.id == "model"
    assert train_statement.value.func.attr == "train"
    capture_index = function.body.index(capture)
    validation = function.body[capture_index + 1]
    assert isinstance(validation, ast.Assign)
    assert len(validation.targets) == 1
    assert isinstance(validation.targets[0], ast.Name)
    assert validation.targets[0].id == "qat_warning_evidence"
    assert isinstance(validation.value, ast.Call)
    assert isinstance(validation.value.func, ast.Name)
    assert validation.value.func.id == "_assert_qat_warning_debt"
    assert len(validation.value.args) == 1
    assert isinstance(validation.value.args[0], ast.Name)
    assert validation.value.args[0].id == "caught"
    assert len(validation.value.keywords) == 1
    assert validation.value.keywords[0].arg == "qat_config"
    assert isinstance(validation.value.keywords[0].value, ast.Name)
    assert validation.value.keywords[0].value.id == "qat_config"
    config_assignments = tuple(
        node
        for node in function.body[:capture_index]
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "qat_config"
            for target in node.targets
        )
    )
    assert len(config_assignments) == 1
    assert isinstance(config_assignments[0].value, ast.Constant)
    assert config_assignments[0].value.value == "8da4w"
    for facade_name in ("QATLifecycleCallback", "qat_train_step_factory"):
        calls = tuple(
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "nnx"
            and node.func.attr == facade_name
        )
        assert len(calls) == 1
        config_keywords = tuple(
            keyword for keyword in calls[0].keywords if keyword.arg == "qat_config"
        )
        assert len(config_keywords) == 1
        assert isinstance(config_keywords[0].value, ast.Name)
        assert config_keywords[0].value.id == "qat_config"


def _assert_no_other_consumer_warning_capture(
    graph_source: str,
    quantization_source: str,
) -> None:
    graph_tree = ast.parse(graph_source)
    quantization_tree = ast.parse(quantization_source)
    graph_captures = tuple(
        node for node in ast.walk(graph_tree) if _warnings_call(node, "catch_warnings")
    )
    quantization_captures = tuple(
        node
        for node in ast.walk(quantization_tree)
        if _warnings_call(node, "catch_warnings")
    )
    graph_filters = tuple(
        node for node in ast.walk(graph_tree) if _warnings_call(node, "simplefilter")
    )
    quantization_filters = tuple(
        node
        for node in ast.walk(quantization_tree)
        if _warnings_call(node, "simplefilter")
    )
    assert not graph_captures
    assert not graph_filters
    assert len(quantization_captures) == 1
    assert len(quantization_filters) == 1
    _assert_qat_warning_capture_is_exact(quantization_source)


CLEAN_QAT_CAPTURE_SOURCE = '''
def test_qat_prepare_train_convert_and_inference():
    qat_config = "8da4w"
    callback = nnx.QATLifecycleCallback(qat_config=qat_config)
    train_step = nnx.qat_train_step_factory(qat_config=qat_config)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run = model.train(callbacks=[callback], train_step_fn=train_step)
    qat_warning_evidence = _assert_qat_warning_debt(caught, qat_config=qat_config)
    logits, classes = model.predict(X=X)
'''


def test_qat_warning_capture_contract_accepts_only_exact_synthetic_fixture() -> None:
    _assert_qat_warning_capture_is_exact(CLEAN_QAT_CAPTURE_SOURCE)
    _assert_no_other_consumer_warning_capture(
        "def test_graph():\n    pass\n",
        CLEAN_QAT_CAPTURE_SOURCE,
    )


@pytest.mark.parametrize(
    ("old", "new"),
    (
        ("record=True", "record=False"),
        ('simplefilter("always")', 'simplefilter("ignore")'),
        ('qat_config = "8da4w"', 'qat_config = "8da4w-next"'),
        (
            "    qat_warning_evidence = _assert_qat_warning_debt(caught, qat_config=qat_config)\n",
            "",
        ),
    ),
)
def test_qat_warning_capture_contract_rejects_shape_mutations(old: str, new: str) -> None:
    mutated = CLEAN_QAT_CAPTURE_SOURCE.replace(old, new, 1)
    assert mutated != CLEAN_QAT_CAPTURE_SOURCE
    with pytest.raises(AssertionError):
        _assert_qat_warning_capture_is_exact(mutated)


@pytest.mark.parametrize(
    "line",
    (
        "    callback = nnx.QATLifecycleCallback(qat_config=qat_config)\n",
        "    logits, classes = model.predict(X=X)\n",
    ),
    ids=("callback-inside-capture", "predict-inside-capture"),
)
def test_qat_warning_capture_contract_rejects_syntactic_broadening(line: str) -> None:
    capture_anchor = '        warnings.simplefilter("always")\n'
    assert CLEAN_QAT_CAPTURE_SOURCE.count(line) == 1
    assert CLEAN_QAT_CAPTURE_SOURCE.count(capture_anchor) == 1
    without_original = CLEAN_QAT_CAPTURE_SOURCE.replace(line, "", 1)
    mutated = without_original.replace(
        capture_anchor,
        capture_anchor + "        " + line.lstrip(),
        1,
    )
    assert mutated != CLEAN_QAT_CAPTURE_SOURCE
    ast.parse(mutated)
    with pytest.raises(AssertionError):
        _assert_qat_warning_capture_is_exact(mutated)


@pytest.mark.parametrize(
    "extra",
    (
        "\ndef other():\n    with warnings.catch_warnings(record=True):\n        pass\n",
        "\ndef other():\n    warnings.simplefilter('always')\n",
    ),
)
def test_qat_warning_capture_contract_rejects_other_consumer_capture(extra: str) -> None:
    with pytest.raises(AssertionError):
        _assert_no_other_consumer_warning_capture(
            "def test_graph():\n    pass\n",
            CLEAN_QAT_CAPTURE_SOURCE + extra,
        )


def test_qat_warning_origin_source_mutation_cannot_delete_is_file_guard(
    tmp_path: Path,
) -> None:
    source_path = (
        REPO_ROOT
        / "tests"
        / "nnx_surface"
        / "test_quantization_mnist_ffnn_pytorch.py"
    )
    source = source_path.read_text(encoding="utf-8")
    mutated = source.replace(" or not origin.is_file()", "", 1)
    assert mutated != source
    module_path = tmp_path / "mutated_qat_warning_debt.py"
    module_path.write_text(mutated, encoding="utf-8")
    module_name = "mutated_qat_warning_debt"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        distributions, record = module._exact_qat_warning(tmp_path / "inventory")
        exact_path = Path(record.filename)
        exact_path.unlink()
        exact_path.mkdir()
        with pytest.raises(AssertionError, match="qat warning debt validation failed"):
            module._assert_qat_warning_debt(
                (module._warning_record(exact_path),),
                qat_config="8da4w",
                distribution=distributions.__getitem__,
            )
    finally:
        sys.modules.pop(module_name, None)


def _call_to_name(node: ast.AST, name: str) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
    )


def _direct_selected_import(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "hooks"
        and node.func.attr == "import_module"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Attribute)
        and isinstance(node.args[0].value, ast.Name)
        and node.args[0].value.id == "pin"
        and node.args[0].attr == "import_name"
        and not node.keywords
    )


def _assert_selected_import_warning_boundary(source: str) -> None:
    tree = ast.parse(source)
    assignments = tuple(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "_IMPORT_WARNING_OUTER_COMPONENTS"
            for target in node.targets
        )
    )
    assert len(assignments) == 1
    value = assignments[0].value
    assert isinstance(value, ast.Call)
    assert isinstance(value.func, ast.Name) and value.func.id == "frozenset"
    assert len(value.args) == 1 and not value.keywords
    assert isinstance(value.args[0], ast.Tuple)
    members = tuple(
        element.value
        for element in value.args[0].elts
        if isinstance(element, ast.Constant) and isinstance(element.value, str)
    )
    assert members == ("torch-geometric", "torch-sparse")
    assert len(members) == len(value.args[0].elts)

    verify_functions = tuple(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "verify_torch_stack"
    )
    assert len(verify_functions) == 1
    verify_function = verify_functions[0]
    branches = tuple(
        node
        for node in ast.walk(verify_function)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Attribute)
        and isinstance(node.test.left.value, ast.Name)
        and node.test.left.value.id == "pin"
        and node.test.left.attr == "distribution"
        and len(node.test.ops) == 1
        and isinstance(node.test.ops[0], ast.In)
        and len(node.test.comparators) == 1
        and isinstance(node.test.comparators[0], ast.Name)
        and node.test.comparators[0].id == "_IMPORT_WARNING_OUTER_COMPONENTS"
    )
    assert len(branches) == 1
    branch = branches[0]
    wrapper_calls = tuple(
        node
        for node in ast.walk(verify_function)
        if _call_to_name(node, "_import_with_selected_warning_boundary")
    )
    direct_calls = tuple(
        node for node in ast.walk(verify_function) if _direct_selected_import(node)
    )
    assert len(wrapper_calls) == 1
    assert len(direct_calls) == 1
    assert any(node is wrapper_calls[0] for statement in branch.body for node in ast.walk(statement))
    assert any(node is direct_calls[0] for statement in branch.orelse for node in ast.walk(statement))

    capture_calls = tuple(
        (function.name, node)
        for function in tree.body
        if isinstance(function, ast.FunctionDef)
        for node in ast.walk(function)
        if _call_to_name(node, "_capture_selected_import")
    )
    assert len(capture_calls) == 1
    assert capture_calls[0][0] == "_import_with_selected_warning_boundary"


def _is_sys_modules(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "sys"
        and node.attr == "modules"
    )


def _mutates_sys_modules(node: ast.AST) -> bool:
    targets: tuple[ast.AST, ...] = ()
    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
        if isinstance(node, ast.Assign):
            targets = tuple(node.targets)
        else:
            targets = (node.target,)
    elif isinstance(node, ast.Delete):
        targets = tuple(node.targets)
    if any(
        _is_sys_modules(target)
        or (isinstance(target, ast.Subscript) and _is_sys_modules(target.value))
        for target in targets
    ):
        return True
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and _is_sys_modules(node.func.value)
        and node.func.attr
        in {"clear", "pop", "popitem", "setdefault", "update", "__delitem__", "__setitem__"}
    )


def _pytest_filterwarnings_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "filterwarnings"
        and isinstance(node.func.value, ast.Attribute)
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id == "pytest"
        and node.func.value.attr == "mark"
    )


def _assert_no_warning_policy_bypass(sources: dict[str, str]) -> None:
    assert set(sources) == {
        "verifier",
        "conftest",
        "graph",
        "quantization",
        "make",
        "ci",
    }
    for source in sources.values():
        assert "PYTHONWARNINGS" not in source
        assert "--disable-warnings" not in source
        assert re.search(r"(?:^|\s)-W\s+ignore(?:\s|$)", source) is None
    for name in ("verifier", "conftest", "graph", "quantization"):
        tree = ast.parse(sources[name])
        for node in ast.walk(tree):
            assert not _warnings_call(node, "filterwarnings")
            assert not _pytest_filterwarnings_call(node)
            assert not _mutates_sys_modules(node)
            if _warnings_call(node, "simplefilter"):
                assert node.args
                action = node.args[0]
                assert not (
                    isinstance(action, ast.Constant)
                    and isinstance(action.value, str)
                    and action.value == "ignore"
                )


def _warning_policy_sources() -> dict[str, str]:
    return {
        "verifier": (REPO_ROOT / "scripts" / "verify_torch_stack.py").read_text(
            encoding="utf-8"
        ),
        "conftest": (REPO_ROOT / "tests" / "nnx_surface" / "conftest.py").read_text(
            encoding="utf-8"
        ),
        "graph": (
            REPO_ROOT
            / "tests"
            / "nnx_surface"
            / "test_node_classification_reddit_gnn_pyg.py"
        ).read_text(encoding="utf-8"),
        "quantization": (
            REPO_ROOT
            / "tests"
            / "nnx_surface"
            / "test_quantization_mnist_ffnn_pytorch.py"
        ).read_text(encoding="utf-8"),
        "make": (REPO_ROOT / "Makefile").read_text(encoding="utf-8"),
        "ci": (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(
            encoding="utf-8"
        ),
    }


def test_selected_import_warning_boundary_is_exactly_local() -> None:
    source = (REPO_ROOT / "scripts" / "verify_torch_stack.py").read_text(
        encoding="utf-8"
    )

    _assert_selected_import_warning_boundary(source)


@pytest.mark.parametrize(
    ("old", "new"),
    (
        (
            "if pin.distribution in _IMPORT_WARNING_OUTER_COMPONENTS:",
            "if True:",
        ),
        (
            "module = hooks.import_module(pin.import_name)",
            "module = _import_with_selected_warning_boundary(\n"
            "                    pin, distribution, torch_distribution, hooks\n"
            "                )",
        ),
        (
            '_IMPORT_WARNING_OUTER_COMPONENTS = frozenset(("torch-geometric", "torch-sparse"))',
            '_IMPORT_WARNING_OUTER_COMPONENTS = frozenset(("torch-geometric", "torch-sparse", "torchao"))',
        ),
    ),
    ids=("unconditional-wrapper", "all-import-wrapper", "broadened-membership"),
)
def test_selected_import_warning_boundary_rejects_source_mutations(
    old: str, new: str
) -> None:
    source = (REPO_ROOT / "scripts" / "verify_torch_stack.py").read_text(
        encoding="utf-8"
    )
    mutated = source.replace(old, new, 1)
    assert mutated != source
    ast.parse(mutated)
    with pytest.raises(AssertionError):
        _assert_selected_import_warning_boundary(mutated)


def test_warning_policy_has_no_global_cli_environment_or_module_bypass() -> None:
    _assert_no_warning_policy_bypass(_warning_policy_sources())


@pytest.mark.parametrize(
    ("consumer", "mutation"),
    (
        ("graph", "\nwarnings.filterwarnings('ignore')\n"),
        ("quantization", "\nwarnings.simplefilter('ignore')\n"),
        (
            "conftest",
            "\npytestmark = pytest.mark.filterwarnings('ignore::DeprecationWarning')\n",
        ),
        ("make", "\nPYTHONWARNINGS=ignore pytest\n"),
        ("ci", "\n      run: pytest -W ignore\n"),
        ("ci", "\n      run: pytest --disable-warnings\n"),
        ("verifier", "\ndel sys.modules['torch_geometric']\n"),
        ("graph", "\nsys.modules['torch_sparse'] = object()\n"),
    ),
    ids=(
        "filterwarnings",
        "simplefilter-ignore",
        "pytest-mark-filterwarnings",
        "pythonwarnings",
        "cli-w-ignore",
        "disable-warnings",
        "sys-modules-delete",
        "sys-modules-replace",
    ),
)
def test_warning_policy_rejects_each_bypass_mutation(
    consumer: str, mutation: str
) -> None:
    sources = _warning_policy_sources()
    sources[consumer] += mutation
    with pytest.raises(AssertionError):
        _assert_no_warning_policy_bypass(sources)
