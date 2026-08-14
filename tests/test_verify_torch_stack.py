from __future__ import annotations

import dataclasses
import importlib.util
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
    "torch-cluster": "torch_cluster",
    "torch-spline-conv": "torch_spline_conv",
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
    "torch-cluster": "1.6.3",
    "torch-spline-conv": "1.2.2",
}
CORE = "torch==2.11.0\ntorchvision==0.26.0\ntorchaudio==2.11.0\n"
ECOSYSTEM = "pytorch-lightning==2.6.1\ntorchmetrics==1.9.0\ntorchao==0.18.0\n"
RUNTIME = (
    "-r torch-ecosystem-requirements.txt\n"
    "--find-links https://data.pyg.org/whl/torch-2.11.0+cpu.html\n"
    "pyg-lib==0.8.0\ntorch-scatter==2.1.2\ntorch-sparse==0.6.18\n"
    "torch-cluster==1.6.3\ntorch-spline-conv==1.2.2\ntorch_geometric==2.8.0.post1\n"
)
AUDIT = "-r torch-core-requirements.txt\n-r torch-ecosystem-requirements.txt\ntorch_geometric==2.8.0.post1\n"
EXTENSIONS = (
    "# Pre-resolved compiled PyG extension supplement for the strict audit.\n"
    "# Runtime source: torch-requirements.txt retains the approved PyG wheel selector.\n"
    "torch-scatter==2.1.2\ntorch-sparse==0.6.18\ntorch-cluster==1.6.3\ntorch-spline-conv==1.2.2\n"
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
    "torch-cluster",
    "torch-spline-conv",
}
BINARY_WHEELS = COMPILED | {"torch", "torchvision", "torchaudio"}
SENSITIVE = (
    "https://user:password@packages.invalid/private?token=secret "
    "/Users/example/private/installer.log Traceback RuntimeError installer-output"
)


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
            self.distributions[distribution] = FakeDistribution(
                distribution,
                EXPECTED_VERSIONS[distribution],
                root,
                f"Wheel-Version: 1.0\nTag: {wheel_tag}\n",
            )
            module = ModuleType(import_name)
            module.__file__ = str(module_path)
            if distribution == "torch":
                module.__version__ = EXPECTED_VERSIONS[distribution]
                module.version = SimpleNamespace(cuda=None)
            self.modules[import_name] = module
        canaries = CanaryHooks(*(self._canary(name) for name in ("scatter", "sparse", "cluster", "sampler", "spline")))
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
    assert stack.calls == ["scatter", "sparse", "cluster", "sampler", "spline", "nnx"]


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
    fake_stack.remove_record_file("torch-cluster", "torch_cluster/__init__.py")

    with pytest.raises(
        TorchStackVerificationError,
        match=r"^torch stack verification failed: torch-cluster: metadata$",
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

    assert fake_stack.calls == ["scatter", "sparse", "cluster", "sampler", "spline", "nnx"]


@pytest.mark.parametrize(
    ("failed", "category"),
    (("scatter", "operator"), ("sparse", "operator"), ("cluster", "operator"), ("sampler", "sampler"), ("spline", "operator")),
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

    with pytest.warns(UserWarning), pytest.raises(
        TorchStackVerificationError,
        match=r"^torch stack verification failed: nnx: nnx$",
    ):
        verify_torch_stack(repo=fake_stack.repo, hooks=fake_stack.hooks)

    assert fake_stack.calls == ["scatter", "sparse", "cluster", "sampler", "spline"]


def test_environment_variables_cannot_bypass_canaries_or_nnx(fake_stack: FakeStack, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("SKIP_TORCH_STACK_VERIFY", "TORCH_STACK_VERIFY", "CI", "NNX_ALLOW_EDITABLE"):
        monkeypatch.setenv(name, "0")

    verify_torch_stack(repo=fake_stack.repo, hooks=fake_stack.hooks)

    assert fake_stack.calls == ["scatter", "sparse", "cluster", "sampler", "spline", "nnx"]


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

    assert fake_stack.calls == ["scatter", "sparse", "cluster", "sampler", "spline"]


def test_injected_torch_version_is_rejected_before_nnx(fake_stack: FakeStack) -> None:
    fake_stack.modules["torch"].__version__ = "2.11.0+cpu\nhttps://token.invalid/private"

    with pytest.raises(
        TorchStackVerificationError,
        match=r"^torch stack verification failed: torch: metadata$",
    ):
        verify_torch_stack(repo=fake_stack.repo, hooks=fake_stack.hooks)

    assert fake_stack.calls == ["scatter", "sparse", "cluster", "sampler", "spline"]


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
        self.cluster_result = _CanaryResult((2, 2), elements=4)
        self.batch = SimpleNamespace(batch_size=1, num_edges=1)
        self.spline_result = _CanaryResult((2, 2))
        rig = self

        class SparseTensor:
            def __init__(inner_self, **kwargs):
                rig.events.append("sparse-init")
                inner_self.kwargs = kwargs

            def matmul(inner_self, matrix):
                rig.events.append("sparse-matmul")
                return rig.sparse_result

        class NeighborLoader:
            def __init__(inner_self, data, **kwargs):
                rig.events.append("neighbor-loader")
                inner_self.data = data
                inner_self.kwargs = kwargs

            def __iter__(inner_self):
                rig.events.append("neighbor-batch")
                return iter((rig.batch,))

        class SplineConv:
            def __init__(inner_self, *args, **kwargs):
                rig.events.append("spline-init")

            def __call__(inner_self, *args):
                rig.events.append("spline-forward")
                return rig.spline_result

        self.modules = {
            "torch": SimpleNamespace(tensor=lambda value: value),
            "torch-scatter": SimpleNamespace(scatter=self._scatter),
            "torch-sparse": SimpleNamespace(SparseTensor=SparseTensor),
            "torch-cluster": SimpleNamespace(knn=self._knn),
            "torch-geometric": SimpleNamespace(
                data=SimpleNamespace(Data=self._data),
                loader=SimpleNamespace(NeighborLoader=NeighborLoader),
                nn=SimpleNamespace(SplineConv=SplineConv),
            ),
        }

    def _scatter(self, *args, **kwargs):
        self.events.append("scatter")
        return self.scatter_result

    def _knn(self, *args, **kwargs):
        self.events.append("cluster-knn")
        return self.cluster_result

    def _data(self, **kwargs):
        self.events.append("graph-data")
        return SimpleNamespace(**kwargs)


def test_default_canary_bodies_execute_all_required_operators() -> None:
    rig = _DefaultCanaryRig()

    verifier_module._scatter_canary(rig.modules)
    verifier_module._sparse_canary(rig.modules)
    verifier_module._cluster_canary(rig.modules)
    verifier_module._sampler_canary(rig.modules)
    verifier_module._spline_canary(rig.modules)

    assert rig.events == [
        "scatter",
        "sparse-init",
        "sparse-matmul",
        "cluster-knn",
        "graph-data",
        "neighbor-loader",
        "neighbor-batch",
        "spline-init",
        "spline-forward",
    ]


@pytest.mark.parametrize(
    ("canary", "mutation"),
    (
        ("scatter", lambda rig: setattr(rig, "scatter_result", _CanaryResult((1,)))),
        ("sparse", lambda rig: setattr(rig, "sparse_result", _CanaryResult((1, 1)))),
        ("cluster", lambda rig: setattr(rig, "cluster_result", _CanaryResult((1, 2), elements=2))),
        ("cluster", lambda rig: setattr(rig, "cluster_result", _CanaryResult((2, 0), elements=0))),
        ("sampler", lambda rig: setattr(rig.batch, "batch_size", 0)),
        ("sampler", lambda rig: setattr(rig.batch, "num_edges", 0)),
        ("spline", lambda rig: setattr(rig, "spline_result", _CanaryResult((2, 1)))),
    ),
    ids=("scatter-shape", "sparse-shape", "cluster-shape", "cluster-empty", "sampler-seed", "sampler-edge", "spline-shape"),
)
def test_default_canary_bodies_reject_empty_or_weakened_results(canary: str, mutation) -> None:
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


@pytest.mark.parametrize("omitted", ("scatter", "sparse", "cluster", "sampler", "spline", "nnx"))
def test_source_mutations_cannot_omit_a_canary_or_nnx(tmp_path: Path, omitted: str) -> None:
    source = (REPO_ROOT / "scripts" / "verify_torch_stack.py").read_text(encoding="utf-8")
    if omitted == "nnx":
        original = "        hooks.nnx_verify()"
        replacement = "        object()"
    else:
        original = "            canary(modules)"
        replacement = f'            if name != "{omitted}":\n                canary(modules)'
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
            assert stack.calls == ["scatter", "sparse", "cluster", "sampler", "spline", "nnx"]
    finally:
        sys.modules.pop(module_name, None)
