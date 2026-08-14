from __future__ import annotations

import dataclasses
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
PYG_INDEX_DISTRIBUTIONS = {"pyg-lib", "torch-scatter", "torch-sparse", "torch-cluster"}
BINARY_DISTRIBUTIONS = PYG_INDEX_DISTRIBUTIONS | {
    "torch",
    "torchvision",
    "torchaudio",
    "torch-spline-conv",
}


class PlatformDistribution:
    def __init__(self, name: str, version: str, root: Path, wheel_tag: str) -> None:
        self.version = version
        self.root = root
        self.wheel_tag = wheel_tag
        package = IMPORTS[name].replace(".", "/") + "/__init__.py"
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
        if filename != "WHEEL":
            return None
        return f"Wheel-Version: 1.0\nTag: {self.wheel_tag}\n"


class PlatformStack:
    def __init__(self, tmp_path: Path, system: str, machine: str) -> None:
        self.system = system
        self.machine = machine
        self.calls: list[str] = []
        self.distributions: dict[str, PlatformDistribution] = {}
        self.modules: dict[str, ModuleType] = {}
        root = tmp_path / "site-packages"
        if system == "Darwin":
            binary_tag = "cp310-abi3-macosx_11_0_arm64"
            pyg_local = "pt211"
        else:
            binary_tag = f"cp311-cp311-manylinux_2_28_{machine}"
            pyg_local = "pt211cpu"
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
            self.distributions[distribution] = PlatformDistribution(
                distribution,
                version,
                root,
                wheel_tag,
            )
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
            canaries=CanaryHooks(
                *(self._canary(name) for name in ("scatter", "sparse", "cluster", "sampler", "spline"))
            ),
        )

    def _canary(self, name: str):
        def canary(modules):
            assert tuple(modules) == tuple(IMPORTS)
            self.calls.append(name)

        return canary

    def _nnx(self) -> object:
        self.calls.append("nnx")
        return object()

    def version(self, distribution: str, version: str) -> None:
        self.distributions[distribution].version = version

    def wheel(self, distribution: str, wheel_tag: str) -> None:
        self.distributions[distribution].wheel_tag = wheel_tag

    def remove_recorded_module(self, distribution: str) -> None:
        selected = self.distributions[distribution]
        assert selected.files is not None
        module_suffix = IMPORTS[distribution].replace(".", "/") + "/__init__.py"
        selected.files = [path for path in selected.files if not str(path).endswith(module_suffix)]


@pytest.mark.parametrize("local", ("pt211", None), ids=("native-local", "no-local"))
def test_darwin_arm64_accepts_selected_pyg_provenance_through_all_runtime_gates(
    tmp_path: Path,
    local: str | None,
) -> None:
    stack = PlatformStack(tmp_path, "Darwin", "arm64")
    stack.version("pyg-lib", "0.8.0" + (f"+{local}" if local else ""))

    evidence = verify_torch_stack(repo=REPO_ROOT, hooks=stack.hooks)

    assert evidence == StackEvidence("Darwin", "arm64", "2.11.0", "pyg-lib")
    assert stack.calls == ["scatter", "sparse", "cluster", "sampler", "spline", "nnx"]


@pytest.mark.parametrize("local", ("pt211cpu", "pt211cu128", "cu128", "arbitrary", "pt210"))
def test_darwin_arm64_rejects_nonselected_pyg_local_versions(tmp_path: Path, local: str) -> None:
    stack = PlatformStack(tmp_path, "Darwin", "arm64")
    stack.version("pyg-lib", f"0.8.0+{local}")

    with pytest.raises(
        TorchStackVerificationError,
        match=r"^torch stack verification failed: pyg-lib: abi$",
    ):
        verify_torch_stack(repo=REPO_ROOT, hooks=stack.hooks)

    assert stack.calls == []


@pytest.mark.parametrize(("system", "machine"), (("Linux", "x86_64"), ("Linux", "aarch64")))
def test_linux_accepts_exact_pt211cpu_pyg_provenance(tmp_path: Path, system: str, machine: str) -> None:
    stack = PlatformStack(tmp_path, system, machine)

    evidence = verify_torch_stack(repo=REPO_ROOT, hooks=stack.hooks)

    assert evidence.system == "Linux"
    assert stack.calls == ["scatter", "sparse", "cluster", "sampler", "spline", "nnx"]


@pytest.mark.parametrize("machine", ("x86_64", "aarch64"))
def test_linux_absent_pyg_local_still_runs_all_runtime_gates(tmp_path: Path, machine: str) -> None:
    stack = PlatformStack(tmp_path, "Linux", machine)
    stack.version("pyg-lib", "0.8.0")

    evidence = verify_torch_stack(repo=REPO_ROOT, hooks=stack.hooks)

    assert evidence == StackEvidence("Linux", machine, "2.11.0+cpu", "pyg-lib")
    assert stack.calls == ["scatter", "sparse", "cluster", "sampler", "spline", "nnx"]


@pytest.mark.parametrize("local", ("pt211", "pt211cu128", "cu128", "arbitrary", "pt210cpu"))
def test_linux_rejects_nonselected_pyg_local_versions(tmp_path: Path, local: str) -> None:
    stack = PlatformStack(tmp_path, "Linux", "x86_64")
    stack.version("pyg-lib", f"0.8.0+{local}")

    with pytest.raises(
        TorchStackVerificationError,
        match=r"^torch stack verification failed: pyg-lib: abi$",
    ):
        verify_torch_stack(repo=REPO_ROOT, hooks=stack.hooks)

    assert stack.calls == []


@pytest.mark.parametrize(
    ("system", "machine", "local"),
    (("Darwin", "arm64", "cpu"), ("Darwin", "arm64", "cu128"), ("Linux", "x86_64", "cu128")),
)
def test_torch_local_versions_remain_platform_strict(
    tmp_path: Path,
    system: str,
    machine: str,
    local: str,
) -> None:
    stack = PlatformStack(tmp_path, system, machine)
    stack.version("torch", f"2.11.0+{local}")
    stack.modules["torch"].__version__ = f"2.11.0+{local}"

    with pytest.raises(
        TorchStackVerificationError,
        match=r"^torch stack verification failed: torch: metadata$",
    ):
        verify_torch_stack(repo=REPO_ROOT, hooks=stack.hooks)

    assert stack.calls == []


@pytest.mark.parametrize(
    ("system", "machine", "local", "wheel_tag"),
    (
        ("Darwin", "arm64", "pt211", "cp310-abi3-macosx_11_0_x86_64"),
        ("Darwin", "arm64", None, "cp310-abi3-macosx_11_0_x86_64"),
        ("Linux", "x86_64", None, "cp311-cp311-manylinux_2_28_aarch64"),
    ),
    ids=("darwin-local", "darwin-no-local", "linux-no-local"),
)
def test_selected_or_absent_local_still_requires_architecture_compatible_wheel(
    tmp_path: Path,
    system: str,
    machine: str,
    local: str | None,
    wheel_tag: str,
) -> None:
    stack = PlatformStack(tmp_path, system, machine)
    stack.version("pyg-lib", "0.8.0" + (f"+{local}" if local else ""))
    stack.wheel("pyg-lib", wheel_tag)

    with pytest.raises(
        TorchStackVerificationError,
        match=r"^torch stack verification failed: pyg-lib: platform$",
    ):
        verify_torch_stack(repo=REPO_ROOT, hooks=stack.hooks)

    assert stack.calls == []


@pytest.mark.parametrize(
    ("system", "machine", "wheel_tag"),
    (
        ("Darwin", "arm64", "cp310-cp310-macosx_11_0_arm64"),
        ("Linux", "x86_64", "cp310-cp310-manylinux_2_28_x86_64"),
    ),
)
def test_absent_local_still_requires_python311_compatible_wheel(
    tmp_path: Path,
    system: str,
    machine: str,
    wheel_tag: str,
) -> None:
    stack = PlatformStack(tmp_path, system, machine)
    stack.version("pyg-lib", "0.8.0")
    stack.wheel("pyg-lib", wheel_tag)

    with pytest.raises(
        TorchStackVerificationError,
        match=r"^torch stack verification failed: pyg-lib: abi$",
    ):
        verify_torch_stack(repo=REPO_ROOT, hooks=stack.hooks)

    assert stack.calls == []


@pytest.mark.parametrize(("system", "machine"), (("Darwin", "arm64"), ("Linux", "x86_64")))
def test_absent_local_still_requires_record_ownership(
    tmp_path: Path,
    system: str,
    machine: str,
) -> None:
    stack = PlatformStack(tmp_path, system, machine)
    stack.version("pyg-lib", "0.8.0")
    stack.remove_recorded_module("pyg-lib")

    with pytest.raises(
        TorchStackVerificationError,
        match=r"^torch stack verification failed: pyg-lib: metadata$",
    ):
        verify_torch_stack(repo=REPO_ROOT, hooks=stack.hooks)

    assert stack.calls == []


@pytest.mark.parametrize(
    ("system", "machine", "local"),
    (("Darwin", "arm64", "pt211"), ("Darwin", "arm64", None), ("Linux", "x86_64", None)),
    ids=("darwin-local", "darwin-no-local", "linux-no-local"),
)
def test_selected_or_absent_local_still_requires_runtime_canaries(
    tmp_path: Path,
    system: str,
    machine: str,
    local: str | None,
) -> None:
    stack = PlatformStack(tmp_path, system, machine)
    stack.version("pyg-lib", "0.8.0" + (f"+{local}" if local else ""))

    def fail_cluster(modules):
        raise RuntimeError

    stack.hooks = dataclasses.replace(
        stack.hooks,
        canaries=dataclasses.replace(stack.hooks.canaries, cluster=fail_cluster),
    )

    with pytest.raises(
        TorchStackVerificationError,
        match=r"^torch stack verification failed: cluster: operator$",
    ):
        verify_torch_stack(repo=REPO_ROOT, hooks=stack.hooks)

    assert stack.calls == ["scatter", "sparse"]


@pytest.mark.parametrize(
    ("system", "machine"),
    (("Darwin", "x86_64"), ("Linux", "arm64"), ("Windows", "AMD64")),
)
def test_local_version_policy_cannot_enable_unsupported_platforms(
    tmp_path: Path,
    system: str,
    machine: str,
) -> None:
    stack = PlatformStack(tmp_path, system, machine)

    with pytest.raises(
        TorchStackVerificationError,
        match=r"^torch stack verification failed: platform: platform$",
    ):
        verify_torch_stack(repo=REPO_ROOT, hooks=stack.hooks)

    assert stack.calls == []


def test_design_and_plan_state_the_same_platform_local_version_policy() -> None:
    design = " ".join(
        (
            REPO_ROOT / "docs/superpowers/specs/2026-08-14-issue-62-torch-stack-upgrade-design.md"
        ).read_text(encoding="utf-8").split()
    )
    plan = " ".join(
        (
            REPO_ROOT / "docs/superpowers/plans/2026-08-14-issue-62-torch-stack-upgrade-implementation-plan.md"
        ).read_text(encoding="utf-8").split()
    )
    required_claims = (
        "Linux PyG-index local versions require the exact `pt211cpu` tag",
        "Darwin arm64 accepts the exact `pt211` local tag or an absent local tag",
        "An absent local tag is accepted only after independent WHEEL/RECORD, Python ABI/platform architecture, and mandatory runtime-canary checks pass",
    )

    for claim in required_claims:
        assert claim in design
        assert claim in plan


def test_task2_plan_inventories_and_runs_the_platform_regression_gate() -> None:
    plan = (
        REPO_ROOT / "docs/superpowers/plans/2026-08-14-issue-62-torch-stack-upgrade-implementation-plan.md"
    ).read_text(encoding="utf-8")
    task2 = plan.split("## 12.22.5 Task 2:", 1)[1].split("## 12.22.6 Task 3:", 1)[0]
    test_path = "tests/test_verify_torch_stack_platform.py"
    pytest_command = (
        "pytest -p no:cacheprovider tests/test_verify_torch_stack.py "
        f"{test_path} tests/test_makefile_contract.py -q"
    )
    ruff_command = (
        "ruff check scripts/verify_torch_stack.py tests/test_verify_torch_stack.py "
        f"{test_path} tests/test_makefile_contract.py"
    )
    git_add_command = (
        "git add scripts/verify_torch_stack.py tests/test_verify_torch_stack.py "
        f"{test_path} Makefile tests/test_makefile_contract.py "
        "docs/superpowers/plans/2026-08-14-issue-62-torch-stack-upgrade-implementation-plan.md"
    )

    assert f"- Create: `{test_path}`" in task2
    assert task2.count(pytest_command) == 2
    assert ruff_command in task2
    assert git_add_command in task2


def test_reported_darwin_wheel_tag_is_python311_compatible() -> None:
    contract = verifier.load_stack_contract(REPO_ROOT, "Darwin", "arm64")
    tag = next(iter(verifier.parse_tag("cp310-abi3-macosx_11_0_arm64")))

    assert verifier._tag_matches_platform(tag, contract)
    assert (tag.interpreter, tag.abi) in verifier._PYTHON_311_ABIS
