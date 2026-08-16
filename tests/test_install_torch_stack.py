from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from scripts.install_torch_stack import (
    InstallCommand,
    InstallStage,
    TorchStackInstallError,
    build_install_commands,
    install_torch_stack,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
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
MANIFESTS = {
    "torch-core-requirements.txt": CORE,
    "torch-ecosystem-requirements.txt": ECOSYSTEM,
    "torch-requirements.txt": RUNTIME_MANIFEST,
    "torch-audit-requirements.txt": AUDIT,
    "pyg-extension-audit-requirements.txt": SUPPLEMENT,
}
UPGRADE_PIP = (sys.executable, "-m", "pip", "install", "--upgrade", "pip")
LINUX_CORE = (
    sys.executable,
    "-m",
    "pip",
    "install",
    "--index-url",
    "https://download.pytorch.org/whl/cpu",
    "-r",
    "torch-core-requirements.txt",
)
DARWIN_CORE = (
    sys.executable,
    "-m",
    "pip",
    "install",
    "-r",
    "torch-core-requirements.txt",
)
RUNTIME = (
    sys.executable,
    "-m",
    "pip",
    "install",
    "--only-binary=pyg-lib,torch-scatter,torch-sparse",
    "-r",
    "torch-requirements.txt",
)
ROOT = (
    sys.executable,
    "-m",
    "pip",
    "install",
    "--only-binary=thekaveh-nnx",
    "-r",
    "requirements.txt",
)
SUPPORTED_HOSTS = (("Linux", "x86_64"), ("Linux", "aarch64"), ("Darwin", "arm64"))
LEGACY_MANIFEST_MUTATIONS = (
    (
        "torch_geometric==2.8.0.post1\n",
        "torch-cluster==1.6.3\ntorch_geometric==2.8.0.post1\n",
    ),
    (
        "torch_geometric==2.8.0.post1\n",
        "torch-spline-conv==1.2.2\ntorch_geometric==2.8.0.post1\n",
    ),
)
LEGACY_ARG_MUTATIONS = (
    (
        "--only-binary=pyg-lib,torch-scatter,torch-sparse",
        "--only-binary=pyg-lib,torch-scatter,torch-sparse,torch-cluster",
    ),
    (
        '"-r",\n                "torch-requirements.txt",',
        '"--no-binary=torch-spline-conv",\n                "--no-build-isolation",\n                "-r",\n                "torch-requirements.txt",',
    ),
    (("--upgrade", "pip"), ("--upgrade", "pip", "wheel")),
)


def _assert_canonical_manifests(repo: Path) -> None:
    for name, expected in MANIFESTS.items():
        assert (repo / name).read_text(encoding="utf-8") == expected
    requirements = (repo / "requirements.txt").read_text(encoding="utf-8")
    assert requirements.count("thekaveh-nnx[lm]==0.2.0") == 1
    package_names = tuple(
        re.sub(r"[-_.]+", "-", line.split("==", 1)[0]).lower()
        for line in requirements.splitlines()
        if line and not line.startswith("#") and "==" in line
    )
    assert "torchao" not in package_names


def _assert_final_command_plan(commands: tuple[InstallCommand, ...], core: tuple[str, ...]) -> None:
    assert tuple(item.stage for item in commands) == tuple(InstallStage)
    assert tuple(item.argv for item in commands) == (UPGRADE_PIP, core, RUNTIME, ROOT)


def test_canonical_manifest_bytes_are_exact() -> None:
    _assert_canonical_manifests(REPO_ROOT)


def test_final_four_stage_plans_are_exact() -> None:
    for system, machine, core in (
        ("Linux", "x86_64", LINUX_CORE),
        ("Linux", "aarch64", LINUX_CORE),
        ("Darwin", "arm64", DARWIN_CORE),
    ):
        _assert_final_command_plan(build_install_commands(sys.executable, system, machine), core)


def test_makefile_installer_and_codespace_recipes_are_exact() -> None:
    lines = (REPO_ROOT / "Makefile").read_text(encoding="utf-8").splitlines()

    def recipe(target: str) -> tuple[str, tuple[str, ...]]:
        index = next(index for index, line in enumerate(lines) if line.startswith(f"{target}:"))
        body: list[str] = []
        for line in lines[index + 1 :]:
            if line and not line.startswith(("\t", "#")):
                break
            if line.startswith("\t"):
                body.append(line[1:])
        return lines[index], tuple(body)

    install_header, install_recipe = recipe("install-torch-stack")
    codespace_header, codespace_recipe = recipe("codespace-setup")
    assert install_header == "install-torch-stack:"
    assert install_recipe == ("$(PYTHON) -m scripts.install_torch_stack",)
    assert codespace_header == "codespace-setup: install-torch-stack"
    assert codespace_recipe == (
        "$(MAKE) nlp-assets",
        "$(PYTHON) -m pip check",
        "$(MAKE) verify-torch-stack",
        "$(MAKE) verify-nnx-install",
    )


@pytest.mark.parametrize(("original", "replacement"), LEGACY_MANIFEST_MUTATIONS)
def test_legacy_manifest_mutations_fail_the_exact_manifest_contract(
    tmp_path: Path, original: str, replacement: str
) -> None:
    for name in MANIFESTS:
        (tmp_path / name).write_text((REPO_ROOT / name).read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "requirements.txt").write_text(
        (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8"), encoding="utf-8"
    )
    runtime = tmp_path / "torch-requirements.txt"
    source = runtime.read_text(encoding="utf-8")
    mutated = source.replace(original, replacement, 1)
    assert mutated != source
    runtime.write_text(mutated, encoding="utf-8")

    with pytest.raises(AssertionError):
        _assert_canonical_manifests(tmp_path)


def _import_mutated_installer(tmp_path: Path, source: str) -> ModuleType:
    installer = tmp_path / "mutated_install_torch_stack.py"
    installer.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("mutated_install_torch_stack", installer)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        assert sys.modules.pop(spec.name, None) is module
    return module


@pytest.mark.parametrize(("original", "replacement"), LEGACY_ARG_MUTATIONS)
def test_legacy_argument_mutations_fail_the_exact_four_stage_contract(
    tmp_path: Path, original: str | tuple[str, ...], replacement: str | tuple[str, ...]
) -> None:
    source = (REPO_ROOT / "scripts" / "install_torch_stack.py").read_text(encoding="utf-8")
    if isinstance(original, tuple):
        original = '"' + '", "'.join(original) + '"'
        replacement = '"' + '", "'.join(replacement) + '"'
    assert original in source
    mutated = source.replace(original, replacement, 1)
    assert mutated != source
    module = _import_mutated_installer(tmp_path, mutated)

    with pytest.raises(AssertionError):
        _assert_final_command_plan(
            module.build_install_commands(sys.executable, "Linux", "x86_64"), LINUX_CORE
        )


@pytest.mark.parametrize(
    ("original", "replacement"),
    (
        (
            "--only-binary=pyg-lib,torch-scatter,torch-sparse",
            "--only-binary=torch-scatter,torch-sparse",
        ),
        (
            "--only-binary=pyg-lib,torch-scatter,torch-sparse",
            "--only-binary=pyg-lib,torch-sparse",
        ),
        (
            "--only-binary=pyg-lib,torch-scatter,torch-sparse",
            "--only-binary=pyg-lib,torch-scatter",
        ),
        ("--only-binary=pyg-lib,torch-scatter,torch-sparse", "--only-binary=:all:"),
    ),
    ids=("missing-pyg-lib", "missing-scatter", "missing-sparse", "broad-binary-policy"),
)
def test_runtime_policy_mutations_fail_the_exact_four_stage_contract(
    tmp_path: Path, original: str, replacement: str
) -> None:
    source = (REPO_ROOT / "scripts" / "install_torch_stack.py").read_text(encoding="utf-8")
    assert original in source
    mutated = source.replace(original, replacement, 1)
    assert mutated != source
    module = _import_mutated_installer(tmp_path, mutated)

    with pytest.raises(AssertionError):
        _assert_final_command_plan(
            module.build_install_commands(sys.executable, "Linux", "x86_64"), LINUX_CORE
        )


def test_stage_order_root_position_and_command_count_mutations_fail_exact_contract(tmp_path: Path) -> None:
    source = (REPO_ROOT / "scripts" / "install_torch_stack.py").read_text(encoding="utf-8")
    upgrade = '        InstallCommand(InstallStage.UPGRADE_PIP, pip + ("--upgrade", "pip")),\n'
    core = "        InstallCommand(InstallStage.CORE, core),\n"
    runtime = (
        "        InstallCommand(\n"
        "            InstallStage.RUNTIME,\n"
        "            pip\n"
        "            + (\n"
        '                "--only-binary=pyg-lib,torch-scatter,torch-sparse",\n'
        '                "-r",\n'
        '                "torch-requirements.txt",\n'
        "            ),\n"
        "        ),\n"
    )
    root = (
        "        InstallCommand(\n"
        "            InstallStage.ROOT,\n"
        '            pip + ("--only-binary=thekaveh-nnx", "-r", "requirements.txt"),\n'
        "        ),\n"
    )
    mutations = (
        source.replace(upgrade + core, core + upgrade, 1),
        source.replace(runtime + root, root + runtime, 1),
        source.replace(root, root + root, 1),
        source.replace('"--only-binary=thekaveh-nnx", ', "", 1),
    )
    for mutated in mutations:
        assert mutated != source
        module = _import_mutated_installer(tmp_path, mutated)
        with pytest.raises(AssertionError):
            _assert_final_command_plan(
                module.build_install_commands(sys.executable, "Linux", "x86_64"), LINUX_CORE
            )


def test_unsupported_platform_and_nonzero_runner_mutations_fail_closed_contract(tmp_path: Path) -> None:
    source = (REPO_ROOT / "scripts" / "install_torch_stack.py").read_text(encoding="utf-8")
    unsupported = source.replace(
        '(("Linux", "x86_64"), ("Linux", "aarch64"), ("Darwin", "arm64"))',
        '(("Linux", "x86_64"), ("Linux", "aarch64"), ("Linux", "arm64"), ("Darwin", "arm64"))',
        1,
    )
    ignores_failure = source.replace("if runner(command.argv, check=False).returncode != 0:", "if False:", 1)
    for mutated in (unsupported, ignores_failure):
        assert mutated != source

    unsupported_module = _import_mutated_installer(tmp_path, unsupported)
    with pytest.raises(pytest.fail.Exception):
        with pytest.raises(unsupported_module.TorchStackInstallError, match="unsupported Torch stack platform"):
            unsupported_module.build_install_commands(sys.executable, "Linux", "arm64")

    runner_module = _import_mutated_installer(tmp_path, ignores_failure)
    with pytest.raises(pytest.fail.Exception):
        with pytest.raises(runner_module.TorchStackInstallError, match="torch stack installation failed: upgrade-pip"):
            runner_module.install_torch_stack(
                runner_module.build_install_commands(sys.executable, "Linux", "x86_64"),
                runner=lambda argv, *, check: SimpleNamespace(returncode=1),
            )


@pytest.mark.parametrize(("system", "machine"), (("Linux", "arm64"), ("Darwin", "x86_64"), ("Windows", "AMD64")))
def test_unsupported_platforms_fail_closed(system: str, machine: str) -> None:
    with pytest.raises(TorchStackInstallError, match="unsupported Torch stack platform"):
        build_install_commands(sys.executable, system, machine)


def test_installer_runs_every_immutable_argv_plan_without_checking_runner() -> None:
    commands = build_install_commands(sys.executable, "Linux", "x86_64")
    seen: list[tuple[tuple[str, ...], bool]] = []

    def runner(argv, *, check):
        seen.append((tuple(argv), check))
        return SimpleNamespace(returncode=0)

    install_torch_stack(commands, runner=runner)

    assert seen == [(command.argv, False) for command in commands]
    assert all(isinstance(command.argv, tuple) for command in commands)


def _assert_upgrade_failure_stops_the_full_plan(module: ModuleType) -> None:
    commands = module.build_install_commands(sys.executable, "Linux", "x86_64")
    seen: list[tuple[str, ...]] = []
    secret = "https://token@example.test/private-output"

    def runner(argv, *, check):
        seen.append(tuple(argv))
        return SimpleNamespace(returncode=1, stdout=secret, stderr=secret)

    with pytest.raises(module.TorchStackInstallError) as error:
        module.install_torch_stack(commands, runner=runner)

    assert str(error.value) == "torch stack installation failed: upgrade-pip"
    assert secret not in str(error.value)
    assert seen == [UPGRADE_PIP]


def test_installer_stops_after_a_failed_upgrade_stage() -> None:
    from scripts import install_torch_stack as module

    _assert_upgrade_failure_stops_the_full_plan(module)


def test_upgrade_failure_mutation_cannot_continue_to_later_stages(tmp_path: Path) -> None:
    source = (REPO_ROOT / "scripts" / "install_torch_stack.py").read_text(encoding="utf-8")
    original = "        if runner(command.argv, check=False).returncode != 0:\n"
    replacement = (
        "        result = runner(command.argv, check=False)\n"
        "        if command.stage is not InstallStage.UPGRADE_PIP and result.returncode != 0:\n"
    )
    assert original in source
    mutated = source.replace(original, replacement, 1)
    assert mutated != source
    module = _import_mutated_installer(tmp_path, mutated)

    with pytest.raises((AssertionError, pytest.fail.Exception)):
        _assert_upgrade_failure_stops_the_full_plan(module)


def test_installer_stops_on_nonzero_with_a_redacted_stable_error() -> None:
    commands = build_install_commands(sys.executable, "Linux", "x86_64")
    secret = "https://token@example.test/private-output"

    def runner(argv, *, check):
        return SimpleNamespace(returncode=1, stdout=secret, stderr=secret)

    with pytest.raises(TorchStackInstallError) as error:
        install_torch_stack(commands[2:], runner=runner)

    assert str(error.value) == "torch stack installation failed: runtime"
    assert secret not in str(error.value)


def test_cli_uses_platform_and_runner_seams(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import install_torch_stack as module

    seen: list[tuple[tuple[str, ...], bool]] = []
    monkeypatch.setattr(module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(module.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(
        module,
        "install_torch_stack",
        lambda commands: seen.extend((command.argv, False) for command in commands),
    )

    assert module.main() == 0
    assert seen == [(UPGRADE_PIP, False), (DARWIN_CORE, False), (RUNTIME, False), (ROOT, False)]
