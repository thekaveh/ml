"""Compile the repository's deterministic, platform-aware dependency lock family."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from scripts.dependency_locks import (
    LockPolicy,
    LockedRequirement,
    PlatformPolicy,
    TargetEnvironment,
    load_policy,
    parse_candidate_lock,
    parse_compiler_identity,
)


Mode = Literal["write", "check", "update-compiler"]


class LockGenerationError(RuntimeError):
    """A stable, redacted dependency-lock generation failure."""


@dataclass(frozen=True)
class CompileCommand:
    target: str
    surface: str
    output: Path
    argv: tuple[str, ...]
    env: Mapping[str, str]


Runner = Callable[[CompileCommand], Any]


_SURFACES = (
    "bootstrap",
    "compiler",
    "docs",
    "audit",
    "atlas-contract",
    "core",
    "runtime",
    "root",
)
_MATRIX_OUTPUTS = {
    "bootstrap": Path("requirements/locks/bootstrap.txt"),
    "compiler": Path("requirements/locks/compiler.txt"),
    "docs": Path("docs-requirements.txt"),
    "audit": Path("requirements/locks/audit.txt"),
    "atlas-contract": Path("requirements/locks/atlas-contract.txt"),
}


def _installed_uv_version() -> str:
    try:
        return importlib.metadata.version("uv")
    except importlib.metadata.PackageNotFoundError as exc:
        raise LockGenerationError("dependency lock generation failed: compiler: missing") from exc


def _validate_compiler_update_transition(
    *,
    old: str,
    manifest: str,
    installed: str,
    atlas: str,
    first: Mapping[Path, bytes],
    second: Mapping[Path, bytes],
) -> None:
    if (
        not old
        or old == manifest
        or installed != manifest
        or atlas != manifest
        or not first
        or first != second
    ):
        raise LockGenerationError("dependency lock generation failed: compiler-update: state")


def _surface_inputs(surface: str) -> tuple[str, ...]:
    return {
        "bootstrap": ("bootstrap-requirements.txt",),
        "compiler": ("compiler-requirements.txt",),
        "docs": ("docs-requirements.in",),
        "audit": ("vulnerability-audit-requirements.txt",),
        "atlas-contract": ("atlas-contract-requirements.txt",),
        "core": ("torch-core-requirements.txt",),
        "runtime": ("torch-core-requirements.txt", "torch-requirements.txt"),
        "root": (
            "bootstrap-requirements.txt",
            "torch-core-requirements.txt",
            "torch-requirements.txt",
            "requirements.txt",
            "docs-requirements.in",
            "nlp-model-requirements.txt",
        ),
    }[surface]


def _candidate_path(root: Path, target: str, surface: str) -> Path:
    return root / target / f"{surface}.txt"


def _build_plan(repo: Path, policy: LockPolicy, candidate_root: Path) -> tuple[CompileCommand, ...]:
    commands: list[CompileCommand] = []
    for platform in policy.platforms:
        for surface in _SURFACES:
            output = _candidate_path(candidate_root, platform.key, surface)
            argv: list[str] = [
                "-m",
                "uv",
                "pip",
                "compile",
                *_surface_inputs(surface),
                "--output-file",
                str(output),
                "--generate-hashes",
                "--no-annotate",
                "--no-header",
                "--no-strip-markers",
                "--python-version",
                str(policy.python_floor),
                "--python-platform",
                platform.uv_platform,
                "--default-index",
                "https://pypi.org/simple",
                "--only-binary",
                ":all:",
                "--no-progress",
                "--refresh",
            ]
            if surface in {"runtime", "root"}:
                argv.extend(("--find-links", platform.pyg_find_links))
            if platform.system == "Linux" and surface in {"core", "runtime", "root"}:
                argv.extend(("--torch-backend", "cpu"))
            if surface == "root":
                argv.extend(
                    (
                        "--no-binary",
                        "python-louvain",
                        "--no-build-isolation-package",
                        "python-louvain",
                    )
                )
            environment = {
                "PIP_CONFIG_FILE": os.devnull,
                "UV_NO_CONFIG": "1",
                "UV_PYTHON_DOWNLOADS": "never",
                "UV_INDEX_STRATEGY": "first-index",
            }
            if platform.system == "Darwin":
                environment["MACOSX_DEPLOYMENT_TARGET"] = policy.macos_deployment_target
            commands.append(
                CompileCommand(
                    target=platform.key,
                    surface=surface,
                    output=output,
                    argv=tuple(argv),
                    env=environment,
                )
            )
    return tuple(commands)


def build_compile_plan(repo: Path, policy: LockPolicy, mode: Mode) -> tuple[CompileCommand, ...]:
    if mode not in {"write", "check", "update-compiler"}:
        raise LockGenerationError("dependency lock generation failed: mode")
    return _build_plan(repo, policy, repo / ".superpowers/sdd/issue63-lock-candidates")


def _run_command(repo: Path, command: CompileCommand) -> subprocess.CompletedProcess[str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("PIP_", "UV_")) and key not in {"PYTHONPATH", "PYTHONHOME"}
    }
    environment.update(command.env)
    return subprocess.run(
        (sys.executable, *command.argv),
        cwd=repo,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _header(repo: Path, policy: LockPolicy, target: str) -> str:
    policy_path = repo / "requirements/lock-policy.toml"
    lines = [
        "# Generated by python -m scripts.lock_dependencies; do not edit.",
        f"# policy-sha256: {_sha256(policy_path)}",
        f"# target: {target}",
    ]
    for relative in policy.inputs:
        lines.append(f"# input-sha256: {relative.as_posix()}={_sha256(repo / relative)}")
    return "\n".join(lines) + "\n"


def _target_marker(targets: tuple[str, ...]) -> str | None:
    all_targets = ("darwin-arm64", "linux-x86_64", "linux-aarch64")
    if targets == all_targets:
        return None
    terms = {
        "darwin-arm64": 'sys_platform == "darwin" and platform_machine == "arm64"',
        "linux-x86_64": 'sys_platform == "linux" and platform_machine == "x86_64"',
        "linux-aarch64": 'sys_platform == "linux" and platform_machine == "aarch64"',
    }
    if targets == ("linux-x86_64", "linux-aarch64"):
        return 'sys_platform == "linux"'
    return " or ".join(f"({terms[target]})" for target in targets)


def _format_requirement(requirement: LockedRequirement) -> str:
    marker = f" ; {requirement.marker}" if requirement.marker else ""
    hashes = " \\\n    ".join(f"--hash=sha256:{value}" for value in requirement.hashes)
    identity = (
        f"{requirement.name} @ {requirement.source}"
        if requirement.source is not None
        else f"{requirement.name}=={requirement.version}"
    )
    return f"{identity}{marker} \\\n    {hashes}\n"


def _merge_matrix_locks(
    candidate_root: Path, platforms: Sequence[PlatformPolicy], surface: str
) -> str:
    by_identity: dict[tuple[str, str], dict[str, LockedRequirement]] = defaultdict(dict)
    target_order = tuple(platform.key for platform in platforms)
    for platform in platforms:
        for requirement in parse_candidate_lock(_candidate_path(candidate_root, platform.key, surface)):
            decisions = tuple(
                _candidate_applies(requirement, platform, patch)
                for patch in ("3.11.0", "3.11.10", "3.11.15")
            )
            if len(set(decisions)) != 1:
                raise LockGenerationError(
                    f"dependency lock generation failed: patch-sensitive marker: {requirement.name}"
                )
            if not decisions[0]:
                continue
            by_identity[(requirement.name, str(requirement.version))][platform.key] = requirement
    by_name: dict[str, set[str]] = defaultdict(set)
    for name, version in by_identity:
        by_name[name].add(version)
    rendered: list[LockedRequirement] = []
    for (name, version), target_records in sorted(by_identity.items()):
        targets = tuple(target for target in target_order if target in target_records)
        hashes = tuple(sorted({value for item in target_records.values() for value in item.hashes}))
        rendered.append(
            LockedRequirement(
                name=name,
                version=next(iter(target_records.values())).version,
                hashes=hashes,
                source=None,
                marker=_target_marker(targets) if len(by_name[name]) > 1 or targets != target_order else None,
            )
        )
    return "".join(_format_requirement(requirement) for requirement in rendered)


def _candidate_applies(
    requirement: LockedRequirement, platform: PlatformPolicy, patch: str
) -> bool:
    if requirement.marker is None:
        return True
    from packaging.markers import Marker

    environment = {
        **dict(platform.marker_environment),
        "implementation_version": patch,
        "python_full_version": patch,
        "extra": "",
        "platform_release": "",
        "platform_version": "",
    }
    target = TargetEnvironment(key=platform.key, marker_environment=environment)
    return Marker(requirement.marker).evaluate(environment=dict(target.marker_environment))


def _normalize_candidate(path: Path, platform: PlatformPolicy) -> str:
    rendered: list[LockedRequirement] = []
    for requirement in parse_candidate_lock(path):
        decisions = tuple(
            _candidate_applies(requirement, platform, patch)
            for patch in ("3.11.0", "3.11.10", "3.11.15")
        )
        if len(set(decisions)) != 1:
            raise LockGenerationError(
                f"dependency lock generation failed: patch-sensitive marker: {requirement.name}"
            )
        if decisions[0]:
            rendered.append(
                LockedRequirement(
                    name=requirement.name,
                    version=requirement.version,
                    hashes=requirement.hashes,
                    source=requirement.source,
                    marker=None,
                )
            )
    return "".join(_format_requirement(requirement) for requirement in rendered)


def _assemble_outputs(
    repo: Path, policy: LockPolicy, candidate_root: Path
) -> dict[Path, bytes]:
    assembled: dict[Path, bytes] = {}
    for surface, relative in _MATRIX_OUTPUTS.items():
        body = _merge_matrix_locks(candidate_root, policy.platforms, surface)
        assembled[relative] = (_header(repo, policy, "supported-matrix") + body).encode()
    for platform in policy.platforms:
        for surface in ("core", "runtime", "root"):
            relative = Path("requirements/locks") / platform.key / f"{surface}.txt"
            body = _normalize_candidate(
                _candidate_path(candidate_root, platform.key, surface), platform
            )
            assembled[relative] = (_header(repo, policy, platform.key) + body).encode()
    if set(assembled) != set(policy.outputs):
        raise LockGenerationError("dependency lock generation failed: output inventory")
    return {relative: assembled[relative] for relative in policy.outputs}


def _transactional_write(repo: Path, outputs: Mapping[Path, bytes]) -> None:
    backups: dict[Path, bytes | None] = {}
    try:
        for relative, content in outputs.items():
            path = repo / relative
            backups[path] = path.read_bytes() if path.exists() else None
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.issue63.tmp")
            temporary.write_bytes(content)
            temporary.replace(path)
    except OSError as exc:
        for path, previous in backups.items():
            if previous is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(previous)
        raise LockGenerationError("dependency lock generation failed: transaction") from exc


def _compile_once(
    repo: Path, policy: LockPolicy, execute: Runner, *, prefix: str
) -> dict[Path, bytes]:
    with tempfile.TemporaryDirectory(prefix=prefix) as temporary:
        candidate_root = Path(temporary)
        for command in _build_plan(repo, policy, candidate_root):
            command.output.parent.mkdir(parents=True, exist_ok=True)
            try:
                result = execute(command)
            except (OSError, UnicodeError) as exc:
                raise LockGenerationError(
                    f"dependency lock generation failed: compile: {command.surface}"
                ) from exc
            if getattr(result, "returncode", 1) != 0 or not command.output.is_file():
                raise LockGenerationError(
                    f"dependency lock generation failed: compile: {command.surface}"
                )
        return _assemble_outputs(repo, policy, candidate_root)


def _atlas_uv_identity(repo: Path) -> str:
    lines = (repo / "atlas-contract-requirements.txt").read_text(encoding="utf-8").splitlines()
    pins = [line.removeprefix("uv==") for line in lines if line.startswith("uv==")]
    return pins[0] if len(pins) == 1 else ""


def _committed_compiler_identity(repo: Path) -> str:
    path = repo / "requirements/locks/compiler.txt"
    if not path.is_file():
        return ""
    try:
        requirements = parse_candidate_lock(path)
    except Exception:
        return ""
    matches = [str(item.version) for item in requirements if item.name == "uv"]
    return matches[0] if len(matches) == 1 else ""


def compile_locks(repo: Path, mode: Mode, runner: Runner | None = None) -> None:
    policy = load_policy(repo)
    if mode not in {"write", "check", "update-compiler"}:
        raise LockGenerationError("dependency lock generation failed: mode")
    installed = _installed_uv_version()
    manifest = str(parse_compiler_identity(repo))
    if installed != manifest:
        raise LockGenerationError("dependency lock generation failed: compiler: identity")
    execute = runner or (lambda command: _run_command(repo, command))
    outputs = _compile_once(repo, policy, execute, prefix="issue63-locks-")
    if mode == "update-compiler":
        second = _compile_once(repo, policy, execute, prefix="issue63-locks-second-")
        _validate_compiler_update_transition(
            old=_committed_compiler_identity(repo),
            manifest=manifest,
            installed=installed,
            atlas=_atlas_uv_identity(repo),
            first=outputs,
            second=second,
        )
    if mode in {"write", "update-compiler"}:
        _transactional_write(repo, outputs)
        return
    mismatched = [
        relative
        for relative, expected in outputs.items()
        if not (repo / relative).is_file() or (repo / relative).read_bytes() != expected
    ]
    if mismatched:
        raise LockGenerationError("dependency lock generation failed: check: stale")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("write", "check", "update-compiler"))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    arguments = parser.parse_args(argv)
    try:
        compile_locks(arguments.repo_root.resolve(), arguments.mode)
    except LockGenerationError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
