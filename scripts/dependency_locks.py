"""Strict, offline models for the repository's dependency lock contracts."""

from __future__ import annotations

import json
import re
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from packaging.markers import InvalidMarker, Marker
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version


class DependencyLockError(ValueError):
    """A stable fail-closed dependency-lock contract error."""


@dataclass(frozen=True)
class PlatformPolicy:
    key: str
    uv_platform: str
    system: str
    machine: str
    torch_index: str
    pyg_find_links: str
    marker_environment: Mapping[str, str]


@dataclass(frozen=True)
class TargetEnvironment:
    key: str
    marker_environment: Mapping[str, str]


@dataclass(frozen=True)
class LockPolicy:
    schema: int
    compiler_input: Path
    python_floor: Version
    macos_deployment_target: str
    exclude_newer: str
    platforms: tuple[PlatformPolicy, ...]
    inputs: tuple[Path, ...]
    outputs: tuple[Path, ...]


@dataclass(frozen=True)
class LockedRequirement:
    name: str
    version: Version
    hashes: tuple[str, ...]
    source: str | None
    marker: str | None


@dataclass(frozen=True)
class ImageRecord:
    consumer: str
    source: str
    index_digest: str
    children: Mapping[str, str]


_EXPECTED_INPUTS = (
    "requirements.txt",
    "bootstrap-requirements.txt",
    "compiler-requirements.txt",
    "nlp-model-requirements.txt",
    "torch-core-requirements.txt",
    "torch-ecosystem-requirements.txt",
    "torch-requirements.txt",
    "torch-audit-requirements.txt",
    "pyg-extension-audit-requirements.txt",
    "vulnerability-audit-requirements.txt",
    "atlas-contract-requirements.txt",
    "docs-requirements.in",
)
_EXPECTED_OUTPUTS = (
    "requirements/locks/bootstrap.txt",
    "requirements/locks/compiler.txt",
    "requirements/locks/audit.txt",
    "requirements/locks/atlas-contract.txt",
    "requirements/locks/darwin-arm64/core.txt",
    "requirements/locks/darwin-arm64/runtime.txt",
    "requirements/locks/darwin-arm64/root.txt",
    "requirements/locks/linux-x86_64/core.txt",
    "requirements/locks/linux-x86_64/runtime.txt",
    "requirements/locks/linux-x86_64/root.txt",
    "requirements/locks/linux-aarch64/core.txt",
    "requirements/locks/linux-aarch64/runtime.txt",
    "requirements/locks/linux-aarch64/root.txt",
    "docs-requirements.txt",
)
_EXPECTED_BOOTSTRAP = (
    "pip==26.1.2",
    "setuptools==81.0.0",
    "wheel==0.47.0",
    "packaging==26.2",
)
_MODEL_URL = (
    "https://github.com/explosion/spacy-models/releases/download/"
    "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
)
_MODEL_HASH = "1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85"
_EXPECTED_MARKER_KEYS = (
    "implementation_name",
    "os_name",
    "platform_machine",
    "platform_python_implementation",
    "platform_system",
    "python_version",
    "sys_platform",
)
_ALLOWED_MARKER_VARIABLES = frozenset(_EXPECTED_MARKER_KEYS)
_CANDIDATE_MARKER_VARIABLES = _ALLOWED_MARKER_VARIABLES | {
    "extra",
    "implementation_version",
    "platform_release",
    "platform_version",
    "python_full_version",
}
_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
_NAME_VERSION_RE = re.compile(
    r"(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)==(?P<version>[^\s;\\]+)"
    r"(?:\s*;\s*(?P<marker>.*?))?(?=\s+--hash=|\s*\Z)"
)
_DIRECT_REQUIREMENT_RE = re.compile(
    r"(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)\s+@\s+(?P<source>https://[^\s;\\]+)"
    r"(?:\s*;\s*(?P<marker>.*?))?(?=\s+--hash=|\s*\Z)"
)
_IMAGE_SOURCE_RE = re.compile(r"[a-z0-9.-]+(?:/[A-Za-z0-9._-]+)+:[A-Za-z0-9._-]+\Z")


def _error(detail: str) -> DependencyLockError:
    return DependencyLockError(detail)


def _require_exact_keys(document: Mapping[str, Any], expected: tuple[str, ...], label: str) -> None:
    if tuple(document) != expected:
        raise _error(f"{label}: unknown, missing, or reordered fields")


def _strict_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _error(f"image ledger: duplicate JSON key {key}")
        result[key] = value
    return result


def _read_lines(path: Path) -> tuple[str, ...]:
    try:
        return tuple(path.read_text(encoding="utf-8").splitlines())
    except OSError as exc:
        raise _error(f"manifest missing or unreadable: {path.name}") from exc


def parse_compiler_identity(repo: Path) -> Version:
    lines = _read_lines(repo / "compiler-requirements.txt")
    if len(lines) != 1 or not lines[0].startswith("uv=="):
        raise _error("compiler manifest must contain exactly one uv pin")
    try:
        return Version(lines[0].removeprefix("uv=="))
    except InvalidVersion as exc:
        raise _error("compiler manifest has an invalid uv version") from exc


def _parse_marker(text: str, *, candidate: bool = False) -> str:
    try:
        marker = Marker(text)
    except InvalidMarker as exc:
        raise _error("lock marker has invalid syntax") from exc
    scrubbed = re.sub(r"(['\"]).*?\1", "", text)
    variables = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", scrubbed))
    variables -= {"and", "or", "in", "not"}
    allowed = _CANDIDATE_MARKER_VARIABLES if candidate else _ALLOWED_MARKER_VARIABLES
    unsupported = variables - allowed
    if unsupported:
        raise _error(f"lock marker uses unsupported variable: {sorted(unsupported)[0]}")
    return str(marker)


def _logical_requirement_lines(path: Path) -> tuple[str, ...]:
    logical: list[str] = []
    current = ""
    for raw in _read_lines(path):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith(("-r ", "--requirement ")):
            continue
        if stripped.startswith("--") and not current:
            raise _error(f"lock contains unsupported global option: {path.name}")
        continuation = stripped.endswith("\\")
        part = stripped[:-1].rstrip() if continuation else stripped
        current = f"{current} {part}".strip()
        if not continuation:
            logical.append(current)
            current = ""
    if current:
        raise _error(f"lock contains an unterminated continuation: {path.name}")
    return tuple(logical)


def _parse_lock_requirements(path: Path, *, candidate: bool) -> tuple[LockedRequirement, ...]:
    requirements: list[LockedRequirement] = []
    for line in _logical_requirement_lines(path):
        match = _NAME_VERSION_RE.match(line)
        direct = False
        if match is None:
            match = _DIRECT_REQUIREMENT_RE.match(line)
            direct = match is not None
        if match is None:
            raise _error(f"lock requirement has unsupported syntax: {path.name}")
        remainder = line[match.end() :]
        hashes = tuple(re.findall(r"--hash=sha256:([0-9a-f]{64})(?=\s|\Z)", remainder))
        if not hashes or remainder.strip() != " ".join(f"--hash=sha256:{value}" for value in hashes):
            raise _error(f"lock requirement has malformed or missing hash: {path.name}")
        if len(set(hashes)) != len(hashes):
            raise _error(f"lock requirement has a duplicate hash: {path.name}")
        if direct:
            source = match.group("source")
            version_match = re.search(r"-([0-9]+(?:\.[0-9]+)+)-[^/]+\.whl\Z", source)
            if version_match is None:
                raise _error(f"lock direct requirement has no wheel version: {path.name}")
            version = Version(version_match.group(1))
        else:
            source = None
            try:
                version = Version(match.group("version"))
            except InvalidVersion as exc:
                raise _error(f"lock requirement has invalid version: {path.name}") from exc
        marker_text = match.group("marker")
        marker = _parse_marker(marker_text, candidate=candidate) if marker_text else None
        requirements.append(
            LockedRequirement(
                name=canonicalize_name(match.group("name")),
                version=version,
                hashes=hashes,
                source=source,
                marker=marker,
            )
        )
    return tuple(requirements)


def parse_candidate_lock(path: Path) -> tuple[LockedRequirement, ...]:
    """Parse a resolver candidate before supported-range marker projection."""
    return _parse_lock_requirements(path, candidate=True)


def parse_lock(path: Path) -> tuple[LockedRequirement, ...]:
    requirements = _parse_lock_requirements(path, candidate=False)
    targets = _supported_target_environments()
    selected_names: dict[tuple[str, str], int] = {}
    for requirement in requirements:
        applicable = tuple(
            target
            for target in targets
            if requirement.marker is None
            or Marker(requirement.marker).evaluate(environment=dict(target.marker_environment))
        )
        if requirement.marker is not None and not applicable:
            raise _error(f"lock marker applies to no supported target: {requirement.name}")
        for target in applicable:
            binding = (requirement.name, target.key)
            selected_names[binding] = selected_names.get(binding, 0) + 1
            if selected_names[binding] > 1:
                raise _error(f"lock marker overlap for {requirement.name} on {target.key}")
    return requirements


def _supported_target_environments() -> tuple[TargetEnvironment, ...]:
    base = {
        "implementation_name": "cpython",
        "implementation_version": "3.11.0",
        "os_name": "posix",
        "platform_python_implementation": "CPython",
        "python_full_version": "3.11.0",
        "python_version": "3.11",
        "extra": "",
    }
    targets = (
        ("darwin-arm64", "arm64", "Darwin", "darwin"),
        ("linux-x86_64", "x86_64", "Linux", "linux"),
        ("linux-aarch64", "aarch64", "Linux", "linux"),
    )
    return tuple(
        TargetEnvironment(
            key=key,
            marker_environment={
                **base,
                "platform_machine": machine,
                "platform_system": system,
                "sys_platform": sys_platform,
            },
        )
        for key, machine, system, sys_platform in targets
    )


def project_for_target(
    requirements: Sequence[LockedRequirement], target: TargetEnvironment
) -> tuple[LockedRequirement, ...]:
    selected: list[LockedRequirement] = []
    names: set[str] = set()
    for requirement in requirements:
        applies = requirement.marker is None or Marker(requirement.marker).evaluate(
            environment=dict(target.marker_environment)
        )
        if not applies:
            continue
        if requirement.name in names:
            raise _error(f"lock marker overlap for {requirement.name} on {target.key}")
        names.add(requirement.name)
        selected.append(requirement)
    return tuple(selected)


def exact_version_projection(
    requirements: Sequence[LockedRequirement], target: TargetEnvironment
) -> str:
    return "".join(f"{item.name}=={item.version}\n" for item in project_for_target(requirements, target))


def _validate_direct_manifests(repo: Path, compiler_version: Version) -> None:
    if _read_lines(repo / "bootstrap-requirements.txt") != _EXPECTED_BOOTSTRAP:
        raise _error("bootstrap manifest does not match the exact locked toolchain")
    if _read_lines(repo / "atlas-contract-requirements.txt") != (
        "nltk==3.10.3",
        "pytest==9.0.3",
        "pyyaml==6.0.3",
        f"uv=={compiler_version}",
    ):
        raise _error("Atlas compiler uv pin does not match compiler manifest")
    model = _read_lines(repo / "nlp-model-requirements.txt")
    expected_model = f"en-core-web-sm @ {_MODEL_URL} --hash=sha256:{_MODEL_HASH}"
    if model != (expected_model,):
        raise _error("NLP model direct URL or hash is not the approved identity")
    torch_runtime = _read_lines(repo / "torch-requirements.txt")
    if any(line.startswith(("--find-links", "--index-url", "--extra-index-url")) for line in torch_runtime):
        raise _error("Torch runtime manifest contains an unsafe source directive")


def _validate_requirement_inputs(repo: Path, inputs: Sequence[Path]) -> None:
    approved_includes = {
        "torch-requirements.txt": ("-r torch-ecosystem-requirements.txt",),
        "torch-audit-requirements.txt": (
            "-r torch-core-requirements.txt",
            "-r torch-ecosystem-requirements.txt",
        ),
    }
    expected_model = f"en-core-web-sm @ {_MODEL_URL} --hash=sha256:{_MODEL_HASH}"
    for relative in inputs:
        for raw in _read_lines(repo / relative):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            line = line.split("  #", 1)[0].rstrip()
            if any(token in line for token in ("${", "$(", "`")):
                raise _error(f"unsafe variable interpolation in {relative.as_posix()}")
            if line.startswith(("-r ", "--requirement ")):
                if line not in approved_includes.get(relative.as_posix(), ()):
                    raise _error(f"unsafe requirement include in {relative.as_posix()}")
                continue
            if line.startswith("-"):
                raise _error(f"unsafe source or editable directive in {relative.as_posix()}")
            if relative.as_posix() == "nlp-model-requirements.txt" and line == expected_model:
                continue
            if " @ " in line:
                raise _error(f"unapproved direct URL in {relative.as_posix()}")
            if "git+" in line or "file:" in line or line.startswith((".", "/", "~")):
                raise _error(f"unsafe local or VCS requirement in {relative.as_posix()}")
            try:
                Requirement(line)
            except InvalidRequirement as exc:
                raise _error(f"unsafe requirement syntax in {relative.as_posix()}") from exc


def load_policy(repo: Path) -> LockPolicy:
    path = repo / "requirements" / "lock-policy.toml"
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise _error("lock policy is missing, duplicate, or invalid TOML") from exc
    expected_keys = (
        "schema_version",
        "compiler_input",
        "python_floor",
        "macos_deployment_target",
        "exclude_newer",
        "inputs",
        "outputs",
        "direct_url_packages",
        "sdist_packages",
        "sources",
        "package_sources",
        "platforms",
    )
    _require_exact_keys(raw, expected_keys, "lock policy")
    if isinstance(raw["schema_version"], bool) or raw["schema_version"] != 1:
        raise _error("lock policy schema must be integer 1")
    if raw["compiler_input"] != "compiler-requirements.txt":
        raise _error("lock policy compiler input is invalid")
    try:
        python_floor = Version(raw["python_floor"])
    except (InvalidVersion, TypeError) as exc:
        raise _error("lock policy Python floor is invalid") from exc
    if python_floor != Version("3.11.0") or raw["macos_deployment_target"] != "13.0":
        raise _error("lock policy interpreter or deployment floor is invalid")
    if raw["exclude_newer"] != "2026-08-17T02:21:18Z":
        raise _error("lock policy resolver cutoff is invalid")
    if tuple(raw["inputs"]) != _EXPECTED_INPUTS or tuple(raw["outputs"]) != _EXPECTED_OUTPUTS:
        raise _error("lock policy has an unknown or missing input/output inventory")
    if raw["direct_url_packages"] != ["en-core-web-sm"]:
        raise _error("lock policy direct URL exception is invalid")
    if raw["sdist_packages"] != ["python-louvain"]:
        raise _error("lock policy sdist exception is invalid")
    expected_sources = {
        "pypi": "https://pypi.org/simple",
        "torch_cpu": "https://download.pytorch.org/whl/cpu",
        "pyg": "https://data.pyg.org/whl/torch-2.11.0+cpu.html",
        "nlp_model": _MODEL_URL,
    }
    if raw["sources"] != expected_sources:
        raise _error("lock policy source binding is invalid")
    expected_package_sources = {
        "default": "pypi",
        "pyg-lib": "pyg",
        "torch-scatter": "pyg",
        "torch-sparse": "pyg",
        "en-core-web-sm": "nlp_model",
        "torch": {
            "darwin-arm64": "pypi",
            "linux-x86_64": "torch_cpu",
            "linux-aarch64": "torch_cpu",
        },
        "torchao": {
            "darwin-arm64": "pypi",
            "linux-x86_64": "torch_cpu",
            "linux-aarch64": "torch_cpu",
        },
    }
    if raw["package_sources"] != expected_package_sources:
        raise _error("lock policy package source binding is invalid")
    inputs = tuple(Path(value) for value in raw["inputs"])
    for relative in inputs:
        if relative.is_absolute() or ".." in relative.parts or not (repo / relative).is_file():
            raise _error(f"lock policy input is missing or unsafe: {relative.as_posix()}")
    compiler_version = parse_compiler_identity(repo)
    _validate_direct_manifests(repo, compiler_version)
    _validate_requirement_inputs(repo, inputs)
    platforms: list[PlatformPolicy] = []
    for record in raw["platforms"]:
        _require_exact_keys(
            record,
            (
                "key",
                "uv_platform",
                "system",
                "machine",
                "torch_index",
                "pyg_find_links",
                "marker_environment",
            ),
            "lock policy platform",
        )
        environment = record["marker_environment"]
        _require_exact_keys(environment, _EXPECTED_MARKER_KEYS, "platform marker environment")
        platforms.append(
            PlatformPolicy(
                key=record["key"],
                uv_platform=record["uv_platform"],
                system=record["system"],
                machine=record["machine"],
                torch_index=record["torch_index"],
                pyg_find_links=record["pyg_find_links"],
                marker_environment=MappingProxyType(dict(environment)),
            )
        )
    expected_platforms = (
        ("darwin-arm64", "aarch64-apple-darwin", "Darwin", "arm64", "https://pypi.org/simple"),
        (
            "linux-x86_64",
            "x86_64-manylinux_2_28",
            "Linux",
            "x86_64",
            "https://download.pytorch.org/whl/cpu",
        ),
        (
            "linux-aarch64",
            "aarch64-manylinux_2_28",
            "Linux",
            "aarch64",
            "https://download.pytorch.org/whl/cpu",
        ),
    )
    if tuple((p.key, p.uv_platform, p.system, p.machine, p.torch_index) for p in platforms) != expected_platforms:
        raise _error("lock policy supported platform matrix is invalid")
    expected_environments = (
        {
            "implementation_name": "cpython",
            "os_name": "posix",
            "platform_machine": "arm64",
            "platform_python_implementation": "CPython",
            "platform_system": "Darwin",
            "python_version": "3.11",
            "sys_platform": "darwin",
        },
        {
            "implementation_name": "cpython",
            "os_name": "posix",
            "platform_machine": "x86_64",
            "platform_python_implementation": "CPython",
            "platform_system": "Linux",
            "python_version": "3.11",
            "sys_platform": "linux",
        },
        {
            "implementation_name": "cpython",
            "os_name": "posix",
            "platform_machine": "aarch64",
            "platform_python_implementation": "CPython",
            "platform_system": "Linux",
            "python_version": "3.11",
            "sys_platform": "linux",
        },
    )
    if tuple(dict(platform.marker_environment) for platform in platforms) != expected_environments:
        raise _error("lock policy platform marker environment is invalid")
    for platform in platforms:
        if platform.pyg_find_links != expected_sources["pyg"]:
            raise _error("lock policy PyG source binding is invalid")
    load_image_ledger(repo)
    return LockPolicy(
        schema=1,
        compiler_input=Path(raw["compiler_input"]),
        python_floor=python_floor,
        macos_deployment_target=raw["macos_deployment_target"],
        exclude_newer=raw["exclude_newer"],
        platforms=tuple(platforms),
        inputs=inputs,
        outputs=tuple(Path(value) for value in raw["outputs"]),
    )


def load_image_ledger(repo: Path) -> tuple[ImageRecord, ...]:
    path = repo / "requirements" / "image-lock.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_strict_json_pairs)
    except DependencyLockError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise _error("image ledger is missing or invalid JSON") from exc
    if not isinstance(raw, dict):
        raise _error("image ledger root must be an object")
    _require_exact_keys(raw, ("schema_version", "images"), "image ledger")
    if isinstance(raw["schema_version"], bool) or raw["schema_version"] != 1:
        raise _error("image ledger schema must be integer 1")
    if not isinstance(raw["images"], list) or len(raw["images"]) != 2:
        raise _error("image ledger must contain exactly two images")
    images: list[ImageRecord] = []
    for record in raw["images"]:
        if not isinstance(record, dict):
            raise _error("image ledger entry must be an object")
        _require_exact_keys(record, ("consumer", "source", "index_digest", "children"), "image")
        source = record["source"]
        index_digest = record["index_digest"]
        children = record["children"]
        if not isinstance(source, str) or _IMAGE_SOURCE_RE.fullmatch(source) is None:
            raise _error("image source must be a tag-only reference")
        if not isinstance(index_digest, str) or _DIGEST_RE.fullmatch(index_digest) is None:
            raise _error("image index digest is malformed")
        if not isinstance(children, dict) or tuple(children) != ("linux/amd64", "linux/arm64"):
            raise _error("image ledger has a missing or reordered architecture")
        for digest in children.values():
            if not isinstance(digest, str) or _DIGEST_RE.fullmatch(digest) is None:
                raise _error("image child digest is malformed")
            if digest == index_digest:
                raise _error("image child digest substitutes the index digest")
        images.append(
            ImageRecord(
                consumer=record["consumer"],
                source=source,
                index_digest=index_digest,
                children=MappingProxyType(dict(children)),
            )
        )
    if tuple(image.consumer for image in images) != ("docker", "devcontainer"):
        raise _error("image ledger consumer order or identity is invalid")
    if len({image.source for image in images}) != len(images):
        raise _error("image ledger contains a duplicate source")
    return tuple(images)
