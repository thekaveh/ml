"""Compare live pip-audit observations with the reviewed advisory baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from packaging.requirements import InvalidRequirement, Requirement
from packaging.version import InvalidVersion, Version

from scripts.dependency_locks import (
    DependencyLockError,
    LockedRequirement,
    TargetEnvironment,
    load_policy,
    parse_lock,
    project_for_target,
)
from scripts.verify_dependency_locks import verify_dependency_locks


SCHEMA_VERSION = 1
PYG_EXTENSION_AUDIT_VERSIONS = (
    ("torch-scatter", "2.1.2"),
    ("torch-sparse", "0.6.18"),
)
SURFACE_ORDER = (
    "combined-runtime",
    "torch",
    "documentation",
    "atlas-contract",
)
AUDIT_FAILURE_CATEGORIES = frozenset(
    {
        "unexpected-exit",
        "execution-error",
        "missing-output",
        "bootstrap-error",
        "resolution-error",
        "service-error",
        "unsupported-package",
        "unavailable-output",
        "invalid-json",
        "invalid-schema",
    }
)


class AdvisoryBaselineError(RuntimeError):
    """The advisory policy or an audit observation is invalid."""


class AuditSurfaceError(AdvisoryBaselineError):
    """A safe, fixed-category failure for one audit surface."""

    def __init__(self, surface: str, category: str) -> None:
        if surface not in SURFACE_ORDER or category not in AUDIT_FAILURE_CATEGORIES:
            raise ValueError("invalid audit failure category")
        self.surface = surface
        self.category = category
        super().__init__(f"{surface}: {category}")


@dataclass(frozen=True, order=True)
class AcceptedAdvisory:
    package: str
    advisory_id: str
    accepted_version: str
    surfaces: tuple[str, ...]


@dataclass(frozen=True)
class Baseline:
    schema_version: int
    audited_surfaces: tuple[str, ...]
    accepted_advisories: tuple[AcceptedAdvisory, ...]


@dataclass(frozen=True, order=True)
class LockInput:
    path: str
    sha256: str


@dataclass(frozen=True, order=True)
class NonPyPIRecord:
    package: str
    version: str
    public_version: str
    source: str
    hashes: tuple[str, ...]
    targets: tuple[str, ...]
    audited: bool = False
    reason: str = "non-pypi"


@dataclass(frozen=True)
class Observation:
    surface: str
    resolved_versions: tuple[tuple[str, str], ...]
    advisories: tuple[tuple[str, str, str], ...]
    lock_inputs: tuple[LockInput, ...] = ()
    non_pypi: tuple[NonPyPIRecord, ...] = ()


@dataclass(frozen=True)
class Comparison:
    errors: tuple[str, ...]
    notices: tuple[str, ...]


@dataclass(frozen=True)
class AuditSurface:
    name: str
    projection_kind: str
    output_name: str


@dataclass(frozen=True)
class PreparedAuditSurface:
    surface: AuditSurface
    projection: Path
    expected_versions: tuple[tuple[str, str], ...]
    lock_inputs: tuple[LockInput, ...]
    non_pypi: tuple[NonPyPIRecord, ...]


AUDIT_SURFACES = (
    AuditSurface("combined-runtime", "main", "combined-runtime-resolver"),
    AuditSurface("combined-runtime", "pyg-extensions", "combined-runtime-pyg-extensions"),
    AuditSurface("torch", "main", "torch-resolver"),
    AuditSurface("torch", "pyg-extensions", "torch-pyg-extensions"),
    AuditSurface("documentation", "main", "documentation"),
    AuditSurface("atlas-contract", "main", "atlas-contract"),
)
AuditRunner = Callable[..., subprocess.CompletedProcess[str]]

_BOOTSTRAP_LOCK = Path("requirements/locks/bootstrap.txt")
_AUDIT_LOCK = Path("requirements/locks/audit.txt")
_ATLAS_LOCK = Path("requirements/locks/atlas-contract.txt")
_DOCS_LOCK = Path("docs-requirements.txt")
_NON_PYPI_PACKAGES = frozenset({"en-core-web-sm", "pyg-lib"})
_PYG_AUDITED_PACKAGES = frozenset({"torch-scatter", "torch-sparse"})


def normalize_package_name(value: str) -> str:
    """Return the PEP 503 form used for advisory identities."""
    return re.sub(r"[-_.]+", "-", value).lower()


def advisory_identity(item: AcceptedAdvisory) -> tuple[str, str, str]:
    """Return the canonical accepted-advisory identity."""
    return (item.package, item.advisory_id, item.accepted_version)


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    document: dict[str, object] = {}
    for key, value in pairs:
        if key in document:
            raise AdvisoryBaselineError("duplicate JSON key")
        document[key] = value
    return document


def _require_exact_keys(
    value: object,
    expected: tuple[str, ...],
    *,
    item: bool = False,
    canonical_order: bool = True,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise AdvisoryBaselineError("advisory item must be an object" if item else "policy root must be an object")
    keys = tuple(value)
    if set(keys) != set(expected):
        if set(keys) - set(expected):
            raise AdvisoryBaselineError("unknown advisory item key" if item else "unknown policy root key")
        raise AdvisoryBaselineError("missing advisory item key" if item else "missing policy root key")
    if canonical_order and keys != expected:
        raise AdvisoryBaselineError("policy key order is not canonical")
    return value


def _require_string(value: object, message: str) -> str:
    if not isinstance(value, str) or not value:
        raise AdvisoryBaselineError(message)
    return value


def _require_surface_list(value: object, *, root: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(surface, str) for surface in value):
        raise AdvisoryBaselineError("audited surfaces must be a list of strings" if root else "advisory surfaces must be a list of strings")
    surfaces = tuple(value)
    if root:
        if surfaces != SURFACE_ORDER:
            raise AdvisoryBaselineError("audited surfaces must match the canonical order")
    elif not surfaces or len(set(surfaces)) != len(surfaces) or any(surface not in SURFACE_ORDER for surface in surfaces):
        raise AdvisoryBaselineError("advisory surfaces must be known and unique")
    elif surfaces != tuple(surface for surface in SURFACE_ORDER if surface in surfaces):
        raise AdvisoryBaselineError("advisory surfaces must use canonical order")
    return surfaces


def load_baseline(path: Path) -> Baseline:
    """Load one strictly canonical accepted-advisory policy."""
    try:
        original = path.read_bytes()
        document = json.loads(original.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AdvisoryBaselineError("invalid advisory policy JSON") from error

    root = _require_exact_keys(document, ("schema_version", "audited_surfaces", "accepted_advisories"))
    schema_version = root["schema_version"]
    if type(schema_version) is not int or schema_version != SCHEMA_VERSION:
        raise AdvisoryBaselineError("unsupported schema version")
    audited_surfaces = _require_surface_list(root["audited_surfaces"], root=True)
    items = root["accepted_advisories"]
    if not isinstance(items, list):
        raise AdvisoryBaselineError("accepted advisories must be a list")

    accepted: list[AcceptedAdvisory] = []
    identities: set[tuple[str, str, str]] = set()
    for raw_item in items:
        item = _require_exact_keys(
            raw_item,
            ("advisory_id", "package", "accepted_version", "surfaces"),
            item=True,
        )
        package = _require_string(item["package"], "advisory package must be a string")
        normalized_package = normalize_package_name(package)
        accepted_item = AcceptedAdvisory(
            package=normalized_package,
            advisory_id=_require_string(item["advisory_id"], "advisory id must be a string"),
            accepted_version=_require_string(item["accepted_version"], "accepted version must be a string"),
            surfaces=_require_surface_list(item["surfaces"]),
        )
        identity = advisory_identity(accepted_item)
        if identity in identities:
            raise AdvisoryBaselineError("duplicate advisory identity")
        if package != normalized_package:
            raise AdvisoryBaselineError("advisory package must use canonical package spelling")
        identities.add(identity)
        accepted.append(accepted_item)

    if accepted != sorted(accepted, key=lambda item: (item.package, item.accepted_version, item.advisory_id)):
        raise AdvisoryBaselineError("accepted advisories must be sorted")
    canonical = json.dumps(document, indent=2, ensure_ascii=False).encode("utf-8") + b"\n"
    if original != canonical:
        raise AdvisoryBaselineError("advisory policy serialization is not canonical")
    return Baseline(SCHEMA_VERSION, audited_surfaces, tuple(accepted))


def normalize_pip_audit(surface: str, payload: object) -> Observation:
    """Reduce one pip-audit JSON observation to policy-relevant data."""
    if surface not in SURFACE_ORDER:
        raise AdvisoryBaselineError("unknown audit surface")
    root = _require_exact_keys(payload, ("dependencies", "fixes"), canonical_order=False)
    dependencies = root["dependencies"]
    if not isinstance(dependencies, list):
        raise AdvisoryBaselineError("pip-audit dependencies must be a list")
    if root["fixes"] != []:
        raise AdvisoryBaselineError("pip-audit fixes must be an empty list")

    versions: dict[str, str] = {}
    advisories: set[tuple[str, str, str]] = set()
    for dependency in dependencies:
        dependency = _require_exact_keys(
            dependency, ("name", "version", "vulns"), item=True, canonical_order=False
        )
        name = _require_string(dependency["name"], "pip-audit dependency name must be a string")
        version = _require_string(dependency["version"], "pip-audit dependency version must be a string")
        vulnerabilities = dependency["vulns"]
        if not isinstance(vulnerabilities, list):
            raise AdvisoryBaselineError("pip-audit dependency vulnerabilities must be a list")
        package = normalize_package_name(name)
        if package in versions:
            raise AdvisoryBaselineError("pip-audit payload has duplicate normalized dependencies")
        versions[package] = version
        for vulnerability in vulnerabilities:
            vulnerability = _require_exact_keys(
                vulnerability,
                ("id", "fix_versions", "aliases"),
                item=True,
                canonical_order=False,
            )
            advisory_id = _require_string(
                vulnerability["id"],
                "pip-audit vulnerability id must be a string",
            )
            for field in ("aliases", "fix_versions"):
                value = vulnerability[field]
                if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
                    raise AdvisoryBaselineError(f"pip-audit vulnerability {field} must be a list of strings")
            advisories.add((package, version, advisory_id))

    return Observation(
        surface=surface,
        resolved_versions=tuple(sorted(versions.items())),
        advisories=tuple(sorted(advisories)),
    )


def _format_surfaces(surfaces: Sequence[str]) -> str:
    return "[" + ", ".join(surfaces) + "]"


def compare_baseline(baseline: Baseline, observations: Sequence[Observation]) -> Comparison:
    """Return deterministic policy errors and reconciliation notices."""
    errors: list[str] = []
    observations_by_surface: dict[str, Observation] = {}
    for observation in observations:
        if observation.surface not in baseline.audited_surfaces:
            errors.append(f"unknown audit surface: {observation.surface}")
        elif observation.surface in observations_by_surface:
            errors.append(f"duplicate audit surface: {observation.surface}")
        else:
            observations_by_surface[observation.surface] = observation
    for surface in baseline.audited_surfaces:
        if surface not in observations_by_surface:
            errors.append(f"missing audit surface: {surface}")

    observed_advisories: dict[tuple[str, str, str], set[str]] = {}
    baseline_by_primary: dict[tuple[str, str], list[AcceptedAdvisory]] = {}
    accepted_versions_by_surface_package: dict[tuple[str, str], set[str]] = {}
    accepted_versions_by_package: dict[str, set[str]] = {}
    accepted_surfaces_by_package_version: dict[tuple[str, str], set[str]] = {}
    for accepted in baseline.accepted_advisories:
        baseline_by_primary.setdefault((accepted.package, accepted.advisory_id), []).append(accepted)
        accepted_versions_by_package.setdefault(accepted.package, set()).add(accepted.accepted_version)
        accepted_surfaces_by_package_version.setdefault(
            (accepted.package, accepted.accepted_version), set()
        ).update(accepted.surfaces)
        for accepted_surface in accepted.surfaces:
            accepted_versions_by_surface_package.setdefault(
                (accepted_surface, accepted.package), set()
            ).add(accepted.accepted_version)
    for surface, observation in observations_by_surface.items():
        for package, version in observation.resolved_versions:
            expected_versions = accepted_versions_by_surface_package.get(
                (surface, package), accepted_versions_by_package.get(package)
            )
            if expected_versions is not None and version not in expected_versions:
                expected = ", ".join(sorted(expected_versions))
                errors.append(f"accepted version drift: {package} expected {expected}; observed {version}")
            expected_surfaces = accepted_surfaces_by_package_version.get((package, version))
            if expected_surfaces is not None and surface not in expected_surfaces:
                expected = tuple(item for item in baseline.audited_surfaces if item in expected_surfaces)
                errors.append(
                    f"surface drift: {package} {version} observed on {surface}; "
                    f"expected {_format_surfaces(expected)}"
                )
        for package, version, advisory_id in observation.advisories:
            observed_advisories.setdefault((package, advisory_id, version), set()).add(surface)
            accepted = baseline_by_primary.get((package, advisory_id))
            if accepted is None:
                errors.append(f"new advisory: {package} {version} {advisory_id} on {surface}")
            elif version not in {item.accepted_version for item in accepted}:
                expected = ", ".join(sorted({item.accepted_version for item in accepted}))
                errors.append(f"accepted version drift: {package} expected {expected}; observed {version}")

    notices: list[str] = []
    for accepted in baseline.accepted_advisories:
        observed = observed_advisories.get(advisory_identity(accepted), set())
        if not observed:
            notices.append(
                f"reconcile removed advisory: {accepted.package} {accepted.accepted_version} {accepted.advisory_id}"
            )
            continue
        actual_surfaces = tuple(surface for surface in baseline.audited_surfaces if surface in observed)
        if actual_surfaces != accepted.surfaces:
            errors.append(
                "surface drift: "
                f"{accepted.package} {accepted.accepted_version} {accepted.advisory_id} "
                f"expected {_format_surfaces(accepted.surfaces)}; observed {_format_surfaces(actual_surfaces)}"
            )

    return Comparison(tuple(sorted(set(errors))), tuple(sorted(set(notices))))


def _audit_command(surface: PreparedAuditSurface, output: Path) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "pip_audit",
        "--disable-pip",
        "--no-deps",
        "-r",
        str(surface.projection),
    ]
    command.extend(
        (
            "--strict",
            "--vulnerability-service",
            "pypi",
            "--format",
            "json",
            "--aliases",
            "on",
            "--desc",
            "off",
            "--progress-spinner",
            "off",
            "--output",
            str(output),
        )
    )
    return command


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _target_for(platform: object) -> TargetEnvironment:
    environment = dict(platform.marker_environment)
    environment.update(
        {
            "implementation_version": "3.11.0",
            "python_full_version": "3.11.0",
            "extra": "",
        }
    )
    return TargetEnvironment(key=platform.key, marker_environment=environment)


def _lock_requirements(
    repo: Path,
    relative: Path,
    targets: Sequence[TargetEnvironment],
) -> tuple[tuple[LockedRequirement, str], ...]:
    requirements = parse_lock(repo / relative)
    selected: list[tuple[LockedRequirement, str]] = []
    for target in targets:
        selected.extend((item, target.key) for item in project_for_target(requirements, target))
    return tuple(selected)


def _surface_lock_paths(policy: object, surface: str) -> tuple[Path, ...]:
    platform_paths = tuple(Path("requirements/locks") / platform.key for platform in policy.platforms)
    if surface == "combined-runtime":
        return (_BOOTSTRAP_LOCK, _AUDIT_LOCK) + tuple(path / "root.txt" for path in platform_paths)
    if surface == "torch":
        return (_BOOTSTRAP_LOCK,) + tuple(
            child
            for path in platform_paths
            for child in (path / "core.txt", path / "runtime.txt")
        )
    if surface == "documentation":
        return (_BOOTSTRAP_LOCK, _DOCS_LOCK)
    if surface == "atlas-contract":
        return (_BOOTSTRAP_LOCK, _ATLAS_LOCK)
    raise AdvisoryBaselineError("unknown audit surface")


def _surface_requirements(
    repo: Path,
    policy: object,
    surface: str,
) -> tuple[tuple[LockedRequirement, str, str], ...]:
    targets = tuple(_target_for(platform) for platform in policy.platforms)
    platform_by_key = {platform.key: platform for platform in policy.platforms}
    records: list[tuple[LockedRequirement, str, str]] = []
    for relative in _surface_lock_paths(policy, surface):
        parts = relative.parts
        if len(parts) >= 4 and parts[:2] == ("requirements", "locks"):
            selected_targets = (next(item for item in targets if item.key == parts[2]),)
        else:
            selected_targets = targets
        for requirement, target in _lock_requirements(repo, relative, selected_targets):
            source = requirement.source or "https://pypi.org/simple"
            if requirement.name == "pyg-lib":
                source = platform_by_key[target].pyg_find_links
            records.append((requirement, target, source))
    return tuple(records)


def _collapse_public_versions(
    records: Sequence[tuple[LockedRequirement, str, str]],
) -> tuple[tuple[str, str], ...]:
    versions: dict[str, str] = {}
    for requirement, _, _ in records:
        if requirement.name in _NON_PYPI_PACKAGES:
            continue
        public = requirement.version.public
        prior = versions.setdefault(requirement.name, public)
        if prior != public:
            raise AdvisoryBaselineError("dependency locks contain conflicting public versions")
    return tuple(sorted(versions.items()))


def _non_pypi_records(
    records: Sequence[tuple[LockedRequirement, str, str]],
) -> tuple[NonPyPIRecord, ...]:
    grouped: dict[tuple[str, str, str, tuple[str, ...]], set[str]] = {}
    for requirement, target, source in records:
        if requirement.name not in _NON_PYPI_PACKAGES:
            continue
        identity = (requirement.name, str(requirement.version), source, requirement.hashes)
        grouped.setdefault(identity, set()).add(target)
    return tuple(
        NonPyPIRecord(
            package=package,
            version=version,
            public_version=Version(version).public,
            source=source,
            hashes=hashes,
            targets=tuple(sorted(targets)),
        )
        for (package, version, source, hashes), targets in sorted(grouped.items())
    )


def _source_audit_projection(
    repo: Path,
    relative: Path,
    *,
    seen: frozenset[Path] = frozenset(),
) -> tuple[tuple[str, str], ...]:
    if relative in seen or relative.is_absolute() or ".." in relative.parts:
        raise AdvisoryBaselineError("source audit projection has an invalid include")
    try:
        lines = (repo / relative).read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise AdvisoryBaselineError("source audit projection is unavailable") from error
    projected: list[tuple[str, str]] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        line = line.split("  #", 1)[0].rstrip()
        if line.startswith("-r "):
            include = Path(line.removeprefix("-r "))
            projected.extend(
                _source_audit_projection(repo, include, seen=seen | {relative})
            )
            continue
        try:
            requirement = Requirement(line)
            specifiers = tuple(requirement.specifier)
            if (
                requirement.url is not None
                or requirement.marker is not None
                or requirement.extras
                or len(specifiers) != 1
                or specifiers[0].operator != "=="
            ):
                raise AdvisoryBaselineError("source audit projection is not exact")
            version = Version(specifiers[0].version).public
        except (InvalidRequirement, InvalidVersion) as error:
            raise AdvisoryBaselineError("source audit projection is invalid") from error
        projected.append((normalize_package_name(requirement.name), version))
    if len(projected) != len(set(projected)):
        raise AdvisoryBaselineError("source audit projection has a duplicate requirement")
    return tuple(sorted(projected))


def _validate_human_audit_projections(
    repo: Path,
    torch_versions: tuple[tuple[str, str], ...],
    pyg_versions: tuple[tuple[str, str], ...],
) -> None:
    locked = dict(torch_versions)
    torch_source = _source_audit_projection(repo, Path("torch-audit-requirements.txt"))
    pyg_source = _source_audit_projection(repo, Path("pyg-extension-audit-requirements.txt"))
    if any(locked.get(package) != version for package, version in torch_source):
        raise AdvisoryBaselineError("human Torch audit projection does not match locks")
    if pyg_source != pyg_versions:
        raise AdvisoryBaselineError("human PyG audit projection does not match locks")


def derive_lock_audit_surfaces(repo: Path, projection_root: Path) -> tuple[PreparedAuditSurface, ...]:
    """Build deterministic no-resolve audit projections from validated lock artifacts."""
    try:
        findings = verify_dependency_locks(repo)
        if findings:
            raise AdvisoryBaselineError("dependency locks are not valid")
        policy = load_policy(repo)
    except (DependencyLockError, OSError) as error:
        raise AdvisoryBaselineError("dependency locks are not valid") from error
    projection_root.mkdir(parents=True, exist_ok=True)
    logical: dict[
        str,
        tuple[tuple[tuple[str, str], ...], tuple[LockInput, ...], tuple[NonPyPIRecord, ...]],
    ] = {}
    for name in SURFACE_ORDER:
        lock_paths = _surface_lock_paths(policy, name)
        records = _surface_requirements(repo, policy, name)
        versions = _collapse_public_versions(records)
        expected_non_pypi = {
            "combined-runtime": {"en-core-web-sm", "pyg-lib"},
            "torch": {"pyg-lib"},
            "documentation": set(),
            "atlas-contract": set(),
        }[name]
        non_pypi = _non_pypi_records(records)
        if {item.package for item in non_pypi} != expected_non_pypi:
            raise AdvisoryBaselineError("dependency locks have invalid non-PyPI evidence")
        lock_inputs = tuple(
            LockInput(relative.as_posix(), _sha256(repo / relative)) for relative in lock_paths
        )
        logical[name] = (versions, lock_inputs, non_pypi)

    prepared: list[PreparedAuditSurface] = []
    for surface in AUDIT_SURFACES:
        versions, lock_inputs, non_pypi = logical[surface.name]
        if surface.projection_kind == "pyg-extensions":
            selected = tuple(item for item in versions if item[0] in _PYG_AUDITED_PACKAGES)
            if selected != PYG_EXTENSION_AUDIT_VERSIONS:
                raise AdvisoryBaselineError("dependency locks have invalid PyG audit versions")
        else:
            selected = tuple(item for item in versions if item[0] not in _PYG_AUDITED_PACKAGES)
        projection = projection_root / f"{surface.output_name}.requirements.txt"
        projection.write_text(
            "".join(f"{package}=={version}\n" for package, version in selected),
            encoding="utf-8",
        )
        prepared.append(
            PreparedAuditSurface(
                surface=surface,
                projection=projection,
                expected_versions=selected,
                lock_inputs=lock_inputs,
                non_pypi=non_pypi,
            )
        )
    torch_main = next(
        item.expected_versions
        for item in prepared
        if item.surface.output_name == "torch-resolver"
    )
    torch_pyg = next(
        item.expected_versions
        for item in prepared
        if item.surface.output_name == "torch-pyg-extensions"
    )
    _validate_human_audit_projections(repo, torch_main, torch_pyg)
    return tuple(prepared)


def _load_pip_audit_output(path: Path) -> object:
    """Parse one pip-audit JSON document without silently accepting duplicate keys."""
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys)


def _classify_missing_output(returncode: int, stderr: object) -> str:
    """Classify only stable pip-audit setup failures without exposing stderr."""
    if returncode == 1 and isinstance(stderr, str):
        if "Failed to upgrade `pip`" in stderr:
            return "bootstrap-error"
        if "Failed to install packages" in stderr:
            return "resolution-error"
        if "Tip: your network may be blocking this service" in stderr:
            return "service-error"
        if "Dependency not found on PyPI and could not be audited" in stderr:
            return "unsupported-package"
    return "missing-output"


def run_audit_surfaces(repo: Path, runner: AuditRunner = subprocess.run) -> tuple[Observation, ...]:
    """Run six physical commands and merge them into four logical observations."""
    observations: dict[str, Observation] = {}
    with tempfile.TemporaryDirectory(prefix="advisory-baseline-") as temporary_directory:
        output_directory = Path(temporary_directory)
        prepared_surfaces = derive_lock_audit_surfaces(repo, output_directory / "projections")
        for prepared in prepared_surfaces:
            surface = prepared.surface
            output = output_directory / f"{surface.output_name}.json"
            try:
                result = runner(
                    _audit_command(prepared, output),
                    cwd=repo,
                    check=False,
                    capture_output=True,
                    text=True,
                )
            except (OSError, UnicodeError) as error:
                raise AuditSurfaceError(surface.name, "execution-error") from error
            if result.returncode not in (0, 1):
                raise AuditSurfaceError(surface.name, "unexpected-exit")
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
            if observation.resolved_versions != prepared.expected_versions:
                raise AuditSurfaceError(surface.name, "invalid-schema")
            observation = Observation(
                surface=observation.surface,
                resolved_versions=observation.resolved_versions,
                advisories=observation.advisories,
                lock_inputs=prepared.lock_inputs,
                non_pypi=prepared.non_pypi,
            )
            prior = observations.get(surface.name)
            if prior is None:
                observations[surface.name] = observation
                continue
            versions = dict(prior.resolved_versions)
            if set(versions) & {package for package, _ in observation.resolved_versions}:
                raise AuditSurfaceError(surface.name, "invalid-schema")
            versions.update(observation.resolved_versions)
            advisories = set(prior.advisories)
            if advisories & set(observation.advisories):
                raise AuditSurfaceError(surface.name, "invalid-schema")
            advisories.update(observation.advisories)
            if prior.lock_inputs != observation.lock_inputs or prior.non_pypi != observation.non_pypi:
                raise AuditSurfaceError(surface.name, "invalid-schema")
            observations[surface.name] = Observation(
                surface.name,
                tuple(sorted(versions.items())),
                tuple(sorted(advisories)),
                prior.lock_inputs,
                prior.non_pypi,
            )
    return tuple(observations[name] for name in SURFACE_ORDER)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare live pip-audit results with the accepted advisory policy.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(), help="repository root (default: current directory)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the policy gate without exposing runner paths or credentials."""
    args = _parser().parse_args(argv)
    try:
        baseline = load_baseline(args.repo_root / "security" / "accepted-advisories.json")
        comparison = compare_baseline(baseline, run_audit_surfaces(args.repo_root))
    except AuditSurfaceError as error:
        print(f"advisory audit failed: {error.surface}: {error.category}", file=sys.stderr)
        return 1
    except (AdvisoryBaselineError, OSError):
        print("advisory audit failed", file=sys.stderr)
        return 1
    for diagnostic in comparison.errors:
        print(diagnostic, file=sys.stderr)
    for notice in comparison.notices:
        print(notice, file=sys.stderr)
    return 1 if comparison.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
