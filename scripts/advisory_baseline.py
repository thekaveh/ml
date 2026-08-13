"""Compare live pip-audit observations with the reviewed advisory baseline."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


SCHEMA_VERSION = 1
SURFACE_ORDER = (
    "combined-runtime",
    "torch",
    "documentation",
    "atlas-contract",
)


class AdvisoryBaselineError(RuntimeError):
    """The advisory policy or an audit observation is invalid."""


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


@dataclass(frozen=True)
class Observation:
    surface: str
    resolved_versions: tuple[tuple[str, str], ...]
    advisories: tuple[tuple[str, str, str], ...]


@dataclass(frozen=True)
class Comparison:
    errors: tuple[str, ...]
    notices: tuple[str, ...]


@dataclass(frozen=True)
class AuditSurface:
    name: str
    requirements: tuple[str, ...]
    disable_pip: bool = False


AUDIT_SURFACES = (
    AuditSurface("combined-runtime", ("requirements.txt", "torch-requirements.txt")),
    AuditSurface("torch", ("torch-requirements.txt",)),
    AuditSurface("documentation", ("docs-requirements.txt",), disable_pip=True),
    AuditSurface("atlas-contract", ("atlas-contract-requirements.txt",)),
)
AuditRunner = Callable[..., subprocess.CompletedProcess[str]]


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


def _require_exact_keys(value: object, expected: tuple[str, ...], *, item: bool = False) -> dict[str, object]:
    if not isinstance(value, dict):
        raise AdvisoryBaselineError("advisory item must be an object" if item else "policy root must be an object")
    keys = tuple(value)
    if set(keys) != set(expected):
        if set(keys) - set(expected):
            raise AdvisoryBaselineError("unknown advisory item key" if item else "unknown policy root key")
        raise AdvisoryBaselineError("missing advisory item key" if item else "missing policy root key")
    if keys != expected:
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

    if accepted != sorted(accepted):
        raise AdvisoryBaselineError("accepted advisories must be sorted")
    canonical = json.dumps(document, indent=2, ensure_ascii=False).encode("utf-8") + b"\n"
    if original != canonical:
        raise AdvisoryBaselineError("advisory policy serialization is not canonical")
    return Baseline(SCHEMA_VERSION, audited_surfaces, tuple(accepted))


def normalize_pip_audit(surface: str, payload: object) -> Observation:
    """Reduce one pip-audit JSON observation to policy-relevant data."""
    if surface not in SURFACE_ORDER:
        raise AdvisoryBaselineError("unknown audit surface")
    if not isinstance(payload, list):
        raise AdvisoryBaselineError("pip-audit payload must be a list")

    versions: dict[str, str] = {}
    advisories: set[tuple[str, str, str]] = set()
    for dependency in payload:
        if not isinstance(dependency, dict):
            raise AdvisoryBaselineError("pip-audit dependency must be an object")
        name = _require_string(dependency.get("name"), "pip-audit dependency name must be a string")
        version = _require_string(dependency.get("version"), "pip-audit dependency version must be a string")
        vulnerabilities = dependency.get("vulns")
        if not isinstance(vulnerabilities, list):
            raise AdvisoryBaselineError("pip-audit dependency vulnerabilities must be a list")
        package = normalize_package_name(name)
        previous_version = versions.get(package)
        if previous_version is not None and previous_version != version:
            raise AdvisoryBaselineError("pip-audit package has conflicting versions")
        versions[package] = version
        for vulnerability in vulnerabilities:
            if not isinstance(vulnerability, dict):
                raise AdvisoryBaselineError("pip-audit vulnerability must be an object")
            advisory_id = _require_string(
                vulnerability.get("id"),
                "pip-audit vulnerability id must be a string",
            )
            for field in ("aliases", "fix_versions"):
                value = vulnerability.get(field)
                if value is not None and (
                    not isinstance(value, list) or any(not isinstance(item, str) for item in value)
                ):
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
    expected_versions_by_surface: dict[tuple[str, str], set[str]] = {}
    for accepted in baseline.accepted_advisories:
        baseline_by_primary.setdefault((accepted.package, accepted.advisory_id), []).append(accepted)
        for surface in accepted.surfaces:
            expected_versions_by_surface.setdefault((surface, accepted.package), set()).add(accepted.accepted_version)
    for surface, observation in observations_by_surface.items():
        for package, version in observation.resolved_versions:
            expected_versions = expected_versions_by_surface.get((surface, package))
            if expected_versions is not None and version not in expected_versions:
                expected = ", ".join(sorted(expected_versions))
                errors.append(f"accepted version drift: {package} expected {expected}; observed {version}")
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


def _audit_command(surface: AuditSurface, output: Path) -> list[str]:
    command = [sys.executable, "-m", "pip_audit"]
    if surface.disable_pip:
        command.append("--disable-pip")
    for requirement in surface.requirements:
        command.extend(("-r", requirement))
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


def run_audit_surfaces(repo: Path, runner: AuditRunner = subprocess.run) -> tuple[Observation, ...]:
    """Run the fixed four-surface audit contract and normalize every result."""
    observations: list[Observation] = []
    with tempfile.TemporaryDirectory(prefix="advisory-baseline-") as temporary_directory:
        output_directory = Path(temporary_directory)
        for surface in AUDIT_SURFACES:
            output = output_directory / f"{surface.name}.json"
            result = runner(
                _audit_command(surface, output),
                cwd=repo,
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode not in (0, 1):
                raise AdvisoryBaselineError("pip-audit did not complete")
            try:
                payload = json.loads(output.read_text(encoding="utf-8"))
                observations.append(normalize_pip_audit(surface.name, payload))
            except (OSError, json.JSONDecodeError, AdvisoryBaselineError) as error:
                raise AdvisoryBaselineError("pip-audit produced an invalid observation") from error
    return tuple(observations)


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
