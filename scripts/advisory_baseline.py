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
TORCH_RUNTIME_REQUIREMENTS = "torch-requirements.txt"
TORCH_AUDIT_REQUIREMENTS = "torch-audit-requirements.txt"
TORCH_CORE_REQUIREMENTS = "-r torch-core-requirements.txt"
PYG_FIND_LINKS = "--find-links https://data.pyg.org/whl/torch-2.4.0+cpu.html"
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
    AuditSurface("combined-runtime", ("requirements.txt", TORCH_AUDIT_REQUIREMENTS)),
    AuditSurface("torch", (TORCH_AUDIT_REQUIREMENTS,)),
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


def _semantic_requirement_lines(path: Path) -> tuple[str, ...]:
    """Return non-comment requirement/include lines with trailing comments removed."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise AdvisoryBaselineError("torch audit projection is unavailable") from error
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        line = line.split(" #", 1)[0].rstrip()
        if line:
            lines.append(line)
    return tuple(lines)


def _validate_torch_audit_projection(repo: Path) -> None:
    """Require the selector-free audit manifest to exactly mirror the runtime pins."""
    runtime = _semantic_requirement_lines(repo / TORCH_RUNTIME_REQUIREMENTS)
    audit = _semantic_requirement_lines(repo / TORCH_AUDIT_REQUIREMENTS)
    if runtime.count(PYG_FIND_LINKS) != 1 or runtime.count(TORCH_CORE_REQUIREMENTS) != 1:
        raise AdvisoryBaselineError("torch audit projection has an invalid runtime selector")
    expected = tuple(line for line in runtime if line != PYG_FIND_LINKS)
    if (
        len(runtime) != len(set(runtime))
        or len(audit) != len(set(audit))
        or audit.count(TORCH_CORE_REQUIREMENTS) != 1
        or any(line.startswith("-") and line != TORCH_CORE_REQUIREMENTS for line in audit)
        or audit != expected
    ):
        raise AdvisoryBaselineError("torch audit projection does not match runtime requirements")


def run_audit_surfaces(repo: Path, runner: AuditRunner = subprocess.run) -> tuple[Observation, ...]:
    """Run the fixed four-surface audit contract and normalize every result."""
    _validate_torch_audit_projection(repo)
    observations: list[Observation] = []
    with tempfile.TemporaryDirectory(prefix="advisory-baseline-") as temporary_directory:
        output_directory = Path(temporary_directory)
        for surface in AUDIT_SURFACES:
            output = output_directory / f"{surface.name}.json"
            try:
                result = runner(
                    _audit_command(surface, output),
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
                observations.append(normalize_pip_audit(surface.name, payload))
            except AdvisoryBaselineError as error:
                raise AuditSurfaceError(surface.name, "invalid-schema") from error
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
