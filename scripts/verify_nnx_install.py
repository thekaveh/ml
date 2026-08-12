"""Verify that the active ``nnx`` import satisfies the repository contract."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from importlib import metadata
from importlib.machinery import ModuleSpec
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname


REPO_ROOT = Path(__file__).resolve().parent.parent
_DISTRIBUTION_NAME = "thekaveh-nnx"
_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_PIN_PATTERN = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)"
    r"\[(?P<extras>[A-Za-z0-9][A-Za-z0-9._-]*(?:\s*,\s*[A-Za-z0-9][A-Za-z0-9._-]*)*)\]"
    r"\s*==\s*"
    r"(?P<version>[0-9]+(?:\.[0-9]+)*(?:(?:a|b|rc)[0-9]+)?(?:\.post[0-9]+)?(?:\.dev[0-9]+)?"
    r"(?:\+[A-Za-z0-9]+(?:[-_.][A-Za-z0-9]+)*)?)$"
)
_REQUIREMENT_ERROR = "NNx requirement is not an exact pin"
_DISTRIBUTION_ERROR = "NNx distribution metadata is not canonical"
_IMPORT_ERROR = "NNx import is not owned by the installed distribution"
_EDITABLE_ERROR = "NNx editable metadata is not valid"
_OVERRIDE_ERROR = "NNx editable override is invalid"


@dataclass(frozen=True)
class NnxPin:
    version: str
    extras: frozenset[str]


@dataclass(frozen=True)
class NnxInstallEvidence:
    mode: str
    distribution: str
    version: str


class VerificationError(RuntimeError):
    """The active NNx installation does not satisfy the selected contract."""


def _normalize_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def parse_nnx_pin(requirements_text: str) -> NnxPin:
    """Return the single exact ``thekaveh-nnx[lm]`` pin from requirements text."""
    candidates: list[str] = []
    for raw_line in requirements_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        line = re.split(r"\s+#", line, maxsplit=1)[0].strip()
        if any(_normalize_name(token) == _DISTRIBUTION_NAME for token in _NAME_PATTERN.findall(line)):
            candidates.append(line)

    if len(candidates) != 1:
        raise VerificationError(_REQUIREMENT_ERROR)

    match = _PIN_PATTERN.fullmatch(candidates[0])
    if match is None or _normalize_name(match.group("name")) != _DISTRIBUTION_NAME:
        raise VerificationError(_REQUIREMENT_ERROR)
    extras = frozenset(_normalize_name(value.strip()) for value in match.group("extras").split(","))
    if extras != frozenset({"lm"}):
        raise VerificationError(_REQUIREMENT_ERROR)
    return NnxPin(version=match.group("version"), extras=extras)


def _matching_distribution(
    distributions: Iterable[metadata.Distribution], expected_version: str
) -> metadata.Distribution:
    try:
        matches = [
            distribution
            for distribution in distributions
            if _normalize_name(str(distribution.metadata["Name"])) == _DISTRIBUTION_NAME
        ]
        if len(matches) != 1:
            raise VerificationError(_DISTRIBUTION_ERROR)
        distribution = matches[0]
        if (
            _normalize_name(str(distribution.metadata["Name"])) != _DISTRIBUTION_NAME
            or str(distribution.version) != expected_version
        ):
            raise VerificationError(_DISTRIBUTION_ERROR)
        return distribution
    except VerificationError:
        raise
    except Exception:
        raise VerificationError(_DISTRIBUTION_ERROR) from None


def _matching_editable_distribution(
    distributions: Iterable[metadata.Distribution], expected_version: str
) -> metadata.Distribution:
    try:
        named = [
            distribution
            for distribution in distributions
            if _normalize_name(str(distribution.metadata["Name"])) == _DISTRIBUTION_NAME
        ]
        pinned = [distribution for distribution in named if str(distribution.version) == expected_version]
        other_versions = [distribution for distribution in named if str(distribution.version) != expected_version]
        if len(pinned) != 1 or any(
            distribution.read_text("direct_url.json") is not None for distribution in other_versions
        ):
            raise VerificationError(_DISTRIBUTION_ERROR)
        return pinned[0]
    except VerificationError:
        raise
    except Exception:
        raise VerificationError(_DISTRIBUTION_ERROR) from None


def _canonical_owned_init(distribution: metadata.Distribution) -> Path:
    try:
        if distribution.read_text("direct_url.json") is not None:
            raise VerificationError(_DISTRIBUTION_ERROR)
        files = distribution.files
        if files is None:
            raise VerificationError(_DISTRIBUTION_ERROR)
        file_paths = tuple(Path(file) for file in files)
        distribution_name = str(distribution.metadata["Name"])
        distribution_version = str(distribution.version)
        dist_info_directory = Path(
            f"{re.sub(r'[-_.]+', '_', distribution_name).lower()}-"
            f"{distribution_version}.dist-info"
        )
        if (
            dist_info_directory / "WHEEL" not in file_paths
            or dist_info_directory / "RECORD" not in file_paths
            or Path("nnx/__init__.py") not in file_paths
        ):
            raise VerificationError(_DISTRIBUTION_ERROR)
        return Path(distribution.locate_file(Path("nnx/__init__.py"))).resolve()
    except VerificationError:
        raise
    except Exception:
        raise VerificationError(_DISTRIBUTION_ERROR) from None


def _resolved_import_origin(find_spec: Callable[[str], ModuleSpec | None]) -> Path:
    try:
        spec = find_spec("nnx")
        if spec is None or spec.origin is None:
            raise VerificationError(_IMPORT_ERROR)
        return Path(spec.origin).resolve()
    except VerificationError:
        raise
    except Exception:
        raise VerificationError(_IMPORT_ERROR) from None


def _editable_source(distribution: metadata.Distribution) -> Path:
    try:
        direct_url = distribution.read_text("direct_url.json")
        value = json.loads(direct_url) if direct_url is not None else None
        if not isinstance(value, dict) or "archive_info" in value or "vcs_info" in value:
            raise VerificationError(_EDITABLE_ERROR)
        dir_info = value.get("dir_info")
        url = value.get("url")
        if not isinstance(dir_info, dict) or dir_info.get("editable") is not True or not isinstance(url, str):
            raise VerificationError(_EDITABLE_ERROR)
        parsed = urlparse(url)
        if (
            parsed.scheme != "file"
            or parsed.netloc
            or parsed.params
            or parsed.query
            or parsed.fragment
            or not parsed.path
        ):
            raise VerificationError(_EDITABLE_ERROR)
        source = Path(url2pathname(parsed.path))
        if not source.is_absolute():
            raise VerificationError(_EDITABLE_ERROR)
        source = source.resolve()
        if not source.is_dir():
            raise VerificationError(_EDITABLE_ERROR)
        return source
    except VerificationError:
        raise
    except Exception:
        raise VerificationError(_EDITABLE_ERROR) from None


def _verify_editable_import(source: Path, find_spec: Callable[[str], ModuleSpec | None]) -> None:
    origin = _resolved_import_origin(find_spec)
    try:
        if (
            not origin.is_file()
            or origin.parts[-2:] != ("nnx", "__init__.py")
            or not origin.is_relative_to(source)
        ):
            raise VerificationError(_IMPORT_ERROR)
    except VerificationError:
        raise
    except Exception:
        raise VerificationError(_IMPORT_ERROR) from None


def verify_nnx_install(
    requirements_path: Path = REPO_ROOT / "requirements.txt",
    *,
    environ: Mapping[str, str] | None = None,
    distributions: Iterable[metadata.Distribution] | None = None,
    find_spec: Callable[[str], ModuleSpec | None] = importlib.util.find_spec,
    repo_root: Path = REPO_ROOT,
) -> NnxInstallEvidence:
    """Verify canonical named-release NNx installation and import ownership."""
    selected_environ = os.environ if environ is None else environ
    try:
        editable_override = selected_environ.get("NNX_ALLOW_EDITABLE")
    except Exception:
        raise VerificationError(_OVERRIDE_ERROR) from None
    if editable_override not in (None, "", "1"):
        raise VerificationError(_OVERRIDE_ERROR)
    try:
        requirements_text = requirements_path.read_text(encoding="utf-8")
    except Exception:
        raise VerificationError(_REQUIREMENT_ERROR) from None
    try:
        pin = parse_nnx_pin(requirements_text)
    except VerificationError:
        raise
    except Exception:
        raise VerificationError(_REQUIREMENT_ERROR) from None
    if distributions is None:
        try:
            selected_distributions = metadata.distributions()
        except Exception:
            raise VerificationError(_DISTRIBUTION_ERROR) from None
    else:
        selected_distributions = distributions
    if editable_override == "1":
        distribution = _matching_editable_distribution(selected_distributions, pin.version)
        source = _editable_source(distribution)
        _verify_editable_import(source, find_spec)
        return NnxInstallEvidence(mode="editable-development", distribution=_DISTRIBUTION_NAME, version=pin.version)
    distribution = _matching_distribution(selected_distributions, pin.version)
    owned_init = _canonical_owned_init(distribution)
    import_origin = _resolved_import_origin(find_spec)
    try:
        resolved_repo_root = repo_root.resolve()
        if not owned_init.is_file() or owned_init != import_origin or owned_init.is_relative_to(resolved_repo_root):
            raise VerificationError(_IMPORT_ERROR)
    except VerificationError:
        raise
    except Exception:
        raise VerificationError(_IMPORT_ERROR) from None
    return NnxInstallEvidence(mode="canonical-wheel", distribution=_DISTRIBUTION_NAME, version=pin.version)


def main() -> int:
    try:
        evidence = verify_nnx_install()
    except VerificationError as error:
        print(str(error), file=sys.stderr)
        return 1
    print(f"NNx installation verified: {evidence.distribution} {evidence.version} ({evidence.mode})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
