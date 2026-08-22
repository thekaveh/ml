#!/usr/bin/env python3
"""Capture safe, machine-readable evidence about an Atlas Python runtime."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.metadata as metadata
import json
import platform
import re
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path

try:
    from scripts.nlp_assets import (
        NLPAssetError,
        load_manifest as load_nlp_asset_manifest,
        verify_vader,
    )
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from nlp_assets import (  # type: ignore[no-redef]
        NLPAssetError,
        load_manifest as load_nlp_asset_manifest,
        verify_vader,
    )


SCHEMA_VERSION = 1
NON_PYTHON_CELL_MAGICS = frozenset(
    {"bash", "html", "javascript", "js", "latex", "perl", "ruby", "sh", "svg"}
)
DEPENDENCIES = (
    ("thekaveh-nnx[lm]", "thekaveh-nnx", "nnx"),
    ("nnx", "thekaveh-nnx", "nnx"),
    ("Torch", "torch", "torch"),
    ("TorchVision", "torchvision", "torchvision"),
    ("TorchAO", "torchao", "torchao"),
    ("PyTorch Geometric", "torch-geometric", "torch_geometric"),
    ("python-louvain", "python-louvain", "community"),
    ("spaCy", "spacy", "spacy"),
    ("NLTK", "nltk", "nltk"),
)
EXTRA_IMPORTS = {"thekaveh-nnx[lm]": ("datasets", "tokenizers")}
DEPENDENCY_BY_LABEL = {
    label: (distribution_name, module_name)
    for label, distribution_name, module_name in DEPENDENCIES
}
ASSET_BY_NAME = {
    "en_core_web_sm": ("en-core-web-sm", "en_core_web_sm"),
    "vader_lexicon": (None, "nltk.sentiment.vader"),
}
IMPORT_STATUSES = frozenset({"ok", "missing_module", "import_error"})
DISTRIBUTION_STATUSES = frozenset({"ok", "missing_distribution", "metadata_error"})
DEPENDENCY_STATUSES = IMPORT_STATUSES | DISTRIBUTION_STATUSES | {"version_mismatch"}
ASSET_STATUSES = DEPENDENCY_STATUSES | {"missing_asset", "asset_identity_mismatch"}
NOTEBOOK_IMPORT_STATUSES = frozenset({"ok", "failed", "scan_error"})
SUMMARY_STATUSES = frozenset({"ok", "failed"})
SAFE_VERSION_RE = re.compile(r"^[0-9][A-Za-z0-9.!+_-]{0,127}$")
SAFE_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SAFE_MODULE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")
SAFE_PYTHON_IMPLEMENTATIONS = frozenset(
    {"CPython", "PyPy", "Jython", "IronPython", "GraalPy"}
)


def evaluate_capability(
    *,
    kind: str,
    available: bool,
    observed_version: str | None = None,
    expected_version: str | None = None,
) -> dict[str, str]:
    """Classify capability evidence without conflating absence and incompatibility."""
    if kind not in {"module", "asset"}:
        raise ValueError(f"unsupported capability kind: {kind!r}")
    if not available:
        return {"status": f"missing_{kind}"}

    result = {"status": "ok"}
    if observed_version is not None:
        result["observed_version"] = observed_version
    if expected_version is not None:
        result["expected_version"] = expected_version
        if observed_version != expected_version:
            result["status"] = "version_mismatch"
    return result


def _mapping(value: object) -> Mapping[str, object] | None:
    return value if isinstance(value, Mapping) else None


def _safe_status(value: object, allowed: frozenset[str], default: str) -> str:
    return value if isinstance(value, str) and value in allowed else default


def _safe_version(value: object) -> str | None:
    return value if isinstance(value, str) and SAFE_VERSION_RE.fullmatch(value) else None


def _safe_count(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 100_000:
        return value
    return None


def _project_import(
    raw: object,
    module_name: str | None = None,
    *,
    include_context_count: bool = False,
) -> dict[str, object] | None:
    item = _mapping(raw)
    if module_name is None:
        if item is None:
            return None
        candidate = item.get("module")
        if not isinstance(candidate, str) or not SAFE_MODULE_RE.fullmatch(candidate):
            return None
        module_name = candidate
    status = _safe_status(
        item.get("status") if item is not None else None,
        IMPORT_STATUSES,
        "import_error",
    )
    result: dict[str, object] = {"module": module_name, "status": status}
    if item is None:
        return result
    if version := _safe_version(item.get("module_version")):
        result["module_version"] = version
    if include_context_count and (count := _safe_count(item.get("context_count"))) is not None:
        result["context_count"] = count
    return result


def _project_distribution(raw: object, distribution_name: str) -> dict[str, object]:
    item = _mapping(raw)
    status = _safe_status(
        item.get("status") if item is not None else None,
        DISTRIBUTION_STATUSES,
        "metadata_error",
    )
    result: dict[str, object] = {"name": distribution_name, "status": status}
    if item is not None and (version := _safe_version(item.get("version"))):
        result["version"] = version
    return result


def _project_dependency(raw: Mapping[str, object], label: str) -> dict[str, object]:
    distribution_name, module_name = DEPENDENCY_BY_LABEL[label]
    result: dict[str, object] = {
        "label": label,
        "status": _safe_status(raw.get("status"), DEPENDENCY_STATUSES, "import_error"),
        "distribution": _project_distribution(raw.get("distribution"), distribution_name),
        "import": _project_import(raw.get("import"), module_name),
    }
    expected_extra_modules = EXTRA_IMPORTS.get(label, ())
    raw_extra_imports = _mapping_by_module(raw.get("extra_imports"))
    if expected_extra_modules:
        result["extra_imports"] = [
            _project_import(raw_extra_imports.get(module), module)
            for module in expected_extra_modules
        ]
    return result


def _mapping_by_module(value: object) -> dict[str, Mapping[str, object]]:
    if not isinstance(value, list):
        return {}
    result: dict[str, Mapping[str, object]] = {}
    for item in value:
        mapping = _mapping(item)
        if mapping is None:
            continue
        module = mapping.get("module")
        if isinstance(module, str) and module not in result:
            result[module] = mapping
    return result


def _project_asset(raw: Mapping[str, object], asset_name: str) -> dict[str, object]:
    distribution_name, module_name = ASSET_BY_NAME[asset_name]
    result: dict[str, object] = {
        "asset": asset_name,
        "status": _safe_status(raw.get("status"), ASSET_STATUSES, "import_error"),
        "import": _project_import(raw.get("import"), module_name),
    }
    if distribution_name is not None:
        result["distribution"] = _project_distribution(raw.get("distribution"), distribution_name)
    if asset_name == "vader_lexicon":
        expected_sha = raw.get("expected_sha256")
        observed_sha = raw.get("observed_sha256")
        expected_size = _safe_count(raw.get("expected_size"))
        observed_size = _safe_count(raw.get("observed_size"))
        member = raw.get("member")
        identity_ok = (
            isinstance(expected_sha, str)
            and SAFE_SHA256_RE.fullmatch(expected_sha) is not None
            and isinstance(observed_sha, str)
            and SAFE_SHA256_RE.fullmatch(observed_sha) is not None
            and expected_sha == observed_sha
            and expected_size is not None
            and observed_size is not None
            and expected_size == observed_size
            and member == "vader_lexicon/vader_lexicon.txt"
        )
        if result["status"] == "ok" and identity_ok:
            result.update({
                "expected_sha256": expected_sha,
                "observed_sha256": observed_sha,
                "expected_size": expected_size,
                "observed_size": observed_size,
                "member": member,
            })
        elif result["status"] == "ok":
            result["status"] = "asset_identity_mismatch"
    return result


def _project_notebook_imports(raw: object) -> dict[str, object] | None:
    item = _mapping(raw)
    if item is None:
        return None
    raw_imports = item.get("imports")
    imports = (
        [
            projected
            for entry in raw_imports
            if (projected := _project_import(entry, include_context_count=True)) is not None
        ]
        if isinstance(raw_imports, list)
        else []
    )
    return {
        "status": _safe_status(item.get("status"), NOTEBOOK_IMPORT_STATUSES, "scan_error"),
        "module_count": len(imports),
        "imports": imports,
    }


def serialize_report(report: Mapping[str, object]) -> str:
    """Project runtime evidence onto the fixed, path-free JSON report schema."""
    result: dict[str, object] = {}
    if report.get("schema_version") == SCHEMA_VERSION:
        result["schema_version"] = SCHEMA_VERSION

    python = _mapping(report.get("python"))
    if python is not None:
        implementation = python.get("implementation")
        version = _safe_version(python.get("version"))
        if implementation in SAFE_PYTHON_IMPLEMENTATIONS and version is not None:
            result["python"] = {"implementation": implementation, "version": version}

    raw_dependencies = report.get("dependencies")
    if isinstance(raw_dependencies, list):
        dependency_by_label = {
            item["label"]: item
            for entry in raw_dependencies
            if (item := _mapping(entry)) is not None
            and isinstance(item.get("label"), str)
            and item["label"] in DEPENDENCY_BY_LABEL
        }
        result["dependencies"] = [
            _project_dependency(dependency_by_label[label], label)
            for label, _, _ in DEPENDENCIES
            if label in dependency_by_label
        ]

    raw_assets = report.get("assets")
    if isinstance(raw_assets, list):
        asset_by_name = {
            item["asset"]: item
            for entry in raw_assets
            if (item := _mapping(entry)) is not None
            and isinstance(item.get("asset"), str)
            and item["asset"] in ASSET_BY_NAME
        }
        result["assets"] = [
            _project_asset(asset_by_name[name], name)
            for name in ASSET_BY_NAME
            if name in asset_by_name
        ]

    if notebook_imports := _project_notebook_imports(report.get("notebook_imports")):
        result["notebook_imports"] = notebook_imports

    summary = _mapping(report.get("summary"))
    if summary is not None:
        status = _safe_status(summary.get("status"), SUMMARY_STATUSES, "failed")
        failed_count = _safe_count(summary.get("failed_capability_count"))
        summary_result: dict[str, object] = {"status": status}
        if failed_count is not None:
            summary_result["failed_capability_count"] = failed_count
        result["summary"] = summary_result
    return json.dumps(result, indent=2, sort_keys=True) + "\n"


def _cell_source(cell: Mapping[str, object]) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(part for part in source if isinstance(part, str))
    return source if isinstance(source, str) else ""


def _python_cell_source(source: str) -> str:
    for line in source.splitlines():
        stripped = line.lstrip()
        if not stripped:
            continue
        if stripped.startswith("%%"):
            magic = stripped[2:].split(None, 1)[0].lower()
            if magic in NON_PYTHON_CELL_MAGICS:
                return ""
        break
    return "\n".join(
        "" if line.lstrip().startswith(("%", "!")) else line
        for line in source.splitlines()
    )


def extract_notebook_imports(
    notebooks: Iterable[Mapping[str, object]],
) -> tuple[str, ...]:
    """Return sorted direct module imports from notebook code cells only."""
    modules: set[str] = set()
    for notebook in notebooks:
        cells = notebook.get("cells", [])
        if not isinstance(cells, list):
            continue
        for cell in cells:
            if not isinstance(cell, Mapping) or cell.get("cell_type") != "code":
                continue
            source = _python_cell_source(_cell_source(cell))
            if not source:
                continue
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules.update(alias.name for alias in node.names)
                elif (
                    isinstance(node, ast.ImportFrom)
                    and node.level == 0
                    and node.module
                ):
                    modules.add(node.module)
    return tuple(sorted(modules))


def _active_notebook_documents(
    repo_root: Path,
) -> tuple[list[Mapping[str, object]], dict[str, tuple[Path, ...]]]:
    documents: list[Mapping[str, object]] = []
    local_contexts: dict[str, list[Path]] = {}
    for notebook_path in sorted((repo_root / "notebooks").glob("*/*.ipynb")):
        if "archive" in notebook_path.relative_to(repo_root / "notebooks").parts:
            continue
        document = json.loads(notebook_path.read_text(encoding="utf-8"))
        if not isinstance(document, Mapping):
            raise ValueError("notebook document is not a JSON object")
        documents.append(document)
        sibling_modules = {
            path.stem
            for path in notebook_path.parent.glob("*.py")
            if path.stem != "__init__"
        }
        for module in extract_notebook_imports((document,)):
            if module.split(".", 1)[0] in sibling_modules:
                contexts = local_contexts.setdefault(module, [])
                if notebook_path.parent not in contexts:
                    contexts.append(notebook_path.parent)
    return documents, {module: tuple(contexts) for module, contexts in local_contexts.items()}


def _runtime_notebook_imports(
    repo_root: Path,
    documents: Iterable[Mapping[str, object]],
) -> tuple[str, ...]:
    """Return every direct module import required by active notebooks."""
    del repo_root
    return extract_notebook_imports(documents)


def _failure_evidence(module_name: str, status: str) -> dict[str, str]:
    return {"module": module_name, "status": status}


def _module_version_evidence(module_name: str, module: object) -> dict[str, str]:
    try:
        module_version = getattr(module, "__version__", None)
    except Exception:
        return _failure_evidence(module_name, "import_error")
    evidence = _failure_evidence(module_name, "ok")
    if isinstance(module_version, str):
        evidence["module_version"] = module_version
    return evidence


def _import_current_module(module_name: str) -> dict[str, str]:
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        status = "missing_module" if exc.name == module_name else "import_error"
        return _failure_evidence(module_name, status)
    except Exception:
        return _failure_evidence(module_name, "import_error")
    return _module_version_evidence(module_name, module)


def _module_origin_is_in_context(module: object, context: Path) -> bool:
    try:
        spec = getattr(module, "__spec__", None)
        origin = getattr(spec, "origin", None) or getattr(module, "__file__", None)
        if not isinstance(origin, str):
            return False
        return Path(origin).resolve().is_relative_to(context.resolve())
    except Exception:
        return False


def _local_module_roots(context: Path, module_name: str) -> set[str]:
    roots = {module_name.split(".", 1)[0]}
    try:
        roots.update(path.stem for path in context.glob("*.py") if path.stem != "__init__")
        roots.update(
            path.name for path in context.iterdir() if path.is_dir() and (path / "__init__.py").is_file()
        )
    except OSError:
        return roots
    return roots


def _is_root_module(module_name: str, roots: set[str]) -> bool:
    return module_name.split(".", 1)[0] in roots


def _contextual_module_evidence(module_name: str, context: Path) -> dict[str, str]:
    roots = _local_module_roots(context, module_name)
    saved_modules = {
        name: module
        for name, module in sys.modules.items()
        if _is_root_module(name, roots) or _module_origin_is_in_context(module, context)
    }
    original_sys_path = sys.path[:]
    for name in saved_modules:
        sys.modules.pop(name, None)

    try:
        sys.path.insert(0, str(context))
        module = importlib.import_module(module_name)
        if not _module_origin_is_in_context(module, context):
            return _failure_evidence(module_name, "import_error")
        return _module_version_evidence(module_name, module)
    except ModuleNotFoundError as exc:
        status = "missing_module" if exc.name == module_name else "import_error"
        return _failure_evidence(module_name, status)
    except Exception:
        return _failure_evidence(module_name, "import_error")
    finally:
        for name, module in list(sys.modules.items()):
            if _is_root_module(name, roots) or _module_origin_is_in_context(module, context):
                sys.modules.pop(name, None)
        sys.modules.update(saved_modules)
        sys.path[:] = original_sys_path


def _normalise_contexts(context: Path | tuple[Path, ...] | None) -> tuple[Path, ...]:
    if context is None:
        return ()
    if isinstance(context, Path):
        return (context,)
    return tuple(item for item in context if isinstance(item, Path))


def _aggregate_status(statuses: Iterable[str]) -> str:
    values = tuple(statuses)
    if "import_error" in values:
        return "import_error"
    if "missing_module" in values:
        return "missing_module"
    return next((status for status in values if status != "ok"), "ok")


def _module_evidence(
    module_name: str,
    context: Path | tuple[Path, ...] | None = None,
) -> dict[str, object]:
    contexts = _normalise_contexts(context)
    if not contexts:
        return _import_current_module(module_name)

    evidence_by_context = [_contextual_module_evidence(module_name, item) for item in contexts]
    result: dict[str, object] = {
        "module": module_name,
        "status": _aggregate_status(item["status"] for item in evidence_by_context),
        "context_count": len(contexts),
    }
    versions = {
        item["module_version"]
        for item in evidence_by_context
        if isinstance(item.get("module_version"), str)
    }
    if result["status"] == "ok" and len(versions) == 1:
        result["module_version"] = versions.pop()
    return result


def _distribution_evidence(
    label: str,
    distribution_name: str,
    module_name: str,
) -> dict[str, object]:
    try:
        version = metadata.version(distribution_name)
        distribution = {
            "name": distribution_name,
            "status": "ok",
            "version": version,
        }
    except metadata.PackageNotFoundError:
        distribution = {"name": distribution_name, "status": "missing_distribution"}
    except Exception:
        distribution = {
            "name": distribution_name,
            "status": "metadata_error",
        }

    module = _module_evidence(module_name)
    status = module["status"]
    if status == "ok" and distribution["status"] != "ok":
        status = distribution["status"]
    result: dict[str, object] = {
        "label": label,
        "status": status,
        "distribution": distribution,
        "import": module,
    }
    if extra_modules := EXTRA_IMPORTS.get(label):
        extra_imports = [_module_evidence(name) for name in extra_modules]
        result["extra_imports"] = extra_imports
        result["status"] = _aggregate_status(
            [str(status), *(str(item["status"]) for item in extra_imports)]
        )
    return result


def _spacy_model_evidence() -> dict[str, object]:
    module = _module_evidence("en_core_web_sm")
    try:
        version = metadata.version("en-core-web-sm")
        distribution: dict[str, str] = {
            "name": "en-core-web-sm",
            "status": "ok",
            "version": version,
        }
    except metadata.PackageNotFoundError:
        distribution = {
            "name": "en-core-web-sm",
            "status": "missing_distribution",
        }
    except Exception:
        distribution = {
            "name": "en-core-web-sm",
            "status": "metadata_error",
        }
    available = module["status"] == "ok" and distribution["status"] == "ok"
    capability = evaluate_capability(
        kind="asset",
        available=available,
        observed_version=distribution.get("version"),
        expected_version="3.8.0",
    )
    if capability["status"] == "ok":
        try:
            import en_core_web_sm

            en_core_web_sm.load()
        except Exception:
            capability = {"status": "import_error"}
    return {
        "asset": "en_core_web_sm",
        **capability,
        "distribution": distribution,
        "import": module,
    }


def _vader_evidence() -> dict[str, object]:
    module = _module_evidence("nltk.sentiment.vader")
    if module["status"] != "ok":
        return {
            "asset": "vader_lexicon",
            "status": module["status"],
            "import": module,
        }
    try:
        import nltk

        asset = load_nlp_asset_manifest(
            Path(__file__).resolve().parents[1] / "requirements/nlp-assets.toml"
        )
        pointer = nltk.data.find(str(asset.resource))
        resolved = Path(str(pointer))
        resource_by_identity: dict[Path, Path] = {}
        for root in nltk.data.path:
            root_path = Path(root)
            if not root_path.is_absolute():
                continue
            candidate = root_path / Path(*asset.resource.parts)
            if not candidate.exists() and not candidate.is_symlink():
                continue
            if candidate.is_symlink() or not candidate.is_file():
                raise NLPAssetError("destination")
            resource_by_identity.setdefault(candidate.resolve(strict=True), root_path)
        if (
            len(resource_by_identity) != 1
            or resolved.is_symlink()
            or not resolved.is_file()
            or resolved.resolve(strict=True) not in resource_by_identity
        ):
            raise NLPAssetError("destination")
        resource, data_root = next(iter(resource_by_identity.items()))
        if hashlib.sha256(resource.read_bytes()).hexdigest() != asset.sha256:
            raise NLPAssetError("hash")
        verified = verify_vader(asset, data_root)
        observed_size = verified.stat().st_size
        observed_sha = hashlib.sha256(verified.read_bytes()).hexdigest()
    except Exception:
        return {
            "asset": "vader_lexicon",
            "status": "asset_identity_mismatch",
            "import": module,
        }
    return {
        "asset": "vader_lexicon",
        "status": "ok",
        "expected_sha256": asset.sha256,
        "observed_sha256": observed_sha,
        "expected_size": asset.size,
        "observed_size": observed_size,
        "member": str(asset.member),
        "import": module,
    }


def _failed_dependency_evidence(
    label: str,
    distribution_name: str,
    module_name: str,
) -> dict[str, object]:
    return {
        "label": label,
        "status": "import_error",
        "distribution": {"name": distribution_name, "status": "metadata_error"},
        "import": _failure_evidence(module_name, "import_error"),
    }


def _safe_distribution_evidence(
    label: str,
    distribution_name: str,
    module_name: str,
) -> dict[str, object]:
    try:
        return _distribution_evidence(label, distribution_name, module_name)
    except Exception:
        return _failed_dependency_evidence(label, distribution_name, module_name)


def _failed_asset_evidence(asset_name: str) -> dict[str, object]:
    distribution_name, module_name = ASSET_BY_NAME[asset_name]
    result: dict[str, object] = {
        "asset": asset_name,
        "status": "import_error",
        "import": _failure_evidence(module_name, "import_error"),
    }
    if distribution_name is not None:
        result["distribution"] = {"name": distribution_name, "status": "metadata_error"}
    return result


def _safe_asset_evidence(asset_name: str) -> dict[str, object]:
    try:
        if asset_name == "en_core_web_sm":
            return _spacy_model_evidence()
        return _vader_evidence()
    except Exception:
        return _failed_asset_evidence(asset_name)


def build_report(repo_root: Path) -> dict[str, object]:
    """Probe the current interpreter and return path-free runtime evidence."""
    dependencies = [
        _safe_distribution_evidence(label, distribution, module)
        for label, distribution, module in DEPENDENCIES
    ]
    assets = [_safe_asset_evidence(name) for name in ASSET_BY_NAME]

    try:
        documents, local_contexts = _active_notebook_documents(repo_root)
        modules = _runtime_notebook_imports(repo_root, documents)
        imports = [
            _module_evidence(module, local_contexts.get(module)) for module in modules
        ]
        notebook_imports: dict[str, object] = {
            "status": (
                "ok" if all(item["status"] == "ok" for item in imports) else "failed"
            ),
            "module_count": len(imports),
            "imports": imports,
        }
    except Exception:
        notebook_imports = {
            "status": "scan_error",
            "module_count": 0,
            "imports": [],
        }

    failed_dependencies = sum(item["status"] != "ok" for item in dependencies)
    failed_assets = sum(item["status"] != "ok" for item in assets)
    imports = notebook_imports.get("imports", [])
    failed_imports = sum(
        item.get("status") != "ok" for item in imports if isinstance(item, Mapping)
    )
    if notebook_imports["status"] == "scan_error":
        failed_imports += 1
    failed = failed_dependencies + failed_assets + failed_imports

    return {
        "schema_version": SCHEMA_VERSION,
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "dependencies": dependencies,
        "assets": assets,
        "notebook_imports": notebook_imports,
        "summary": {
            "status": "ok" if failed == 0 else "failed",
            "failed_capability_count": failed,
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture safe Atlas runtime compatibility evidence."
    )
    parser.add_argument("--json", required=True, type=Path, dest="json_path")
    return parser.parse_args(argv)


def probe_exit_code(report: Mapping[str, object]) -> int:
    summary = report.get("summary")
    return 0 if isinstance(summary, Mapping) and summary.get("status") == "ok" else 1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    report = build_report(repo_root)
    args.json_path.write_text(serialize_report(report), encoding="utf-8")
    failed = report["summary"]["failed_capability_count"]  # type: ignore[index]
    print(f"Atlas runtime probe: {report['summary']['status']} ({failed} failed checks)")
    return probe_exit_code(report)


if __name__ == "__main__":
    raise SystemExit(main())
