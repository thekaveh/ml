#!/usr/bin/env python3
"""Capture safe, machine-readable evidence about an Atlas Python runtime."""

from __future__ import annotations

import argparse
import ast
import importlib
import importlib.metadata as metadata
import json
import platform
import re
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path


SCHEMA_VERSION = 1
NON_PYTHON_CELL_MAGICS = frozenset(
    {"bash", "html", "javascript", "js", "latex", "perl", "ruby", "sh", "svg"}
)
UNSAFE_KEY_PARTS = frozenset(
    {
        "connection",
        "credential",
        "credentials",
        "cwd",
        "directory",
        "env",
        "environment",
        "home",
        "host",
        "password",
        "path",
        "secret",
        "token",
        "uri",
        "url",
        "username",
        "workspace",
    }
)
SECRET_VALUE_RE = re.compile(
    r"(?:token|password|passwd|secret|credential|api[_-]?key)\s*[=:]",
    re.IGNORECASE,
)
UNSAFE_VALUE = object()

DEPENDENCIES = (
    ("thekaveh-nnx[lm]", "thekaveh-nnx", "nnx"),
    ("nnx", "thekaveh-nnx", "nnx"),
    ("Torch", "torch", "torch"),
    ("TorchVision", "torchvision", "torchvision"),
    ("TorchAudio", "torchaudio", "torchaudio"),
    ("TorchAO", "torchao", "torchao"),
    ("PyTorch Geometric", "torch-geometric", "torch_geometric"),
    ("python-louvain", "python-louvain", "community"),
    ("spaCy", "spacy", "spacy"),
    ("NLTK", "nltk", "nltk"),
)
EXTRA_IMPORTS = {"thekaveh-nnx[lm]": ("datasets", "tokenizers")}


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


def _unsafe_key(key: object) -> bool:
    words = set(re.findall(r"[a-z0-9]+", str(key).lower()))
    return bool(words & UNSAFE_KEY_PARTS)


def _unsafe_string(value: str) -> bool:
    return (
        "/" in value
        or "\\" in value
        or value.startswith("~")
        or "://" in value
        or SECRET_VALUE_RE.search(value) is not None
    )


def _safe_value(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return UNSAFE_VALUE if _unsafe_string(value) else value
    if isinstance(value, Path):
        return UNSAFE_VALUE
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if _unsafe_key(key):
                continue
            safe_item = _safe_value(item)
            if safe_item is not UNSAFE_VALUE:
                result[str(key)] = safe_item
        return result
    if isinstance(value, (list, tuple)):
        return [
            safe_item
            for item in value
            if (safe_item := _safe_value(item)) is not UNSAFE_VALUE
        ]
    return UNSAFE_VALUE


def serialize_report(report: Mapping[str, object]) -> str:
    """Serialize an allowlisted value vocabulary, dropping sensitive material."""
    safe_report = _safe_value(report)
    if not isinstance(safe_report, dict):
        raise TypeError("report must serialize to a JSON object")
    return json.dumps(safe_report, indent=2, sort_keys=True) + "\n"


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
) -> tuple[list[Mapping[str, object]], dict[str, Path]]:
    documents: list[Mapping[str, object]] = []
    local_contexts: dict[str, Path] = {}
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
                local_contexts.setdefault(module, notebook_path.parent)
    return documents, local_contexts


def _module_evidence(module_name: str, context: Path | None = None) -> dict[str, str]:
    inserted_context = False
    if context is not None:
        sys.path.insert(0, str(context))
        inserted_context = True
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        return {
            "module": module_name,
            **evaluate_capability(kind="module", available=False),
            "error_type": type(exc).__name__,
        }
    except Exception as exc:
        return {
            "module": module_name,
            "status": "import_error",
            "error_type": type(exc).__name__,
        }
    finally:
        if inserted_context:
            sys.path.pop(0)

    evidence = {
        "module": module_name,
        **evaluate_capability(kind="module", available=True),
    }
    module_version = getattr(module, "__version__", None)
    if isinstance(module_version, str):
        evidence["module_version"] = module_version
    return evidence


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
    except Exception as exc:
        distribution = {
            "name": distribution_name,
            "status": "metadata_error",
            "error_type": type(exc).__name__,
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
        if status == "ok" and any(item["status"] != "ok" for item in extra_imports):
            result["status"] = "missing_module"
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
    except Exception as exc:
        distribution = {
            "name": "en-core-web-sm",
            "status": "metadata_error",
            "error_type": type(exc).__name__,
        }
    available = module["status"] == "ok" and distribution["status"] == "ok"
    return {
        "asset": "en_core_web_sm",
        **evaluate_capability(kind="asset", available=available),
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

        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.data.find("sentiment/vader_lexicon")
    except Exception as exc:
        return {
            "asset": "vader_lexicon",
            **evaluate_capability(kind="asset", available=False),
            "import": module,
            "error_type": type(exc).__name__,
        }
    return {"asset": "vader_lexicon", "status": "ok", "import": module}


def build_report(repo_root: Path) -> dict[str, object]:
    """Probe the current interpreter and return path-free runtime evidence."""
    dependencies = [
        _distribution_evidence(label, distribution, module)
        for label, distribution, module in DEPENDENCIES
    ]
    assets = [_spacy_model_evidence(), _vader_evidence()]

    try:
        documents, local_contexts = _active_notebook_documents(repo_root)
        modules = extract_notebook_imports(documents)
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
    except Exception as exc:
        notebook_imports = {
            "status": "scan_error",
            "module_count": 0,
            "imports": [],
            "error_type": type(exc).__name__,
        }

    failed_dependencies = sum(item["status"] != "ok" for item in dependencies)
    failed_assets = sum(item["status"] != "ok" for item in assets)
    failed_imports = sum(
        item["status"] != "ok" for item in notebook_imports["imports"]  # type: ignore[union-attr]
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
