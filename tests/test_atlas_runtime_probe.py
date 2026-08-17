from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path, PurePosixPath
from types import ModuleType

import pytest

from scripts import atlas_runtime_probe as probe
from scripts.atlas_runtime_probe import (
    evaluate_capability,
    extract_notebook_imports,
    probe_exit_code,
    serialize_report,
)
from scripts.nlp_assets import VaderAsset


def _write_vader_archive(
    path: Path,
    *,
    member: str = "vader_lexicon/vader_lexicon.txt",
    contents: str = "good\t2.0\t0.0\t[2]\nbad\t-2.0\t0.0\t[-2]",
) -> VaderAsset:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(member, contents)
    payload = path.read_bytes()
    return VaderAsset(
        url=(
            "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/"
            "packages/sentiment/vader_lexicon.zip"
        ),
        sha256=hashlib.sha256(payload).hexdigest(),
        size=len(payload),
        resource=PurePosixPath("sentiment/vader_lexicon.zip"),
        member=PurePosixPath(member),
        license="MIT License",
    )


@pytest.mark.parametrize(
    ("kind", "available", "observed_version", "expected_version", "expected_status"),
    [
        ("module", False, None, None, "missing_module"),
        ("asset", False, None, None, "missing_asset"),
        ("module", True, "2.10.0", "2.11.0", "version_mismatch"),
    ],
)
def test_evaluator_distinguishes_missing_capabilities_from_version_mismatch(
    kind: str,
    available: bool,
    observed_version: str | None,
    expected_version: str | None,
    expected_status: str,
) -> None:
    result = evaluate_capability(
        kind=kind,
        available=available,
        observed_version=observed_version,
        expected_version=expected_version,
    )

    assert result["status"] == expected_status


def test_serialization_omits_secrets_environment_and_paths() -> None:
    report = {
        "schema_version": 1,
        "summary": {"status": "failed"},
        "connection_token": "jupyter-token-value",
        "credentials": {"username": "jovyan", "password": "password-value"},
        "environment": {"JUPYTERHUB_TOKEN": "env-token-value"},
        "home": "/home/jovyan",
        "workspace": "/home/jovyan/work/ml-eng-lab",
        "nested": {
            "safe": "missing_module",
            "url": "http://127.0.0.1:8888/?token=query-token-value",
            "source": "notebooks/task/notebook.ipynb",
        },
    }

    serialized = serialize_report(report)
    parsed = json.loads(serialized)

    assert parsed == {
        "schema_version": 1,
        "summary": {"status": "failed"},
    }
    for forbidden in (
        "jupyter-token-value",
        "password-value",
        "env-token-value",
        "/home/jovyan",
        "notebooks/task",
        "query-token-value",
    ):
        assert forbidden not in serialized


def test_serialization_projects_only_known_safe_schema_fields() -> None:
    report = {
        "schema_version": 1,
        "api_key": "top-level-secret",
        "filename": ".env",
        "python": {
            "implementation": "CPython",
            "version": "3.11.0",
            "api_key": "python-secret",
        },
        "dependencies": [
            {
                "label": "Torch",
                "status": "ok",
                "filename": ".env",
                "distribution": {
                    "name": "torch",
                    "status": "ok",
                    "version": "2.11.0",
                    "api_key": "distribution-secret",
                },
                "import": {
                    "module": "torch",
                    "status": "ok",
                    "exception_payload": {"message": "exception-secret"},
                },
            }
        ],
        "notebook_imports": {
            "status": "ok",
            "module_count": 1,
            "imports": [
                {
                    "module": "torch",
                    "status": "ok",
                    "filename": ".env",
                }
            ],
        },
        "summary": {
            "status": "ok",
            "failed_capability_count": 0,
            "exception_payload": "summary-secret",
        },
    }

    serialized = serialize_report(report)

    assert json.loads(serialized) == {
        "schema_version": 1,
        "python": {"implementation": "CPython", "version": "3.11.0"},
        "dependencies": [
            {
                "label": "Torch",
                "status": "ok",
                "distribution": {"name": "torch", "status": "ok", "version": "2.11.0"},
                "import": {"module": "torch", "status": "ok"},
            }
        ],
        "notebook_imports": {
            "status": "ok",
            "module_count": 1,
            "imports": [{"module": "torch", "status": "ok"}],
        },
        "summary": {"status": "ok", "failed_capability_count": 0},
    }
    for forbidden in (
        "top-level-secret",
        "python-secret",
        "distribution-secret",
        "exception-secret",
        "summary-secret",
        ".env",
    ):
        assert forbidden not in serialized


def test_notebook_import_extraction_is_deterministic_and_ignores_outputs() -> None:
    notebooks = [
        {
            "cells": [
                {
                    "cell_type": "code",
                    "source": [
                        "%matplotlib inline\n",
                        "import torch, torchvision as vision\n",
                        "from nnx.nn.dataset.nn_dataset import NNDataset\n",
                    ],
                    "outputs": [
                        {
                            "output_type": "stream",
                            "text": "import leaked_output\n/home/jovyan/private.py",
                        }
                    ],
                }
            ]
        },
        {
            "cells": [
                {
                    "cell_type": "code",
                    "source": "from nltk.sentiment.vader import SentimentIntensityAnalyzer\nimport torch\n",
                }
            ]
        },
    ]

    assert extract_notebook_imports(notebooks) == (
        "nltk.sentiment.vader",
        "nnx.nn.dataset.nn_dataset",
        "torch",
        "torchvision",
    )


@pytest.mark.parametrize(
    ("summary_status", "expected_exit_code"),
    [("ok", 0), ("failed", 1)],
)
def test_probe_exit_code_distinguishes_success_from_failed_capability(
    summary_status: str,
    expected_exit_code: int,
) -> None:
    assert probe_exit_code({"summary": {"status": summary_status}}) == expected_exit_code


@pytest.mark.parametrize(
    ("missing_name", "expected_status"),
    [
        ("target_module", "missing_module"),
        ("transitive_dependency", "import_error"),
    ],
)
def test_module_evidence_distinguishes_target_and_transitive_missing_modules(
    monkeypatch, missing_name: str, expected_status: str
) -> None:
    def missing_transitive_dependency(_: str) -> ModuleType:
        raise ModuleNotFoundError(
            f"No module named {missing_name!r}", name=missing_name
        )

    monkeypatch.setattr(probe.importlib, "import_module", missing_transitive_dependency)

    assert probe._module_evidence("target_module")["status"] == expected_status


def test_extra_import_error_is_not_collapsed_to_missing_module(monkeypatch) -> None:
    def evidence(module: str, context=None) -> dict[str, str]:
        if module == "datasets":
            return {"module": module, "status": "import_error"}
        return {"module": module, "status": "ok"}

    monkeypatch.setattr(probe, "_module_evidence", evidence)
    monkeypatch.setattr(probe.metadata, "version", lambda _: "0.2.0")

    result = probe._distribution_evidence("thekaveh-nnx[lm]", "thekaveh-nnx", "nnx")

    assert result["status"] == "import_error"


def test_local_sibling_import_uses_its_context_despite_a_cached_module(
    tmp_path, monkeypatch
) -> None:
    module_name = "atlas_probe_cached_collision"
    context = tmp_path / "notebook"
    context.mkdir()
    (context / f"{module_name}.py").write_text('__version__ = "1.2.3"\n', encoding="utf-8")
    cached = ModuleType(module_name)
    cached.__file__ = str(tmp_path / "third_party.py")
    cached.__version__ = "9.9.9"
    monkeypatch.setitem(sys.modules, module_name, cached)

    result = probe._module_evidence(module_name, context)

    assert result["status"] == "ok"
    assert result["module_version"] == "1.2.3"
    assert sys.modules[module_name] is cached


def test_local_sibling_import_rejects_an_origin_outside_its_context(
    tmp_path, monkeypatch
) -> None:
    module_name = "atlas_probe_foreign_origin"
    context = tmp_path / "notebook"
    context.mkdir()
    foreign = ModuleType(module_name)
    foreign.__file__ = str(tmp_path / "third_party.py")
    foreign.__version__ = "1.2.3"
    monkeypatch.setattr(probe.importlib, "import_module", lambda _: foreign)

    assert probe._module_evidence(module_name, context)["status"] == "import_error"


def test_local_sibling_evidence_checks_each_context(tmp_path) -> None:
    module_name = "atlas_probe_multiple_contexts"
    first_context = tmp_path / "first"
    second_context = tmp_path / "second"
    for context, version in ((first_context, "1.0.0"), (second_context, "2.0.0")):
        context.mkdir()
        (context / f"{module_name}.py").write_text(
            f'__version__ = "{version}"\n', encoding="utf-8"
        )

    result = probe._module_evidence(module_name, (first_context, second_context))

    assert result["status"] == "ok"
    assert result["context_count"] == 2
    assert module_name not in sys.modules


def test_active_notebook_discovery_keeps_all_local_sibling_contexts(tmp_path) -> None:
    module_name = "atlas_probe_shared_sibling"
    repo = tmp_path / "repo"
    contexts = []
    for task in ("first", "second"):
        context = repo / "notebooks" / task
        context.mkdir(parents=True)
        contexts.append(context)
        (context / f"{module_name}.py").write_text("VALUE = 1\n", encoding="utf-8")
        (context / "notebook.ipynb").write_text(
            json.dumps(
                {"cells": [{"cell_type": "code", "source": f"import {module_name}\n"}]}
            ),
            encoding="utf-8",
        )

    _, local_contexts = probe._active_notebook_documents(repo)

    assert local_contexts[module_name] == tuple(contexts)


def _configure_vader_probe(
    monkeypatch: pytest.MonkeyPatch,
    *,
    asset: VaderAsset,
    roots: list[Path],
    pointer: Path,
) -> None:
    import nltk

    monkeypatch.setattr(
        probe,
        "_module_evidence",
        lambda module: {"module": module, "status": "ok"},
    )
    monkeypatch.setattr(probe, "load_nlp_asset_manifest", lambda _: asset)
    monkeypatch.setattr(nltk.data, "path", [str(root) for root in roots])
    original_find = nltk.data.find
    monkeypatch.setattr(
        nltk.data,
        "find",
        lambda resource: (
            str(pointer) if str(resource) == str(asset.resource) else original_find(resource)
        ),
    )


def test_vader_evidence_reports_verified_archive_identity(tmp_path, monkeypatch) -> None:
    root = tmp_path / "nltk-data"
    archive = root / "sentiment/vader_lexicon.zip"
    asset = _write_vader_archive(archive)
    _configure_vader_probe(monkeypatch, asset=asset, roots=[root], pointer=archive)

    evidence = probe._vader_evidence()

    assert evidence == {
        "asset": "vader_lexicon",
        "status": "ok",
        "expected_sha256": asset.sha256,
        "observed_sha256": asset.sha256,
        "expected_size": asset.size,
        "observed_size": asset.size,
        "member": "vader_lexicon/vader_lexicon.txt",
        "import": {"module": "nltk.sentiment.vader", "status": "ok"},
    }


@pytest.mark.parametrize("mutation", ("wrong_hash", "wrong_size", "malformed", "smoke"))
def test_vader_evidence_rejects_corrupt_archive_identity(
    tmp_path, monkeypatch, mutation: str
) -> None:
    root = tmp_path / "nltk-data"
    archive = root / "sentiment/vader_lexicon.zip"
    if mutation == "malformed":
        archive.parent.mkdir(parents=True)
        archive.write_bytes(b"not-a-zip")
        payload = archive.read_bytes()
        asset = VaderAsset(
            url="https://example.invalid/vader.zip",
            sha256=hashlib.sha256(payload).hexdigest(),
            size=len(payload),
            resource=PurePosixPath("sentiment/vader_lexicon.zip"),
            member=PurePosixPath("vader_lexicon/vader_lexicon.txt"),
            license="MIT License",
        )
    else:
        contents = "neutral\t0.0\t0.0\t[0]\n" if mutation == "smoke" else (
            "good\t2.0\t0.0\t[2]\nbad\t-2.0\t0.0\t[-2]"
        )
        asset = _write_vader_archive(archive, contents=contents)
    if mutation == "wrong_hash":
        asset = VaderAsset(
            asset.url,
            "0" * 64,
            asset.size,
            asset.resource,
            asset.member,
            asset.license,
        )
    elif mutation == "wrong_size":
        asset = VaderAsset(
            asset.url,
            asset.sha256,
            asset.size + 1,
            asset.resource,
            asset.member,
            asset.license,
        )
    _configure_vader_probe(monkeypatch, asset=asset, roots=[root], pointer=archive)

    assert probe._vader_evidence()["status"] == "asset_identity_mismatch"


@pytest.mark.parametrize("mutation", ("symlink", "directory", "escape", "duplicate"))
def test_vader_evidence_rejects_unsafe_or_ambiguous_resource_paths(
    tmp_path, monkeypatch, mutation: str
) -> None:
    first = tmp_path / "first"
    archive = first / "sentiment/vader_lexicon.zip"
    if mutation == "symlink":
        target = tmp_path / "outside.zip"
        asset = _write_vader_archive(target)
        archive.parent.mkdir(parents=True)
        archive.symlink_to(target)
        roots = [first]
        pointer = archive
    elif mutation == "directory":
        archive.mkdir(parents=True)
        asset = VaderAsset(
            "https://example.invalid/vader.zip",
            "0" * 64,
            1,
            PurePosixPath("sentiment/vader_lexicon.zip"),
            PurePosixPath("vader_lexicon/vader_lexicon.txt"),
            "MIT License",
        )
        roots = [first]
        pointer = archive
    elif mutation == "escape":
        pointer = tmp_path / "outside.zip"
        asset = _write_vader_archive(pointer)
        roots = [first]
    else:
        asset = _write_vader_archive(archive)
        second = tmp_path / "second"
        duplicate = second / "sentiment/vader_lexicon.zip"
        duplicate.parent.mkdir(parents=True)
        duplicate.write_bytes(b"different")
        roots = [first, second]
        pointer = archive
    _configure_vader_probe(monkeypatch, asset=asset, roots=roots, pointer=pointer)

    assert probe._vader_evidence()["status"] == "asset_identity_mismatch"


def test_vader_projection_requires_exact_hash_size_and_member() -> None:
    digest = "a" * 64
    raw = {
        "asset": "vader_lexicon",
        "status": "ok",
        "expected_sha256": digest,
        "observed_sha256": digest,
        "expected_size": 90486,
        "observed_size": 90486,
        "member": "vader_lexicon/vader_lexicon.txt",
        "import": {"module": "nltk.sentiment.vader", "status": "ok"},
    }
    assert probe._project_asset(raw, "vader_lexicon")["status"] == "ok"
    for key, value in (
        ("observed_sha256", "b" * 64),
        ("observed_size", 90485),
        ("member", "vader_lexicon/other.txt"),
    ):
        mutated = dict(raw)
        mutated[key] = value
        assert (
            probe._project_asset(mutated, "vader_lexicon")["status"]
            == "asset_identity_mismatch"
        )


def test_vader_projection_preserves_a_missing_import_status() -> None:
    projected = probe._project_asset(
        {
            "asset": "vader_lexicon",
            "status": "missing_module",
            "import": {"module": "nltk.sentiment.vader", "status": "missing_module"},
        },
        "vader_lexicon",
    )
    assert projected["status"] == "missing_module"


def test_spacy_asset_requires_exact_version_and_load_success(monkeypatch) -> None:
    model = ModuleType("en_core_web_sm")
    model.load = lambda: object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "en_core_web_sm", model)
    monkeypatch.setattr(
        probe,
        "_module_evidence",
        lambda module: {"module": module, "status": "ok"},
    )
    monkeypatch.setattr(probe.metadata, "version", lambda _: "3.8.0")
    assert probe._spacy_model_evidence()["status"] == "ok"

    monkeypatch.setattr(probe.metadata, "version", lambda _: "3.7.0")
    assert probe._spacy_model_evidence()["status"] == "version_mismatch"

    monkeypatch.setattr(probe.metadata, "version", lambda _: "3.8.0")
    model.load = lambda: (_ for _ in ()).throw(RuntimeError("load failed"))  # type: ignore[attr-defined]
    assert probe._spacy_model_evidence()["status"] == "import_error"


def test_main_writes_json_when_module_version_access_raises(tmp_path, monkeypatch) -> None:
    class VersionRaises(ModuleType):
        def __getattr__(self, name: str) -> object:
            if name == "__version__":
                raise RuntimeError("version lookup must not prevent evidence")
            raise AttributeError(name)

    output = tmp_path / "probe.json"
    monkeypatch.setattr(probe, "DEPENDENCIES", (("Torch", "torch", "torch"),))
    monkeypatch.setattr(probe, "EXTRA_IMPORTS", {})
    monkeypatch.setattr(probe.metadata, "version", lambda _: "2.11.0")
    monkeypatch.setattr(probe.importlib, "import_module", lambda name: VersionRaises(name))
    monkeypatch.setattr(
        probe,
        "_spacy_model_evidence",
        lambda: {
            "asset": "en_core_web_sm",
            "status": "ok",
            "distribution": {"name": "en-core-web-sm", "status": "ok", "version": "3.8.0"},
            "import": {"module": "en_core_web_sm", "status": "ok"},
        },
    )
    monkeypatch.setattr(
        probe,
        "_vader_evidence",
        lambda: {
            "asset": "vader_lexicon",
            "status": "ok",
            "import": {"module": "nltk.sentiment.vader", "status": "ok"},
        },
    )
    monkeypatch.setattr(probe, "_active_notebook_documents", lambda _: ([], {}))

    try:
        exit_code = probe.main(["--json", str(output)])
    except RuntimeError:
        exit_code = None

    assert exit_code == 1
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["dependencies"] == [
        {
            "label": "Torch",
            "status": "import_error",
            "distribution": {"name": "torch", "status": "ok", "version": "2.11.0"},
            "import": {"module": "torch", "status": "import_error"},
        }
    ]


def test_main_writes_json_when_a_dependency_probe_raises(tmp_path, monkeypatch) -> None:
    def raise_probe_failure(*_) -> dict[str, object]:
        raise RuntimeError("probe failure")

    output = tmp_path / "probe.json"
    monkeypatch.setattr(probe, "DEPENDENCIES", (("Torch", "torch", "torch"),))
    monkeypatch.setattr(probe, "EXTRA_IMPORTS", {})
    monkeypatch.setattr(probe, "_distribution_evidence", raise_probe_failure)
    monkeypatch.setattr(
        probe,
        "_spacy_model_evidence",
        lambda: {"asset": "en_core_web_sm", "status": "ok"},
    )
    monkeypatch.setattr(
        probe,
        "_vader_evidence",
        lambda: {"asset": "vader_lexicon", "status": "ok"},
    )
    monkeypatch.setattr(probe, "_active_notebook_documents", lambda _: ([], {}))

    assert probe.main(["--json", str(output)]) == 1
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["dependencies"][0]["status"] == "import_error"
