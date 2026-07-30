from __future__ import annotations

import json

import pytest

from scripts.atlas_runtime_probe import (
    evaluate_capability,
    extract_notebook_imports,
    probe_exit_code,
    serialize_report,
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
        "nested": {"safe": "missing_module"},
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
