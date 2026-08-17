from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "notebooks/sentiment_classification-vader-mlp-pytorch/notebook.ipynb"


def _make_recipe(target: str) -> tuple[str, ...]:
    lines = (REPO / "Makefile").read_text(encoding="utf-8").splitlines()
    header = next(index for index, line in enumerate(lines) if line.startswith(f"{target}:"))
    recipe: list[str] = []
    for line in lines[header + 1 :]:
        if line.startswith("\t"):
            recipe.append(line[1:])
            continue
        if line.strip() and not line.startswith("#"):
            break
    return tuple(recipe)


def _workflow() -> dict:
    return yaml.safe_load((REPO / ".github/workflows/ci.yml").read_text(encoding="utf-8"))


def _step(job: dict, name: str) -> dict:
    matches = [step for step in job["steps"] if step.get("name") == name]
    assert len(matches) == 1
    return matches[0]


def _notebook_code() -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


def _qualified_calls(source: str) -> set[str]:
    tree = ast.parse(source)
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                aliases[alias.asname or alias.name] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"

    def name(node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            root = name(node.value)
            return f"{root}.{node.attr}" if root else None
        return None

    return {
        qualified
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and (qualified := name(node.func)) is not None
    }


def test_makefile_uses_one_installer_and_one_offline_verifier() -> None:
    assert _make_recipe("nlp-assets") == ("$(PYTHON) -m scripts.nlp_assets install",)
    assert _make_recipe("verify-nlp-assets") == (
        "$(PYTHON) -m scripts.nlp_assets verify",
    )
    assert _make_recipe("codespace-setup") == (
        "$(MAKE) nlp-assets",
        "$(MAKE) verify-nlp-assets",
        "$(PYTHON) -m pip check",
        "$(MAKE) verify-torch-stack",
        "$(MAKE) verify-nnx-install",
    )
    makefile = (REPO / "Makefile").read_text(encoding="utf-8")
    assert "verify-nlp-assets" in next(
        line for line in makefile.splitlines() if line.startswith(".PHONY:")
    )


def test_docker_installs_then_verifies_assets_before_environment_gates() -> None:
    dockerfile = (REPO / "Dockerfile").read_text(encoding="utf-8")
    order = (
        "make install-torch-stack",
        "make nlp-assets",
        "make verify-nlp-assets",
        "python -m pip check",
        "python -m scripts.verify_torch_stack",
        "python -m scripts.verify_nnx_install",
    )
    positions = [dockerfile.index(item) for item in order]
    assert positions == sorted(positions)
    assert dockerfile.count("make nlp-assets") == 1
    assert dockerfile.count("make verify-nlp-assets") == 1


def test_tier_a_installs_then_verifies_assets_before_workload() -> None:
    job = _workflow()["jobs"]["tier-a-papermill"]
    install = _step(job, "Install dependencies")
    run = install["run"].splitlines()
    assert run == [
        "make install-torch-stack",
        "make nlp-assets",
        "make verify-nlp-assets",
    ]
    assert job["steps"].index(install) < job["steps"].index(
        _step(job, "Check and verify canonical Torch and NNx stack")
    ) < job["steps"].index(_step(job, "Run Tier-A notebooks (papermill)"))
    for key in ("continue-on-error", "shell"):
        assert key not in install


def test_active_sentiment_notebook_requires_preinstalled_asset_without_network() -> None:
    code = _notebook_code()
    assert 'nltk.data.find("sentiment/vader_lexicon.zip")' in code
    assert "make nlp-assets" in code
    assert "nltk.download" not in _qualified_calls(code)
    assert "nltk.downloader" not in code
    assert "spacy download" not in code.lower()
    assert "getattr(nltk, \"download\")" not in code


def test_sentiment_spec_describes_verified_preinstall() -> None:
    spec = yaml.safe_load(
        (REPO / "notebooks/sentiment_classification-vader-mlp-pytorch/docs/spec.yaml").read_text(
            encoding="utf-8"
        )
    )
    walkthrough = {item["anchor"]: item["note"] for item in spec["code_walkthrough"]}
    assert walkthrough['nltk.data.find("sentiment/vader_lexicon.zip")'] == (
        "requires the hash-verified asset installed by make nlp-assets"
    )
    assert all("download" not in item.lower() for item in spec["pitfalls"])


def test_current_consumers_contain_no_legacy_asset_downloader() -> None:
    paths = (
        REPO / "Makefile",
        REPO / "Dockerfile",
        REPO / ".github/workflows/ci.yml",
        REPO / ".devcontainer/devcontainer.json",
        NOTEBOOK,
    )
    forbidden = (
        "nltk.download(",
        "python -m nltk.downloader",
        "spacy download en_core_web_sm",
    )
    for path in paths:
        text = path.read_text(encoding="utf-8").lower()
        for token in forbidden:
            assert token not in text, (path, token)


def test_lock_verifier_and_repository_gate_consume_the_manifest() -> None:
    dependency = (REPO / "scripts/verify_dependency_locks.py").read_text(encoding="utf-8")
    repository = (REPO / "scripts/verify_repo.py").read_text(encoding="utf-8")
    assert "load_manifest" in dependency
    assert "requirements/nlp-assets.toml" in dependency
    assert "D11.nlp_asset_contract" in repository
    assert re.search(r'["\']assets["\']', repository)


def test_atlas_probe_reports_exact_vader_identity_and_spacy_version() -> None:
    source = (REPO / "scripts/atlas_runtime_probe.py").read_text(encoding="utf-8")
    for token in (
        "expected_sha256",
        "observed_sha256",
        "expected_size",
        "observed_size",
        "asset_identity_mismatch",
        'expected_version="3.8.0"',
        "en_core_web_sm.load",
    ):
        assert token in source
