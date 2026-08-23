"""NLP asset contract validation."""

from __future__ import annotations

import ast
import json
from pathlib import Path

try:
    import yaml as _yaml
except ImportError:
    _yaml = None

from .models import CheckResult, Finding, VerifierConfig


def _qualified_name(node: ast.AST, aliases: dict[str, str]) -> str | None:
    if isinstance(node, ast.Name):
        return aliases.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        root = _qualified_name(node.value, aliases)
        return f"{root}.{node.attr}" if root is not None else None
    return None


def _notebook_nlp_asset_contract(code_cells: list[str]) -> bool:
    aliases: dict[str, str] = {}
    trees: list[ast.Module] = []
    for source in code_cells:
        for line in source.splitlines():
            stripped = line.lstrip().lower()
            if stripped.startswith(("!", "%")) and "download" in stripped:
                return False
        parse_source = "\n".join("" if line.lstrip().startswith(("!", "%")) else line for line in source.splitlines())
        try:
            tree = ast.parse(parse_source)
        except SyntaxError:
            return False
        trees.append(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for item in node.names:
                    aliases[item.asname or item.name] = item.name
            elif isinstance(node, ast.ImportFrom) and node.module:
                for item in node.names:
                    aliases[item.asname or item.name] = f"{node.module}.{item.name}"

    find_calls = 0
    has_guidance = False
    for tree in trees:
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                has_guidance |= "make nlp-assets" in node.value
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
                if any(_qualified_name(target, aliases) == "nltk.download" for target in targets):
                    return False
            if not isinstance(node, ast.Call):
                continue
            qualified = _qualified_name(node.func, aliases)
            if qualified in {"nltk.download", "nltk.downloader.download"}:
                return False
            if qualified in {"getattr", "setattr"} and len(node.args) >= 2:
                receiver = _qualified_name(node.args[0], aliases)
                member = node.args[1]
                if receiver == "nltk" and isinstance(member, ast.Constant) and member.value == "download":
                    return False
            if qualified == "nltk.data.find" and len(node.args) == 1:
                argument = node.args[0]
                if isinstance(argument, ast.Constant) and argument.value == "sentiment/vader_lexicon.zip":
                    find_calls += 1
    return find_calls == 1 and has_guidance


def _nlp_asset_contract_findings(repo: Path) -> list[Finding]:
    """Validate current NLP asset consumers without network access."""

    def failed(location: str, message: str) -> list[Finding]:
        return [
            Finding(id="D11.nlp_asset_contract", check="assets", severity="error", location=location, message=message)
        ]

    try:
        from scripts.nlp_assets import load_manifest

        asset = load_manifest(repo / "requirements/nlp-assets.toml")
        makefile = (repo / "Makefile").read_text(encoding="utf-8")
        dockerfile = (repo / "Dockerfile").read_text(encoding="utf-8")
        workflow_source = (repo / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        devcontainer = (repo / ".devcontainer/devcontainer.json").read_text(encoding="utf-8")
        notebook_path = repo / "notebooks/sentiment_classification-vader-mlp-pytorch/notebook.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        code_cells = ["".join(cell.get("source", [])) for cell in notebook["cells"] if cell.get("cell_type") == "code"]
        code = "\n".join(code_cells)
        dependency_verifier = (repo / "scripts/verify_dependency_locks.py").read_text(encoding="utf-8")
        model_requirement = (repo / "nlp-model-requirements.txt").read_text(encoding="utf-8")
        runtime_probe = (repo / "scripts/atlas_runtime_probe.py").read_text(encoding="utf-8")
        workflow = _yaml.safe_load(workflow_source) if _yaml is not None else None
    except Exception:
        return failed("requirements/nlp-assets.toml", "NLP asset contract is unreadable")
    if (
        asset.sha256 != "8adba4294eef3964d820bf655e37e61bdc3a341994356af59b74fb3b4a36ce5c"
        or asset.size != 90486
        or str(asset.resource) != "sentiment/vader_lexicon.zip"
        or str(asset.member) != "vader_lexicon/vader_lexicon.txt"
    ):
        return failed("requirements/nlp-assets.toml", "VADER identity drifted")
    expected_make = (
        "nlp-assets:\n\t$(PYTHON) -m scripts.nlp_assets install",
        "verify-nlp-assets:\n\t$(PYTHON) -m scripts.nlp_assets verify",
        "codespace-setup: install-torch-stack\n\t$(MAKE) nlp-assets\n\t$(MAKE) verify-nlp-assets\n\t$(PYTHON) -m pip check\n\t$(MAKE) verify-torch-stack\n\t$(MAKE) verify-nnx-install",
    )
    if any(makefile.count(block) != 1 for block in expected_make):
        return failed("Makefile", "NLP asset Make contract drifted")
    docker_order = (
        "make install-torch-stack",
        "make nlp-assets",
        "make verify-nlp-assets",
        "python -m pip check",
        "python -m scripts.verify_torch_stack",
        "python -m scripts.verify_nnx_install",
    )
    if any(item not in dockerfile for item in docker_order) or [
        dockerfile.index(item) for item in docker_order
    ] != sorted(dockerfile.index(item) for item in docker_order):
        return failed("Dockerfile", "NLP asset Docker order drifted")
    try:
        tier_a = workflow["jobs"]["tier-a-papermill"]
        install_steps = [step for step in tier_a["steps"] if step.get("name") == "Install dependencies"]
        if len(install_steps) != 1 or install_steps[0].get("run", "").splitlines() != [
            "make install-torch-stack",
            "make nlp-assets",
            "make verify-nlp-assets",
        ]:
            raise ValueError
    except (KeyError, TypeError, ValueError):
        return failed(".github/workflows/ci.yml", "NLP asset CI order drifted")
    if '"postCreateCommand": "make codespace-setup"' not in devcontainer:
        return failed(".devcontainer/devcontainer.json", "Codespaces asset contract drifted")
    forbidden = ("nltk.download(", "python -m nltk.downloader", "spacy download en_core_web_sm")
    for location, source in (
        ("Makefile", makefile),
        ("Dockerfile", dockerfile),
        (".github/workflows/ci.yml", workflow_source),
        (".devcontainer/devcontainer.json", devcontainer),
        (notebook_path.relative_to(repo).as_posix(), code),
    ):
        if any(token in source.lower() for token in forbidden):
            return failed(location, "legacy NLP downloader remains active")
    if any("scripts.nlp_assets" in source for source in (dockerfile, workflow_source, devcontainer)):
        return failed("Makefile", "NLP asset consumers bypass Make")
    if not _notebook_nlp_asset_contract(code_cells):
        return failed(notebook_path.relative_to(repo).as_posix(), "notebook asset guard drifted")
    if "load_nlp_asset_manifest" not in dependency_verifier:
        return failed("scripts/verify_dependency_locks.py", "lock verifier omits NLP manifest")
    expected_model_requirement = (
        "en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/"
        "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl "
        "--hash=sha256:1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85\n"
    )
    if model_requirement != expected_model_requirement:
        return failed("nlp-model-requirements.txt", "spaCy model identity drifted")
    required_probe = (
        "expected_sha256",
        "observed_sha256",
        "expected_size",
        "observed_size",
        "asset_identity_mismatch",
        'expected_version="3.8.0"',
        "en_core_web_sm.load",
    )
    if any(token not in runtime_probe for token in required_probe):
        return failed("scripts/atlas_runtime_probe.py", "Atlas asset evidence drifted")
    return []


def check_assets(repo: Path, config: VerifierConfig) -> CheckResult:
    result = CheckResult(name="assets")
    result.findings.extend(_nlp_asset_contract_findings(repo))
    return result
