from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_SUBPROCESS_TIMEOUT = 30
_DOCKER_INSTALL_BLOCK = """RUN /opt/conda/bin/python -m venv "$VIRTUAL_ENV" \\
  && make install-torch-stack \\
  && make nlp-assets \\
  && make verify-nlp-assets \\
  && python -m pip check \\
  && python -m scripts.verify_torch_stack \\
  && python -m scripts.verify_nnx_install"""
_DOCKER_IMAGE = (
    "quay.io/jupyter/datascience-notebook:python-3.11@"
    "sha256:14379f24c840f27375e3a4b29a9fa55e449633ac85c9cf3806ca62d11e5603ec"
)
_DEVCONTAINER_IMAGE = (
    "mcr.microsoft.com/devcontainers/python:3.11-bookworm@"
    "sha256:8e95c16fbc98a4a6a8f11f5b5bd152d0ffcd4fd0f4b31bd03e95965c777d2577"
)
_EXECUTED_NOTEBOOK_SOURCE = "print('executed')\n"
_EXECUTED_NOTEBOOK_SOURCE_HASH = hashlib.sha256(_EXECUTED_NOTEBOOK_SOURCE.encode("utf-8")).hexdigest()


def _target_recipe(makefile: str, target: str) -> tuple[str, ...]:
    lines = makefile.splitlines()
    definitions = tuple(
        index
        for index, line in enumerate(lines)
        if ":" in line and line.partition(":")[0].strip() == target
    )
    assert len(definitions) == 1
    start = definitions[0]
    recipes: list[str] = []
    for line in lines[start + 1 :]:
        if line.startswith("\t"):
            recipes.append(line.removeprefix("\t"))
            continue
        if line and not line.startswith((" ", "#")):
            break
    return tuple(recipes)


def _write_fake_papermill(path: Path) -> None:
    document = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": 1,
                "id": "executed-cell",
                "metadata": {},
                "outputs": [{"name": "stdout", "output_type": "stream", "text": "executed\\n"}],
                "source": _EXECUTED_NOTEBOOK_SOURCE,
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "from pathlib import Path\n"
        f"payload = {json.dumps(json.dumps(document))}\n"
        "output = Path(sys.argv[-1])\n"
        "output.parent.mkdir(parents=True, exist_ok=True)\n"
        "output.write_text(payload + '\\n', encoding='utf-8')\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _assert_source_hash_stamp(path: Path) -> None:
    document = json.loads(path.read_text(encoding="utf-8"))
    cell = document["cells"][0]
    assert cell["source"] == _EXECUTED_NOTEBOOK_SOURCE
    assert cell["execution_count"] == 1
    assert cell["outputs"]
    assert cell["metadata"]["source_hash"] == _EXECUTED_NOTEBOOK_SOURCE_HASH


def _write_failing_papermill(path: Path) -> None:
    error_output = {
        "ename": "RuntimeError",
        "evalue": "simulated Papermill failure",
        "output_type": "error",
        "traceback": ["RuntimeError: simulated Papermill failure"],
    }
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import json\n"
        "import sys\n"
        "from pathlib import Path\n"
        "source = Path(sys.argv[-2])\n"
        "output = Path(sys.argv[-1])\n"
        "document = json.loads(source.read_text(encoding='utf-8'))\n"
        "document['cells'][0]['execution_count'] = None\n"
        f"document['cells'][0]['outputs'] = [{error_output!r}]\n"
        "output.parent.mkdir(parents=True, exist_ok=True)\n"
        "output.write_text(json.dumps(document, indent=1) + '\\n', encoding='utf-8')\n"
        "raise SystemExit(19)\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _write_stamper_sentinel(path: Path) -> None:
    path.write_text(
        "#!/usr/bin/env python3\n"
        "from pathlib import Path\n"
        "Path('.source-hash-stamper-ran').touch()\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _write_clearer_sentinel(path: Path) -> None:
    command = [sys.executable, str(REPO_ROOT / "scripts" / "stamp_notebook_source_hashes.py"), "--clear"]
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import subprocess\n"
        "import sys\n"
        "from pathlib import Path\n"
        "Path('.source-hash-clearer-ran').touch()\n"
        f"command = {command!r}\n"
        "raise SystemExit(subprocess.run([*command, *sys.argv[1:]]).returncode)\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _write_hashed_input_notebook(path: Path) -> None:
    document = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": 1,
                "id": "executed-cell",
                "metadata": {"source_hash": _EXECUTED_NOTEBOOK_SOURCE_HASH},
                "outputs": [{"name": "stdout", "output_type": "stream", "text": "executed\n"}],
                "source": _EXECUTED_NOTEBOOK_SOURCE,
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")


def _stamper_command() -> str:
    return f"{sys.executable} {REPO_ROOT / 'scripts' / 'stamp_notebook_source_hashes.py'}"


def test_issue63_locked_install_targets_and_nlp_assets_are_exact() -> None:
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    expected = {
        "install-bootstrap": "$(PYTHON) -m scripts.install_locked_requirements bootstrap",
        "install-compiler-lock": "$(PYTHON) -m scripts.install_locked_requirements compiler",
        "install-docs-lock": "$(PYTHON) -m scripts.install_locked_requirements docs",
        "install-audit-lock": "$(PYTHON) -m scripts.install_locked_requirements audit",
        "install-atlas-contract-lock": (
            "$(PYTHON) -m scripts.install_locked_requirements atlas-contract"
        ),
        "install-torch-stack": "$(PYTHON) -m scripts.install_torch_stack",
    }
    for target, command in expected.items():
        assert _target_recipe(makefile, target) == (command,)
    nlp = _target_recipe(makefile, "nlp-assets")
    assert nlp == ("$(PYTHON) -m scripts.nlp_assets install",)
    assert _target_recipe(makefile, "verify-nlp-assets") == (
        "$(PYTHON) -m scripts.nlp_assets verify",
    )
    assert "spacy download" not in "\n".join(nlp)
    assert "pip install" not in "\n".join(nlp)


def _assert_tier_inventory_contract(makefile: Path, cwd: Path) -> None:
    source = makefile.read_text(encoding="utf-8")
    lines = source.splitlines()
    phony_members = [
        member
        for line in lines
        if line.startswith(".PHONY:")
        for member in line.removeprefix(".PHONY:").split()
    ]
    expected_counts = {"a": 18, "b": 7, "c": 4}
    for tier, count in expected_counts.items():
        target = f"print-tier-{tier}"
        variable = f"TIER_{tier.upper()}"
        assert phony_members.count(target) == 1
        assert _target_recipe(source, target) == (f"@printf '%s\\n' $({variable})",)
        result = subprocess.run(
            ["make", "-f", str(makefile), "--no-print-directory", "-s", target],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            timeout=TEST_SUBPROCESS_TIMEOUT,
        )
        inventory = tuple(result.stdout.splitlines())
        assert len(inventory) == count
        assert len(set(inventory)) == count
        assert all(item.startswith("notebooks/") and item.endswith(".ipynb") for item in inventory)
        assert result.stderr == ""


def _assert_smoke_output_environment_override_contract(makefile: Path, cwd: Path) -> None:
    fake_papermill = cwd / "papermill"
    _write_fake_papermill(fake_papermill)

    for tier in ("b", "c"):
        notebook = cwd / "notebooks" / f"tier-{tier}" / "notebook.ipynb"
        notebook.parent.mkdir(parents=True)
        notebook.write_text(f"tier-{tier} source\n", encoding="utf-8")
        output_root = cwd / "isolated" / f"tier-{tier}"
        env = {
            **os.environ,
            "JUPYTER_PATH": str(cwd / "isolated" / "jupyter"),
            "SMOKE_OUT": str(output_root),
        }
        result = subprocess.run(
            [
                "make",
                "-f",
                str(makefile),
                f"smoke-tier-{tier}",
                f"TIER_{tier.upper()}={notebook.relative_to(cwd)}",
                f"PAPERMILL={fake_papermill}",
                f"SOURCE_HASH_STAMPER={_stamper_command()}",
            ],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=TEST_SUBPROCESS_TIMEOUT,
        )

        assert result.returncode == 0, result.stderr
        assert notebook.read_text(encoding="utf-8") == f"tier-{tier} source\n"
        output = output_root / notebook.name
        assert output.is_file()
        _assert_source_hash_stamp(output)


def _assert_docker_and_codespaces_contract(
    docker: str,
    makefile: str,
    devcontainer: str,
) -> None:
    import json

    assert docker.startswith(f"FROM {_DOCKER_IMAGE}\n")
    assert "ENV VIRTUAL_ENV=/home/jovyan/.venvs/ml-eng-lab\n" in docker
    assert "ENV CONDA_AUTO_ACTIVATE_BASE=false\n" in docker
    assert 'ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"\n' in docker
    assert "--platform" not in docker
    assert docker.count("RUN ") == 1
    assert docker[docker.index("RUN ") :].strip() == _DOCKER_INSTALL_BLOCK
    codespace_definitions = tuple(
        line
        for line in makefile.splitlines()
        if ":" in line and line.partition(":")[0].strip() == "codespace-setup"
    )
    assert codespace_definitions == ("codespace-setup: install-torch-stack",)
    assert _target_recipe(makefile, "codespace-setup") == (
        "$(MAKE) nlp-assets",
        "$(MAKE) verify-nlp-assets",
        "$(PYTHON) -m pip check",
        "$(MAKE) verify-torch-stack",
        "$(MAKE) verify-nnx-install",
    )
    payload = json.loads(
        "\n".join(
            line
            for line in devcontainer.splitlines()
            if not line.lstrip().startswith("//")
        )
    )
    assert payload["image"] == _DEVCONTAINER_IMAGE
    assert payload["postCreateCommand"] == "make codespace-setup"


def _assert_torch_stack_verifier_target(makefile: Path, cwd: Path, python: str = "python") -> None:
    source = makefile.read_text(encoding="utf-8")
    lines = source.splitlines()
    phony_members = [
        member
        for line in lines
        if line.startswith(".PHONY:")
        for member in line.removeprefix(".PHONY:").split()
    ]
    assert phony_members.count("verify-torch-stack") == 1
    assert lines.count('\t@echo "  verify-torch-stack Verify the active canonical Torch stack."') == 1
    assert lines.count("verify-torch-stack:") == 1
    target_index = lines.index("verify-torch-stack:")
    assert lines[target_index + 1] == "\t$(PYTHON) -m scripts.verify_torch_stack"
    result = subprocess.run(
        [
            "make",
            "-f",
            str(makefile),
            "--no-print-directory",
            "-n",
            "verify-torch-stack",
            f"PYTHON={python}",
        ],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
        timeout=TEST_SUBPROCESS_TIMEOUT,
    )
    assert result.stdout == f"{python} -m scripts.verify_torch_stack\n"
    assert result.stderr == ""
    failure_probe = subprocess.run(
        [
            "make",
            "-f",
            str(makefile),
            "--no-print-directory",
            "verify-torch-stack",
            "PYTHON=false",
        ],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=TEST_SUBPROCESS_TIMEOUT,
    )
    assert failure_probe.returncode != 0


def _assert_audit_advisories_contract(makefile: Path, cwd: Path) -> None:
    source = makefile.read_text(encoding="utf-8")
    lines = source.splitlines()
    target_lines = [line for line in lines if line.startswith("audit-advisories:")]
    assert target_lines == ["audit-advisories:"]
    target_index = lines.index("audit-advisories:")
    assert lines[target_index + 1] == "\t$(PYTHON) -m scripts.advisory_baseline"
    for line in lines:
        directive = line.split("#", 1)[0].strip()
        ignore_match = re.fullmatch(r"\.IGNORE\s*:\s*(.*)", directive)
        if ignore_match is not None:
            ignored_targets = ignore_match.group(1).split()
            assert ignored_targets and "audit-advisories" not in ignored_targets
    result = subprocess.run(
        ["make", "-f", str(makefile), "--no-print-directory", "-n", "audit-advisories"],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
        timeout=TEST_SUBPROCESS_TIMEOUT,
    )
    assert result.stdout == "python -m scripts.advisory_baseline\n"
    assert result.stderr == ""
    failure_probe = subprocess.run(
        ["make", "-f", str(makefile), "--no-print-directory", "audit-advisories", "PYTHON=false"],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=TEST_SUBPROCESS_TIMEOUT,
    )
    assert failure_probe.returncode != 0


def _is_nnx_import(node: ast.stmt) -> bool:
    if isinstance(node, ast.Import):
        return any(alias.name == "nnx" or alias.name.startswith("nnx.") for alias in node.names)
    return (
        isinstance(node, ast.ImportFrom)
        and node.module is not None
        and (node.module == "nnx" or node.module.startswith("nnx."))
    )


def _assert_nnx_collection_verifier_contract(source: str) -> None:
    tree = ast.parse(source)
    expected_imports = {
        "scripts.verify_torch_stack": "verify_torch_stack",
        "scripts.verify_nnx_install": "verify_nnx_install",
    }
    for module_name, binding in expected_imports.items():
        imports = tuple(
            node
            for node in tree.body
            if isinstance(node, ast.ImportFrom)
            and node.module == module_name
            and node.level == 0
        )
        assert len(imports) == 1
        assert not any(
            isinstance(node, ast.Import)
            and any(alias.name == module_name for alias in node.names)
            for node in tree.body
        )
        assert len(imports[0].names) == 1
        assert imports[0].names[0].name == binding
        assert imports[0].names[0].asname is None
    assert not [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_verify_nnx_installation_contract"
    ]
    calls = {
        name: tuple(
            node
            for node in tree.body
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == name
        )
        for name in ("verify_torch_stack", "verify_nnx_install")
    }
    assert len(calls["verify_torch_stack"]) == 1
    assert len(calls["verify_nnx_install"]) == 1
    assert not calls["verify_torch_stack"][0].value.args
    assert not calls["verify_torch_stack"][0].value.keywords
    assert not calls["verify_nnx_install"][0].value.args
    assert not calls["verify_nnx_install"][0].value.keywords
    nnx_imports = tuple(node for node in tree.body if _is_nnx_import(node))
    assert nnx_imports
    assert tree.body.index(calls["verify_torch_stack"][0]) < tree.body.index(
        calls["verify_nnx_install"][0]
    )
    assert tree.body.index(calls["verify_nnx_install"][0]) < tree.body.index(nnx_imports[0])


def test_mkdocs_commands_suppress_only_the_upstream_material_banner():
    result = subprocess.run(
        ["make", "--no-print-directory", "-n", "docs-build", "docs-serve", "docs-check"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=TEST_SUBPROCESS_TIMEOUT,
    )
    mkdocs_lines = [line for line in result.stdout.splitlines() if "mkdocs " in line]

    assert mkdocs_lines
    assert all(line.startswith("NO_MKDOCS_2_WARNING=1 mkdocs ") for line in mkdocs_lines)


def test_torch_installer_target_is_one_exact_command_and_codespace_has_no_late_pip_install():
    custom_python = "/opt/custom/bin/python"
    result = subprocess.run(
        [
            "make",
            "--no-print-directory",
            "-n",
            f"PYTHON={custom_python}",
            "install-torch-stack",
            "codespace-setup",
            "nlp-assets",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=TEST_SUBPROCESS_TIMEOUT,
    )
    lines = result.stdout.splitlines()

    assert lines.count(f"{custom_python} -m scripts.install_torch_stack") == 1
    assert f"{custom_python} -m spacy download en_core_web_sm" not in lines
    assert lines.count(f"{custom_python} -m scripts.nlp_assets install") == 2
    assert lines.count(f"{custom_python} -m scripts.nlp_assets verify") == 1
    assert not any(" -m pip install" in line for line in lines)


def test_docker_and_codespaces_verify_after_the_last_package_change():
    docker = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    devcontainer = (REPO_ROOT / ".devcontainer" / "devcontainer.json").read_text(
        encoding="utf-8"
    )

    _assert_docker_and_codespaces_contract(docker, makefile, devcontainer)


def test_devcontainer_uses_only_the_one_shot_codespace_target():
    devcontainer = (REPO_ROOT / ".devcontainer" / "devcontainer.json").read_text(
        encoding="utf-8"
    )
    _assert_docker_and_codespaces_contract(
        (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8"),
        (REPO_ROOT / "Makefile").read_text(encoding="utf-8"),
        devcontainer,
    )


@pytest.mark.parametrize(
    "replacement",
    (
        "RUN python -m pip install -r requirements.txt",
        _DOCKER_INSTALL_BLOCK + " \\\n  && python -m pip install package",
        "RUN make install-torch-stack \\\n  && docker compose up -d",
        "RUN make install-torch-stack \\\n  && jupyter lab",
        "RUN make install-torch-stack \\\n  && ollama serve",
        "RUN make install-torch-stack \\\n  && comfyui --listen",
        "RUN make install-torch-stack \\\n  && make atlas-setup",
    ),
)
def test_docker_contract_rejects_direct_late_install_or_service_mutations(replacement):
    docker = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    mutated = docker.replace(_DOCKER_INSTALL_BLOCK, replacement, 1)
    assert mutated != docker
    with pytest.raises(AssertionError):
        _assert_docker_and_codespaces_contract(
            mutated,
            (REPO_ROOT / "Makefile").read_text(encoding="utf-8"),
            (REPO_ROOT / ".devcontainer" / "devcontainer.json").read_text(
                encoding="utf-8"
            ),
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "\t$(PYTHON) -m pip install package\n",
        "\tdocker compose up -d\n",
        "\tjupyter lab\n",
        "\tollama serve\n",
        "\tcomfyui --listen\n",
        "\t$(MAKE) atlas-setup\n",
    ),
)
def test_codespace_contract_rejects_late_install_or_service_mutations(mutation):
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    anchor = "\t$(MAKE) verify-nnx-install\n"
    mutated = makefile.replace(anchor, anchor + mutation, 1)
    assert mutated != makefile
    with pytest.raises(AssertionError):
        _assert_docker_and_codespaces_contract(
            (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8"),
            mutated,
            (REPO_ROOT / ".devcontainer" / "devcontainer.json").read_text(
                encoding="utf-8"
            ),
        )


@pytest.mark.parametrize(
    "prerequisites",
    (
        "",
        "nlp-assets",
        "install-torch-stack atlas-setup",
        "install-torch-stack ollama",
        "install-torch-stack install-extra",
        "install-torch-stack install-torch-stack",
    ),
)
def test_codespace_contract_rejects_noncanonical_prerequisites(prerequisites):
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    original = "codespace-setup: install-torch-stack"
    mutated_header = f"codespace-setup: {prerequisites}".rstrip()
    mutated = makefile.replace(original, mutated_header, 1)
    assert mutated != makefile

    with pytest.raises(AssertionError):
        _assert_docker_and_codespaces_contract(
            (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8"),
            mutated,
            (REPO_ROOT / ".devcontainer" / "devcontainer.json").read_text(
                encoding="utf-8"
            ),
        )


def test_codespace_contract_rejects_duplicate_target_definition():
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    duplicate = (
        "\ncodespace-setup: install-torch-stack\n"
        "\t$(MAKE) nlp-assets\n"
        "\t$(MAKE) verify-nlp-assets\n"
        "\t$(PYTHON) -m pip check\n"
        "\t$(MAKE) verify-torch-stack\n"
        "\t$(MAKE) verify-nnx-install\n"
    )
    mutated = makefile + duplicate
    assert mutated != makefile

    with pytest.raises(AssertionError):
        _assert_docker_and_codespaces_contract(
            (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8"),
            mutated,
            (REPO_ROOT / ".devcontainer" / "devcontainer.json").read_text(
                encoding="utf-8"
            ),
        )


def test_verify_nnx_install_target_is_public_and_uses_selected_python():
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    phony_members = [
        member
        for line in makefile.splitlines()
        if line.startswith(".PHONY:")
        for member in line.removeprefix(".PHONY:").split()
    ]

    assert phony_members.count("verify-nnx-install") == 1
    assert (
        '\t@echo "  verify-nnx-install Verify the active NNx installation provenance."'
        in makefile.splitlines()
    )

    result = subprocess.run(
        ["make", "--no-print-directory", "-n", "verify-nnx-install"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=TEST_SUBPROCESS_TIMEOUT,
    )

    assert result.stdout == "python -m scripts.verify_nnx_install\n"
    assert result.stderr == ""


def test_verify_torch_stack_target_is_one_public_fail_closed_command() -> None:
    _assert_torch_stack_verifier_target(
        REPO_ROOT / "Makefile",
        REPO_ROOT,
        python="/opt/qualified/bin/python",
    )


@pytest.mark.parametrize(
    ("original", "replacement"),
    (
        ("\t$(PYTHON) -m scripts.verify_torch_stack", "\t-$(PYTHON) -m scripts.verify_torch_stack"),
        ("\t$(PYTHON) -m scripts.verify_torch_stack", "\t$(PYTHON) -m scripts.verify_torch_stack || true"),
        ("verify-torch-stack:\n", "verify-torch-stack: verify-nnx-install\n"),
        ("\t$(PYTHON) -m scripts.verify_torch_stack", "\t$(PYTHON) -m scripts.verify_torch_stack\n\t@echo extra"),
    ),
    ids=("ignored-failure", "shell-mask", "prerequisite", "extra-command"),
)
def test_verify_torch_stack_target_rejects_fail_open_mutations(
    tmp_path: Path, original: str, replacement: str
) -> None:
    source = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    mutated = source.replace(original, replacement, 1)
    assert mutated != source
    makefile = tmp_path / "Makefile"
    makefile.write_text(mutated, encoding="utf-8")

    with pytest.raises(AssertionError):
        _assert_torch_stack_verifier_target(makefile, tmp_path)


def test_audit_advisories_target_is_one_unsuppressed_command() -> None:
    _assert_audit_advisories_contract(REPO_ROOT / "Makefile", REPO_ROOT)


@pytest.mark.parametrize(
    "original,mutation",
    [
        ("\t$(PYTHON) -m scripts.advisory_baseline", "\t$(PYTHON) -m scripts.advisory_baseline || true"),
        ("\t$(PYTHON) -m scripts.advisory_baseline", "\t-$(PYTHON) -m scripts.advisory_baseline"),
        ("\t$(PYTHON) -m scripts.advisory_baseline", "\t@$(PYTHON) -m scripts.advisory_baseline"),
        ("\t$(PYTHON) -m scripts.advisory_baseline", "\t+$(PYTHON) -m scripts.advisory_baseline"),
        ("audit-advisories:\n", "audit-advisories: requirements.txt\n"),
        ("\t$(PYTHON) -m scripts.advisory_baseline", "\t$(PYTHON) -m scripts.advisory_baseline\n\t@echo extra"),
        ("\n\nlint:\n", "\n\naudit-advisories:\n\t@echo duplicate\n\nlint:\n"),
        ("\n\nlint:\n", "\n\n.IGNORE: audit-advisories # fail-open\nlint:\n"),
        ("\n\nlint:\n", "\n\n  .IGNORE: # fail-open\nlint:\n"),
        ("\n\nlint:\n", "\n\n.IGNORE : audit-advisories # fail-open\nlint:\n"),
        ("\n\nlint:\n", "\n\n.IGNORE: \\\n  audit-advisories\nlint:\n"),
        ("\n\nlint:\n", "\n\nAUDIT_TARGET := audit-advisories\n.IGNORE: $(AUDIT_TARGET)\nlint:\n"),
        ("\n\nlint:\n", "\n\n.IGNORE: harmless\\#name audit-advisories\nlint:\n"),
    ],
    ids=(
        "failure-suppressed",
        "recipe-error-prefix",
        "recipe-silent-prefix",
        "recipe-recursive-prefix",
        "prerequisite-added",
        "extra-recipe",
        "duplicate-target",
        "target-ignore-directive",
        "global-ignore-directive",
        "spaced-target-ignore-directive",
        "continued-target-ignore-directive",
        "expanded-target-ignore-directive",
        "escaped-comment-target-ignore-directive",
    ),
)
def test_audit_advisories_contract_rejects_makefile_mutations(
    tmp_path: Path, original: str, mutation: str
) -> None:
    source = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    mutated = source.replace(original, mutation, 1)
    assert mutated != source
    makefile = tmp_path / "Makefile"
    makefile.write_text(mutated, encoding="utf-8")

    with pytest.raises(AssertionError):
        _assert_audit_advisories_contract(makefile, tmp_path)


def test_nnx_surface_verifies_stack_then_nnx_once_before_collection_imports():
    source = (REPO_ROOT / "tests" / "nnx_surface" / "conftest.py").read_text(encoding="utf-8")

    _assert_nnx_collection_verifier_contract(source)


@pytest.mark.parametrize(
    ("original", "mutation"),
    (
        (
            "from scripts.verify_torch_stack import verify_torch_stack",
            "from scripts.verify_torch_stack import verify_torch_stack as stack_verify",
        ),
        (
            "from scripts.verify_nnx_install import verify_nnx_install",
            "from scripts.verify_nnx_install import *",
        ),
        (
            "from scripts.verify_torch_stack import verify_torch_stack",
            "from scripts.verify_other import verify_torch_stack",
        ),
        (
            "from scripts.verify_nnx_install import verify_nnx_install",
            "def import_verifier():\n    from scripts.verify_nnx_install import verify_nnx_install",
        ),
        (
            "from scripts.verify_torch_stack import verify_torch_stack",
            "from scripts.verify_torch_stack import verify_torch_stack\n"
            "from scripts.verify_torch_stack import verify_torch_stack",
        ),
        (
            "from scripts.verify_nnx_install import verify_nnx_install",
            "import scripts.verify_nnx_install",
        ),
        ("verify_torch_stack()\n", ""),
        ("verify_nnx_install()\n", ""),
        (
            "verify_torch_stack()\nverify_nnx_install()",
            "verify_nnx_install()\nverify_torch_stack()",
        ),
        (
            "verify_torch_stack()\nverify_nnx_install()",
            "verify_torch_stack()\nverify_torch_stack()\nverify_nnx_install()",
        ),
        (
            "verify_torch_stack()\nverify_nnx_install()",
            "verify_torch_stack()\nverify_nnx_install()\nverify_nnx_install()",
        ),
        (
            "verify_torch_stack()\nverify_nnx_install()\n\nimport nnx",
            "import nnx\n\nverify_torch_stack()\nverify_nnx_install()",
        ),
        (
            "verify_torch_stack()\n",
            "def verify_during_collection():\n    verify_torch_stack()\n",
        ),
        (
            "verify_nnx_install()\n",
            "try:\n    verify_nnx_install()\nexcept Exception:\n    pass\n",
        ),
        (
            "verify_torch_stack()\n",
            "if ENABLE_VERIFY:\n    verify_torch_stack()\n",
        ),
        (
            "import nnx  # noqa: E402  # both provenance gates precede collection imports",
            "import nnx  # noqa: E402  # both provenance gates precede collection imports\n\n"
            "@pytest.fixture(scope=\"session\", autouse=True)\n"
            "def _verify_nnx_installation_contract():\n"
            "    verify_nnx_install()",
        ),
    ),
    ids=(
        "torch-import-alias",
        "nnx-star-import",
        "torch-wrong-module",
        "nnx-import-inside-function",
        "torch-import-duplicated",
        "nnx-module-import",
        "torch-call-deleted",
        "nnx-call-deleted",
        "calls-reversed",
        "torch-call-duplicated",
        "nnx-call-duplicated",
        "calls-after-nnx",
        "torch-call-inside-function",
        "nnx-call-inside-try",
        "torch-call-conditional",
        "autouse-fixture-restored",
    ),
)
def test_nnx_collection_verifier_contract_rejects_mutations(original: str, mutation: str):
    source = (REPO_ROOT / "tests" / "nnx_surface" / "conftest.py").read_text(encoding="utf-8")
    mutated = source.replace(original, mutation, 1)

    assert mutated != source
    with pytest.raises(AssertionError):
        _assert_nnx_collection_verifier_contract(mutated)


def test_smoke_tier_a_writes_to_temporary_outputs_without_mutating_sources(
    tmp_path: Path,
) -> None:
    sources = (
        tmp_path / "notebooks" / "first" / "notebook.ipynb",
        tmp_path / "notebooks" / "second" / "notebook.ipynb",
    )
    for index, source in enumerate(sources, start=1):
        source.parent.mkdir(parents=True)
        source.write_text(f"source notebook {index}\n", encoding="utf-8")
    output_root = tmp_path / "tier-a-output"
    fake_papermill = tmp_path / "papermill"
    _write_fake_papermill(fake_papermill)

    result = subprocess.run(
        [
            "make",
            "-f",
            str(REPO_ROOT / "Makefile"),
            "smoke-tier-a",
            "TIER_A=notebooks/first/notebook.ipynb notebooks/second/notebook.ipynb",
            f"TIER_A_OUT={output_root}",
            f"PAPERMILL={fake_papermill}",
            f"SOURCE_HASH_STAMPER={_stamper_command()}",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=TEST_SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 0, result.stderr
    assert tuple(source.read_text(encoding="utf-8") for source in sources) == (
        "source notebook 1\n",
        "source notebook 2\n",
    )
    for task in ("first", "second"):
        _assert_source_hash_stamp(output_root / "notebooks" / task / "notebook.ipynb")


def test_run_tier_a_stamps_successful_in_place_execution(tmp_path: Path) -> None:
    notebook = tmp_path / "notebooks" / "tier-a" / "notebook.ipynb"
    notebook.parent.mkdir(parents=True)
    notebook.write_text("source notebook\n", encoding="utf-8")
    fake_papermill = tmp_path / "papermill"
    _write_fake_papermill(fake_papermill)

    result = subprocess.run(
        [
            "make",
            "-f",
            str(REPO_ROOT / "Makefile"),
            "run-tier-a",
            f"TIER_A={notebook.relative_to(tmp_path)}",
            f"PAPERMILL={fake_papermill}",
            f"SOURCE_HASH_STAMPER={_stamper_command()}",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=TEST_SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 0, result.stderr
    _assert_source_hash_stamp(notebook)


def test_execution_targets_configure_and_order_source_hash_stamping() -> None:
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    assert "SOURCE_HASH_STAMPER ?= $(PYTHON) scripts/stamp_notebook_source_hashes.py" in makefile
    assert "SOURCE_HASH_CLEARER ?= $(PYTHON) scripts/stamp_notebook_source_hashes.py --clear" in makefile

    expected_paths = {
        "run-tier-a": "$$nb",
        "smoke-tier-a": "$$out",
        "smoke-tier-b": "$$out",
        "smoke-tier-c": "$$out",
    }
    for target, path in expected_paths.items():
        recipe = "\n".join(_target_recipe(makefile, target))
        stamp = f'$(SOURCE_HASH_STAMPER) "{path}"'
        clear = f'$(SOURCE_HASH_CLEARER) "{path}"'
        assert recipe.count(stamp) == 1
        assert recipe.count(clear) == 1
        assert recipe.index("$(PAPERMILL)") < recipe.index(stamp)
        assert recipe.index("$(PAPERMILL)") < recipe.index(clear) < recipe.index(stamp)
        assert f'if [ -f "{path}" ]' in recipe
        assert "papermill_status=$$?" in recipe
        assert "exit $$papermill_status" in recipe
        assert f"{stamp} || exit 1;" in recipe


@pytest.mark.parametrize(
    ("target", "tier_assignment"),
    (
        ("run-tier-a", "TIER_A=notebooks/tier-a/notebook.ipynb"),
        ("smoke-tier-a", "TIER_A=notebooks/tier-a/notebook.ipynb"),
    ),
    ids=("in-place-tier-a", "temporary-output-smoke"),
)
def test_failed_papermill_clears_inherited_hashes_without_invoking_source_hash_stamper(
    tmp_path: Path, target: str, tier_assignment: str
) -> None:
    notebook = tmp_path / "notebooks" / "tier-a" / "notebook.ipynb"
    notebook.parent.mkdir(parents=True)
    _write_hashed_input_notebook(notebook)
    fake_papermill = tmp_path / "papermill"
    _write_failing_papermill(fake_papermill)
    stamper_sentinel = tmp_path / "stamper-sentinel"
    _write_stamper_sentinel(stamper_sentinel)
    clearer_sentinel = tmp_path / "clearer-sentinel"
    _write_clearer_sentinel(clearer_sentinel)
    output_root = tmp_path / "tier-a-output"

    result = subprocess.run(
        [
            "make",
            "-f",
            str(REPO_ROOT / "Makefile"),
            target,
            tier_assignment,
            f"TIER_A_OUT={output_root}",
            f"PAPERMILL={fake_papermill}",
            f"SOURCE_HASH_STAMPER={stamper_sentinel}",
            f"SOURCE_HASH_CLEARER={clearer_sentinel}",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=TEST_SUBPROCESS_TIMEOUT,
    )

    assert result.returncode != 0
    assert not (tmp_path / ".source-hash-stamper-ran").exists()
    assert (tmp_path / ".source-hash-clearer-ran").is_file()
    artifact = notebook if target == "run-tier-a" else output_root / notebook.relative_to(tmp_path)
    failed_document = json.loads(artifact.read_text(encoding="utf-8"))
    assert sum(
        "source_hash" in cell["metadata"]
        for cell in failed_document["cells"]
        if cell["cell_type"] == "code"
    ) == 0
    assert sum(
        output["output_type"] == "error"
        for cell in failed_document["cells"]
        if cell["cell_type"] == "code"
        for output in cell["outputs"]
    ) == 1


@pytest.mark.parametrize(
    ("target", "tier_assignment"),
    (
        ("run-tier-a", "TIER_A=notebooks/tier-a/notebook.ipynb"),
        ("smoke-tier-a", "TIER_A=notebooks/tier-a/notebook.ipynb"),
    ),
    ids=("in-place-tier-a", "temporary-output-smoke"),
)
def test_failure_clearer_error_cannot_turn_failed_execution_green(
    tmp_path: Path, target: str, tier_assignment: str
) -> None:
    notebook = tmp_path / "notebooks" / "tier-a" / "notebook.ipynb"
    notebook.parent.mkdir(parents=True)
    _write_hashed_input_notebook(notebook)
    fake_papermill = tmp_path / "papermill"
    _write_failing_papermill(fake_papermill)

    result = subprocess.run(
        [
            "make",
            "-f",
            str(REPO_ROOT / "Makefile"),
            target,
            tier_assignment,
            f"TIER_A_OUT={tmp_path / 'tier-a-output'}",
            f"PAPERMILL={fake_papermill}",
            "SOURCE_HASH_CLEARER=false",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=TEST_SUBPROCESS_TIMEOUT,
    )

    assert result.returncode != 0


@pytest.mark.parametrize(
    ("target", "tier_assignment"),
    (
        ("run-tier-a", "TIER_A=notebooks/tier-a/notebook.ipynb"),
        ("smoke-tier-a", "TIER_A=notebooks/tier-a/notebook.ipynb"),
    ),
    ids=("in-place-tier-a", "temporary-output-smoke"),
)
def test_source_hash_stamper_failure_fails_execution_target(
    tmp_path: Path, target: str, tier_assignment: str
) -> None:
    notebook = tmp_path / "notebooks" / "tier-a" / "notebook.ipynb"
    notebook.parent.mkdir(parents=True)
    notebook.write_text("source notebook\n", encoding="utf-8")
    fake_papermill = tmp_path / "papermill"
    _write_fake_papermill(fake_papermill)

    result = subprocess.run(
        [
            "make",
            "-f",
            str(REPO_ROOT / "Makefile"),
            target,
            tier_assignment,
            f"PAPERMILL={fake_papermill}",
            "SOURCE_HASH_STAMPER=false",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=TEST_SUBPROCESS_TIMEOUT,
    )

    assert result.returncode != 0


def test_makefile_exposes_exact_tier_inventory_targets() -> None:
    _assert_tier_inventory_contract(REPO_ROOT / "Makefile", REPO_ROOT)


def test_task7_smoke_tiers_honor_environment_output_roots(tmp_path: Path) -> None:
    _assert_smoke_output_environment_override_contract(REPO_ROOT / "Makefile", tmp_path)


def test_smoke_output_contract_rejects_hard_assignment_mutation(tmp_path: Path) -> None:
    source = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    mutated = source.replace(
        "SMOKE_OUT ?= /tmp/ml-smoke",
        "SMOKE_OUT := ignored-smoke-output",
        1,
    )
    assert mutated != source
    makefile = tmp_path / "Makefile"
    makefile.write_text(mutated, encoding="utf-8")

    with pytest.raises(AssertionError):
        _assert_smoke_output_environment_override_contract(makefile, tmp_path)


@pytest.mark.parametrize(
    ("original", "mutation"),
    (
        (
            "print-tier-a:\n\t@printf '%s\\n' $(TIER_A)",
            "print-tier-a:\n\t@printf '%s\\n' $(TIER_B)",
        ),
        (
            "print-tier-b:\n\t@printf '%s\\n' $(TIER_B)",
            "print-tier-b:\n\t@printf '%s\\n' $(TIER_C)",
        ),
        (
            "print-tier-c:\n\t@printf '%s\\n' $(TIER_C)",
            "print-tier-c:\n\t@printf '%s\\n' $(TIER_A)",
        ),
    ),
    ids=("tier-a-wrong-variable", "tier-b-wrong-variable", "tier-c-wrong-variable"),
)
def test_tier_inventory_contract_rejects_wrong_variable_mutations(
    tmp_path: Path, original: str, mutation: str
) -> None:
    source = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    mutated = source.replace(original, mutation, 1)
    assert mutated != source
    makefile = tmp_path / "Makefile"
    makefile.write_text(mutated, encoding="utf-8")

    with pytest.raises(AssertionError):
        _assert_tier_inventory_contract(makefile, tmp_path)


def test_check_tier_a_artifacts_accepts_every_nonempty_mirrored_output(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "tier-a-output"
    tier_a = "notebooks/first/notebook.ipynb notebooks/second/notebook.ipynb"
    for task in ("first", "second"):
        output = output_root / "notebooks" / task / "notebook.ipynb"
        output.parent.mkdir(parents=True)
        output.write_text(f"{task} output\n", encoding="utf-8")

    result = subprocess.run(
        [
            "make",
            "-f",
            str(REPO_ROOT / "Makefile"),
            "check-tier-a-artifacts",
            f"TIER_A={tier_a}",
            f"TIER_A_OUT={output_root}",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=TEST_SUBPROCESS_TIMEOUT,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("empty_output", (False, True))
def test_check_tier_a_artifacts_reports_a_missing_or_empty_mirrored_output(
    tmp_path: Path, empty_output: bool
) -> None:
    output_root = tmp_path / "tier-a-output"
    existing = output_root / "notebooks" / "first" / "notebook.ipynb"
    existing.parent.mkdir(parents=True)
    existing.write_text("first output\n", encoding="utf-8")
    if empty_output:
        empty = output_root / "notebooks" / "second" / "notebook.ipynb"
        empty.parent.mkdir(parents=True)
        empty.write_text("", encoding="utf-8")
    tier_a = "notebooks/first/notebook.ipynb notebooks/second/notebook.ipynb"

    result = subprocess.run(
        [
            "make",
            "-f",
            str(REPO_ROOT / "Makefile"),
            "check-tier-a-artifacts",
            f"TIER_A={tier_a}",
            f"TIER_A_OUT={output_root}",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=TEST_SUBPROCESS_TIMEOUT,
    )

    assert result.returncode != 0
    assert "missing expected Tier-A notebook output" in result.stderr


def test_audit_advisories_target_uses_the_focused_pinned_tool() -> None:
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")

    assert (REPO_ROOT / "vulnerability-audit-requirements.txt").read_text(encoding="utf-8") == "pip-audit==2.10.0\n"
    assert "audit-advisories:" in makefile
    assert "$(PYTHON) -m scripts.advisory_baseline" in makefile
