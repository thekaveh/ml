from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_SUBPROCESS_TIMEOUT = 30


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
    assert f"{custom_python} -m spacy download en_core_web_sm" in lines
    assert any(line.startswith(f"{custom_python} -c ") for line in lines)
    assert not any(" -m pip install" in line for line in lines)


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
    fake_papermill.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'input="${@: -2:1}"\n'
        'output="${@: -1}"\n'
        'mkdir -p "$(dirname "$output")"\n'
        'printf "rendered:%s\\n" "$input" > "$output"\n',
        encoding="utf-8",
    )
    fake_papermill.chmod(0o755)

    result = subprocess.run(
        [
            "make",
            "-f",
            str(REPO_ROOT / "Makefile"),
            "smoke-tier-a",
            "TIER_A=notebooks/first/notebook.ipynb notebooks/second/notebook.ipynb",
            f"TIER_A_OUT={output_root}",
            f"PAPERMILL={fake_papermill}",
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
    assert tuple(
        (output_root / "notebooks" / task / "notebook.ipynb").read_text(encoding="utf-8")
        for task in ("first", "second")
    ) == ("rendered:notebook.ipynb\n", "rendered:notebook.ipynb\n")


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
