from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_SUBPROCESS_TIMEOUT = 30


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


def _assert_nnx_install_fixture_contract(source: str) -> None:
    tree = ast.parse(source)
    verifier_imports = [
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "scripts.verify_nnx_install"
        and any(alias.name == "verify_nnx_install" for alias in node.names)
    ]
    fixtures = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_verify_nnx_installation_contract"
    ]
    initial_verifier_calls = [
        node
        for node in tree.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "verify_nnx_install"
        and not node.value.args
        and not node.value.keywords
    ]
    nnx_imports = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Import)
            and any(
                alias.name == "nnx" or alias.name.startswith("nnx.")
                for alias in node.names
            )
        )
        or (
            isinstance(node, ast.ImportFrom)
            and node.module is not None
            and (node.module == "nnx" or node.module.startswith("nnx."))
        )
    ]

    assert len(verifier_imports) == 1
    assert len(initial_verifier_calls) == 1
    assert len(nnx_imports) == 1
    assert tree.body.index(verifier_imports[0]) < tree.body.index(initial_verifier_calls[0])
    assert tree.body.index(initial_verifier_calls[0]) < tree.body.index(nnx_imports[0])
    assert len(fixtures) == 1
    fixture = fixtures[0]
    assert len(fixture.decorator_list) == 1
    decorator = fixture.decorator_list[0]
    assert isinstance(decorator, ast.Call)
    assert isinstance(decorator.func, ast.Attribute)
    assert isinstance(decorator.func.value, ast.Name)
    assert (decorator.func.value.id, decorator.func.attr) == ("pytest", "fixture")
    assert not decorator.args
    assert {keyword.arg: ast.literal_eval(keyword.value) for keyword in decorator.keywords} == {
        "scope": "session",
        "autouse": True,
    }
    assert not fixture.args.args
    assert len(fixture.body) == 1
    invocation = fixture.body[0]
    assert isinstance(invocation, ast.Expr)
    assert isinstance(invocation.value, ast.Call)
    assert isinstance(invocation.value.func, ast.Name)
    assert invocation.value.func.id == "verify_nnx_install"
    assert not invocation.value.args
    assert not invocation.value.keywords


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


def test_setup_targets_use_selected_python_interpreter():
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

    assert f"{custom_python} -m pip install --upgrade pip" in lines
    assert f"{custom_python} -m pip install -r torch-core-requirements.txt" in lines
    assert f"{custom_python} -m pip install --no-build-isolation -r torch-requirements.txt" in lines
    assert f"{custom_python} -m pip install -r requirements.txt" in lines
    assert f"{custom_python} -m spacy download en_core_web_sm" in lines
    assert any(line.startswith(f"{custom_python} -c ") for line in lines)
    assert not any(line.startswith("pip install") or line.startswith("python ") for line in lines)


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


def test_nnx_surface_has_a_session_autouse_installation_verifier():
    source = (REPO_ROOT / "tests" / "nnx_surface" / "conftest.py").read_text(encoding="utf-8")

    _assert_nnx_install_fixture_contract(source)


@pytest.mark.parametrize(
    ("original", "mutation"),
    (
        ('scope="session"', 'scope="function"'),
        ("autouse=True", "autouse=False"),
        (
            "    verify_nnx_install()",
            "    try:\n        verify_nnx_install()\n    except VerificationError:\n        pass",
        ),
        (
            "    verify_nnx_install()",
            '    os.environ["NNX_ALLOW_EDITABLE"] = "1"\n    verify_nnx_install()',
        ),
        (
            "verify_nnx_install()\n\nimport nnx",
            "import nnx\n\nverify_nnx_install()",
        ),
        ("verify_nnx_install()\n\nimport nnx", "import nnx"),
        (
            "verify_nnx_install()\n\nimport nnx",
            "from nnx.utils import seed\n\nverify_nnx_install()\n\nimport nnx",
        ),
        (
            "verify_nnx_install()\n\nimport nnx",
            "import nnx.utils\n\nverify_nnx_install()\n\nimport nnx",
        ),
    ),
    ids=(
        "function-scope",
        "autouse-disabled",
        "error-swallowed",
        "environment-mutated",
        "initial-verifier-reordered",
        "initial-verifier-removed",
        "submodule-from-import-before-verifier",
        "submodule-import-before-verifier",
    ),
)
def test_nnx_surface_installation_fixture_contract_rejects_mutations(original: str, mutation: str):
    source = (REPO_ROOT / "tests" / "nnx_surface" / "conftest.py").read_text(encoding="utf-8")
    mutated = source.replace(original, mutation, 1)

    assert mutated != source
    with pytest.raises(AssertionError):
        _assert_nnx_install_fixture_contract(mutated)


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
