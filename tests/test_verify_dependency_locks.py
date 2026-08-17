from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from scripts import verify_dependency_locks as verifier_module
from scripts.dependency_locks import load_policy
from scripts.verify_dependency_locks import main, verify_dependency_locks


REPO_ROOT = Path(__file__).resolve().parent.parent


def _copy_lock_contract(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "requirements").mkdir(parents=True)
    policy = load_policy(REPO_ROOT)
    for relative in policy.inputs + policy.outputs:
        destination = repo / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / relative, destination)
    shutil.copy2(REPO_ROOT / "requirements/lock-policy.toml", repo / "requirements")
    shutil.copy2(REPO_ROOT / "requirements/image-lock.json", repo / "requirements")
    shutil.copy2(REPO_ROOT / "Makefile", repo)
    shutil.copy2(REPO_ROOT / "Dockerfile", repo)
    (repo / ".devcontainer").mkdir()
    shutil.copy2(
        REPO_ROOT / ".devcontainer/devcontainer.json",
        repo / ".devcontainer/devcontainer.json",
    )
    (repo / ".github/workflows").mkdir(parents=True)
    for name in ("ci.yml", "docs.yml", "pages.yml", "atlas-contract.yml"):
        shutil.copy2(
            REPO_ROOT / ".github/workflows" / name,
            repo / ".github/workflows" / name,
        )
    (repo / "scripts").mkdir()
    for name in ("install_torch_stack.py", "install_locked_requirements.py"):
        shutil.copy2(REPO_ROOT / "scripts" / name, repo / "scripts")
    for relative in verifier_module._DOCUMENTATION_CONTRACT:
        destination = repo / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / relative, destination)
    return repo


def _replace_once(path: Path, old: str, new: str) -> None:
    source = path.read_text(encoding="utf-8")
    mutated = source.replace(old, new, 1)
    assert mutated != source
    path.write_text(mutated, encoding="utf-8")


def test_live_generated_lock_family_passes_offline_verification(capsys: object) -> None:
    assert verify_dependency_locks(REPO_ROOT) == ()
    assert main(["--repo-root", str(REPO_ROOT)]) == 0


@pytest.mark.parametrize(
    ("relative", "old", "new"),
    (
        ("README.md", "make lock-check", "pip install -r requirements.txt"),
        ("CONTRIBUTING.md", "make lock-write", "uv pip compile"),
        ("SECURITY.md", "requirements/lock-policy.toml", "requirements.txt"),
        (
            "docs/env-setup.md",
            "reproducible for that qualified",
            "perfectly reproducible across every platform",
        ),
        (
            "docs/architecture.md",
            "CONDA_AUTO_ACTIVATE_BASE=false",
            "CONDA_AUTO_ACTIVATE_BASE=true",
        ),
            (
                "docs/dependency-contracts.md",
                "exact resolver cutoff",
                "floating resolver horizon",
            ),
        (
            "docs/conventions.md",
            "hash-required Linux lock",
            "requirements.txt with --only-binary=thekaveh-nnx",
        ),
        (
            "docs/jupyterhub-integration.md",
            "requirements/locks/atlas-contract.txt",
            "focused dependency manifest",
        ),
        (
            "docs/nnx-library.md",
            "make install-torch-stack",
            "pip install -r requirements.txt",
        ),
        (
            "docs/notebooks/text_classification-agnews-spacy-mlp-pytorch.md",
            "en-core-web-sm==3.8.0",
            "python -m spacy download en_core_web_sm",
        ),
    ),
)
def test_offline_verifier_rejects_stale_current_documentation(
    tmp_path: Path,
    relative: str,
    old: str,
    new: str,
) -> None:
    repo = _copy_lock_contract(tmp_path)
    _replace_once(repo / relative, old, new)

    assert any(
        finding.category == "documentation" and finding.path == relative
        for finding in verify_dependency_locks(repo)
    )


@pytest.mark.parametrize(
    ("relative", "old", "new", "category"),
    (
        (
            "requirements/locks/bootstrap.txt",
            "# policy-sha256: ",
            "# policy-sha256: 0",
            "header",
        ),
        (
            "requirements/locks/bootstrap.txt",
            "# input-sha256: requirements.txt=",
            "# input-sha256: requirements.txt=0",
            "header",
        ),
        (
            "requirements/locks/bootstrap.txt",
            "--hash=sha256:",
            "--hash=sha256:xyz",
            "content",
        ),
        (
            "requirements/locks/darwin-arm64/core.txt",
            "torch==2.11.0",
            "torch==2.10.0",
            "pin",
        ),
        (
            "requirements/locks/linux-aarch64/runtime.txt",
            "pyg-lib==0.8.0+pt211cpu",
            "pyg-lib==0.8.0+pt210cpu",
            "pin",
        ),
        (
            "requirements/locks/darwin-arm64/root.txt",
            "en_core_web_sm-3.8.0-py3-none-any.whl",
            "en_core_web_sm-3.7.0-py3-none-any.whl",
            "source",
        ),
        (
            "requirements/locks/darwin-arm64/root.txt",
            "# target: darwin-arm64",
            "# target: linux-aarch64",
            "header",
        ),
        (
            "requirements/locks/darwin-arm64/root.txt",
            "# Generated by",
            "# /private/tmp/secret Generated by",
            "content",
        ),
        (
            "requirements/lock-policy.toml",
            'exclude_newer = "2026-08-17T02:21:18Z"',
            'exclude_newer = "2026-08-18T02:21:18Z"',
            "policy",
        ),
    ),
)
def test_offline_verifier_rejects_authority_and_content_mutations(
    tmp_path: Path, relative: str, old: str, new: str, category: str
) -> None:
    repo = _copy_lock_contract(tmp_path)
    _replace_once(repo / relative, old, new)

    findings = verify_dependency_locks(repo)

    assert category in {finding.category for finding in findings}
    assert all(not finding.path.startswith("/") for finding in findings)


def test_valid_but_wrong_artifact_hash_is_offline_syntactically_coherent(tmp_path: Path) -> None:
    repo = _copy_lock_contract(tmp_path)
    lock = repo / "requirements/locks/bootstrap.txt"
    source = lock.read_text(encoding="utf-8")
    start = source.index("--hash=sha256:") + len("--hash=sha256:")
    mutated = source[:start] + "0" * 64 + source[start + 64 :]
    assert mutated != source
    lock.write_text(mutated, encoding="utf-8")

    assert verify_dependency_locks(repo) == ()


def test_missing_generated_lock_has_stable_inventory_finding(tmp_path: Path) -> None:
    repo = _copy_lock_contract(tmp_path)
    (repo / "requirements/locks/compiler.txt").unlink()

    findings = verify_dependency_locks(repo)

    assert findings
    assert any(
        finding.category == "inventory"
        and finding.path == "requirements/locks/compiler.txt"
        for finding in findings
    )


def test_cli_failure_is_stable_and_redacted(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo = _copy_lock_contract(tmp_path)
    lock = repo / "requirements/locks/bootstrap.txt"
    lock.write_text("/private/tmp/secret\n", encoding="utf-8")

    assert main(["--repo-root", str(repo)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "/private" not in captured.err
    assert "dependency lock verification failed:" in captured.err


@pytest.mark.parametrize(
    ("relative", "old", "new"),
    (
        ("scripts/install_torch_stack.py", '"--require-hashes",', ""),
        ("scripts/install_torch_stack.py", '"--only-binary=:all:",', ""),
        ("scripts/install_torch_stack.py", '"--no-build-isolation",', ""),
        (
            "Makefile",
            "install-docs-lock:\n\t$(PYTHON) -m scripts.install_locked_requirements docs",
            "install-docs-lock:\n\t$(PYTHON) -m pip install -r docs-requirements.in",
        ),
    ),
)
def test_offline_verifier_rejects_installer_and_make_consumer_drift(
    tmp_path: Path, relative: str, old: str, new: str
) -> None:
    repo = _copy_lock_contract(tmp_path)
    _replace_once(repo / relative, old, new)

    findings = verify_dependency_locks(repo)

    assert "consumer" in {finding.category for finding in findings}


@pytest.mark.parametrize("event", ("schedule", "workflow_dispatch"))
def test_network_check_classifier_always_runs_both_for_unbounded_events(event: str) -> None:
    scope = verifier_module.classify_network_checks(event, ())

    assert scope.lock_check is True
    assert scope.image_check is True


@pytest.mark.parametrize(
    ("paths", "lock_check", "image_check"),
    (
        (("requirements/locks/bootstrap.txt",), True, False),
        (("bootstrap-requirements.txt",), True, False),
        (("scripts/lock_dependencies.py",), True, False),
        (("requirements/image-lock.json",), False, True),
        (("Dockerfile",), True, True),
        ((".devcontainer/devcontainer.json",), True, True),
        (("notebooks/example/notebook.ipynb",), False, False),
    ),
)
def test_network_check_classifier_maps_exact_path_groups(
    paths: tuple[str, ...], lock_check: bool, image_check: bool
) -> None:
    scope = verifier_module.classify_network_checks("pull_request", paths)

    assert (scope.lock_check, scope.image_check) == (lock_check, image_check)


@pytest.mark.parametrize("paths", (None, ("/absolute",), ("../escape",), ("",)))
def test_network_check_classifier_fails_closed_for_missing_or_ambiguous_paths(
    paths: tuple[str, ...] | None,
) -> None:
    scope = verifier_module.classify_network_checks("push", paths)

    assert scope.lock_check is True
    assert scope.image_check is True


def test_live_image_and_python_consumers_are_exactly_pinned() -> None:
    docker = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    devcontainer = (REPO_ROOT / ".devcontainer" / "devcontainer.json").read_text(
        encoding="utf-8"
    )
    workflows = tuple((REPO_ROOT / ".github" / "workflows").glob("*.yml"))

    assert docker.startswith(
        "FROM quay.io/jupyter/datascience-notebook:python-3.11@"
        "sha256:14379f24c840f27375e3a4b29a9fa55e449633ac85c9cf3806ca62d11e5603ec\n"
    )
    assert "ENV VIRTUAL_ENV=/home/jovyan/.venvs/ml-eng-lab\n" in docker
    assert "ENV CONDA_AUTO_ACTIVATE_BASE=false\n" in docker
    assert 'ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"\n' in docker
    assert 'RUN /opt/conda/bin/python -m venv "$VIRTUAL_ENV" \\\n' in docker
    assert (
        '"image": "mcr.microsoft.com/devcontainers/python:3.11-bookworm@'
        'sha256:8e95c16fbc98a4a6a8f11f5b5bd152d0ffcd4fd0f4b31bd03e95965c777d2577"'
        in devcontainer
    )
    for path in workflows:
        source = path.read_text(encoding="utf-8")
        if "actions/setup-python@" in source:
            assert 'python-version: "3.11"' not in source
            assert 'python-version: "3.11.15"' in source


def test_classifier_cli_writes_exact_fail_closed_outputs(tmp_path: Path) -> None:
    output = tmp_path / "github-output"

    assert (
        main(
            [
                "--repo-root",
                str(REPO_ROOT),
                "--classify-event",
                "schedule",
                "--github-output",
                str(output),
            ]
        )
        == 0
    )
    assert output.read_text(encoding="utf-8") == (
        "lock-check=true\nimage-check=true\n"
    )


@pytest.mark.parametrize(
    ("relative", "old", "new"),
    (
        (
            ".github/workflows/ci.yml",
            'python-version: "3.11.15"',
            'python-version: "3.11"',
        ),
        (
            ".github/workflows/ci.yml",
            "requirements/locks/linux-x86_64/root.txt\n",
            "",
        ),
        (
            ".github/workflows/docs.yml",
            "make install-docs-lock",
            "pip install -r docs-requirements.txt",
        ),
        (
            ".github/workflows/atlas-contract.yml",
            "make install-atlas-contract-lock",
            "python -m pip install uv==0.11.19",
        ),
        (
            ".github/workflows/ci.yml",
            'steps.dependency-scope.outputs.image-check == \'true\'',
            'steps.dependency-scope.outputs.image-check == \'false\'',
        ),
        (
            "Dockerfile",
            "@sha256:14379f24c840f27375e3a4b29a9fa55e449633ac85c9cf3806ca62d11e5603ec",
            "",
        ),
            (
                "Dockerfile",
                'RUN /opt/conda/bin/python -m venv "$VIRTUAL_ENV"',
                "RUN docker build --platform linux/amd64 .\n"
                'RUN /opt/conda/bin/python -m venv "$VIRTUAL_ENV"',
            ),
        (
            ".devcontainer/devcontainer.json",
            "@sha256:8e95c16fbc98a4a6a8f11f5b5bd152d0ffcd4fd0f4b31bd03e95965c777d2577",
            "",
        ),
    ),
)
def test_offline_verifier_rejects_workflow_and_image_consumer_mutations(
    tmp_path: Path, relative: str, old: str, new: str
) -> None:
    repo = _copy_lock_contract(tmp_path)
    _replace_once(repo / relative, old, new)

    findings = verify_dependency_locks(repo)

    assert any(finding.category == "consumer" for finding in findings)
