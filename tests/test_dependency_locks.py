from __future__ import annotations

import json
import shutil
import tomllib
from collections.abc import Callable
from pathlib import Path

import pytest
from packaging.version import Version

from scripts.dependency_locks import (
    DependencyLockError,
    LockedRequirement,
    TargetEnvironment,
    load_image_ledger,
    load_policy,
    parse_compiler_identity,
    parse_lock,
    project_for_target,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_URL = (
    "https://github.com/explosion/spacy-models/releases/download/"
    "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
)
MODEL_SHA256 = "1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85"
INPUTS = (
    "requirements.txt",
    "bootstrap-requirements.txt",
    "compiler-requirements.txt",
    "nlp-model-requirements.txt",
    "torch-core-requirements.txt",
    "torch-ecosystem-requirements.txt",
    "torch-requirements.txt",
    "torch-audit-requirements.txt",
    "pyg-extension-audit-requirements.txt",
    "vulnerability-audit-requirements.txt",
    "atlas-contract-requirements.txt",
    "docs-requirements.in",
)
OUTPUTS = (
    "requirements/locks/bootstrap.txt",
    "requirements/locks/compiler.txt",
    "requirements/locks/audit.txt",
    "requirements/locks/atlas-contract.txt",
    "requirements/locks/darwin-arm64/core.txt",
    "requirements/locks/darwin-arm64/runtime.txt",
    "requirements/locks/darwin-arm64/root.txt",
    "requirements/locks/linux-x86_64/core.txt",
    "requirements/locks/linux-x86_64/runtime.txt",
    "requirements/locks/linux-x86_64/root.txt",
    "requirements/locks/linux-aarch64/core.txt",
    "requirements/locks/linux-aarch64/runtime.txt",
    "requirements/locks/linux-aarch64/root.txt",
    "docs-requirements.txt",
)
TARGETS = {
    "darwin-arm64": {
        "uv_platform": "aarch64-apple-darwin",
        "system": "Darwin",
        "machine": "arm64",
        "marker_environment": {
            "implementation_name": "cpython",
            "os_name": "posix",
            "platform_machine": "arm64",
            "platform_python_implementation": "CPython",
            "platform_system": "Darwin",
            "python_version": "3.11",
            "sys_platform": "darwin",
        },
    },
    "linux-x86_64": {
        "uv_platform": "x86_64-manylinux_2_28",
        "system": "Linux",
        "machine": "x86_64",
        "marker_environment": {
            "implementation_name": "cpython",
            "os_name": "posix",
            "platform_machine": "x86_64",
            "platform_python_implementation": "CPython",
            "platform_system": "Linux",
            "python_version": "3.11",
            "sys_platform": "linux",
        },
    },
    "linux-aarch64": {
        "uv_platform": "aarch64-manylinux_2_28",
        "system": "Linux",
        "machine": "aarch64",
        "marker_environment": {
            "implementation_name": "cpython",
            "os_name": "posix",
            "platform_machine": "aarch64",
            "platform_python_implementation": "CPython",
            "platform_system": "Linux",
            "python_version": "3.11",
            "sys_platform": "linux",
        },
    },
}


def _copy_policy_inputs(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "requirements").mkdir()
    for relative in INPUTS:
        shutil.copy2(REPO_ROOT / relative, repo / relative)
    shutil.copy2(REPO_ROOT / "requirements" / "lock-policy.toml", repo / "requirements")
    shutil.copy2(REPO_ROOT / "requirements" / "image-lock.json", repo / "requirements")
    return repo


def _target(key: str, *, python_full_version: str = "3.11.0") -> TargetEnvironment:
    values = dict(TARGETS[key]["marker_environment"])
    # packaging.Marker requires these standard keys even though lock expressions may not use them.
    values["implementation_version"] = python_full_version
    values["python_full_version"] = python_full_version
    values["extra"] = ""
    return TargetEnvironment(key=key, marker_environment=values)


def test_live_policy_has_the_exact_supported_matrix_and_inventory() -> None:
    policy = load_policy(REPO_ROOT)

    assert policy.schema == 1
    assert policy.compiler_input == Path("compiler-requirements.txt")
    assert policy.python_floor == Version("3.11.0")
    assert policy.macos_deployment_target == "13.0"
    assert tuple(path.as_posix() for path in policy.inputs) == INPUTS
    assert tuple(path.as_posix() for path in policy.outputs) == OUTPUTS
    assert tuple(platform.key for platform in policy.platforms) == tuple(TARGETS)
    for platform in policy.platforms:
        expected = TARGETS[platform.key]
        assert platform.uv_platform == expected["uv_platform"]
        assert platform.system == expected["system"]
        assert platform.machine == expected["machine"]
        assert dict(platform.marker_environment) == expected["marker_environment"]


def test_live_direct_inputs_have_one_compiler_authority_and_exact_exceptions() -> None:
    assert parse_compiler_identity(REPO_ROOT) == Version("0.11.19")
    assert (REPO_ROOT / "bootstrap-requirements.txt").read_text(encoding="utf-8") == (
        "pip==26.1.2\nsetuptools==81.0.0\nwheel==0.47.0\npackaging==26.2\n"
    )
    assert (REPO_ROOT / "compiler-requirements.txt").read_text(encoding="utf-8") == "uv==0.11.19\n"
    assert (REPO_ROOT / "nlp-model-requirements.txt").read_text(encoding="utf-8") == (
        f"en-core-web-sm @ {MODEL_URL} --hash=sha256:{MODEL_SHA256}\n"
    )
    assert (REPO_ROOT / "atlas-contract-requirements.txt").read_text(encoding="utf-8") == (
        "pytest==9.0.3\npyyaml==6.0.3\nuv==0.11.19\n"
    )
    assert "--find-links" not in (REPO_ROOT / "torch-requirements.txt").read_text(encoding="utf-8")
    policy = tomllib.loads(
        (REPO_ROOT / "requirements/lock-policy.toml").read_text(encoding="utf-8")
    )
    assert policy["package_sources"]["torchao"] == {
        "darwin-arm64": "pypi",
        "linux-x86_64": "torch_cpu",
        "linux-aarch64": "torch_cpu",
    }


def test_live_image_ledger_has_two_exact_native_multi_platform_identities() -> None:
    images = load_image_ledger(REPO_ROOT)

    assert tuple(image.consumer for image in images) == ("docker", "devcontainer")
    assert images[0].source == "quay.io/jupyter/datascience-notebook:python-3.11"
    assert images[0].index_digest == (
        "sha256:14379f24c840f27375e3a4b29a9fa55e449633ac85c9cf3806ca62d11e5603ec"
    )
    assert images[1].source == "mcr.microsoft.com/devcontainers/python:3.11-bookworm"
    assert images[1].index_digest == (
        "sha256:8e95c16fbc98a4a6a8f11f5b5bd152d0ffcd4fd0f4b31bd03e95965c777d2577"
    )
    for image in images:
        assert tuple(image.children) == ("linux/amd64", "linux/arm64")
        assert all(value.startswith("sha256:") and len(value) == 71 for value in image.children.values())


def test_lock_parser_preserves_and_projects_supported_markers(tmp_path: Path) -> None:
    lock = tmp_path / "lock.txt"
    lock.write_text(
        "demo==1.0 ; sys_platform == 'darwin' \\\n    --hash=sha256:" + "a" * 64 + "\n"
        "demo==2.0 ; sys_platform == 'linux' \\\n    --hash=sha256:" + "b" * 64 + "\n",
        encoding="utf-8",
    )

    parsed = parse_lock(lock)

    assert tuple(item.marker for item in parsed) == (
        'sys_platform == "darwin"',
        'sys_platform == "linux"',
    )
    assert tuple(item.version for item in project_for_target(parsed, _target("darwin-arm64"))) == (
        Version("1.0"),
    )
    assert tuple(item.version for item in project_for_target(parsed, _target("linux-aarch64"))) == (
        Version("2.0"),
    )


@pytest.mark.parametrize(
    "marker",
    (
        "python_full_version >= '3.11.10'",
        "python_full_version >= '3.11.15'",
        "implementation_version >= '3.11.10'",
        "implementation_version >= '3.11.15'",
        "extra == 'test'",
        "platform_release == 'unsafe'",
    ),
)
def test_lock_parser_rejects_patch_sensitive_or_unbounded_markers(
    tmp_path: Path, marker: str
) -> None:
    lock = tmp_path / "lock.txt"
    lock.write_text(
        f"demo==1.0 ; {marker} \\\n    --hash=sha256:" + "c" * 64 + "\n",
        encoding="utf-8",
    )

    with pytest.raises(DependencyLockError, match="marker"):
        parse_lock(lock)


def test_projection_rejects_overlapping_same_name_branches() -> None:
    requirements = (
        LockedRequirement("demo", Version("1.0"), ("a" * 64,), None, None),
        LockedRequirement("demo", Version("2.0"), ("b" * 64,), None, 'sys_platform == "linux"'),
    )

    with pytest.raises(DependencyLockError, match="overlap"):
        project_for_target(requirements, _target("linux-x86_64"))


def test_lock_parser_accepts_only_disjoint_same_name_branches(tmp_path: Path) -> None:
    lock = tmp_path / "lock.txt"
    lock.write_text(
        "demo==1.0 ; sys_platform == 'darwin' \\\n    --hash=sha256:" + "a" * 64 + "\n"
        "demo==2.0 ; sys_platform == 'linux' \\\n    --hash=sha256:" + "b" * 64 + "\n",
        encoding="utf-8",
    )

    parsed = parse_lock(lock)

    for key in TARGETS:
        for patch in ("3.11.0", "3.11.10", "3.11.15"):
            selected = project_for_target(parsed, _target(key, python_full_version=patch))
            assert len(selected) == 1


@pytest.mark.parametrize(
    "requirements",
    (
        "demo==1.0 \\\n    --hash=sha256:{hash}\n"
        "demo==2.0 ; sys_platform == 'linux' \\\n    --hash=sha256:{hash}\n",
        "demo==1.0 ; sys_platform == 'linux' \\\n    --hash=sha256:{hash}\n"
        "demo==2.0 ; platform_system == 'Linux' \\\n    --hash=sha256:{hash}\n",
    ),
)
def test_lock_parser_rejects_overlapping_branches(tmp_path: Path, requirements: str) -> None:
    lock = tmp_path / "lock.txt"
    lock.write_text(requirements.format(hash="d" * 64), encoding="utf-8")

    with pytest.raises(DependencyLockError, match="overlap"):
        parse_lock(lock)


def test_lock_parser_rejects_a_marker_matching_no_supported_target(tmp_path: Path) -> None:
    lock = tmp_path / "lock.txt"
    lock.write_text(
        "demo==1.0 ; sys_platform == 'win32' \\\n    --hash=sha256:" + "e" * 64 + "\n",
        encoding="utf-8",
    )

    with pytest.raises(DependencyLockError, match="no supported target"):
        parse_lock(lock)


@pytest.mark.parametrize(
    "requirement",
    (
        "demo==1.0",
        "demo==1.0 --hash=sha256:abc",
        "demo==1.0 --hash=sha512:{hash}",
        "demo==1.0 --hash=sha256:{hash} --hash=sha256:{hash}",
    ),
)
def test_lock_parser_rejects_missing_malformed_or_duplicate_hashes(
    tmp_path: Path, requirement: str
) -> None:
    lock = tmp_path / "lock.txt"
    lock.write_text(requirement.format(hash="f" * 64) + "\n", encoding="utf-8")

    with pytest.raises(DependencyLockError, match="hash"):
        parse_lock(lock)


def test_policy_rejects_a_second_compiler_version_authority(tmp_path: Path) -> None:
    repo = _copy_policy_inputs(tmp_path)
    policy = repo / "requirements" / "lock-policy.toml"
    source = policy.read_text(encoding="utf-8")
    mutated = source.replace("schema_version = 1", 'schema_version = 1\ncompiler_version = "0.11.19"', 1)
    assert mutated != source
    policy.write_text(mutated, encoding="utf-8")

    with pytest.raises(DependencyLockError, match="unknown|compiler"):
        load_policy(repo)


def test_policy_rejects_duplicate_toml_keys(tmp_path: Path) -> None:
    repo = _copy_policy_inputs(tmp_path)
    policy = repo / "requirements" / "lock-policy.toml"
    source = policy.read_text(encoding="utf-8")
    mutated = source.replace("schema_version = 1", "schema_version = 1\nschema_version = 1", 1)
    assert mutated != source
    policy.write_text(mutated, encoding="utf-8")

    with pytest.raises(DependencyLockError, match="duplicate|invalid TOML"):
        load_policy(repo)


def test_policy_rejects_atlas_compiler_version_drift(tmp_path: Path) -> None:
    repo = _copy_policy_inputs(tmp_path)
    atlas = repo / "atlas-contract-requirements.txt"
    source = atlas.read_text(encoding="utf-8")
    mutated = source.replace("uv==0.11.19", "uv==0.11.18", 1)
    assert mutated != source
    atlas.write_text(mutated, encoding="utf-8")

    with pytest.raises(DependencyLockError, match="compiler|uv"):
        load_policy(repo)


@pytest.mark.parametrize(
    ("old", "new", "match"),
    (
        ('direct_url_packages = ["en-core-web-sm"]', 'direct_url_packages = ["rogue"]', "direct URL"),
        ('sdist_packages = ["python-louvain"]', 'sdist_packages = ["rogue"]', "sdist"),
        ('pypi = "https://pypi.org/simple"', 'pypi = "https://mirror.invalid/simple"', "source"),
        (
            'pyg = "https://data.pyg.org/whl/torch-2.11.0+cpu.html"',
            'pyg = "https://data.pyg.org/whl/torch-2.10.0+cpu.html"',
            "source",
        ),
        ('uv_platform = "aarch64-apple-darwin"', 'uv_platform = "x86_64-apple-darwin"', "platform"),
        ('python_floor = "3.11.0"', 'python_floor = "3.11.10"', "floor"),
        ('system = "Darwin"', 'system = "Linux"', "platform"),
        ('machine = "arm64"', 'machine = "x86_64"', "platform"),
        ('sys_platform = "darwin"', 'sys_platform = "linux"', "marker"),
    ),
)
def test_policy_rejects_authoritative_identity_mutations(
    tmp_path: Path, old: str, new: str, match: str
) -> None:
    repo = _copy_policy_inputs(tmp_path)
    policy = repo / "requirements" / "lock-policy.toml"
    source = policy.read_text(encoding="utf-8")
    mutated = source.replace(old, new, 1)
    assert mutated != source
    policy.write_text(mutated, encoding="utf-8")

    with pytest.raises(DependencyLockError, match=match):
        load_policy(repo)


@pytest.mark.parametrize(
    "unsafe",
    (
        "--extra-index-url https://mirror.invalid/simple",
        "--index-url https://mirror.invalid/simple",
        "--find-links https://mirror.invalid/wheels",
        "-e git+https://github.com/example/repo.git#egg=rogue",
        "rogue @ git+https://github.com/example/repo.git@deadbeef",
        "rogue @ file:///tmp/rogue.whl",
        "../rogue",
        "${LOCK_OVERRIDE}",
        "$(touch unsafe)",
        "`touch unsafe`",
        "rogue @ https://example.invalid/rogue-1.0-py3-none-any.whl",
    ),
)
def test_policy_rejects_unsafe_requirement_input_syntax(tmp_path: Path, unsafe: str) -> None:
    repo = _copy_policy_inputs(tmp_path)
    manifest = repo / "requirements.txt"
    manifest.write_text(manifest.read_text(encoding="utf-8") + unsafe + "\n", encoding="utf-8")

    with pytest.raises(DependencyLockError, match="unsafe|direct URL|source"):
        load_policy(repo)


def test_policy_rejects_missing_and_unknown_inventory_entries(tmp_path: Path) -> None:
    repo = _copy_policy_inputs(tmp_path)
    policy = repo / "requirements" / "lock-policy.toml"
    source = policy.read_text(encoding="utf-8")
    mutated = source.replace('  "docs-requirements.in",\n]', '  "rogue-requirements.in",\n]', 1)
    assert mutated != source
    policy.write_text(mutated, encoding="utf-8")

    with pytest.raises(DependencyLockError, match="inventory"):
        load_policy(repo)


def test_policy_rejects_unknown_output_inventory_entry(tmp_path: Path) -> None:
    repo = _copy_policy_inputs(tmp_path)
    policy = repo / "requirements" / "lock-policy.toml"
    source = policy.read_text(encoding="utf-8")
    mutated = source.replace(
        '  "docs-requirements.txt",\n]', '  "requirements/locks/rogue.txt",\n]', 1
    )
    assert mutated != source
    policy.write_text(mutated, encoding="utf-8")

    with pytest.raises(DependencyLockError, match="inventory"):
        load_policy(repo)


def test_policy_rejects_model_hash_drift(tmp_path: Path) -> None:
    repo = _copy_policy_inputs(tmp_path)
    model = repo / "nlp-model-requirements.txt"
    source = model.read_text(encoding="utf-8")
    mutated = source.replace(MODEL_SHA256, "0" * 64, 1)
    assert mutated != source
    model.write_text(mutated, encoding="utf-8")

    with pytest.raises(DependencyLockError, match="model|hash"):
        load_policy(repo)


def test_policy_rejects_stale_pyg_source_directive(tmp_path: Path) -> None:
    repo = _copy_policy_inputs(tmp_path)
    runtime = repo / "torch-requirements.txt"
    source = runtime.read_text(encoding="utf-8")
    mutated = source.replace(
        "-r torch-ecosystem-requirements.txt\n",
        "-r torch-ecosystem-requirements.txt\n"
        "--find-links https://data.pyg.org/whl/torch-2.11.0+cpu.html\n",
        1,
    )
    assert mutated != source
    runtime.write_text(mutated, encoding="utf-8")

    with pytest.raises(DependencyLockError, match="source"):
        load_policy(repo)


def _mutate_missing_architecture(document: dict[str, object]) -> None:
    images = document["images"]
    assert isinstance(images, list)
    images[0]["children"].pop("linux/arm64")


def _mutate_malformed_digest(document: dict[str, object]) -> None:
    images = document["images"]
    assert isinstance(images, list)
    images[0]["index_digest"] = "sha256:abc"


def _mutate_child_index_substitution(document: dict[str, object]) -> None:
    images = document["images"]
    assert isinstance(images, list)
    images[0]["children"]["linux/amd64"] = images[0]["index_digest"]


def _mutate_tag_only_violation(document: dict[str, object]) -> None:
    images = document["images"]
    assert isinstance(images, list)
    images[0]["source"] += "@" + images[0]["index_digest"]


@pytest.mark.parametrize(
    ("mutator", "match"),
    (
        (_mutate_missing_architecture, "architecture"),
        (_mutate_malformed_digest, "digest"),
        (_mutate_child_index_substitution, "substitutes"),
        (_mutate_tag_only_violation, "tag-only"),
    ),
)
def test_image_ledger_rejects_identity_and_architecture_mutations(
    tmp_path: Path, mutator: Callable[[dict[str, object]], None], match: str
) -> None:
    repo = _copy_policy_inputs(tmp_path)
    ledger = repo / "requirements" / "image-lock.json"
    document = json.loads(ledger.read_text(encoding="utf-8"))
    before = json.dumps(document, sort_keys=True)
    mutator(document)
    after = json.dumps(document, sort_keys=True)
    assert after != before
    ledger.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(DependencyLockError, match=match):
        load_image_ledger(repo)


def test_image_ledger_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    repo = _copy_policy_inputs(tmp_path)
    ledger = repo / "requirements" / "image-lock.json"
    document = json.loads(ledger.read_text(encoding="utf-8"))
    source = json.dumps(document, separators=(",", ":"))
    mutated = source.replace('"schema_version":1', '"schema_version":1,"schema_version":1', 1)
    assert mutated != source
    ledger.write_text(mutated + "\n", encoding="utf-8")

    with pytest.raises(DependencyLockError, match="duplicate"):
        load_image_ledger(repo)
