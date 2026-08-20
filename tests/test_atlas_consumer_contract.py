"""Static contract for the parent repository's Atlas consumer configuration."""

from __future__ import annotations

import configparser
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml


REPO = Path(__file__).resolve().parents[1]
MANIFEST_NAME = "atlas.consumer.yml"
OVERLAY_NAME = "compose/ml-eng-lab-atlas.yml"
PINNED_ATLAS_REVISION = "41ba856f7cd35f0b559d6875e08443eac3e98a98"
ATLAS_BUILD = Path("services/jupyterhub/build")
NLP_PROJECTIONS = {
    "requirements/nlp-assets.toml": ATLAS_BUILD / "nlp-assets.toml",
    "scripts/nlp_assets.py": ATLAS_BUILD / "install_nlp_assets.py",
    "nlp-model-requirements.txt": ATLAS_BUILD / "nlp-model-requirements.txt",
}
EXPECTED_NLP_DOCKER_BLOCK = """COPY --chown=${NB_UID}:${NB_GID} nlp-model-requirements.txt /tmp/nlp-model-requirements.txt
COPY --chown=${NB_UID}:${NB_GID} nlp-assets.toml /tmp/nlp-assets.toml
COPY --chown=${NB_UID}:${NB_GID} install_nlp_assets.py /tmp/install_nlp_assets.py
RUN python -m pip install --no-cache-dir --no-deps --require-hashes -r /tmp/nlp-model-requirements.txt \\
 && python /tmp/install_nlp_assets.py install --manifest /tmp/nlp-assets.toml --data-dir /home/jovyan/nltk_data \\
 && python /tmp/install_nlp_assets.py verify --manifest /tmp/nlp-assets.toml --data-dir /home/jovyan/nltk_data \\
 && rm -f /tmp/nlp-model-requirements.txt /tmp/nlp-assets.toml /tmp/install_nlp_assets.py

ENV NLTK_DATA=/home/jovyan/nltk_data"""

EXPECTED_MANIFEST = {
    "name": "ml-eng-lab",
    "project_name": "ml-eng-lab",
    "profile": "dev",
    "brand": {"name": "ML Eng Lab"},
    "env": {
        "file": "./atlas.env.user",
        "values": {
            "BASE_PORT": "auto",
            "JUPYTERHUB_SOURCE": "container",
            "LLM_PROVIDER_SOURCE": "ollama-localhost",
        },
    },
    "compose_overlays": ["./compose/ml-eng-lab-atlas.yml"],
}


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def validate_manifest(path: Path) -> None:
    manifest = load_yaml(path)
    assert manifest == EXPECTED_MANIFEST
    assert "track" not in manifest
    assert manifest["env"]["values"]["BASE_PORT"] == "auto"
    assert manifest["env"]["values"]["JUPYTERHUB_SOURCE"] == "container"
    assert manifest["env"]["values"]["LLM_PROVIDER_SOURCE"] == "ollama-localhost"
    source_keys = [key for key in manifest["env"]["values"] if key.endswith("_SOURCE")]
    assert source_keys == ["JUPYTERHUB_SOURCE", "LLM_PROVIDER_SOURCE"]


def validate_overlay(path: Path) -> None:
    overlay = load_yaml(path)
    assert overlay == {
        "services": {
            "jupyterhub": {
                "volumes": ["${ML_ENG_LAB_REPO_PATH}:/home/jovyan/work/ml-eng-lab"]
            }
        }
    }
    mount = overlay["services"]["jupyterhub"]["volumes"][0]
    source, target = mount.split(":", maxsplit=1)
    assert source == "${ML_ENG_LAB_REPO_PATH}"
    assert target == "/home/jovyan/work/ml-eng-lab"
    assert "ssh" not in mount.lower()


def copy_contract(tmp_path: Path) -> tuple[Path, Path]:
    manifest = tmp_path / MANIFEST_NAME
    overlay = tmp_path / OVERLAY_NAME
    overlay.parent.mkdir()
    shutil.copy(REPO / MANIFEST_NAME, manifest)
    shutil.copy(REPO / OVERLAY_NAME, overlay)
    return manifest, overlay


def write_yaml(path: Path, value: dict) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def test_real_atlas_consumer_files_match_the_contract():
    validate_manifest(REPO / MANIFEST_NAME)
    validate_overlay(REPO / OVERLAY_NAME)
    assert (REPO / "atlas.env.user.example").read_text(encoding="utf-8") == (
        "# Copy/create this as atlas.env.user; it is ignored and machine-local.\n"
        "ML_ENG_LAB_REPO_PATH=/absolute/path/to/ml-eng-lab\n"
        "# Optional: only when the native Ollama daemon uses a non-default port.\n"
        "# OLLAMA_LOCALHOST_PORT=11434\n"
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda manifest: manifest.update({"track": "main"}), "track"),
        (
            lambda manifest: manifest["env"]["values"].update({"BASE_PORT": 8000}),
            "BASE_PORT",
        ),
        (
            lambda manifest: manifest["env"]["values"].update({"JUPYTERHUB_SOURCE": "host"}),
            "JUPYTERHUB_SOURCE",
        ),
        (
            lambda manifest: manifest["env"]["values"].update({"OPEN_WEBUI_SOURCE": "container"}),
            "OPEN_WEBUI_SOURCE",
        ),
        (
            lambda manifest: manifest["env"]["values"].pop("LLM_PROVIDER_SOURCE"),
            "LLM_PROVIDER_SOURCE",
        ),
        *[
            (
                lambda manifest, source=source: manifest["env"]["values"].update(
                    {"LLM_PROVIDER_SOURCE": source}
                ),
                "LLM_PROVIDER_SOURCE",
            )
            for source in ("auto", "ollama-container-cpu", "ollama-container-gpu")
        ],
        *[
            (
                lambda manifest, source=source: manifest["env"]["values"].update(
                    {"COMFYUI_SOURCE": source}
                ),
                "COMFYUI_SOURCE",
            )
            for source in ("container-cpu", "container-gpu")
        ],
        (
            lambda manifest: manifest["compose_overlays"].append("./compose/extra.yml"),
            "compose_overlays",
        ),
    ],
)
def test_manifest_rejects_prohibited_configuration(tmp_path, mutate, message):
    manifest_path, _ = copy_contract(tmp_path)
    manifest = load_yaml(manifest_path)
    mutate(manifest)
    write_yaml(manifest_path, manifest)

    with pytest.raises(AssertionError, match=message):
        validate_manifest(manifest_path)


@pytest.mark.parametrize(
    ("mount", "message"),
    [
        ("${HOME}/.ssh:/home/jovyan/.ssh", "services"),
        ("./:/home/jovyan/work/ml-eng-lab", "services"),
        ("${ML_ENG_LAB_REPO_PATH}:/home/jovyan/work", "services"),
    ],
)
def test_overlay_rejects_unsafe_or_ambiguous_mounts(tmp_path, mount, message):
    _, overlay_path = copy_contract(tmp_path)
    overlay = load_yaml(overlay_path)
    overlay["services"]["jupyterhub"]["volumes"] = [mount]
    write_yaml(overlay_path, overlay)

    with pytest.raises(AssertionError, match=message):
        validate_overlay(overlay_path)


def parse_gitmodules(path: Path) -> configparser.ConfigParser:
    parser = configparser.ConfigParser()
    parser.read(path, encoding="utf-8")
    return parser


def validate_atlas_submodule(gitmodules_path: Path) -> None:
    parser = parse_gitmodules(gitmodules_path)
    section = parser["submodule \"infra\""]
    assert "branch" not in section
    assert dict(section) == {"path": "infra", "url": "https://github.com/thekaveh/atlas.git"}


def validate_atlas_nlp_projection(
    *, atlas_root: Path, actual_revision: str, ledger: str
) -> None:
    assert actual_revision == PINNED_ATLAS_REVISION
    assert f"Current Atlas `infra` gitlink SHA: `{PINNED_ATLAS_REVISION}`." in ledger
    for parent_relative, atlas_relative in NLP_PROJECTIONS.items():
        assert (atlas_root / atlas_relative).read_bytes() == (
            REPO / parent_relative
        ).read_bytes()
    dockerfile = (atlas_root / ATLAS_BUILD / "Dockerfile").read_text(encoding="utf-8")
    assert dockerfile.count(EXPECTED_NLP_DOCKER_BLOCK) == 1


def test_atlas_submodule_is_detached_at_the_required_revision(tmp_path):
    gitmodules = tmp_path / ".gitmodules"
    shutil.copy(REPO / ".gitmodules", gitmodules)
    validate_atlas_submodule(gitmodules)
    assert (
        subprocess.run(
            ["git", "rev-parse", ":infra"], cwd=REPO, check=True, text=True, capture_output=True
        ).stdout.strip()
        == PINNED_ATLAS_REVISION
    )


def test_atlas_nlp_projection_is_byte_exact_and_ordered() -> None:
    validate_atlas_nlp_projection(
        atlas_root=REPO / "infra",
        actual_revision=subprocess.run(
            ["git", "rev-parse", ":infra"],
            cwd=REPO,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip(),
        ledger=(REPO / "docs/dependency-contracts.md").read_text(encoding="utf-8"),
    )


@pytest.mark.parametrize(
    ("relative", "old", "new"),
    (
        (ATLAS_BUILD / "nlp-assets.toml", b"raw.githubusercontent.com", b"example.invalid"),
        (ATLAS_BUILD / "install_nlp_assets.py", b"sha256", b"sha265"),
        (ATLAS_BUILD / "nlp-model-requirements.txt", b"en_core_web_sm-3.8.0", b"en_core_web_sm-3.7.0"),
        (ATLAS_BUILD / "Dockerfile", b" --chown=${NB_UID}:${NB_GID}", b""),
        (ATLAS_BUILD / "Dockerfile", b"install_nlp_assets.py verify", b"install_nlp_assets.py verify-later"),
    ),
)
def test_atlas_nlp_projection_rejects_independent_mutations(
    tmp_path: Path, relative: Path, old: bytes, new: bytes
) -> None:
    atlas = tmp_path / "infra"
    shutil.copytree(REPO / "infra" / ATLAS_BUILD, atlas / ATLAS_BUILD)
    target = atlas / relative
    source = target.read_bytes()
    assert old in source
    target.write_bytes(source.replace(old, new, 1))

    with pytest.raises(AssertionError):
        validate_atlas_nlp_projection(
            atlas_root=atlas,
            actual_revision=PINNED_ATLAS_REVISION,
            ledger=f"Current Atlas `infra` gitlink SHA: `{PINNED_ATLAS_REVISION}`.",
        )


def test_atlas_nlp_projection_rejects_gitlink_and_ledger_drift() -> None:
    ledger = (REPO / "docs/dependency-contracts.md").read_text(encoding="utf-8")
    with pytest.raises(AssertionError):
        validate_atlas_nlp_projection(
            atlas_root=REPO / "infra", actual_revision="0" * 40, ledger=ledger
        )
    with pytest.raises(AssertionError):
        validate_atlas_nlp_projection(
            atlas_root=REPO / "infra",
            actual_revision=PINNED_ATLAS_REVISION,
            ledger=ledger.replace(PINNED_ATLAS_REVISION, "0" * 40),
        )


def test_atlas_submodule_rejects_a_moving_branch(tmp_path):
    gitmodules = tmp_path / ".gitmodules"
    shutil.copy(REPO / ".gitmodules", gitmodules)
    gitmodules.write_text(
        gitmodules.read_text(encoding="utf-8").replace(
            "url = https://github.com/thekaveh/atlas.git",
            "url = https://github.com/thekaveh/atlas.git\n\tbranch = main",
        ),
        encoding="utf-8",
    )

    with pytest.raises(AssertionError, match="branch"):
        validate_atlas_submodule(gitmodules)
