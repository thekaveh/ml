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
PINNED_ATLAS_REVISION = "61c7c5103660e2226bf107c115dae42bf46f8374"

EXPECTED_MANIFEST = {
    "name": "ml-eng-lab",
    "project_name": "ml-eng-lab",
    "profile": "dev",
    "brand": {"name": "ML Eng Lab"},
    "env": {
        "file": "./atlas.env.user",
        "values": {"BASE_PORT": "auto", "JUPYTERHUB_SOURCE": "container"},
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
    source_keys = [key for key in manifest["env"]["values"] if key.endswith("_SOURCE")]
    assert source_keys == ["JUPYTERHUB_SOURCE"]


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
