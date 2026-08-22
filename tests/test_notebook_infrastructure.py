from __future__ import annotations

from pathlib import Path

import pytest

from scripts.docs.manifest import load_manifest, parse_manifest
from scripts.docs.notebook_infrastructure import (
    AtlasTaskContract,
    NotebookInfrastructureError,
    load_atlas_task_contracts,
    render_atlas_task_table,
    verify_atlas_task_table,
    write_atlas_task_table,
)


def _manifest(tasks: list[str]):
    entries = "\n".join(
        f"""  - task: {task}
    number: \"1\"
    family: test
    depth: full
    doc: docs/notebooks/{task}.md
    spec: notebooks/{task}/docs/spec.yaml"""
        for task in tasks
    )
    return parse_manifest(
        f"""surfaces: [repo, site, wiki]
numbering: baked
sections: []
notebooks:
{entries}
diagrams: []
"""
    )


def _spec(atlas: str) -> str:
    return f"""title: Example task
tier: A
atlas:
{atlas}
"""


def _write_spec(repo: Path, task: str, atlas: str) -> None:
    path = repo / f"notebooks/{task}/docs/spec.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_spec(atlas), encoding="utf-8")


def _write_active_tasks(repo: Path, tasks: list[str]) -> None:
    config = repo / "scripts/verify_repo_config.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text("active_task_dirs:\n" + "\n".join(f"  - {task}" for task in tasks) + "\n", encoding="utf-8")


VALID_ATLAS = """  executor: jupyterhub
  default_mode: vscode-remote
  required_services: [jupyterhub, postgres]
  workspace_access: remote
  artifact_policy: atlas-jupyter-volume
  constraints: [\"Use a dedicated kernel\", \"Keep data private\"]"""

MOUNTED_WORKSPACE_ATLAS = (
    VALID_ATLAS.replace("default_mode: vscode-remote", "default_mode: mounted-workspace")
    .replace("workspace_access: remote", "workspace_access: mounted-required")
    .replace("artifact_policy: atlas-jupyter-volume", "artifact_policy: task-local-ignored-paths")
)


def test_loads_contracts_in_manifest_order(tmp_path):
    _write_spec(tmp_path, "second", VALID_ATLAS)
    _write_spec(tmp_path, "first", VALID_ATLAS)
    _write_active_tasks(tmp_path, ["first", "second"])

    contracts = load_atlas_task_contracts(tmp_path, _manifest(["second", "first"]))

    assert [contract.task for contract in contracts] == ["second", "first"]
    assert contracts[0] == AtlasTaskContract(
        task="second",
        title="Example task",
        tier="A",
        executor="jupyterhub",
        default_mode="vscode-remote",
        required_services=("jupyterhub", "postgres"),
        workspace_access="remote",
        artifact_policy="atlas-jupyter-volume",
        constraints=("Use a dedicated kernel", "Keep data private"),
    )


def test_loads_mounted_workspace_contracts_with_a_matching_default_mode(tmp_path):
    _write_spec(tmp_path, "task", MOUNTED_WORKSPACE_ATLAS)
    _write_active_tasks(tmp_path, ["task"])

    contracts = load_atlas_task_contracts(tmp_path, _manifest(["task"]))

    assert contracts[0].default_mode == "mounted-workspace"
    assert contracts[0].workspace_access == "mounted-required"


def test_repository_declares_contracts_for_every_manifest_notebook():
    repo_root = Path(__file__).resolve().parent.parent

    contracts = load_atlas_task_contracts(
        repo_root, load_manifest(repo_root / "docs/manifest.yaml", repo_root)
    )

    assert len(contracts) == 21
    assert [contract.task for contract in contracts] == [
        "tabular_classification-iris-mlp-pytorch",
        "tabular_regression-diabetes-mlp-pytorch",
        "image_classification-mnist-ffnn-numpy",
        "image_classification-mnist-ffnn-pytorch",
        "model_surgery-mnist-ffnn-pytorch",
        "knowledge_distillation-mnist-ffnn-pytorch",
        "pruning-mnist-ffnn-pytorch",
        "quantization-mnist-ffnn-pytorch",
        "moe-fmnist-mixture-of-experts-pytorch",
        "diffusion-mnist-ddpm-pytorch",
        "self_supervised-fmnist-jepa-pytorch",
        "peft-mnist-to-fmnist-dora-vs-lora-pytorch",
        "node_classification-reddit-gnn-pyg",
        "link_prediction-karate-graphsage-pyg",
        "community_detection-karate-louvain-vs-gnn-pyg",
        "text_generation-tinyshakespeare-transformer-pytorch",
        "text_classification-agnews-spacy-mlp-pytorch",
        "sentiment_classification-vader-mlp-pytorch",
        "preference_alignment-toy-dpo-pytorch",
        "dim_reduction-iris-autoencoder-pytorch",
        "clustering-iris-kmeans-vs-ae-pytorch",
    ]
    quantization = next(
        contract
        for contract in contracts
        if contract.task == "quantization-mnist-ffnn-pytorch"
    )
    assert quantization.tier.lower() == "b"
    assert all("Issue #66" not in constraint for constraint in quantization.constraints)
    assert any("one epoch" in constraint for constraint in quantization.constraints)
    assert any("checkpoint" in constraint for constraint in quantization.constraints)


def test_load_rejects_manifest_tasks_that_drift_from_active_task_config(tmp_path):
    _write_spec(tmp_path, "task", VALID_ATLAS)
    _write_active_tasks(tmp_path, ["other-task"])

    with pytest.raises(NotebookInfrastructureError, match="active_task_dirs"):
        load_atlas_task_contracts(tmp_path, _manifest(["task"]))


@pytest.mark.parametrize(
    ("atlas", "message"),
    [
        ("  - executor: jupyterhub", "atlas must be a mapping"),
        (VALID_ATLAS.replace("executor: jupyterhub", "executor: local"), "executor"),
        (VALID_ATLAS.replace("default_mode: vscode-remote", "default_mode: terminal"), "default_mode"),
        (VALID_ATLAS.replace("[jupyterhub, postgres]", "[]"), "required_services"),
        (VALID_ATLAS.replace("[jupyterhub, postgres]", "[postgres]"), "jupyterhub"),
        (VALID_ATLAS.replace("[jupyterhub, postgres]", "[jupyterhub, jupyterhub]"), "unique"),
        (VALID_ATLAS.replace("[jupyterhub, postgres]", "[jupyterhub, bad service]"), "service ID"),
        (VALID_ATLAS.replace("workspace_access: remote", "workspace_access: local"), "workspace_access"),
        (MOUNTED_WORKSPACE_ATLAS.replace("workspace_access: mounted-required", "workspace_access: remote"), "mounted-required"),
        (VALID_ATLAS.replace("workspace_access: remote", "workspace_access: mounted-required"), "mounted-workspace"),
        (VALID_ATLAS.replace("artifact_policy: atlas-jupyter-volume", "artifact_policy: shared"), "artifact_policy"),
        (VALID_ATLAS.replace("constraints: [\"Use a dedicated kernel\", \"Keep data private\"]", "constraints: no"), "constraints"),
        (VALID_ATLAS.replace("constraints: [\"Use a dedicated kernel\", \"Keep data private\"]", "constraints: [\"\"]"), "non-empty"),
    ],
)
def test_load_rejects_invalid_atlas_fields(tmp_path, atlas, message):
    _write_spec(tmp_path, "task", atlas)
    _write_active_tasks(tmp_path, ["task"])

    with pytest.raises(NotebookInfrastructureError, match=message):
        load_atlas_task_contracts(tmp_path, _manifest(["task"]))


def test_renders_deterministic_markdown_table():
    table = render_atlas_task_table(
        [
            AtlasTaskContract("task-b", "Task B", "B", "jupyterhub", "mounted-workspace", ("jupyterhub", "mlflow"), "mounted-required", "task-local-ignored-paths", ()),
            AtlasTaskContract("task-a", "Task A", "A", "jupyterhub", "vscode-remote", ("jupyterhub",), "remote", "atlas-jupyter-volume", ("One", "Two")),
        ]
    )

    assert table == """| Task | Tier | Default mode | Workspace access | Required Atlas services | Artifact policy | Constraints |
| --- | --- | --- | --- | --- | --- | --- |
| task-b | B | mounted-workspace | mounted-required | jupyterhub, mlflow | task-local-ignored-paths | — |
| task-a | A | vscode-remote | remote | jupyterhub | atlas-jupyter-volume | One<br>Two |"""


def test_renderer_escapes_constraint_markdown_cell_characters():
    table = render_atlas_task_table(
        [
            AtlasTaskContract(
                "task",
                "Task",
                "A",
                "jupyterhub",
                "vscode-remote",
                ("jupyterhub",),
                "remote",
                "atlas-jupyter-volume",
                ("A | B", "First line\nsecond line"),
            )
        ]
    )

    assert table.endswith("| A \\| B<br>First line<br>second line |")


@pytest.mark.parametrize("content", ["# Contracts\n", "<!-- atlas-task-contracts:end -->\n<!-- atlas-task-contracts:start -->\n"])
def test_verify_rejects_missing_or_reversed_markers(tmp_path, content):
    doc = tmp_path / "contracts.md"
    doc.write_text(content, encoding="utf-8")

    with pytest.raises(NotebookInfrastructureError, match="marker"):
        verify_atlas_task_table(doc, "| expected |")


def test_verify_rejects_duplicate_markers_and_drift(tmp_path):
    doc = tmp_path / "contracts.md"
    doc.write_text(
        """<!-- atlas-task-contracts:start -->
| stale |
<!-- atlas-task-contracts:end -->
<!-- atlas-task-contracts:start -->
| stale |
<!-- atlas-task-contracts:end -->
""",
        encoding="utf-8",
    )
    with pytest.raises(NotebookInfrastructureError, match="duplicated"):
        verify_atlas_task_table(doc, "| expected |")

    doc.write_text("<!-- atlas-task-contracts:start -->\n| stale |\n<!-- atlas-task-contracts:end -->\n", encoding="utf-8")
    with pytest.raises(NotebookInfrastructureError, match="drifted"):
        verify_atlas_task_table(doc, "| expected |")


def test_write_replaces_only_marker_contents_and_then_verifies(tmp_path):
    doc = tmp_path / "contracts.md"
    doc.write_text("# Intro\n<!-- atlas-task-contracts:start -->\nold\n<!-- atlas-task-contracts:end -->\n# Outro\n", encoding="utf-8")

    write_atlas_task_table(doc, "| expected |")

    assert doc.read_text(encoding="utf-8") == "# Intro\n<!-- atlas-task-contracts:start -->\n| expected |\n<!-- atlas-task-contracts:end -->\n# Outro\n"
    assert verify_atlas_task_table(doc, "| expected |") == []
