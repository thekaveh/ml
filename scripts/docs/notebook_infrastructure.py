"""Atlas notebook-runtime contract parsing and canonical table rendering."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import yaml

from scripts.docs.manifest import Manifest, load_manifest


_START_MARKER = "<!-- atlas-task-contracts:start -->"
_END_MARKER = "<!-- atlas-task-contracts:end -->"
_SERVICE_ID_RE = re.compile(r"^[a-z][a-z0-9-]*$")


class NotebookInfrastructureError(ValueError):
    """Raised when an Atlas notebook-runtime contract is malformed."""


@dataclass(frozen=True)
class AtlasTaskContract:
    task: str
    title: str
    tier: str
    executor: str
    default_mode: str
    required_services: tuple[str, ...]
    workspace_access: str
    artifact_policy: str
    constraints: tuple[str, ...]


def _require_string(value: object, field: str, task: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise NotebookInfrastructureError(f"{task}: {field} must be a non-empty string")
    return value


def _parse_contract(task: str, spec_path: Path) -> AtlasTaskContract:
    try:
        data = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise NotebookInfrastructureError(f"{task}: spec is not valid YAML: {error}") from error
    if not isinstance(data, dict):
        raise NotebookInfrastructureError(f"{task}: spec must be a mapping")

    atlas = data.get("atlas")
    if not isinstance(atlas, dict):
        raise NotebookInfrastructureError(f"{task}: atlas must be a mapping")

    executor = _require_string(atlas.get("executor"), "atlas.executor", task)
    if executor != "jupyterhub":
        raise NotebookInfrastructureError(f"{task}: atlas.executor must be 'jupyterhub'")
    default_mode = _require_string(atlas.get("default_mode"), "atlas.default_mode", task)
    if default_mode not in {"vscode-remote", "mounted-workspace"}:
        raise NotebookInfrastructureError(
            f"{task}: atlas.default_mode must be 'vscode-remote' or 'mounted-workspace'"
        )

    services = atlas.get("required_services")
    if not isinstance(services, list) or not services:
        raise NotebookInfrastructureError(f"{task}: atlas.required_services must be a non-empty list")
    if not all(isinstance(service, str) and _SERVICE_ID_RE.fullmatch(service) for service in services):
        raise NotebookInfrastructureError(f"{task}: atlas.required_services must contain valid service IDs")
    if len(set(services)) != len(services):
        raise NotebookInfrastructureError(f"{task}: atlas.required_services must be unique")
    if "jupyterhub" not in services:
        raise NotebookInfrastructureError(f"{task}: atlas.required_services must contain jupyterhub")

    workspace_access = _require_string(atlas.get("workspace_access"), "atlas.workspace_access", task)
    if workspace_access not in {"remote", "mounted-required"}:
        raise NotebookInfrastructureError(f"{task}: atlas.workspace_access must be remote or mounted-required")
    if default_mode == "mounted-workspace" and workspace_access != "mounted-required":
        raise NotebookInfrastructureError(
            f"{task}: atlas.default_mode 'mounted-workspace' requires workspace_access 'mounted-required'"
        )
    if workspace_access == "mounted-required" and default_mode != "mounted-workspace":
        raise NotebookInfrastructureError(
            f"{task}: atlas.workspace_access 'mounted-required' requires default_mode 'mounted-workspace'"
        )
    artifact_policy = _require_string(atlas.get("artifact_policy"), "atlas.artifact_policy", task)
    if artifact_policy not in {"atlas-jupyter-volume", "task-local-ignored-paths"}:
        raise NotebookInfrastructureError(
            f"{task}: atlas.artifact_policy must be atlas-jupyter-volume or task-local-ignored-paths"
        )
    constraints = atlas.get("constraints")
    if not isinstance(constraints, list):
        raise NotebookInfrastructureError(f"{task}: atlas.constraints must be a list of non-empty strings")
    if not all(isinstance(constraint, str) and constraint.strip() for constraint in constraints):
        raise NotebookInfrastructureError(f"{task}: atlas.constraints must contain non-empty strings")

    return AtlasTaskContract(
        task=task,
        title=_require_string(data.get("title"), "title", task),
        tier=_require_string(data.get("tier"), "tier", task),
        executor=executor,
        default_mode=default_mode,
        required_services=tuple(services),
        workspace_access=workspace_access,
        artifact_policy=artifact_policy,
        constraints=tuple(constraints),
    )


def load_atlas_task_contracts(repo_root: Path, manifest: Manifest) -> list[AtlasTaskContract]:
    """Load contracts in the canonical notebook order from ``manifest``."""
    config_path = repo_root / "scripts" / "verify_repo_config.yaml"
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise NotebookInfrastructureError(f"{config_path}: active_task_dirs configuration is missing") from error
    except yaml.YAMLError as error:
        raise NotebookInfrastructureError(f"{config_path}: configuration is not valid YAML: {error}") from error
    active_tasks = config.get("active_task_dirs") if isinstance(config, dict) else None
    if not isinstance(active_tasks, list) or not all(isinstance(task, str) for task in active_tasks):
        raise NotebookInfrastructureError(f"{config_path}: active_task_dirs must be a list of task names")
    manifest_tasks = {notebook.task for notebook in manifest.notebooks}
    if manifest_tasks != set(active_tasks):
        raise NotebookInfrastructureError(
            f"{config_path}: manifest task set must exactly match active_task_dirs"
        )
    return [_parse_contract(notebook.task, repo_root / notebook.spec) for notebook in manifest.notebooks]


def render_atlas_task_table(contracts: Sequence[AtlasTaskContract]) -> str:
    """Render the canonical, deterministic Markdown Atlas contract table."""
    rows = [
        "| Task | Tier | Default mode | Workspace access | Required Atlas services | Artifact policy | Constraints |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for contract in contracts:
        constraints = "<br>".join(_markdown_cell(constraint) for constraint in contract.constraints) if contract.constraints else "—"
        rows.append(
            "| "
            + " | ".join(
                (
                    contract.task,
                    contract.tier,
                    contract.default_mode,
                    contract.workspace_access,
                    ", ".join(contract.required_services),
                    contract.artifact_policy,
                    constraints,
                )
            )
            + " |"
        )
    return "\n".join(rows)


def _markdown_cell(value: str) -> str:
    """Keep valid text within one Markdown table cell."""
    return value.replace("|", r"\|").replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br>")


def _marker_bounds(text: str, doc_path: Path) -> tuple[int, int]:
    starts = [match.start() for match in re.finditer(re.escape(_START_MARKER), text)]
    ends = [match.start() for match in re.finditer(re.escape(_END_MARKER), text)]
    if not starts or not ends:
        raise NotebookInfrastructureError(f"{doc_path}: atlas task-contract markers are missing")
    if len(starts) != 1 or len(ends) != 1:
        raise NotebookInfrastructureError(f"{doc_path}: atlas task-contract markers are duplicated")
    if starts[0] > ends[0]:
        raise NotebookInfrastructureError(f"{doc_path}: atlas task-contract markers are reversed")
    return starts[0], ends[0]


def verify_atlas_task_table(doc_path: Path, expected_table: str) -> list[str]:
    """Return no findings when the marked table matches; otherwise raise an error."""
    text = doc_path.read_text(encoding="utf-8")
    start, end = _marker_bounds(text, doc_path)
    actual = text[start + len(_START_MARKER):end].strip()
    if actual != expected_table.strip():
        raise NotebookInfrastructureError(f"{doc_path}: atlas task-contract table drifted from rendered contracts")
    return []


def write_atlas_task_table(doc_path: Path, expected_table: str) -> None:
    """Replace only the contents of a valid canonical marker pair."""
    text = doc_path.read_text(encoding="utf-8")
    start, end = _marker_bounds(text, doc_path)
    replacement = f"{_START_MARKER}\n{expected_table.strip()}\n{_END_MARKER}"
    doc_path.write_text(text[:start] + replacement + text[end + len(_END_MARKER):], encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    try:
        contracts = load_atlas_task_contracts(
            repo_root, load_manifest(repo_root / "docs/manifest.yaml", repo_root)
        )
        table = render_atlas_task_table(contracts)
        doc_path = repo_root / "docs/notebook-infrastructure.md"
        if args.write:
            write_atlas_task_table(doc_path, table)
        else:
            verify_atlas_task_table(doc_path, table)
    except NotebookInfrastructureError as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
