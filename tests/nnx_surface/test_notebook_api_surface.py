"""Static guards over committed notebooks — catch defects the papermill tiers miss.

Background
----------
The nnx 0.2.0 PyPI migration (2026-06-14) split the plotting helpers
(`multi_line_plot`, `scatter_plot`, `get_scatter_plot_vm`, ...) off
`nnx.utils.Utils` onto `nnx.vis_utils.VisUtils`. Seven node-classification
notebooks kept calling them as `Utils.multi_line_plot(...)`, which raises
``AttributeError`` at runtime. `verify_repo.py --check structure` passed clean
(it resolves *imports*, not attribute access), so the breakage only surfaced in
the weekly `smoke-tier-b` / `smoke-tier-c` cron — and stayed hidden between
merges because those jobs run on `schedule` only. See CHANGELOG 2026-06-19.

These tests close that gap with cheap, execution-free static scans that run in
CI's fast `make test-nnx-surface` job on *every* PR:

1. ``test_no_visutils_method_called_via_Utils`` — the migration guard. The
   forbidden-method set is derived live from the real nnx surface, so it tracks
   future Utils/VisUtils reshuffles automatically.
2. ``test_no_committed_error_outputs`` — no committed error/traceback outputs
   (e.g. a stray ``KeyboardInterrupt`` from a manually-aborted run).
3. ``test_no_transient_worktree_paths`` — no transient ``.claude/worktrees``
   dev paths leaked into committed cell outputs.

Each scan is paired with a synthetic-notebook unit test proving the checker
actually fires, so a green suite means "checked", not "vacuously passed".
"""
from __future__ import annotations

import ast
import importlib
import io
import inspect
import json
import re
import subprocess
import tokenize
from pathlib import Path

import pytest
import yaml

import nnx
from nnx import NNGraphDataset, Utils, VisUtils, set_seed
from scripts.verify_repo import (
    _NON_PYTHON_CELL_MAGICS,
    _cell_magic_name,
    _is_ipython_magic_or_help_line,
)

# Repo root resolved from this file (the autouse conftest fixture chdirs tests
# into a tmp_path, so cwd is NOT the repo root — never rely on it here).
REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_SUBPROCESS_TIMEOUT = 30

# Bare `Utils.<attr>` access, excluding the `VisUtils.` suffix-match.
_UTILS_ATTR_RE = re.compile(r"(?<![A-Za-z0-9_])Utils\.([A-Za-z_]\w*)")
# Transient per-worktree path that must never be committed in an output.
_TRANSIENT_PATH_RE = re.compile(r"\.claude/worktrees/")

_ATLAS_020_DEFAULT_RUNTIME_TASKS = {
    "notebooks/knowledge_distillation-mnist-ffnn-pytorch/docs/spec.yaml": (
        "notebooks/knowledge_distillation-mnist-ffnn-pytorch/notebook.ipynb",
    ),
    "notebooks/peft-mnist-to-fmnist-dora-vs-lora-pytorch/docs/spec.yaml": (
        "notebooks/peft-mnist-to-fmnist-dora-vs-lora-pytorch/notebook.ipynb",
    ),
    "notebooks/node_classification-reddit-gnn-pyg/docs/spec.yaml": (
        "notebooks/node_classification-reddit-gnn-pyg/phase2-model-selection-notebook1.ipynb",
        "notebooks/node_classification-reddit-gnn-pyg/phase2-model-selection-notebook2.ipynb",
    ),
}
_UNSUPPORTED_ATLAS_020_TRAIN_KWARGS = {"salt", "overwrite_existing", "data_id"}

_PUBLIC_NNX_IMPORTS = {
    "nnx.nn.dataset.nn_dataset": ("NNDataset",),
    "nnx.nn.dataset.nn_graph_dataset": ("NNGraphDataset",),
    "nnx.nn.enum.activations": ("Activations",),
    "nnx.nn.enum.devices": ("Devices",),
    "nnx.nn.enum.losses": ("Losses",),
    "nnx.nn.enum.nets": ("Nets",),
    "nnx.nn.enum.optims": ("Optims",),
    "nnx.nn.net.feed_fwd_nn": ("FeedFwdNN",),
    "nnx.nn.nn_model": ("NNModel",),
    "nnx.nn.params.nn_model_params": ("NNModelParams",),
    "nnx.nn.params.nn_optim_params": ("NNOptimParams",),
    "nnx.nn.params.nn_params": ("NNParams",),
    "nnx.nn.params.nn_train_params": ("NNTrainParams",),
    "nnx.seeding": ("set_seed",),
    "nnx.utils": ("Utils",),
    "nnx.vis_utils": ("VisUtils",),
}

_NNDATASET_BATCHING_CONTRACTS = {
    "notebooks/diffusion-mnist-ddpm-pytorch/notebook.ipynb": {
        "batch_sizes": (128, None, None),
        "loader_aliases": {"train_loader": "train_loader"},
    },
    "notebooks/moe-fmnist-mixture-of-experts-pytorch/notebook.ipynb": {
        "batch_sizes": (128, None, None),
        "loader_aliases": {"train_loader": "train_loader"},
    },
    "notebooks/self_supervised-fmnist-jepa-pytorch/notebook.ipynb": {
        "batch_sizes": (128, 128, None),
        "loader_aliases": {
            "train_loader": "train_loader",
            "val_loader": "val_loader",
        },
    },
}

_REDDIT_GRAPH_DATASET_NOTEBOOKS = (
    "notebooks/node_classification-reddit-gnn-pyg/phase2-model-selection-notebook1.ipynb",
    "notebooks/node_classification-reddit-gnn-pyg/phase2-model-selection-notebook2.ipynb",
    "notebooks/node_classification-reddit-gnn-pyg/phase2-model-selection-notebook3.ipynb",
    "notebooks/node_classification-reddit-gnn-pyg/phase2-model-selection-notebook4.ipynb",
    "notebooks/node_classification-reddit-gnn-pyg/phase3-main-model-training-and-eval-notebook.ipynb",
    "notebooks/node_classification-reddit-gnn-pyg/phase3-main-model-training-and-eval-notebook2.ipynb",
    "notebooks/node_classification-reddit-gnn-pyg/phase3-main-model-training-and-eval-notebook3.ipynb",
    "notebooks/node_classification-reddit-gnn-pyg/phase3-main-model-training-and-eval-notebook4.ipynb",
)


def _public_attrs(cls: object) -> set[str]:
    return {n for n in dir(cls) if not n.startswith("_")}


def _visutils_only_methods() -> set[str]:
    """Methods that live on VisUtils but NOT on Utils — illegal as ``Utils.<m>``."""
    return _public_attrs(VisUtils) - _public_attrs(Utils)


def _tracked_ipynb_files(repo_root: Path = REPO_ROOT) -> list[str]:
    return subprocess.run(
        ["git", "ls-files", "--", "*.ipynb"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
        timeout=TEST_SUBPROCESS_TIMEOUT,
    ).stdout.splitlines()


def _active_notebooks(repo_root: Path = REPO_ROOT) -> list[Path]:
    """Every git-tracked active notebook under notebooks/, excluding archive + checkpoints."""
    tracked = _tracked_ipynb_files(repo_root)
    return sorted(
        repo_root / rel
        for rel in tracked
        if rel.startswith("notebooks/")
        and not rel.startswith("notebooks/archive/")
        and ".ipynb_checkpoints" not in Path(rel).parts
    )


def _archive_cross_language_notebooks(repo_root: Path = REPO_ROOT) -> list[Path]:
    tracked = _tracked_ipynb_files(repo_root)
    prefix = "notebooks/archive/codexglue_summarization/codexglue-summarization-cross-"
    return sorted(repo_root / rel for rel in tracked if rel.startswith(prefix))


def _archive_roberta_notebooks(repo_root: Path = REPO_ROOT) -> list[Path]:
    tracked = _tracked_ipynb_files(repo_root)
    prefix = "notebooks/archive/codexglue_summarization/codexglue-summarization-roberta-"
    return sorted(repo_root / rel for rel in tracked if rel.startswith(prefix))


def _code_cells(nb: dict) -> list[dict]:
    return [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]


def _source_lines(cell: dict) -> list[str]:
    source = cell.get("source", [])
    if isinstance(source, str):
        return source.splitlines(keepends=True)
    return list(source)


def _live_lines(cell: dict) -> list[str]:
    """Source lines that are not pure comments (commented-out historical code is
    deliberately preserved verbatim in some Tier-C cells and must not be flagged)."""
    return [ln for ln in _source_lines(cell) if not ln.lstrip().startswith("#")]


def _executable_lines(cell: dict) -> list[str]:
    """Code lines with comments and string literals blanked for regex guards."""
    source = "".join(_source_lines(cell))
    lines = source.splitlines(keepends=True)
    masked = [list(line) for line in lines]
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for token in tokens:
            if token.type not in {tokenize.COMMENT, tokenize.STRING}:
                continue
            (start_line, start_col), (end_line, end_col) = token.start, token.end
            for line_no in range(start_line, end_line + 1):
                row = masked[line_no - 1]
                start = start_col if line_no == start_line else 0
                end = end_col if line_no == end_line else len(row)
                for idx in range(start, min(end, len(row))):
                    if row[idx] not in "\r\n":
                        row[idx] = " "
    except tokenize.TokenError:
        return _live_lines(cell)
    return ["".join(line) for line in masked if "".join(line).strip()]


def _output_text(cell: dict) -> str:
    chunks: list[str] = []
    for o in cell.get("outputs", []):
        txt = o.get("text", "")
        chunks.append("".join(txt) if isinstance(txt, list) else str(txt))
        tp = o.get("data", {}).get("text/plain", "")
        chunks.append("".join(tp) if isinstance(tp, list) else str(tp))
    return "\n".join(chunks)


def _python_cell_source(cell: dict) -> str:
    """Return Python source while preserving line numbers around IPython syntax."""
    lines = _source_lines(cell)
    for line in lines:
        if not line.strip():
            continue
        if _cell_magic_name(line) in _NON_PYTHON_CELL_MAGICS:
            return ""
        break
    return "".join(
        "\n" if _is_ipython_magic_or_help_line(line) else line
        for line in lines
    )


def find_reddit_graph_dataset_seed_contract_violations(nb: dict) -> list[str]:
    """Require the released NNx seeding boundary around graph dataset creation."""
    findings: list[str] = []
    imports_set_seed: list[tuple[int, int]] = []
    seed_assignments: list[tuple[int, int, ast.Assign]] = []
    dataset_assignments: list[tuple[int, int, ast.Assign, list[ast.stmt]]] = []

    for cell_index, cell in enumerate(_code_cells(nb)):
        tree = ast.parse(_python_cell_source(cell))
        for statement_index, statement in enumerate(tree.body):
            if (
                isinstance(statement, ast.ImportFrom)
                and statement.module == "nnx"
                and any(alias.name == "set_seed" for alias in statement.names)
            ):
                imports_set_seed.append((cell_index, statement_index))
            if (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and statement.targets[0].id == "SEED"
            ):
                seed_assignments.append((cell_index, statement_index, statement))
            if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
                continue
            target = statement.targets[0]
            value = statement.value
            if (
                isinstance(target, ast.Name)
                and target.id == "ds"
                and isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id == "NNGraphDataset"
            ):
                dataset_assignments.append((cell_index, statement_index, statement, tree.body))

    if not imports_set_seed:
        findings.append("set_seed must be imported from the public nnx facade")

    if len(seed_assignments) != 1:
        findings.append(f"expected exactly one top-level SEED assignment; found {len(seed_assignments)}")
    elif not (
        isinstance(seed_assignments[0][2].value, ast.Constant)
        and seed_assignments[0][2].value.value == 0
        and type(seed_assignments[0][2].value.value) is int
    ):
        findings.append("SEED must be the integer literal 0")

    if len(dataset_assignments) != 1:
        findings.append(
            f"expected exactly one top-level ds = NNGraphDataset(...); found {len(dataset_assignments)}"
        )
        return findings

    cell_index, statement_index, assignment, statements = dataset_assignments[0]
    dataset_position = (cell_index, statement_index)
    if imports_set_seed and not any(
        import_position < dataset_position for import_position in imports_set_seed
    ):
        findings.append("set_seed import must precede graph dataset construction")
    if len(seed_assignments) == 1 and seed_assignments[0][:2] >= dataset_position:
        findings.append("SEED assignment must precede graph dataset construction")
    if any(keyword.arg == "seed" for keyword in assignment.value.keywords):
        findings.append("NNGraphDataset does not accept a seed keyword in NNx 0.2.0")
    if statement_index == 0:
        findings.append(f"code_cell[{cell_index}] must call set_seed(SEED) immediately before NNGraphDataset")
        return findings
    previous = statements[statement_index - 1]
    if not (
        isinstance(previous, ast.Expr)
        and isinstance(previous.value, ast.Call)
        and isinstance(previous.value.func, ast.Name)
        and previous.value.func.id == "set_seed"
        and len(previous.value.args) == 1
        and isinstance(previous.value.args[0], ast.Name)
        and previous.value.args[0].id == "SEED"
        and not previous.value.keywords
    ):
        findings.append(f"code_cell[{cell_index}] must call set_seed(SEED) immediately before NNGraphDataset")
    return findings


def find_deep_public_nnx_imports(nb: dict) -> list[str]:
    findings: list[str] = []
    for cell_index, cell in enumerate(_code_cells(nb)):
        source = _python_cell_source(cell)
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            raise AssertionError(
                f"code_cell[{cell_index}] remains unparseable after masking IPython lines: {exc}"
            ) from exc
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.module not in _PUBLIC_NNX_IMPORTS:
                continue
            public_names = set(_PUBLIC_NNX_IMPORTS[node.module])
            for alias in node.names:
                if alias.name == "*" or alias.name in public_names:
                    findings.append(
                        f"code_cell[{cell_index}]:line[{node.lineno}] "
                        f"from {node.module} import {alias.name}"
                    )
    return findings


def _assert_atlas_020_default_runtime_compatibility(
    requirements_text: str,
    dependency_ledger: str,
    task_sources: dict[str, str],
) -> None:
    pins = [line for line in requirements_text.splitlines() if line.startswith("thekaveh-nnx")]
    assert pins == ["thekaveh-nnx[lm]==0.2.0"]
    assert "| NNx + language extras | `thekaveh-nnx` / `nnx` 0.2.0;" in dependency_ledger

    checked_train_calls = 0
    for spec_path, notebook_paths in _ATLAS_020_DEFAULT_RUNTIME_TASKS.items():
        spec = yaml.safe_load(task_sources[spec_path])
        assert spec["atlas"]["executor"] == "jupyterhub"
        assert spec["atlas"]["default_mode"] == "vscode-remote"
        for notebook_path in notebook_paths:
            notebook = json.loads(task_sources[notebook_path])
            for cell in _code_cells(notebook):
                try:
                    tree = ast.parse("".join(_source_lines(cell)))
                except SyntaxError:
                    continue
                for call in (
                    node
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "train"
                ):
                    checked_train_calls += 1
                    keyword_names = {keyword.arg for keyword in call.keywords}
                    assert not (_UNSUPPORTED_ATLAS_020_TRAIN_KWARGS & keyword_names)
    assert checked_train_calls >= 5


def _atlas_020_default_runtime_sources() -> dict[str, str]:
    paths = {
        path
        for spec_path, notebook_paths in _ATLAS_020_DEFAULT_RUNTIME_TASKS.items()
        for path in (spec_path, *notebook_paths)
    }
    return {path: (REPO_ROOT / path).read_text(encoding="utf-8") for path in paths}


# --- scan checkers (also exercised directly by the synthetic unit tests) -----

def find_misplaced_utils_attrs(nb: dict, forbidden: set[str]) -> list[str]:
    out: list[str] = []
    for idx, cell in enumerate(_code_cells(nb)):
        for line in _executable_lines(cell):
            for m in _UTILS_ATTR_RE.finditer(line):
                if m.group(1) in forbidden:
                    out.append(f"code_cell[{idx}]: Utils.{m.group(1)} (moved to VisUtils)")
    return out


def find_error_outputs(nb: dict) -> list[str]:
    out: list[str] = []
    for idx, cell in enumerate(_code_cells(nb)):
        for o in cell.get("outputs", []):
            if o.get("output_type") == "error":
                out.append(f"code_cell[{idx}]: committed {o.get('ename', 'error')} output")
    return out


def find_transient_paths(nb: dict) -> list[str]:
    out: list[str] = []
    for idx, cell in enumerate(_code_cells(nb)):
        if _TRANSIENT_PATH_RE.search(_output_text(cell)):
            out.append(f"code_cell[{idx}]: '.claude/worktrees' path leaked into output")
    return out


# --- real-notebook scans (parametrized per notebook) -------------------------

_NOTEBOOKS = _active_notebooks()
_IDS = [p.relative_to(REPO_ROOT).as_posix() for p in _NOTEBOOKS]


def test_active_notebooks_discovered():
    """Guard against the glob silently matching nothing (which would make every
    parametrized scan vacuously pass)."""
    assert len(_NOTEBOOKS) >= 25, f"expected the full active notebook set, found {len(_NOTEBOOKS)}"


def test_nnx_public_facade_preserves_classified_object_identity():
    classified_names = {
        name
        for names in _PUBLIC_NNX_IMPORTS.values()
        for name in names
    }
    assert len(classified_names) == 16
    for module_name, names in _PUBLIC_NNX_IMPORTS.items():
        deep_module = importlib.import_module(module_name)
        for name in names:
            assert getattr(nnx, name) is getattr(deep_module, name)


def test_active_notebooks_use_public_nnx_facade():
    violations = []
    affected_notebooks = 0
    for path in _NOTEBOOKS:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        findings = find_deep_public_nnx_imports(notebook)
        if findings:
            affected_notebooks += 1
            violations.extend(
                f"{path.relative_to(REPO_ROOT)}: {finding}"
                for finding in findings
            )
    assert not violations, (
        f"found {len(violations)} classified deep NNx imports in "
        f"{affected_notebooks} active notebooks; use `from nnx import ...`:\n  "
        + "\n  ".join(violations)
    )


def test_atlas_020_default_runtime_notebooks_use_only_supported_train_kwargs():
    _assert_atlas_020_default_runtime_compatibility(
        (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8"),
        (REPO_ROOT / "docs/dependency-contracts.md").read_text(encoding="utf-8"),
        _atlas_020_default_runtime_sources(),
    )


@pytest.mark.parametrize("unsupported_kwarg", sorted(_UNSUPPORTED_ATLAS_020_TRAIN_KWARGS))
def test_atlas_020_default_runtime_contract_rejects_unsupported_train_kwarg_mutations(
    unsupported_kwarg: str,
):
    sources = _atlas_020_default_runtime_sources()
    path = "notebooks/knowledge_distillation-mnist-ffnn-pytorch/notebook.ipynb"
    notebook = json.loads(sources[path])
    original = "single_gen.train(params=train_params())"
    replacement = f'single_gen.train(params=train_params(), {unsupported_kwarg}="mutated")'
    matching_cells = [
        cell
        for cell in _code_cells(notebook)
        if original in "".join(_source_lines(cell))
    ]
    assert len(matching_cells) == 1
    matching_cells[0]["source"] = "".join(_source_lines(matching_cells[0])).replace(
        original, replacement, 1
    )
    sources[path] = json.dumps(notebook)

    with pytest.raises(AssertionError):
        _assert_atlas_020_default_runtime_compatibility(
            "thekaveh-nnx[lm]==0.2.0\n",
            "| NNx + language extras | `thekaveh-nnx` / `nnx` 0.2.0;\n",
            sources,
        )


@pytest.mark.parametrize(
    "mutation",
    ("root_pin", "atlas_pin", "executor", "default_mode"),
)
def test_atlas_020_default_runtime_contract_rejects_boundary_mutations(mutation: str):
    requirements = "thekaveh-nnx[lm]==0.2.0\n"
    ledger = "| NNx + language extras | `thekaveh-nnx` / `nnx` 0.2.0;\n"
    sources = _atlas_020_default_runtime_sources()
    spec_path = "notebooks/knowledge_distillation-mnist-ffnn-pytorch/docs/spec.yaml"

    if mutation == "root_pin":
        requirements = requirements.replace("0.2.0", "0.2.2")
    elif mutation == "atlas_pin":
        ledger = ledger.replace("0.2.0", "0.2.2")
    else:
        spec = yaml.safe_load(sources[spec_path])
        if mutation == "executor":
            spec["atlas"]["executor"] = "local"
        elif mutation == "default_mode":
            spec["atlas"]["default_mode"] = "mounted-workspace"
        else:
            raise AssertionError(f"unhandled mutation: {mutation}")
        sources[spec_path] = yaml.safe_dump(spec)

    with pytest.raises(AssertionError):
        _assert_atlas_020_default_runtime_compatibility(requirements, ledger, sources)


def test_active_notebooks_uses_git_tracked_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Untracked scratch notebooks on disk must not affect the static surface scans."""
    tracked_nb = tmp_path / "notebooks" / "task" / "notebook.ipynb"
    archive_nb = tmp_path / "notebooks" / "archive" / "old.ipynb"
    checkpoint_nb = tmp_path / "notebooks" / "task" / ".ipynb_checkpoints" / "scratch.ipynb"
    old_root_nb = tmp_path / "task" / "notebook.ipynb"
    untracked_nb = tmp_path / "notebooks" / "task" / "scratch.ipynb"
    for path in (tracked_nb, archive_nb, checkpoint_nb, old_root_nb, untracked_nb):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")

    def fake_run(cmd, cwd, capture_output, text, check, timeout):
        assert cmd == ["git", "ls-files", "--", "*.ipynb"]
        assert cwd == tmp_path
        assert capture_output is True
        assert text is True
        assert check is True
        assert timeout == TEST_SUBPROCESS_TIMEOUT
        stdout = "\n".join([
            "notebooks/task/notebook.ipynb",
            "notebooks/archive/old.ipynb",
            "notebooks/task/.ipynb_checkpoints/scratch.ipynb",
            "task/notebook.ipynb",
        ])
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert _active_notebooks(tmp_path) == [tracked_nb]


def test_archive_notebook_guards_use_git_tracked_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Untracked archive scratch notebooks must not affect archive guard scans."""
    cross_nb = tmp_path / "notebooks/archive/codexglue_summarization/codexglue-summarization-cross-java-on-go/notebook.ipynb"
    roberta_nb = tmp_path / "notebooks/archive/codexglue_summarization/codexglue-summarization-roberta-codebert-java/notebook.ipynb"
    untracked_cross = tmp_path / "notebooks/archive/codexglue_summarization/codexglue-summarization-cross-scratch/notebook.ipynb"
    untracked_roberta = tmp_path / "notebooks/archive/codexglue_summarization/codexglue-summarization-roberta-scratch/notebook.ipynb"
    for path in (cross_nb, roberta_nb, untracked_cross, untracked_roberta):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")

    def fake_run(cmd, cwd, capture_output, text, check, timeout):
        assert cmd == ["git", "ls-files", "--", "*.ipynb"]
        assert cwd == tmp_path
        assert capture_output is True
        assert text is True
        assert check is True
        assert timeout == TEST_SUBPROCESS_TIMEOUT
        stdout = "\n".join([
            "notebooks/archive/codexglue_summarization/codexglue-summarization-cross-java-on-go/notebook.ipynb",
            "notebooks/archive/codexglue_summarization/codexglue-summarization-roberta-codebert-java/notebook.ipynb",
        ])
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert _archive_cross_language_notebooks(tmp_path) == [cross_nb]
    assert _archive_roberta_notebooks(tmp_path) == [roberta_nb]


def test_archive_cross_language_notebooks_guard_missing_model_artifacts():
    """Archived CodeXGLUE transfer notebooks should rerun as references even
    when historical model outputs are absent from the archive snapshot."""
    notebooks = _archive_cross_language_notebooks()
    assert len(notebooks) == 10
    required_fragments = (
        "ARCHIVE_SUBPROCESS_TIMEOUT_SECONDS = 6 * 60 * 60",
        "if not os.path.exists(run_py_path):",
        "elif not os.path.exists(checkpoint_path):",
        "elif not os.path.exists(test_filename_path):",
        "timeout=ARCHIVE_SUBPROCESS_TIMEOUT_SECONDS",
        "if not os.path.exists(bleu_score_path):",
        "if not os.path.exists(_gold_path):",
        "if not os.path.exists(_output_path):",
    )
    for path in notebooks:
        nb = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join("".join(_source_lines(cell)) for cell in _code_cells(nb))
        missing = [fragment for fragment in required_fragments if fragment not in source]
        assert not missing, f"{path.relative_to(REPO_ROOT)} missing archive guards: {missing}"


def test_archive_cross_language_missing_model_outputs_match_tracked_artifacts():
    """Missing tracked model artifacts should not retain stale historical result outputs."""
    notebooks = _archive_cross_language_notebooks()
    assert len(notebooks) == 10
    required_artifacts = (
        "model/bleu_score.test",
        "model/test_0.gold",
        "model/test_0.output",
    )
    expected_guards = {
        "model/bleu_score.test": "Archived BLEU score file not found",
        "model/test_0.gold": "Archived ground-truth summary file not found",
        "model/test_0.output": "Archived prediction file not found",
    }
    stale_result_markers = ("Test Bleu score:", "bleu-4 =")
    stale = []
    for path in notebooks:
        missing_artifacts = [
            rel
            for rel in required_artifacts
            if subprocess.run(
                ["git", "ls-files", "--error-unmatch", str(path.parent / rel)],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
                timeout=TEST_SUBPROCESS_TIMEOUT,
            ).returncode
        ]
        if not missing_artifacts:
            continue
        nb = json.loads(path.read_text(encoding="utf-8"))
        output_text = "\n".join(
            str(output.get("text", ""))
            for cell in nb.get("cells", [])
            for output in cell.get("outputs", [])
            if isinstance(output, dict)
        )
        missing_guards = [expected_guards[rel] for rel in missing_artifacts if expected_guards[rel] not in output_text]
        stale_markers = [marker for marker in stale_result_markers if marker in output_text]
        leaked_repo_root = str(REPO_ROOT) in output_text
        if missing_guards or stale_markers or leaked_repo_root:
            stale.append((
                path.relative_to(REPO_ROOT),
                missing_artifacts,
                missing_guards,
                stale_markers,
                leaked_repo_root,
            ))
    assert not stale


def test_archive_cross_language_notebooks_require_explicit_rerun_opt_in():
    """Cross-language CodeXGLUE notebooks should not write model outputs by default."""
    notebooks = _archive_cross_language_notebooks()
    assert len(notebooks) == 10
    required_fragments = (
        "ARCHIVE_RERUN_ENABLED = False",
        "if not ARCHIVE_RERUN_ENABLED:",
        "elif not os.path.exists(run_py_path):",
        "timeout=ARCHIVE_SUBPROCESS_TIMEOUT_SECONDS",
    )
    for path in notebooks:
        nb = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join("".join(_source_lines(cell)) for cell in _code_cells(nb))
        missing = [fragment for fragment in required_fragments if fragment not in source]
        assert not missing, f"{path.relative_to(REPO_ROOT)} missing archive rerun guards: {missing}"
        assert "subprocess.check_call(" in source
        assert source.index("ARCHIVE_RERUN_ENABLED = False") < source.index("subprocess.check_call("), (
            f"{path.relative_to(REPO_ROOT)} defines rerun opt-in after write-capable subprocess calls"
        )


def test_archive_notebooks_do_not_depend_on_kernel_cwd():
    """Archive notebooks should resolve their own folder rather than trusting cwd."""
    notebooks = _archive_cross_language_notebooks() + _archive_roberta_notebooks()
    assert len(notebooks) == 22
    for path in notebooks:
        nb = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join("".join(_source_lines(cell)) for cell in _code_cells(nb))
        assert "os.path.abspath(os.curdir)" not in source, (
            f"{path.relative_to(REPO_ROOT)} derives archive paths from kernel cwd"
        )
        assert path.parent.name in source, (
            f"{path.relative_to(REPO_ROOT)} should anchor paths to its archive folder name"
        )


def test_archive_roberta_notebooks_require_explicit_rerun_opt_in():
    """Archived RoBERTa CodeXGLUE notebooks should preserve historical model
    artifacts by default and only clone/download/train/test when explicitly enabled."""
    notebooks = _archive_roberta_notebooks()
    assert len(notebooks) == 12
    required_fragments = (
        "ARCHIVE_RERUN_ENABLED = False",
        "ARCHIVE_SUBPROCESS_TIMEOUT_SECONDS = 6 * 60 * 60",
        "if ARCHIVE_RERUN_ENABLED and os.path.exists(path=os.path.join(model_path)):",
        "if not ARCHIVE_RERUN_ENABLED:",
        "elif not os.path.exists(run_py_path):",
        "timeout=ARCHIVE_SUBPROCESS_TIMEOUT_SECONDS",
        "if not os.path.exists(_train_losses_path):",
        "if not os.path.exists(_bleu_score_path):",
        "if not os.path.exists(_gold_path):",
        "if not os.path.exists(_output_path):",
    )
    for path in notebooks:
        nb = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join("".join(_source_lines(cell)) for cell in _code_cells(nb))
        missing = [fragment for fragment in required_fragments if fragment not in source]
        assert not missing, f"{path.relative_to(REPO_ROOT)} missing archive rerun guards: {missing}"
        assert source.index("ARCHIVE_RERUN_ENABLED = False") < source.index("shutil.copy("), (
            f"{path.relative_to(REPO_ROOT)} copies archive helpers before rerun opt-in is defined"
        )
        assert "os.chdir(dataset_path)" not in source, (
            f"{path.relative_to(REPO_ROOT)} mutates notebook cwd during archive preprocessing"
        )
        assert "cwd=dataset_path" in source, (
            f"{path.relative_to(REPO_ROOT)} should run preprocessing with subprocess cwd"
        )


@pytest.mark.parametrize("nb_path", _NOTEBOOKS, ids=_IDS)
def test_no_visutils_method_called_via_Utils(nb_path: Path):
    forbidden = _visutils_only_methods()
    assert forbidden, "expected VisUtils to expose methods absent from Utils"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    violations = find_misplaced_utils_attrs(nb, forbidden)
    assert not violations, (
        f"{nb_path.relative_to(REPO_ROOT)} calls VisUtils-only method(s) via Utils:\n  "
        + "\n  ".join(violations)
    )


@pytest.mark.parametrize("nb_path", _NOTEBOOKS, ids=_IDS)
def test_no_committed_error_outputs(nb_path: Path):
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    violations = find_error_outputs(nb)
    assert not violations, (
        f"{nb_path.relative_to(REPO_ROOT)} has committed error output(s):\n  "
        + "\n  ".join(violations)
    )


@pytest.mark.parametrize("nb_path", _NOTEBOOKS, ids=_IDS)
def test_no_transient_worktree_paths(nb_path: Path):
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    violations = find_transient_paths(nb)
    assert not violations, (
        f"{nb_path.relative_to(REPO_ROOT)} leaks transient worktree path(s):\n  "
        + "\n  ".join(violations)
    )


# --- self-validation: prove each checker fires on a known-bad notebook -------

def _synthetic_nb(code_cell: dict) -> dict:
    return {"cells": [code_cell], "metadata": {}, "nbformat": 4, "nbformat_minor": 2}


def test_migration_guard_catches_bad_call():
    forbidden = _visutils_only_methods()
    assert "multi_line_plot" in forbidden, "fixture assumes multi_line_plot is VisUtils-only"
    bad = _synthetic_nb({"cell_type": "code", "source": ["Utils.multi_line_plot(x=[1])\n"], "outputs": []})
    assert find_misplaced_utils_attrs(bad, forbidden)


def test_public_facade_guard_catches_single_and_parenthesized_imports():
    bad = _synthetic_nb({
        "cell_type": "code",
        "source": [
            "%matplotlib inline\n",
            "from nnx.seeding import set_seed\n",
            "from nnx.nn.enum.devices import (\n",
            "    Devices,\n",
            ")\n",
        ],
        "outputs": [],
    })
    assert find_deep_public_nnx_imports(bad) == [
        "code_cell[0]:line[2] from nnx.seeding import set_seed",
        "code_cell[0]:line[3] from nnx.nn.enum.devices import Devices",
    ]


def test_public_facade_guard_ignores_comments_strings_and_unclassified_imports():
    good = _synthetic_nb({
        "cell_type": "code",
        "source": [
            "# from nnx.seeding import set_seed\n",
            "example = 'from nnx.utils import Utils'\n",
            "from nnx.nn.net.graph_att_nn import GraphAttNN\n",
            "from nnx import Devices, Utils, set_seed\n",
        ],
        "outputs": [],
    })
    assert not find_deep_public_nnx_imports(good)


def test_public_facade_guard_ignores_non_python_cell_magic_and_ipython_help():
    good = {
        "cells": [
            {
                "cell_type": "code",
                "source": [
                    "%%bash\n",
                    "from nnx.seeding import set_seed\n",
                ],
                "outputs": [],
            },
            {
                "cell_type": "code",
                "source": [
                    "NNModel?\n",
                    "from nnx import NNModel\n",
                ],
                "outputs": [],
            },
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 2,
    }
    assert not find_deep_public_nnx_imports(good)


def test_released_nnx_graph_dataset_uses_public_global_seed_contract():
    dataset_parameters = inspect.signature(NNGraphDataset).parameters
    seed_parameters = inspect.signature(set_seed).parameters

    assert "seed" not in dataset_parameters
    assert seed_parameters["seed"].default is inspect.Parameter.empty
    assert seed_parameters["strict"].default is False


@pytest.mark.parametrize("relative", _REDDIT_GRAPH_DATASET_NOTEBOOKS)
def test_reddit_graph_dataset_construction_is_explicitly_seeded(relative: str):
    notebook = json.loads((REPO_ROOT / relative).read_text(encoding="utf-8"))

    assert not find_reddit_graph_dataset_seed_contract_violations(notebook), relative


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "from nnx import NNGraphDataset\nSEED = 0\nset_seed(SEED)\nds = NNGraphDataset(ds_class=R, n_neighbors=[2])\n",
            "set_seed must be imported",
        ),
        (
            "from nnx import NNGraphDataset, set_seed\nSEED = 1\nset_seed(SEED)\nds = NNGraphDataset(ds_class=R, n_neighbors=[2])\n",
            "integer literal 0",
        ),
        (
            "from nnx import NNGraphDataset, set_seed\nSEED = 0\nif enabled:\n    set_seed(SEED)\nds = NNGraphDataset(ds_class=R, n_neighbors=[2])\n",
            "immediately before",
        ),
        (
            "from nnx import NNGraphDataset, set_seed\nSEED = 0\nset_seed(SEED)\nprepare()\nds = NNGraphDataset(ds_class=R, n_neighbors=[2])\n",
            "immediately before",
        ),
        (
            "from nnx import NNGraphDataset, set_seed\nSEED = 0\nset_seed(SEED)\nds = NNGraphDataset(ds_class=R, n_neighbors=[2], seed=SEED)\n",
            "does not accept a seed keyword",
        ),
        (
            "from nnx import NNGraphDataset, set_seed\nset_seed(SEED)\nds = NNGraphDataset(ds_class=R, n_neighbors=[2])\nSEED = 0\n",
            "SEED assignment must precede",
        ),
        (
            "SEED = 0\nset_seed(SEED)\nds = NNGraphDataset(ds_class=R, n_neighbors=[2])\nfrom nnx import NNGraphDataset, set_seed\n",
            "set_seed import must precede",
        ),
    ],
)
def test_reddit_graph_dataset_seed_guard_rejects_regressions(source: str, expected: str):
    notebook = _synthetic_nb({"cell_type": "code", "source": source, "outputs": []})

    assert any(
        expected in finding
        for finding in find_reddit_graph_dataset_seed_contract_violations(notebook)
    )


def test_reddit_graph_dataset_seed_guard_accepts_public_boundary():
    notebook = _synthetic_nb({
        "cell_type": "code",
        "source": (
            "from nnx import NNGraphDataset, set_seed\n"
            "SEED = 0\n"
            "set_seed(SEED)\n"
            "ds = NNGraphDataset(ds_class=R, n_neighbors=[2])\n"
        ),
        "outputs": [],
    })

    assert not find_reddit_graph_dataset_seed_contract_violations(notebook)


def test_public_facade_guard_rejects_wildcard_import_from_classified_module():
    bad = _synthetic_nb({
        "cell_type": "code",
        "source": ["from nnx.seeding import *\n"],
        "outputs": [],
    })

    assert find_deep_public_nnx_imports(bad) == [
        "code_cell[0]:line[1] from nnx.seeding import *",
    ]


def test_migration_guard_catches_string_source_bad_call():
    forbidden = _visutils_only_methods()
    bad = _synthetic_nb({"cell_type": "code", "source": "Utils.multi_line_plot(x=[1])\n", "outputs": []})
    assert find_misplaced_utils_attrs(bad, forbidden)


def test_migration_guard_allows_correct_call_and_real_utils_methods():
    forbidden = _visutils_only_methods()
    good = _synthetic_nb({
        "cell_type": "code",
        "source": ["VisUtils.multi_line_plot(x=[1])\n", "Utils.print_table(rows)\n"],
        "outputs": [],
    })
    assert not find_misplaced_utils_attrs(good, forbidden)


def test_migration_guard_ignores_string_literals_and_docstrings():
    forbidden = _visutils_only_methods()
    quoted = _synthetic_nb({
        "cell_type": "code",
        "source": [
            "\"\"\"Example: Utils.multi_line_plot(x=[1])\"\"\"\n",
            "msg = 'Utils.multi_line_plot(x=[1])'\n",
        ],
        "outputs": [],
    })
    assert not find_misplaced_utils_attrs(quoted, forbidden)


def test_migration_guard_ignores_commented_out_reference():
    forbidden = _visutils_only_methods()
    commented = _synthetic_nb({
        "cell_type": "code",
        "source": ["# original: Utils.scatter_plot(...)  preserved for reference\n"],
        "outputs": [],
    })
    assert not find_misplaced_utils_attrs(commented, forbidden)


def test_error_output_guard_catches_traceback():
    bad = _synthetic_nb({
        "cell_type": "code", "source": ["train()\n"],
        "outputs": [{"output_type": "error", "ename": "KeyboardInterrupt", "evalue": "", "traceback": []}],
    })
    assert find_error_outputs(bad)


def test_transient_path_guard_catches_leak():
    bad = _synthetic_nb({
        "cell_type": "code", "source": ["run.save()\n"],
        "outputs": [{"output_type": "stream", "name": "stdout",
                     "text": ["Run saved to /Users/x/.claude/worktrees/wt/runs/abc\n"]}],
    })
    assert find_transient_paths(bad)


# --- nnx constructor signature-completeness guard ----------------------------
#
# The Utils->VisUtils break wasn't the only thing the nnx 0.2.0 migration left
# stranded: NNOptimParams gained a required keyword-only `momentum` arg, so seven
# notebooks calling NNOptimParams(name=, max_lr=, weight_decay=) raised TypeError
# at execution. Attribute-surface scanning can't see that — this guard parses
# each code cell's AST and checks every call to an nnx `NN*` constructor supplies
# all of that constructor's required keyword-only params (resolved live from the
# real nnx signatures, so it tracks future signature changes). Calls with
# positional args or **kwargs unpacking are skipped (not statically resolvable).


def _nnx_required_kwonly_params() -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for name in dir(nnx):
        obj = getattr(nnx, name)
        if not (inspect.isclass(obj) and name.startswith("NN")):
            continue
        try:
            sig = inspect.signature(obj.__init__)
        except (ValueError, TypeError):
            continue
        out[name] = {
            pname for pname, p in sig.parameters.items()
            if p.kind is p.KEYWORD_ONLY and p.default is p.empty and pname != "self"
        }
    return out


def _nnx_ctor_accepted_params() -> dict[str, set[str]]:
    """All accepted kwarg names per top-level nnx `NN*` constructor, resolved
    live from the INSTALLED nnx. Classes whose ``__init__`` accepts ``**kwargs``
    are omitted (their kwarg set is unbounded → can't validate).

    NOTE: this resolves against whatever nnx is installed, so a kwarg that
    exists only in a local *dev* checkout but not in the released
    ``thekaveh-nnx`` PyPI build will pass locally and FAIL in CI — which is
    exactly the point: it converts dev-vs-release drift (e.g. an unreleased
    ``NNGraphDataset(seed=)``) into a fast pytest-nnx-surface failure instead of
    a slow smoke-tier-b/c crash.
    """
    out: dict[str, set[str]] = {}
    for name in dir(nnx):
        obj = getattr(nnx, name)
        if not (inspect.isclass(obj) and name.startswith("NN")):
            continue
        try:
            sig = inspect.signature(obj.__init__)
        except (ValueError, TypeError):
            continue
        if any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values()):
            continue
        out[name] = {pname for pname, p in sig.parameters.items() if pname != "self"}
    return out


def find_nnx_unknown_kwargs(nb: dict, accepted: dict[str, set[str]]) -> list[str]:
    out: list[str] = []
    for idx, cell in enumerate(_code_cells(nb)):
        lines = [ln for ln in "".join(cell.get("source", [])).splitlines()
                 if not ln.lstrip().startswith(("%", "!"))]
        try:
            tree = ast.parse("\n".join(lines))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _called_name(node)
            if name not in accepted:
                continue
            if any(kw.arg is None for kw in node.keywords):
                continue  # **kwargs unpacking — can't statically resolve
            bad = {kw.arg for kw in node.keywords if kw.arg} - accepted[name]
            if bad:
                out.append(f"code_cell[{idx}]: {name}(...) unknown kwarg(s) {sorted(bad)} (not in installed nnx signature)")
    return out


@pytest.mark.parametrize("nb_path", _NOTEBOOKS, ids=_IDS)
def test_nnx_constructor_calls_use_known_kwargs(nb_path: Path):
    accepted = _nnx_ctor_accepted_params()
    assert accepted.get("NNGraphDataset"), "expected NNGraphDataset to resolve a signature"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    violations = find_nnx_unknown_kwargs(nb, accepted)
    assert not violations, (
        f"{nb_path.relative_to(REPO_ROOT)} calls an nnx constructor with a kwarg absent from the "
        f"installed nnx (dev-vs-release drift?):\n  " + "\n  ".join(violations)
    )


def test_nnx_unknown_kwarg_guard_catches_bad_kwarg():
    bad = _synthetic_nb({
        "cell_type": "code",
        "source": ["d = NNGraphDataset(ds_class=R, n_neighbors=[2], totally_made_up_kwarg=1)\n"],
        "outputs": [],
    })
    assert find_nnx_unknown_kwargs(bad, {"NNGraphDataset": {"ds_class", "n_neighbors", "n_workers", "transform", "batch_sizes", "root_dir"}})


def test_nnx_unknown_kwarg_guard_allows_real_kwargs():
    good = _synthetic_nb({
        "cell_type": "code",
        "source": ["d = NNGraphDataset(ds_class=R, n_neighbors=[2], n_workers=4, transform=t)\n"],
        "outputs": [],
    })
    assert not find_nnx_unknown_kwargs(good, {"NNGraphDataset": {"ds_class", "n_neighbors", "n_workers", "transform", "batch_sizes", "root_dir"}})


def _called_name(node: ast.Call) -> str | None:
    f = node.func
    if isinstance(f, ast.Attribute):
        return f.attr
    if isinstance(f, ast.Name):
        return f.id
    return None


def find_nndataset_batching_violations(
    nb: dict,
    *,
    expected_batch_sizes: tuple[int | None, int | None, int | None],
    expected_loader_aliases: dict[str, str],
) -> list[str]:
    trees: list[tuple[int, ast.Module]] = []
    for idx, cell in enumerate(_code_cells(nb)):
        source = _python_cell_source(cell)
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        trees.append((idx, tree))

    def resolve_tuple(
        node: ast.AST,
        constants: dict[str, object],
    ) -> tuple[object, ...] | None:
        if not isinstance(node, ast.Tuple):
            return None
        resolved: list[object] = []
        for element in node.elts:
            if isinstance(element, ast.Name) and element.id in constants:
                resolved.append(constants[element.id])
                continue
            try:
                resolved.append(ast.literal_eval(element))
            except (ValueError, TypeError):
                return None
        return tuple(resolved)

    violations: list[str] = []
    constants: dict[str, object] = {}
    dataset_calls: list[tuple[int, ast.Call, bool, tuple[object, ...] | None]] = []
    for idx, tree in trees:
        for statement in tree.body:
            for node in ast.walk(statement):
                if not isinstance(node, ast.Call) or _called_name(node) != "NNDataset":
                    continue
                values = [
                    keyword.value for keyword in node.keywords
                    if keyword.arg == "batch_sizes"
                ]
                actual = (
                    resolve_tuple(values[0], constants)
                    if len(values) == 1
                    else None
                )
                is_direct_ds_assignment = (
                    isinstance(statement, ast.Assign)
                    and len(statement.targets) == 1
                    and isinstance(statement.targets[0], ast.Name)
                    and statement.targets[0].id == "ds"
                    and statement.value is node
                )
                dataset_calls.append((idx, node, is_direct_ds_assignment, actual))

            if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
                continue
            targets = (
                statement.targets
                if isinstance(statement, ast.Assign)
                else [statement.target]
            )
            value = statement.value
            for target in targets:
                if not isinstance(target, ast.Name):
                    continue
                if value is None:
                    constants.pop(target.id, None)
                    continue
                try:
                    constants[target.id] = ast.literal_eval(value)
                except (ValueError, TypeError):
                    constants.pop(target.id, None)

    if len(dataset_calls) != 1:
        violations.append(f"expected exactly one NNDataset call, found {len(dataset_calls)}")
    else:
        idx, _call, is_direct_ds_assignment, actual = dataset_calls[0]
        if not is_direct_ds_assignment:
            violations.append(
                f"code_cell[{idx}]: NNDataset call must be assigned directly to ds at top level"
            )
        if actual != expected_batch_sizes:
            violations.append(
                f"code_cell[{idx}]: expected batch_sizes={expected_batch_sizes}, got {actual}"
            )

    top_level_node_ids = {
        id(statement)
        for _idx, tree in trees
        for statement in tree.body
    }
    alias_assignments: dict[str, list[tuple[int, ast.AST, bool]]] = {
        alias: [] for alias in expected_loader_aliases
    }
    for idx, tree in trees:
        for node in ast.walk(tree):
            targets: list[ast.AST] = []
            if isinstance(node, ast.Assign):
                targets = list(node.targets)
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
                targets = [node.target]
            for target in targets:
                for descendant in ast.walk(target):
                    if (
                        isinstance(descendant, ast.Name)
                        and descendant.id in alias_assignments
                    ):
                        alias_assignments[descendant.id].append(
                            (idx, node, id(node) in top_level_node_ids)
                        )

            if (
                isinstance(node, ast.Attribute)
                and node.attr == "dataset"
                and isinstance(node.value, ast.Attribute)
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id == "ds"
                and node.value.attr in {"train_loader", "val_loader", "test_loader"}
            ):
                violations.append(
                    f"code_cell[{idx}]: accesses ds.{node.value.attr}.dataset"
                )

    for alias, expected_loader in expected_loader_aliases.items():
        assignments = alias_assignments[alias]
        top_level_assignments = [item for item in assignments if item[2]]
        if assignments and not top_level_assignments:
            violations.append(
                f"code_cell[{assignments[0][0]}]: {alias} must be a top-level assignment"
            )
            continue
        if len(assignments) != 1 or len(top_level_assignments) != 1:
            violations.append(
                f"expected exactly one top-level assignment to {alias}, "
                f"found {len(top_level_assignments)} top-level and {len(assignments)} total"
            )
            continue
        idx, assignment, _is_top_level = assignments[0]
        if not (
            isinstance(assignment, ast.Assign)
            and len(assignment.targets) == 1
            and isinstance(assignment.targets[0], ast.Name)
            and isinstance(assignment.value, ast.Attribute)
            and isinstance(assignment.value.value, ast.Name)
            and assignment.value.value.id == "ds"
            and assignment.value.attr == expected_loader
        ):
            violations.append(
                f"code_cell[{idx}]: expected {alias} = ds.{expected_loader}"
            )
    return violations


def test_issue69_notebooks_use_nndataset_batching_contract():
    violations = []
    for relative_path, contract in _NNDATASET_BATCHING_CONTRACTS.items():
        notebook = json.loads((REPO_ROOT / relative_path).read_text(encoding="utf-8"))
        violations.extend(
            f"{relative_path}: {violation}"
            for violation in find_nndataset_batching_violations(
                notebook,
                expected_batch_sizes=contract["batch_sizes"],
                expected_loader_aliases=contract["loader_aliases"],
            )
        )
    assert not violations, "\n".join(violations)


def _issue69_batching_violations(source: str) -> list[str]:
    return find_nndataset_batching_violations(
        _synthetic_nb({
            "cell_type": "code",
            "source": source.splitlines(keepends=True),
            "outputs": [],
        }),
        expected_batch_sizes=(128, None, None),
        expected_loader_aliases={"train_loader": "train_loader"},
    )


def test_issue69_batching_guard_allows_direct_dataset_loader_alias():
    source = """\
BATCH_SIZE = 128
ds = NNDataset(batch_sizes=(BATCH_SIZE, None, None))
train_loader = ds.train_loader
"""
    assert not _issue69_batching_violations(source)


@pytest.mark.parametrize(
    ("source", "expected_fragment"),
    [
        (
            """\
BATCH_SIZE = 128
ds = NNDataset(batch_sizes=(BATCH_SIZE, None, None))
raw_dataset = ds.train_loader.dataset
train_loader = DataLoader(raw_dataset, batch_size=64)
""",
            "accesses ds.train_loader.dataset",
        ),
        (
            """\
BATCH_SIZE = 128
ds = NNDataset(batch_sizes=(BATCH_SIZE, None, None))
train_loader = ds.train_loader
train_loader = DataLoader(other_dataset, batch_size=64)
""",
            "expected exactly one top-level assignment to train_loader",
        ),
        (
            """\
BATCH_SIZE = 128
ds = NNDataset(batch_sizes=(BATCH_SIZE, None, None))
if False:
    train_loader = ds.train_loader
""",
            "must be a top-level assignment",
        ),
        (
            """\
BATCH_SIZE = 64
ds = NNDataset(batch_sizes=(BATCH_SIZE, None, None))
BATCH_SIZE = 128
train_loader = ds.train_loader
""",
            "expected batch_sizes=(128, None, None), got (64, None, None)",
        ),
        (
            """\
BATCH_SIZE = 128
ds = NNDataset(batch_sizes=(64, None, None))
train_loader = ds.train_loader
""",
            "expected batch_sizes=(128, None, None), got (64, None, None)",
        ),
    ],
    ids=(
        "indirect-loader-rebuild",
        "loader-alias-overwrite",
        "dead-code-loader-alias",
        "late-batch-size-rebind",
        "wrong-batch-size-tuple",
    ),
)
def test_issue69_batching_guard_rejects_contract_bypasses(
    source: str,
    expected_fragment: str,
):
    violations = _issue69_batching_violations(source)
    assert any(expected_fragment in violation for violation in violations), violations


def find_signature_violations(nb: dict, required: dict[str, set[str]]) -> list[str]:
    out: list[str] = []
    for idx, cell in enumerate(_code_cells(nb)):
        # drop ipython magics / shell-escapes that aren't valid python
        lines = [ln for ln in "".join(cell.get("source", [])).splitlines()
                 if not ln.lstrip().startswith(("%", "!"))]
        try:
            tree = ast.parse("\n".join(lines))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _called_name(node)
            if name not in required:
                continue
            if node.args or any(kw.arg is None for kw in node.keywords):
                continue  # positional / **kwargs — can't statically verify
            provided = {kw.arg for kw in node.keywords}
            missing = required[name] - provided
            if missing:
                out.append(f"code_cell[{idx}]: {name}(...) missing required kwarg(s) {sorted(missing)}")
    return out


@pytest.mark.parametrize("nb_path", _NOTEBOOKS, ids=_IDS)
def test_nnx_constructor_calls_supply_required_kwargs(nb_path: Path):
    required = _nnx_required_kwonly_params()
    assert required.get("NNOptimParams"), "expected NNOptimParams to have required keyword-only params"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    violations = find_signature_violations(nb, required)
    assert not violations, (
        f"{nb_path.relative_to(REPO_ROOT)} calls an nnx constructor with missing required kwarg(s):\n  "
        + "\n  ".join(violations)
    )


def test_signature_guard_catches_missing_momentum():
    required = _nnx_required_kwonly_params()
    assert "momentum" in required["NNOptimParams"], "fixture assumes momentum is required"
    bad = _synthetic_nb({
        "cell_type": "code",
        "source": ["p = NNOptimParams(name=Optims.ADAM, max_lr=1e-2, weight_decay=5e-4)\n"],
        "outputs": [],
    })
    assert find_signature_violations(bad, required)


def test_signature_guard_allows_complete_call():
    required = _nnx_required_kwonly_params()
    good = _synthetic_nb({
        "cell_type": "code",
        "source": ["p = NNOptimParams(name=Optims.ADAM, max_lr=1e-2, weight_decay=5e-4, momentum=(0.9, 0.999))\n"],
        "outputs": [],
    })
    assert not find_signature_violations(good, required)


# --- v0.2.0 stale-API guards (nnx usage-conformance review, 2026-06-29) -------
#
# The nnx 0.2.0 data model removed the flat per-iteration metric fields and the
# snapshot fields from the intermediate data point, and `NNModel.train` returns
# a single `NNRun` (no tuple to unpack). Seven node-classification notebooks
# (phase2 nb1-3, phase3 nb1-4) plus the image_classification baseline still
# referenced the stale shapes — `idp.train_loss` / `idp.val_error` raise
# `AttributeError`, and `NNRun.load("best")` is not the v0.2.0 idiom (load
# takes a real run id; the BEST checkpoint is reached via
# `NNCheckpoint.load(run=<id>, type=Checkpoints.BEST)` or `run.checkpoints()`).
# These execution-free scans catch the stale shapes on EVERY PR — the phase2/3
# notebooks live in the smoke-only Tier-B/C lanes, so the papermill tiers don't
# exercise them on a normal PR.

# Attribute access of a removed flat IDP metric field (NOT the nested
# `train_edp.loss` / `val_edp.error` form). The leading `.` requires attribute
# access; the trailing `\b` keeps `train_loss_history`-style names from matching.
_STALE_IDP_FIELD_RE = re.compile(
    r"\.(train_loss|train_error|val_loss|val_error|snapshot_x|snapshot_y_hat|snapshot_y)\b"
)
def find_stale_idp_fields(nb: dict) -> list[str]:
    out: list[str] = []
    for idx, cell in enumerate(_code_cells(nb)):
        for line in _executable_lines(cell):
            for m in _STALE_IDP_FIELD_RE.finditer(line):
                out.append(f"code_cell[{idx}]: .{m.group(1)} (removed; use train_edp/val_edp.<loss|error>)")
    return out


def find_nnrun_load_best(nb: dict) -> list[str]:
    out: list[str] = []
    for idx, cell in enumerate(_code_cells(nb)):
        lines = [
            ln for ln in "".join(_live_lines(cell)).splitlines()
            if not ln.lstrip().startswith(("%", "!"))
        ]
        try:
            tree = ast.parse("\n".join(lines))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (
                isinstance(func, ast.Attribute)
                and func.attr == "load"
                and isinstance(func.value, ast.Name)
                and func.value.id == "NNRun"
            ):
                continue
            first_arg = node.args[0] if node.args else None
            if isinstance(first_arg, ast.Constant) and first_arg.value == "best":
                out.append(f"code_cell[{idx}]: NNRun.load(\"best\") (use NNCheckpoint.load(run=<id>, type=Checkpoints.BEST))")
    return out


def find_nnrun_all_calls(nb: dict) -> list[str]:
    out: list[str] = []
    for idx, cell in enumerate(_code_cells(nb)):
        lines = [
            ln for ln in "".join(_live_lines(cell)).splitlines()
            if not ln.lstrip().startswith(("%", "!"))
        ]
        try:
            tree = ast.parse("\n".join(lines))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "all"
                and isinstance(func.value, ast.Name)
                and func.value.id == "NNRun"
            ):
                out.append(f"code_cell[{idx}]: NNRun.all() (use this notebook's local runs list)")
    return out


def find_sparse_tensor_edge_index_drops(nb: dict) -> list[str]:
    out: list[str] = []
    for idx, cell in enumerate(_code_cells(nb)):
        lines = [ln for ln in "".join(_live_lines(cell)).splitlines()
                 if not ln.lstrip().startswith(("%", "!"))]
        try:
            tree = ast.parse("\n".join(lines))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or _called_name(node) != "ToSparseTensor":
                continue
            preserves_edge_index = any(
                kw.arg == "remove_edge_index"
                and isinstance(kw.value, ast.Constant)
                and kw.value.value is False
                for kw in node.keywords
            )
            if not preserves_edge_index:
                out.append(f"code_cell[{idx}]: ToSparseTensor(...) drops edge_index by default")
    return out


@pytest.mark.parametrize("nb_path", _NOTEBOOKS, ids=_IDS)
def test_no_stale_flat_idp_fields(nb_path: Path):
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    violations = find_stale_idp_fields(nb)
    assert not violations, (
        f"{nb_path.relative_to(REPO_ROOT)} accesses removed flat IDP/snapshot field(s):\n  "
        + "\n  ".join(violations)
    )


@pytest.mark.parametrize("nb_path", _NOTEBOOKS, ids=_IDS)
def test_no_nnrun_load_best(nb_path: Path):
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    violations = find_nnrun_load_best(nb)
    assert not violations, (
        f"{nb_path.relative_to(REPO_ROOT)} calls NNRun.load(\"best\"):\n  "
        + "\n  ".join(violations)
    )


def test_phase2_notebook4_ranks_local_runs_not_cross_experiment_registry():
    nb_path = REPO_ROOT / "notebooks/node_classification-reddit-gnn-pyg/phase2-model-selection-notebook4.ipynb"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    violations = find_nnrun_all_calls(nb)
    assert not violations, (
        f"{nb_path.relative_to(REPO_ROOT)} ranks the shared NNRun registry instead of its local runs:\n  "
        + "\n  ".join(violations)
    )


@pytest.mark.parametrize("nb_path", _NOTEBOOKS, ids=_IDS)
def test_no_tosparsetensor_default_edge_index_drop(nb_path: Path):
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    violations = find_sparse_tensor_edge_index_drops(nb)
    assert not violations, (
        f"{nb_path.relative_to(REPO_ROOT)} calls ToSparseTensor without preserving edge_index:\n  "
        + "\n  ".join(violations)
    )


def test_stale_idp_guard_catches_flat_fields():
    bad = _synthetic_nb({
        "cell_type": "code",
        "source": ["losses = [idp.train_loss for idp in run.idps]\n", "e = min(run.idps, key=lambda i: i.val_error).val_error\n"],
        "outputs": [],
    })
    assert len(find_stale_idp_fields(bad)) >= 2


def test_stale_idp_guard_catches_string_source_flat_fields():
    bad = _synthetic_nb({
        "cell_type": "code",
        "source": "losses = [idp.train_loss for idp in run.idps]\n",
        "outputs": [],
    })
    assert find_stale_idp_fields(bad)


def test_stale_idp_guard_allows_nested_form_and_similar_names():
    good = _synthetic_nb({
        "cell_type": "code",
        "source": [
            "losses = [idp.train_edp.loss for idp in run.idps]\n",
            "errs = [i.val_edp.error for i in run.idps if i.val_edp is not None]\n",
            "history = run.train_loss_history\n",  # different attr — must NOT match
        ],
        "outputs": [],
    })
    assert not find_stale_idp_fields(good)


def test_stale_idp_guard_ignores_commented_snapshot_block():
    commented = _synthetic_nb({
        "cell_type": "code",
        "source": ["# if idp.snapshot_y_hat is None: continue  (legacy, removed)\n"],
        "outputs": [],
    })
    assert not find_stale_idp_fields(commented)


def test_stale_idp_guard_ignores_string_literals_and_docstrings():
    quoted = _synthetic_nb({
        "cell_type": "code",
        "source": [
            "\"\"\"Legacy note: idp.val_error\"\"\"\n",
            "msg = 'idp.train_loss'\n",
        ],
        "outputs": [],
    })
    assert not find_stale_idp_fields(quoted)


def test_nnrun_load_best_guard_catches_call():
    bad = _synthetic_nb({
        "cell_type": "code",
        "source": ["for c in NNRun.load(\"best\").checkpoints():\n", "    pass\n"],
        "outputs": [],
    })
    assert find_nnrun_load_best(bad)


def test_nnrun_load_best_guard_catches_string_source_call():
    bad = _synthetic_nb({
        "cell_type": "code",
        "source": "run = NNRun.load(\"best\")\n",
        "outputs": [],
    })
    assert find_nnrun_load_best(bad)


def test_nnrun_load_best_guard_allows_real_id_load():
    good = _synthetic_nb({
        "cell_type": "code",
        "source": ["run = NNRun.load(top_runs[0].id)\n", "ckpt = NNCheckpoint.load(run=top_runs[0].id, type=Checkpoints.BEST)\n"],
        "outputs": [],
    })
    assert not find_nnrun_load_best(good)


def test_nnrun_load_best_guard_ignores_string_literal_example():
    quoted = _synthetic_nb({
        "cell_type": "code",
        "source": ["msg = 'NNRun.load(\"best\")'\n"],
        "outputs": [],
    })
    assert not find_nnrun_load_best(quoted)


def test_tosparsetensor_guard_catches_default_edge_index_drop():
    bad = _synthetic_nb({
        "cell_type": "code",
        "source": ["transform = pyg.transforms.ToSparseTensor()\n"],
        "outputs": [],
    })
    assert find_sparse_tensor_edge_index_drops(bad)


def test_tosparsetensor_guard_allows_preserving_edge_index():
    good = _synthetic_nb({
        "cell_type": "code",
        "source": ["transform = pyg.transforms.ToSparseTensor(remove_edge_index=False)\n"],
        "outputs": [],
    })
    assert not find_sparse_tensor_edge_index_drops(good)


def test_tosparsetensor_guard_catches_multiline_default_edge_index_drop():
    bad = _synthetic_nb({
        "cell_type": "code",
        "source": [
            "transform = pyg.transforms.ToSparseTensor(\n",
            "    fill_cache=False,\n",
            ")\n",
        ],
        "outputs": [],
    })
    assert find_sparse_tensor_edge_index_drops(bad)


def test_tosparsetensor_guard_catches_string_source_default_edge_index_drop():
    bad = _synthetic_nb({
        "cell_type": "code",
        "source": "# keep this historical note\ntransform = pyg.transforms.ToSparseTensor()\n",
        "outputs": [],
    })
    assert find_sparse_tensor_edge_index_drops(bad)


def test_tosparsetensor_guard_allows_spaced_keyword_assignment():
    good = _synthetic_nb({
        "cell_type": "code",
        "source": ["transform = pyg.transforms.ToSparseTensor(remove_edge_index = False)\n"],
        "outputs": [],
    })
    assert not find_sparse_tensor_edge_index_drops(good)


def test_tosparsetensor_guard_allows_multiline_preserving_edge_index():
    good = _synthetic_nb({
        "cell_type": "code",
        "source": [
            "transform = pyg.transforms.ToSparseTensor(\n",
            "    fill_cache=False,\n",
            "    remove_edge_index=False,\n",
            ")\n",
        ],
        "outputs": [],
    })
    assert not find_sparse_tensor_edge_index_drops(good)


# --- VisUtils call-signature guard (nnx plotting-API drift) -------------------
#
# The plotting helpers on `nnx.vis_utils.VisUtils` are a recurring source of
# silent drift: a wide nnx version bump renames/removes a kwarg (e.g.
# `scatter_plot(figsize=)` → `fig_size=`), and the only callers that hit it are
# the smoke-only Tier-B/C node-classification notebooks, which the per-PR
# papermill tier never executes — so the `TypeError` only surfaces on the weekly
# cron (or never, if an earlier cell already crashes). This AST guard validates
# every `VisUtils.<m>(...)` call's keyword-arg NAMES against the live signature,
# so a renamed/removed kwarg fails on every PR. (It can't check positional
# structure like `yss_legend`'s (group_labels, line_labels) tuple — that's a
# value shape, not a kwarg name.)


def _visutils_method_params() -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for name in dir(VisUtils):
        if name.startswith("_"):
            continue
        obj = getattr(VisUtils, name)
        if not callable(obj):
            continue
        try:
            sig = inspect.signature(obj)
        except (ValueError, TypeError):
            continue
        # Skip methods that accept **kwargs — their kwarg set is unbounded.
        if any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values()):
            continue
        out[name] = {p.name for p in sig.parameters.values() if p.name != "self"}
    return out


def find_visutils_kwarg_violations(nb: dict, params: dict[str, set[str]]) -> list[str]:
    out: list[str] = []
    for idx, cell in enumerate(_code_cells(nb)):
        lines = [ln for ln in "".join(cell.get("source", [])).splitlines()
                 if not ln.lstrip().startswith(("%", "!"))]
        try:
            tree = ast.parse("\n".join(lines))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            if not (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "VisUtils"):
                continue
            if f.attr not in params:
                continue
            provided = {kw.arg for kw in node.keywords if kw.arg}
            bad = provided - params[f.attr]
            if bad:
                out.append(f"code_cell[{idx}]: VisUtils.{f.attr}(...) invalid kwarg(s) {sorted(bad)}")
    return out


@pytest.mark.parametrize("nb_path", _NOTEBOOKS, ids=_IDS)
def test_visutils_calls_use_valid_kwargs(nb_path: Path):
    params = _visutils_method_params()
    assert params.get("multi_line_plot"), "expected VisUtils.multi_line_plot to resolve a signature"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    violations = find_visutils_kwarg_violations(nb, params)
    assert not violations, (
        f"{nb_path.relative_to(REPO_ROOT)} calls VisUtils with invalid kwarg(s):\n  "
        + "\n  ".join(violations)
    )


def test_visutils_guard_catches_bad_kwarg():
    params = _visutils_method_params()
    assert "figsize" not in params.get("scatter_plot", set()), "fixture assumes scatter_plot uses fig_size, not figsize"
    bad = _synthetic_nb({
        "cell_type": "code",
        "source": ["VisUtils.scatter_plot(vm=vm, figsize=(25, 20))\n"],
        "outputs": [],
    })
    assert find_visutils_kwarg_violations(bad, params)


def test_visutils_guard_allows_valid_kwargs():
    params = _visutils_method_params()
    good = _synthetic_nb({
        "cell_type": "code",
        "source": ["VisUtils.scatter_plot(vm=vm, fig_size=(25, 20))\n", "VisUtils.multi_line_plot(x=x, yss=y, title=t, yss_legend=g, x_axis_label='a', y_axis_label='b')\n"],
        "outputs": [],
    })
    assert not find_visutils_kwarg_violations(good, params)
