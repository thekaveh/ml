"""Tests for deterministic notebook source-hash stamping."""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.stamp_notebook_source_hashes import (
    NotebookStampError,
    active_notebook_paths,
    clear_document,
    clear_path,
    compute_source_hash,
    logical_source,
    stamp_document,
    stamp_path,
)


REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "stamp_notebook_source_hashes.py"


def _code(source: object, *, outputs: list[dict] | None = None, metadata: dict | None = None) -> dict:
    return {
        "cell_type": "code",
        "execution_count": 1 if outputs else None,
        "id": "code-cell",
        "metadata": metadata or {},
        "outputs": outputs if outputs is not None else [],
        "source": source,
    }


def _document(*cells: dict) -> dict:
    return {
        "cells": list(cells),
        "metadata": {"kernelspec": {"name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _stream_output(text: str = "ok\n") -> dict:
    return {"name": "stdout", "output_type": "stream", "text": text}


def _error_output() -> dict:
    return {
        "ename": "RuntimeError",
        "evalue": "boom",
        "output_type": "error",
        "traceback": ["Traceback (most recent call last):", "RuntimeError: boom"],
    }


def _write_json(path: Path, document: dict) -> bytes:
    encoded = (json.dumps(document, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
    path.write_bytes(encoded)
    return encoded


def test_logical_source_accepts_string_and_concatenates_string_list() -> None:
    assert logical_source("print('x')\n") == "print('x')\n"
    assert logical_source(["print(", "'x')", "\n"]) == "print('x')\n"


@pytest.mark.parametrize("source", [["ok", 1], [1], {"source": "x"}, None])
def test_logical_source_rejects_non_string_source_values(source: object) -> None:
    with pytest.raises(NotebookStampError, match="source"):
        logical_source(source)


def test_compute_source_hash_is_sha256_of_exact_utf8_source() -> None:
    source = "# café\r\nprint('x')"
    expected = hashlib.sha256(source.encode("utf-8")).hexdigest()
    assert compute_source_hash(source) == expected
    assert len(compute_source_hash(source)) == 64
    assert compute_source_hash(source).islower()


def test_stamp_document_only_marks_code_cells_with_outputs_and_removes_orphans() -> None:
    output = _stream_output()
    document = _document(
        _code("print('output')\n", outputs=[output], metadata={"keep": "yes"}),
        _code("print('none')\n", metadata={"source_hash": "orphan", "keep": 3}),
        {"cell_type": "markdown", "metadata": {"source_hash": "leave"}, "source": "# hi"},
        {"cell_type": "raw", "metadata": {}, "source": ["raw"]},
    )

    changed = stamp_document(document)

    assert changed == 2
    assert document["cells"][0]["metadata"]["source_hash"] == compute_source_hash("print('output')\n")
    assert document["cells"][0]["metadata"]["keep"] == "yes"
    assert "source_hash" not in document["cells"][1]["metadata"]
    assert document["cells"][1]["metadata"]["keep"] == 3
    assert document["cells"][2]["metadata"]["source_hash"] == "leave"


def test_stamp_document_is_idempotent() -> None:
    document = _document(
        _code(
            "x = 1",
            outputs=[{"output_type": "execute_result", "data": {}, "metadata": {}, "execution_count": 1}],
        )
    )
    assert stamp_document(document) == 1
    assert stamp_document(document) == 0


def test_outputless_code_without_metadata_is_left_untouched() -> None:
    document = _document(
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": "x = 1", "outputs": []}
    )

    assert stamp_document(document) == 0
    assert document["cells"][0]["metadata"] == {}


def test_clear_document_removes_hashes_from_all_code_cells_in_failed_artifact() -> None:
    failed = _code(
        "raise RuntimeError('boom')",
        outputs=[_error_output()],
        metadata={"source_hash": "a" * 64, "keep": {"nested": True}},
    )
    outputless = _code("x = 1", metadata={"source_hash": "b" * 64, "keep": 3})
    markdown = {"cell_type": "markdown", "metadata": {"source_hash": "leave"}, "source": "# hi"}
    document = _document(failed, outputless, markdown)
    error_output = json.loads(json.dumps(failed["outputs"][0]))

    assert clear_document(document) == 2
    assert "source_hash" not in failed["metadata"]
    assert failed["metadata"]["keep"] == {"nested": True}
    assert failed["outputs"][0] == error_output
    assert "source_hash" not in outputless["metadata"]
    assert outputless["metadata"]["keep"] == 3
    assert markdown["metadata"]["source_hash"] == "leave"


def test_clear_document_accepts_output_bearing_code_cell_with_null_execution_count() -> None:
    cell = _code("x = 1", outputs=[_stream_output()], metadata={"source_hash": "a" * 64})
    cell["execution_count"] = None
    document = _document(cell)

    assert clear_document(document) == 1
    assert "source_hash" not in cell["metadata"]


def test_clear_path_preserves_unrelated_fields_permissions_and_is_byte_idempotent(tmp_path: Path) -> None:
    cell = _code(
        "raise RuntimeError('boom')",
        outputs=[_error_output()],
        metadata={"source_hash": "a" * 64, "custom": [1, {"two": 2}]},
    )
    cell["execution_count"] = None
    document = _document(
        cell,
        {"cell_type": "raw", "id": "raw-cell", "metadata": {"custom": "raw"}, "source": ["raw\n"]},
    )
    path = tmp_path / "failed.ipynb"
    _write_json(path, document)
    os.chmod(path, 0o640)
    expected = json.loads(json.dumps(document))
    del expected["cells"][0]["metadata"]["source_hash"]

    assert clear_path(path) == 1
    cleared = path.read_bytes()
    assert json.loads(cleared) == expected
    assert stat.S_IMODE(path.stat().st_mode) == 0o640
    assert clear_path(path) == 0
    assert path.read_bytes() == cleared


@pytest.mark.parametrize(
    ("document", "message"),
    (
        ([], "root"),
        (_document(None), "cell"),
        (_document(_code("x", outputs=[{"output_type": "error", "ename": "Error", "evalue": "x"}])), "error"),
        (_document(_code("x", outputs=[{"output_type": "stream", "name": "stdout", "text": 3}])), "output"),
    ),
    ids=("malformed-root", "malformed-cell", "malformed-error", "malformed-output"),
)
def test_clear_path_rejects_malformed_notebook_without_changing_bytes(
    tmp_path: Path, document: object, message: str
) -> None:
    path = tmp_path / "bad.ipynb"
    original = _write_json(path, document)

    with pytest.raises(NotebookStampError, match=message):
        clear_path(path)

    assert path.read_bytes() == original


def test_stamp_path_preserves_unrelated_fields_and_is_byte_stable_on_repeat(tmp_path: Path) -> None:
    document = _document(
        _code("x = 1", outputs=[_stream_output("x\n")], metadata={"custom": [1]}),
        {"cell_type": "markdown", "metadata": {"custom": "markdown"}, "source": "hello"},
    )
    path = tmp_path / "notebook.ipynb"
    original = _write_json(path, document)

    assert stamp_path(path) == 1
    stamped = path.read_bytes()
    assert stamped != original
    assert stamp_path(path) == 0
    assert path.read_bytes() == stamped
    loaded = json.loads(stamped)
    assert loaded["cells"][0]["outputs"][0]["text"] == "x\n"
    assert loaded["cells"][0]["metadata"]["custom"] == [1]
    assert loaded["cells"][1] == document["cells"][1]


def test_stamp_path_preserves_permission_bits(tmp_path: Path) -> None:
    path = tmp_path / "notebook.ipynb"
    _write_json(path, _document(_code("x", outputs=[_stream_output()])))
    os.chmod(path, 0o640)

    stamp_path(path)

    assert stat.S_IMODE(path.stat().st_mode) == 0o640


def test_malformed_notebook_is_rejected_without_changing_bytes(tmp_path: Path) -> None:
    path = tmp_path / "bad.ipynb"
    original = b'{"cells": [}'
    path.write_bytes(original)

    with pytest.raises(NotebookStampError, match="JSON"):
        stamp_path(path)

    assert path.read_bytes() == original


def test_structurally_invalid_notebook_is_rejected_without_changing_bytes(tmp_path: Path) -> None:
    path = tmp_path / "bad.ipynb"
    original = _write_json(path, {"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5})
    # A code cell without its required outputs field is not a valid notebook.
    bad = json.loads(original)
    bad["cells"] = [{"cell_type": "code", "execution_count": None, "metadata": {}, "source": "x"}]
    original = _write_json(path, bad)

    with pytest.raises(NotebookStampError, match="outputs"):
        stamp_path(path)

    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "cell_update, message",
    [
        (
            {
                "outputs": [
                    {
                        "output_type": "error",
                        "ename": "RuntimeError",
                        "evalue": "boom",
                        "traceback": [],
                    }
                ]
            },
            "error",
        ),
        ({"outputs": [_stream_output()], "execution_count": None}, "execution_count"),
        ({"outputs": [_stream_output()], "missing_execution_count": True}, "execution_count"),
    ],
)
def test_failed_or_partial_output_is_rejected_before_any_mutation(cell_update: dict, message: str) -> None:
    cell = _code("x = 1", outputs=[_stream_output()])
    cell.update(cell_update)
    if cell.pop("missing_execution_count", False):
        cell.pop("execution_count")
    document = _document(cell)
    before = json.loads(json.dumps(document))

    with pytest.raises(NotebookStampError, match=message):
        stamp_document(document)

    assert document == before


@pytest.mark.parametrize(
    "document_update, message",
    [
        ({"nbformat": None}, "nbformat"),
        ({"nbformat_minor": "5"}, "nbformat_minor"),
        ({"metadata": None}, "metadata"),
        (
            {"cells": [{"cell_type": "code", "source": "x", "outputs": [], "metadata": None}]},
            "metadata",
        ),
        ({"cells": [{**_code("x", outputs=[_stream_output()]), "outputs": [{}]}]}, "output"),
        (
            {"cells": [{**_code("x", outputs=[_stream_output()]), "outputs": [{"output_type": "mystery"}]}]},
            "output",
        ),
    ],
)
def test_malformed_versions_metadata_or_outputs_are_rejected_without_writing(
    tmp_path: Path, document_update: dict, message: str
) -> None:
    document = _document(_code("x", outputs=[_stream_output()]))
    document.update(document_update)
    path = tmp_path / "bad.ipynb"
    original = _write_json(path, document)

    with pytest.raises(NotebookStampError, match=message):
        stamp_path(path)

    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "cell",
    [
        {"cell_type": "markdown", "metadata": {}, "source": "![x](x.png)", "attachments": []},
        {"cell_type": "markdown", "metadata": {}, "source": "![x](x.png)", "attachments": {"": {"image/png": "x"}}},
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "![x](x.png)",
            "attachments": {"x.png": []},
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "![x](x.png)",
            "attachments": {"x.png": {"image/png": None}},
        },
        {
            "cell_type": "raw",
            "metadata": {},
            "source": "raw",
            "attachments": {"x.png": {"image/png": "x"}},
        },
    ],
)
def test_malformed_or_misplaced_attachments_are_rejected_without_writing(tmp_path: Path, cell: dict) -> None:
    path = tmp_path / "bad-attachments.ipynb"
    original = _write_json(path, _document(cell, _code("x", outputs=[_stream_output()])))

    with pytest.raises(NotebookStampError, match="attachment"):
        stamp_path(path)

    assert path.read_bytes() == original


def test_failed_replace_cleans_up_sibling_temporary_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import stamp_notebook_source_hashes as stamper

    path = tmp_path / "notebook.ipynb"
    _write_json(path, _document(_code("x", outputs=[_stream_output()])))
    original_replace = stamper.os.replace

    def fail_replace(src: str, dst: str) -> None:
        raise OSError("simulated replacement failure")

    monkeypatch.setattr(stamper.os, "replace", fail_replace)
    with pytest.raises(NotebookStampError, match="replace"):
        stamp_path(path)
    assert path.exists()
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []
    monkeypatch.setattr(stamper.os, "replace", original_replace)


def test_active_notebook_inventory_is_sorted_and_excludes_archive(tmp_path: Path) -> None:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "notebooks" / "task-b").mkdir(parents=True)
    (tmp_path / "notebooks" / "task-a").mkdir(parents=True)
    (tmp_path / "notebooks" / "archive").mkdir(parents=True)
    (tmp_path / "scripts" / "verify_repo_config.yaml").write_text(
        "active_task_dirs:\n  - task-b\n  - task-a\n", encoding="utf-8"
    )
    (tmp_path / "notebooks" / "task-b" / "z.ipynb").write_text("{}", encoding="utf-8")
    (tmp_path / "notebooks" / "task-a" / "a.ipynb").write_text("{}", encoding="utf-8")
    (tmp_path / "notebooks" / "archive" / "old.ipynb").write_text("{}", encoding="utf-8")

    paths = active_notebook_paths(tmp_path)

    assert paths == (
        tmp_path / "notebooks" / "task-a" / "a.ipynb",
        tmp_path / "notebooks" / "task-b" / "z.ipynb",
    )
    assert all("archive" not in path.parts for path in paths)


def test_live_active_notebook_source_hash_inventory_is_current() -> None:
    inventory = {
        "notebooks": 0,
        "output_bearing_cells": 0,
        "current_markers": 0,
        "missing_markers": 0,
        "invalid_markers": 0,
        "stale_markers": 0,
        "orphan_markers": 0,
        "error_output_cells": 0,
    }
    for notebook in active_notebook_paths(REPO):
        inventory["notebooks"] += 1
        document = json.loads(notebook.read_text(encoding="utf-8"))
        for cell in document["cells"]:
            if cell.get("cell_type") != "code":
                continue
            metadata = cell["metadata"]
            outputs = cell["outputs"]
            marker = metadata.get("source_hash")
            inventory["error_output_cells"] += sum(
                output.get("output_type") == "error" for output in outputs
            )
            if not outputs:
                inventory["orphan_markers"] += "source_hash" in metadata
            else:
                inventory["output_bearing_cells"] += 1
                if marker is None:
                    inventory["missing_markers"] += 1
                elif not isinstance(marker, str) or re.fullmatch(r"[0-9a-f]{64}", marker) is None:
                    inventory["invalid_markers"] += 1
                elif marker != compute_source_hash(cell["source"]):
                    inventory["stale_markers"] += 1
                else:
                    inventory["current_markers"] += 1

    assert inventory == {
        "notebooks": 29,
        "output_bearing_cells": 189,
        "current_markers": 189,
        "missing_markers": 0,
        "invalid_markers": 0,
        "stale_markers": 0,
        "orphan_markers": 0,
        "error_output_cells": 0,
    }


def test_cli_requires_exactly_one_of_paths_or_all_active(tmp_path: Path) -> None:
    commands = (
        [sys.executable, str(SCRIPT)],
        [sys.executable, str(SCRIPT), "--all-active", str(tmp_path / "x.ipynb")],
    )
    for command in commands:
        result = subprocess.run(command, cwd=REPO, capture_output=True, text=True, timeout=30)
        assert result.returncode != 0
        assert "path" in (result.stderr + result.stdout).lower()


def test_cli_clear_requires_explicit_paths_and_rejects_all_active() -> None:
    commands = (
        [sys.executable, str(SCRIPT), "--clear"],
        [sys.executable, str(SCRIPT), "--clear", "--all-active"],
    )
    for command in commands:
        result = subprocess.run(command, cwd=REPO, capture_output=True, text=True, timeout=30)
        assert result.returncode != 0
        message = (result.stderr + result.stdout).lower()
        assert "clear" in message
        assert "path" in message


def test_cli_clears_explicit_failed_notebook_path(tmp_path: Path) -> None:
    path = tmp_path / "failed.ipynb"
    _write_json(
        path,
        _document(_code("raise RuntimeError", outputs=[_error_output()], metadata={"source_hash": "a" * 64})),
    )

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--clear", str(path)], capture_output=True, text=True, timeout=30
    )

    assert result.returncode == 0, result.stderr
    assert "source_hash" not in json.loads(path.read_text(encoding="utf-8"))["cells"][0]["metadata"]


def test_cli_stamps_explicit_path(tmp_path: Path) -> None:
    path = tmp_path / "notebook.ipynb"
    _write_json(path, _document(_code("x", outputs=[_stream_output()])))

    result = subprocess.run([sys.executable, str(SCRIPT), str(path)], capture_output=True, text=True, timeout=30)

    assert result.returncode == 0, result.stderr
    assert json.loads(path.read_text(encoding="utf-8"))["cells"][0]["metadata"]["source_hash"]
