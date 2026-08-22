from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "verify_smoke_outputs.py"
QUANTIZATION_SOURCE = "notebooks/quantization-mnist-ffnn-pytorch/notebook.ipynb"
QUANTIZATION_OUTPUT_NAME = "quantization-mnist-ffnn-pytorch.ipynb"
QUANTIZATION_MARKER_PREFIX = "ISSUE66_QUANTIZATION_CONTRACT="
QUANTIZATION_SMOKE_CONTRACT = {
    "artifacts_nonempty": True,
    "checkpoint_evaluation_finite": True,
    "checkpoint_metadata_parity": True,
    "checkpoint_reloaded": True,
    "checkpoint_state_parity": True,
    "epochs": 1,
    "ptq_validated": True,
    "qat_converted": True,
    "qat_prepared": True,
    "qat_quantized_module": True,
    "schema_version": 1,
    "seed": 0,
    "smoke_test": True,
}


def _module() -> ModuleType:
    return importlib.import_module("scripts.verify_smoke_outputs")


def _sources(tier: str) -> tuple[str, ...]:
    counts = {"a": 18, "b": 7, "c": 4}
    if tier == "a":
        return tuple(f"notebooks/task-{index:02d}/notebook.ipynb" for index in range(counts[tier]))
    if tier == "b":
        return (QUANTIZATION_SOURCE,) + tuple(
            f"notebooks/task-{index:02d}/notebook-{index:02d}.ipynb"
            for index in range(counts[tier] - 1)
        )
    return tuple(
        f"notebooks/task-{index:02d}/notebook-{index:02d}.ipynb"
        for index in range(counts[tier])
    )


def _output_path(root: Path, tier: str, source: str) -> Path:
    if tier == "a":
        return root / source
    if tier == "b" and source == QUANTIZATION_SOURCE:
        return root / QUANTIZATION_OUTPUT_NAME
    return root / Path(source).name


def write_executed_notebook(
    path: Path, *, code_cells: int = 1, output_text: str = "ok\n"
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "code",
                        "execution_count": index + 1,
                        "metadata": {},
                        "outputs": [
                            {
                                "output_type": "stream",
                                "name": "stdout",
                                "text": [output_text],
                            }
                        ],
                        "source": ["print('ok')\n"],
                    }
                    for index in range(code_cells)
                ],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        ),
        encoding="utf-8",
    )


def _write_valid_outputs(root: Path, tier: str, sources: tuple[str, ...]) -> None:
    for source in sources:
        output_text = "ok\n"
        if source == QUANTIZATION_SOURCE:
            output_text = (
                QUANTIZATION_MARKER_PREFIX
                + json.dumps(QUANTIZATION_SMOKE_CONTRACT, sort_keys=True)
                + "\n"
            )
        write_executed_notebook(
            _output_path(root, tier, source), output_text=output_text
        )


def _assert_category(error: Exception, tier: str, category: str) -> None:
    assert str(error) == f"smoke output verification failed: {tier}: {category}"


@pytest.mark.parametrize(("tier", "count"), (("a", 18), ("b", 7), ("c", 4)))
def test_valid_inventory_returns_every_executed_artifact(
    tmp_path: Path, tier: str, count: int
) -> None:
    module = _module()
    root = tmp_path / f"tier-{tier}"
    sources = _sources(tier)
    _write_valid_outputs(root, tier, sources)

    artifacts = module.verify_smoke_outputs(tier, root, lambda selected: sources)

    assert len(artifacts) == count
    assert tuple(artifact.source for artifact in artifacts) == sources
    assert tuple(artifact.output for artifact in artifacts) == tuple(
        _output_path(root, tier, source) for source in sources
    )
    assert all(artifact.code_cells == 1 for artifact in artifacts)


def test_real_tier_b_inventory_maps_quantization_to_a_unique_output(
    tmp_path: Path,
) -> None:
    module = _module()
    sources = tuple(module.load_make_inventory("b"))
    _write_valid_outputs(tmp_path, "b", sources)

    artifacts = module.verify_smoke_outputs("b", tmp_path)

    output_names = tuple(artifact.output.name for artifact in artifacts)
    assert "notebook.ipynb" in output_names
    assert "quantization-mnist-ffnn-pytorch.ipynb" in output_names
    assert len(output_names) == len(set(output_names)) == 7


@pytest.mark.parametrize(
    "mutation",
    ("missing", "false-field", "duplicate", "malformed", "extra-field"),
)
def test_quantization_output_contract_is_fail_closed(
    tmp_path: Path, mutation: str
) -> None:
    module = _module()
    sources = _sources("b")
    _write_valid_outputs(tmp_path, "b", sources)
    target = _output_path(tmp_path, "b", QUANTIZATION_SOURCE)
    document = json.loads(target.read_text(encoding="utf-8"))
    if mutation == "missing":
        text = "ok\n"
    elif mutation == "malformed":
        text = QUANTIZATION_MARKER_PREFIX + "{\n"
    else:
        payload = dict(QUANTIZATION_SMOKE_CONTRACT)
        if mutation == "false-field":
            payload["checkpoint_reloaded"] = False
        elif mutation == "extra-field":
            payload["unreviewed"] = True
        marker = QUANTIZATION_MARKER_PREFIX + json.dumps(payload, sort_keys=True) + "\n"
        text = marker * (2 if mutation == "duplicate" else 1)
    document["cells"][0]["outputs"][0]["text"] = [text]
    target.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(module.SmokeOutputError) as caught:
        module.verify_smoke_outputs("b", tmp_path, lambda selected: sources)

    _assert_category(caught.value, "b", "semantic")


@pytest.mark.parametrize("delta", (-1, 1), ids=("missing-item", "extra-item"))
def test_inventory_requires_the_exact_tier_count(tmp_path: Path, delta: int) -> None:
    module = _module()
    sources = list(_sources("b"))
    if delta < 0:
        sources.pop()
    else:
        sources.append("notebooks/extra/extra.ipynb")

    with pytest.raises(module.SmokeOutputError) as caught:
        module.verify_smoke_outputs("b", tmp_path, lambda selected: tuple(sources))

    _assert_category(caught.value, "b", "inventory")


def test_inventory_rejects_duplicate_sources(tmp_path: Path) -> None:
    module = _module()
    sources = list(_sources("c"))
    sources[1] = sources[0]

    with pytest.raises(module.SmokeOutputError) as caught:
        module.verify_smoke_outputs("c", tmp_path, lambda selected: tuple(sources))

    _assert_category(caught.value, "c", "inventory")


def test_inventory_rejects_duplicate_mapped_outputs(tmp_path: Path) -> None:
    module = _module()
    sources = list(_sources("b"))
    sources[2] = "notebooks/other/notebook-00.ipynb"

    with pytest.raises(module.SmokeOutputError) as caught:
        module.verify_smoke_outputs("b", tmp_path, lambda selected: tuple(sources))

    _assert_category(caught.value, "b", "inventory")


def test_recursive_extra_notebook_is_inventory_drift(tmp_path: Path) -> None:
    module = _module()
    sources = _sources("a")
    _write_valid_outputs(tmp_path, "a", sources)
    write_executed_notebook(tmp_path / "unexpected" / "nested" / "extra.ipynb")

    with pytest.raises(module.SmokeOutputError) as caught:
        module.verify_smoke_outputs("a", tmp_path, lambda selected: sources)

    _assert_category(caught.value, "a", "inventory")


def test_absent_expected_output_is_missing(tmp_path: Path) -> None:
    module = _module()
    sources = _sources("c")
    _write_valid_outputs(tmp_path, "c", sources[:-1])

    with pytest.raises(module.SmokeOutputError) as caught:
        module.verify_smoke_outputs("c", tmp_path, lambda selected: sources)

    _assert_category(caught.value, "c", "missing")


@pytest.mark.parametrize(
    ("case", "category"),
    (
        ("empty", "empty"),
        ("invalid-json", "invalid"),
        ("non-notebook-json", "invalid"),
        ("zero-code-cells", "invalid"),
        ("null-execution-count", "unexecuted"),
        ("error-output", "error"),
    ),
)
def test_notebook_validation_is_fail_closed(
    tmp_path: Path, case: str, category: str
) -> None:
    module = _module()
    sources = _sources("b")
    _write_valid_outputs(tmp_path, "b", sources)
    target = _output_path(tmp_path, "b", sources[0])
    if case == "empty":
        target.write_text("", encoding="utf-8")
    elif case == "invalid-json":
        target.write_text("{", encoding="utf-8")
    elif case == "non-notebook-json":
        target.write_text(json.dumps(["notebook"]), encoding="utf-8")
    elif case == "zero-code-cells":
        target.write_text(
            json.dumps(
                {
                    "cells": [{"cell_type": "markdown", "metadata": {}, "source": ["text"]}],
                    "metadata": {},
                    "nbformat": 4,
                    "nbformat_minor": 5,
                }
            ),
            encoding="utf-8",
        )
    else:
        document = json.loads(target.read_text(encoding="utf-8"))
        if case == "null-execution-count":
            document["cells"][0]["execution_count"] = None
        else:
            document["cells"][0]["outputs"].append(
                {
                    "output_type": "error",
                    "ename": "SensitiveError",
                    "evalue": "token=secret",
                    "traceback": ["token=secret"],
                }
            )
        target.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(module.SmokeOutputError) as caught:
        module.verify_smoke_outputs("b", tmp_path, lambda selected: sources)

    _assert_category(caught.value, "b", category)
    assert "secret" not in str(caught.value)


@pytest.mark.parametrize(
    ("case", "malformed"),
    (
        ("non-mapping-cell", "cell"),
        ("non-list-outputs", "outputs"),
        ("non-mapping-output-string", "output-string"),
        ("non-mapping-output-null", "output-null"),
    ),
)
def test_notebook_validation_rejects_malformed_cell_and_output_shapes(
    tmp_path: Path, case: str, malformed: str
) -> None:
    module = _module()
    sources = _sources("c")
    _write_valid_outputs(tmp_path, "c", sources)
    target = _output_path(tmp_path, "c", sources[0])
    document = json.loads(target.read_text(encoding="utf-8"))
    if malformed == "cell":
        document["cells"][0] = "not-a-cell"
    elif malformed == "outputs":
        document["cells"][0]["outputs"] = {"output_type": "stream"}
    elif malformed == "output-string":
        document["cells"][0]["outputs"] = ["not-an-output"]
    else:
        document["cells"][0]["outputs"] = [None]
    target.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(module.SmokeOutputError) as caught:
        module.verify_smoke_outputs("c", tmp_path, lambda selected: sources)

    _assert_category(caught.value, "c", "invalid")


@pytest.mark.parametrize("execution_count", (True, "1", 1.5))
def test_notebook_validation_rejects_non_integer_execution_counts(
    tmp_path: Path, execution_count: object
) -> None:
    module = _module()
    sources = _sources("a")
    _write_valid_outputs(tmp_path, "a", sources)
    target = _output_path(tmp_path, "a", sources[0])
    document = json.loads(target.read_text(encoding="utf-8"))
    document["cells"][0]["execution_count"] = execution_count
    target.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(module.SmokeOutputError) as caught:
        module.verify_smoke_outputs("a", tmp_path, lambda selected: sources)

    _assert_category(caught.value, "a", "unexecuted")


def test_missing_error_redacts_the_output_path(tmp_path: Path) -> None:
    module = _module()
    sources = list(_sources("b"))
    sources[0] = "notebooks/private/token=secret.ipynb"

    with pytest.raises(module.SmokeOutputError) as caught:
        module.verify_smoke_outputs("b", tmp_path, lambda selected: tuple(sources))

    _assert_category(caught.value, "b", "missing")
    assert "secret" not in str(caught.value)


def test_make_inventory_failure_is_stable_and_redacted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    (tmp_path / "Makefile").write_text(
        "print-tier-a:\n\t@printf 'token=secret\\n' >&2\n\t@false\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    with pytest.raises(module.SmokeOutputError) as caught:
        module.load_make_inventory("a")

    _assert_category(caught.value, "a", "inventory")
    assert "secret" not in str(caught.value)


def test_cli_accepts_only_an_absolute_root(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _module()

    assert module.main(["--tier", "a", "--root", "relative"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "smoke output verification failed: a: inventory\n"


def test_cli_success_prints_only_a_stable_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _module()
    sources = _sources("c")
    root = tmp_path / "outputs"
    _write_valid_outputs(root, "c", sources)
    (tmp_path / "Makefile").write_text(
        "TIER_C := " + " ".join(sources) + "\n"
        "print-tier-c:\n\t@printf '%s\\n' $(TIER_C)\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    assert module.main(["--tier", "c", "--root", str(root)]) == 0
    captured = capsys.readouterr()
    assert captured.out == "smoke output verification ok: c: notebooks=4 code_cells=4\n"
    assert captured.err == ""


@pytest.mark.parametrize(
    "argv",
    (
        (),
        ("--tier", "d", "--root", "/tmp/out"),
        ("--tier", "a"),
        ("--root", "/tmp/out", "--tier", "a", "extra"),
    ),
)
def test_cli_rejects_malformed_arguments_without_usage_payload(
    argv: tuple[str, ...], capsys: pytest.CaptureFixture[str]
) -> None:
    module = _module()

    assert module.main(list(argv)) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "smoke output verification failed: invalid\n"


def _load_mutated_module(tmp_path: Path, source: str, name: str) -> ModuleType:
    path = tmp_path / f"{name}.py"
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def _assert_extra_output_rejected(module: Any, root: Path) -> None:
    sources = _sources("a")
    _write_valid_outputs(root, "a", sources)
    write_executed_notebook(root / "extra" / "nested.ipynb")
    try:
        module.verify_smoke_outputs("a", root, lambda selected: sources)
    except module.SmokeOutputError as error:
        assert str(error).endswith(": inventory")
    else:
        raise AssertionError("extra recursive notebook was accepted")


def _assert_zero_code_cells_rejected(module: Any, root: Path) -> None:
    sources = _sources("c")
    _write_valid_outputs(root, "c", sources)
    target = _output_path(root, "c", sources[0])
    target.write_text(
        json.dumps({"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}),
        encoding="utf-8",
    )
    try:
        module.verify_smoke_outputs("c", root, lambda selected: sources)
    except module.SmokeOutputError as error:
        assert str(error).endswith(": invalid")
    else:
        raise AssertionError("notebook with zero code cells was accepted")


def _assert_malformed_shape_rejected(module: Any, root: Path, case: str) -> None:
    sources = _sources("b")
    _write_valid_outputs(root, "b", sources)
    target = _output_path(root, "b", sources[0])
    document = json.loads(target.read_text(encoding="utf-8"))
    expected = "invalid"
    if case == "cell":
        document["cells"][0] = "not-a-cell"
    elif case == "outputs-list":
        document["cells"][0]["outputs"] = None
    elif case == "output-mapping":
        document["cells"][0]["outputs"] = ["not-an-output"]
    else:
        document["cells"][0]["execution_count"] = True
        expected = "unexecuted"
    target.write_text(json.dumps(document), encoding="utf-8")
    try:
        module.verify_smoke_outputs("b", root, lambda selected: sources)
    except module.SmokeOutputError as error:
        assert str(error).endswith(f": {expected}")
    except BaseException as error:
        raise AssertionError("malformed notebook leaked an unstable exception") from error
    else:
        raise AssertionError("malformed notebook was accepted")


def test_recursive_output_set_gate_kills_its_removal_mutation(tmp_path: Path) -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    original = (
        "    if actual_outputs - expected_outputs:\n"
        "        raise SmokeOutputError(tier, \"inventory\")\n"
    )
    mutated = source.replace(original, "", 1)
    assert mutated != source
    module = _load_mutated_module(tmp_path, mutated, "mutated_recursive_oracle")

    with pytest.raises(AssertionError):
        _assert_extra_output_rejected(module, tmp_path / "outputs")


def test_nonempty_code_cell_gate_kills_its_removal_mutation(tmp_path: Path) -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    original = (
        "    if not code_cells:\n"
        "        raise SmokeOutputError(tier, \"invalid\")\n"
    )
    mutated = source.replace(original, "", 1)
    assert mutated != source
    module = _load_mutated_module(tmp_path, mutated, "mutated_code_cell_oracle")

    with pytest.raises(AssertionError):
        _assert_zero_code_cells_rejected(module, tmp_path / "outputs")


@pytest.mark.parametrize(
    ("case", "original"),
    (
        (
            "cell",
            "    if any(not isinstance(cell, Mapping) for cell in cells):\n"
            "        raise SmokeOutputError(tier, \"invalid\")\n",
        ),
        (
            "outputs-list",
            "    if any(not isinstance(cell.get(\"outputs\"), list) for cell in code_cells):\n"
            "        raise SmokeOutputError(tier, \"invalid\")\n",
        ),
        (
            "output-mapping",
            "    if any(\n"
            "        not isinstance(output_item, Mapping)\n"
            "        for cell in code_cells\n"
            "        for output_item in cell[\"outputs\"]\n"
            "    ):\n"
            "        raise SmokeOutputError(tier, \"invalid\")\n",
        ),
        (
            "execution-count",
            "    if any(\n"
            "        not isinstance(cell.get(\"execution_count\"), int)\n"
            "        or isinstance(cell.get(\"execution_count\"), bool)\n"
            "        for cell in code_cells\n"
            "    ):\n"
            "        raise SmokeOutputError(tier, \"unexecuted\")\n",
        ),
    ),
)
def test_notebook_shape_gates_kill_their_removal_mutations(
    tmp_path: Path, case: str, original: str
) -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    mutated = source.replace(original, "", 1)
    assert mutated != source
    module = _load_mutated_module(tmp_path, mutated, f"mutated_{case.replace('-', '_')}")

    with pytest.raises(AssertionError):
        _assert_malformed_shape_rejected(module, tmp_path / "outputs", case)
