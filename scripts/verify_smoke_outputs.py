"""Verify that a smoke tier produced exactly its complete executed notebook set."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, cast


Tier = Literal["a", "b", "c"]
EXPECTED_COUNTS: Mapping[Tier, int] = {"a": 18, "b": 7, "c": 4}
QUANTIZATION_SOURCE = "notebooks/quantization-mnist-ffnn-pytorch/notebook.ipynb"
QUANTIZATION_OUTPUT_NAME = "quantization-mnist-ffnn-pytorch.ipynb"
QUANTIZATION_MARKER_PREFIX = "ISSUE66_QUANTIZATION_CONTRACT="
QUANTIZATION_SMOKE_CONTRACT: Mapping[str, object] = {
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


class InventoryLoader(Protocol):
    def __call__(self, tier: Tier) -> Sequence[str]: ...


@dataclass(frozen=True)
class NotebookArtifact:
    source: str
    output: Path
    code_cells: int


class SmokeOutputError(RuntimeError):
    """A smoke output set does not satisfy its fail-closed tier contract."""

    def __init__(self, tier: Tier, category: str) -> None:
        super().__init__(f"smoke output verification failed: {tier}: {category}")


def _validate_quantization_contract(
    tier: Tier, source: str, code_cells: Sequence[Mapping[str, object]]
) -> None:
    if tier != "b" or source != QUANTIZATION_SOURCE:
        return
    markers: list[str] = []
    for cell in code_cells:
        outputs = cast(Sequence[Mapping[str, object]], cell["outputs"])
        for output_item in outputs:
            if output_item.get("output_type") != "stream":
                continue
            text = output_item.get("text")
            chunks = (text,) if isinstance(text, str) else text
            if not isinstance(chunks, Sequence) or isinstance(chunks, (str, bytes)):
                continue
            for chunk in chunks:
                if not isinstance(chunk, str):
                    continue
                markers.extend(
                    line.removeprefix(QUANTIZATION_MARKER_PREFIX)
                    for line in chunk.splitlines()
                    if line.startswith(QUANTIZATION_MARKER_PREFIX)
                )
    if len(markers) != 1:
        raise SmokeOutputError(tier, "semantic")
    try:
        payload = json.loads(markers[0])
    except (json.JSONDecodeError, TypeError, ValueError):
        raise SmokeOutputError(tier, "semantic") from None
    if payload != QUANTIZATION_SMOKE_CONTRACT:
        raise SmokeOutputError(tier, "semantic")


def _validate_notebook(tier: Tier, source: str, output: Path) -> NotebookArtifact:
    if not output.is_file():
        raise SmokeOutputError(tier, "missing")
    if output.stat().st_size == 0:
        raise SmokeOutputError(tier, "empty")
    try:
        document = json.loads(output.read_text(encoding="utf-8"))
        cells = document["cells"]
        if document["nbformat"] != 4 or not isinstance(cells, list):
            raise ValueError
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        raise SmokeOutputError(tier, "invalid") from None
    if any(not isinstance(cell, Mapping) for cell in cells):
        raise SmokeOutputError(tier, "invalid")
    code_cells = tuple(cell for cell in cells if cell.get("cell_type") == "code")
    if not code_cells:
        raise SmokeOutputError(tier, "invalid")
    if any(not isinstance(cell.get("outputs"), list) for cell in code_cells):
        raise SmokeOutputError(tier, "invalid")
    if any(
        not isinstance(output_item, Mapping)
        for cell in code_cells
        for output_item in cell["outputs"]
    ):
        raise SmokeOutputError(tier, "invalid")
    if any(
        not isinstance(cell.get("execution_count"), int)
        or isinstance(cell.get("execution_count"), bool)
        for cell in code_cells
    ):
        raise SmokeOutputError(tier, "unexecuted")
    if any(
        output_item.get("output_type") == "error"
        for cell in code_cells
        for output_item in cell["outputs"]
    ):
        raise SmokeOutputError(tier, "error")
    _validate_quantization_contract(tier, source, code_cells)
    return NotebookArtifact(source, output, len(code_cells))


def load_make_inventory(tier: Tier) -> Sequence[str]:
    result = subprocess.run(
        ("make", "--no-print-directory", "-s", f"print-tier-{tier}"),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SmokeOutputError(tier, "inventory")
    return tuple(line.strip() for line in result.stdout.splitlines() if line.strip())


def _mapped_output(root: Path, tier: Tier, source: str) -> Path:
    if tier == "a":
        return root / source
    if tier == "b" and source == QUANTIZATION_SOURCE:
        return root / QUANTIZATION_OUTPUT_NAME
    return root / Path(source).name


def verify_smoke_outputs(
    tier: Tier,
    root: Path,
    inventory_loader: InventoryLoader = load_make_inventory,
) -> tuple[NotebookArtifact, ...]:
    sources = tuple(inventory_loader(tier))
    if len(sources) != EXPECTED_COUNTS[tier] or len(set(sources)) != len(sources):
        raise SmokeOutputError(tier, "inventory")
    outputs = tuple(_mapped_output(root, tier, source) for source in sources)
    if len(set(outputs)) != len(outputs):
        raise SmokeOutputError(tier, "inventory")
    expected_outputs = {output.resolve() for output in outputs}
    actual_outputs = {output.resolve() for output in root.rglob("*.ipynb")}
    if actual_outputs - expected_outputs:
        raise SmokeOutputError(tier, "inventory")
    return tuple(
        _validate_notebook(tier, source, output)
        for source, output in zip(sources, outputs, strict=True)
    )


def _parse_args(args: Sequence[str]) -> tuple[Tier, Path] | None:
    if len(args) != 4:
        return None
    values: dict[str, str] = {}
    for index in (0, 2):
        option, value = args[index : index + 2]
        if option not in {"--tier", "--root"} or option in values:
            return None
        values[option] = value
    tier_value = values.get("--tier")
    root_value = values.get("--root")
    if tier_value not in EXPECTED_COUNTS or root_value is None:
        return None
    return cast(Tier, tier_value), Path(root_value)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    parsed = _parse_args(args)
    if parsed is None:
        print("smoke output verification failed: invalid", file=sys.stderr)
        return 1
    tier, root = parsed
    try:
        if not root.is_absolute():
            raise SmokeOutputError(tier, "inventory")
        artifacts = verify_smoke_outputs(tier, root)
    except SmokeOutputError as error:
        print(error, file=sys.stderr)
        return 1
    except BaseException:
        print(f"smoke output verification failed: {tier}: invalid", file=sys.stderr)
        return 1
    print(
        f"smoke output verification ok: {tier}: notebooks={len(artifacts)} "
        f"code_cells={sum(artifact.code_cells for artifact in artifacts)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
