"""NNx-surface contracts for the Tier-B quantization notebook.

The canonical Torch 2.11 stack makes torchao mandatory for complete PTQ, QAT,
checkpoint reconstruction, and converted-inference execution.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import os
import pickle
import re
import warnings
from collections.abc import Callable, Sequence
from importlib.metadata import PackagePath
from pathlib import Path

import numpy as np
import pytest
import torch
from packaging.version import InvalidVersion, Version

import nnx
from nnx import (
    Checkpoints,
    Devices,
    Losses,
    NNCheckpoint,
    NNModel,
    NNModelParams,
    NNParams,
    NNTrainParams,
    Nets,
)


QAT_WARNING_DEBT_KEY = ("2.11.0", "0.18.0", "0.2.0", "8da4w")
QAT_WARNING_MESSAGE = (
    "Deprecation: TorchAODType is deprecated, please use the torch.intN dtype instead "
    "(e.g. TorchAODType.INT4 -> torch.int4)"
)
QAT_WARNING_RECORD_PATH = "torchao/quantization/quant_primitives.py"
REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = (
    REPO_ROOT / "notebooks/quantization-mnist-ffnn-pytorch/notebook.ipynb"
)
QUANTIZATION_MARKER_PREFIX = "ISSUE66_QUANTIZATION_CONTRACT="
DistributionProvider = Callable[[str], importlib.metadata.Distribution]


class _FakeDistribution:
    def __init__(self, root: Path, version: str, *, owns_warning: bool = False) -> None:
        self.root = root
        self.version = version
        self.files: list[PackagePath] = []
        if owns_warning:
            warning_file = root / QAT_WARNING_RECORD_PATH
            warning_file.parent.mkdir(parents=True, exist_ok=True)
            warning_file.touch()
            entry = PackagePath(QAT_WARNING_RECORD_PATH)
            entry.dist = self
            self.files.append(entry)

    def locate_file(self, path: PackagePath) -> Path:
        return self.root / path


def _qat_distributions(tmp_path: Path) -> dict[str, _FakeDistribution]:
    return {
        "torch": _FakeDistribution(tmp_path / "torch", "2.11.0"),
        "torchao": _FakeDistribution(tmp_path / "torchao", "0.18.0", owns_warning=True),
        "thekaveh-nnx": _FakeDistribution(tmp_path / "nnx", "0.2.0"),
    }


def _warning_record(
    origin: Path,
    *,
    category: type[Warning] = UserWarning,
    message: str = QAT_WARNING_MESSAGE,
) -> warnings.WarningMessage:
    return warnings.WarningMessage(category(message), category, str(origin), 96)


class _UserWarningSubclass(UserWarning):
    pass


def _exact_qat_warning(
    tmp_path: Path,
) -> tuple[dict[str, _FakeDistribution], warnings.WarningMessage]:
    distributions = _qat_distributions(tmp_path)
    torchao_distribution = distributions["torchao"]
    origin = torchao_distribution.locate_file(torchao_distribution.files[0])
    return distributions, _warning_record(origin)


def test_qat_warning_debt_validator_accepts_exact_record(tmp_path: Path) -> None:
    distributions, record = _exact_qat_warning(tmp_path)
    evidence = _assert_qat_warning_debt(
        (record,),
        qat_config="8da4w",
        distribution=distributions.__getitem__,
    )
    assert evidence["debt_key"] == {
        "torch": "2.11.0",
        "torchao": "0.18.0",
        "thekaveh-nnx": "0.2.0",
        "qat_config": "8da4w",
    }
    assert evidence["count"] == 1
    assert evidence["category"] == "builtins.UserWarning"
    assert evidence["message"] == QAT_WARNING_MESSAGE
    assert evidence["origin_inventory_path"] == QAT_WARNING_RECORD_PATH
    assert re.fullmatch(r"[0-9a-f]{64}", evidence["origin_sha256"])
    assert evidence["global_warning_action"] == "error"
    assert evidence["local_capture_action"] == "always"


@pytest.mark.parametrize(
    ("distribution_name", "version"),
    (
        ("torch", "2.11.0+cpu"),
        ("torchao", "0.18.0+linux"),
        ("thekaveh-nnx", "0.2.0+linux"),
    ),
)
def test_qat_warning_debt_key_normalizes_pep440_local_versions(
    tmp_path: Path,
    distribution_name: str,
    version: str,
) -> None:
    distributions, record = _exact_qat_warning(tmp_path)
    distributions[distribution_name].version = version

    evidence = _assert_qat_warning_debt(
        (record,),
        qat_config="8da4w",
        distribution=distributions.__getitem__,
    )

    assert evidence["debt_key"] == {
        "torch": "2.11.0",
        "torchao": "0.18.0",
        "thekaveh-nnx": "0.2.0",
        "qat_config": "8da4w",
    }


@pytest.mark.parametrize("distribution_name", ("torch", "torchao", "thekaveh-nnx"))
def test_qat_warning_debt_key_rejects_malformed_distribution_versions(
    tmp_path: Path,
    distribution_name: str,
) -> None:
    distributions, record = _exact_qat_warning(tmp_path)
    distributions[distribution_name].version = "not a version"

    with pytest.raises(AssertionError, match="qat warning debt validation failed"):
        _assert_qat_warning_debt(
            (record,),
            qat_config="8da4w",
            distribution=distributions.__getitem__,
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "zero",
        "two",
        "mixed",
        "runtime",
        "subclass",
        "prefix",
        "punctuation",
        "dtype",
        "same-basename",
        "matching-suffix",
    ),
)
def test_qat_warning_debt_validator_rejects_record_mutations(
    tmp_path: Path, mutation: str
) -> None:
    distributions, exact = _exact_qat_warning(tmp_path)
    records = (exact,)
    if mutation == "zero":
        records = ()
    elif mutation == "two":
        records = (exact, exact)
    elif mutation == "mixed":
        records = (exact, _warning_record(Path(exact.filename), message="extra"))
    elif mutation == "runtime":
        records = (_warning_record(Path(exact.filename), category=RuntimeWarning),)
    elif mutation == "subclass":
        records = (_warning_record(Path(exact.filename), category=_UserWarningSubclass),)
    elif mutation == "prefix":
        records = (_warning_record(Path(exact.filename), message=QAT_WARNING_MESSAGE[:-1]),)
    elif mutation == "punctuation":
        records = (_warning_record(Path(exact.filename), message=QAT_WARNING_MESSAGE + "."),)
    elif mutation == "dtype":
        records = (
            _warning_record(
                Path(exact.filename),
                message=QAT_WARNING_MESSAGE.replace("INT4", "INT8"),
            ),
        )
    else:
        outsider = (
            tmp_path / "outsider" / "quant_primitives.py"
            if mutation == "same-basename"
            else tmp_path / "outsider" / QAT_WARNING_RECORD_PATH
        )
        outsider.parent.mkdir(parents=True, exist_ok=True)
        outsider.touch()
        records = (_warning_record(outsider),)
    error = (
        "qat warning debt retirement required"
        if mutation == "zero"
        else "qat warning debt validation failed"
    )
    with pytest.raises(AssertionError, match=error):
        _assert_qat_warning_debt(
            records,
            qat_config="8da4w",
            distribution=distributions.__getitem__,
        )


@pytest.mark.parametrize(
    ("distribution_name", "version", "qat_config"),
    (
        ("torch", "2.11.1", "8da4w"),
        ("torchao", "0.18.1", "8da4w"),
        ("thekaveh-nnx", "0.2.1", "8da4w"),
        (None, None, "8da4w-next"),
    ),
)
def test_qat_warning_debt_validator_requires_immutable_key(
    tmp_path: Path,
    distribution_name: str | None,
    version: str | None,
    qat_config: str,
) -> None:
    distributions, record = _exact_qat_warning(tmp_path)
    if distribution_name is not None:
        assert version is not None
        distributions[distribution_name].version = version
    with pytest.raises(AssertionError, match="qat warning debt retirement required"):
        _assert_qat_warning_debt(
            (record,),
            qat_config=qat_config,
            distribution=distributions.__getitem__,
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "none",
        "missing",
        "duplicate",
        "foreign-owner",
        "missing-file",
        "directory-at-exact-path",
        "locate-error",
    ),
)
def test_qat_warning_debt_validator_requires_exact_record_ownership(
    tmp_path: Path, mutation: str
) -> None:
    distributions, record = _exact_qat_warning(tmp_path)
    torchao_distribution = distributions["torchao"]
    entry = torchao_distribution.files[0]
    if mutation == "none":
        torchao_distribution.files = None  # type: ignore[assignment]
    elif mutation == "missing":
        torchao_distribution.files = []
    elif mutation == "duplicate":
        duplicate = PackagePath(QAT_WARNING_RECORD_PATH)
        duplicate.dist = torchao_distribution
        torchao_distribution.files.append(duplicate)
    elif mutation == "foreign-owner":
        entry.dist = _FakeDistribution(tmp_path / "foreign", "0.18.0")
    elif mutation == "missing-file":
        Path(record.filename).unlink()
    elif mutation == "directory-at-exact-path":
        Path(record.filename).unlink()
        Path(record.filename).mkdir()
    else:

        def fail_locate(path: PackagePath) -> Path:
            raise OSError("unlocatable")

        torchao_distribution.locate_file = fail_locate  # type: ignore[method-assign]
    with pytest.raises(AssertionError, match="qat warning debt validation failed"):
        _assert_qat_warning_debt(
            (record,),
            qat_config="8da4w",
            distribution=distributions.__getitem__,
        )


def _torchao_qat_warning_origin(
    distribution: importlib.metadata.Distribution,
) -> Path:
    files = distribution.files
    if files is None:
        raise AssertionError("qat warning debt validation failed")
    matches = tuple(
        path for path in files if path.as_posix() == QAT_WARNING_RECORD_PATH
    )
    if len(matches) != 1 or getattr(matches[0], "dist", None) is not distribution:
        raise AssertionError("qat warning debt validation failed")
    try:
        origin = distribution.locate_file(matches[0]).resolve(strict=True)
        owned_origin = matches[0].locate().resolve(strict=True)
    except (OSError, RuntimeError, TypeError, ValueError):
        raise AssertionError("qat warning debt validation failed") from None
    if origin != owned_origin or not origin.is_file():
        raise AssertionError("qat warning debt validation failed")
    return origin


def _public_distribution_version(
    distribution: importlib.metadata.Distribution,
) -> str:
    try:
        return Version(distribution.version).public
    except (InvalidVersion, TypeError):
        raise AssertionError("qat warning debt validation failed") from None


def _assert_qat_warning_debt(
    caught: Sequence[warnings.WarningMessage],
    *,
    qat_config: str,
    distribution: DistributionProvider = importlib.metadata.distribution,
) -> dict[str, object]:
    selected = {
        name: distribution(name) for name in ("torch", "torchao", "thekaveh-nnx")
    }
    key = (
        _public_distribution_version(selected["torch"]),
        _public_distribution_version(selected["torchao"]),
        _public_distribution_version(selected["thekaveh-nnx"]),
        qat_config,
    )
    if key != QAT_WARNING_DEBT_KEY or not caught:
        raise AssertionError("qat warning debt retirement required")
    if len(caught) != 1:
        raise AssertionError("qat warning debt validation failed")
    record = caught[0]
    expected_origin = _torchao_qat_warning_origin(selected["torchao"])
    try:
        actual_origin = Path(record.filename).resolve(strict=True)
    except (OSError, RuntimeError):
        raise AssertionError("qat warning debt validation failed") from None
    if (
        record.category is not UserWarning
        or str(record.message) != QAT_WARNING_MESSAGE
        or actual_origin != expected_origin
    ):
        raise AssertionError("qat warning debt validation failed")
    try:
        origin_sha256 = hashlib.sha256(expected_origin.read_bytes()).hexdigest()
    except OSError:
        raise AssertionError("qat warning debt validation failed") from None
    return {
        "debt_key": {
            "torch": key[0],
            "torchao": key[1],
            "thekaveh-nnx": key[2],
            "qat_config": key[3],
        },
        "count": 1,
        "category": "builtins.UserWarning",
        "message": QAT_WARNING_MESSAGE,
        "origin_inventory_path": QAT_WARNING_RECORD_PATH,
        "origin_sha256": origin_sha256,
        "global_warning_action": "error",
        "local_capture_action": "always",
    }


def test_quantization_facade_signatures_match_notebook_contract():
    assert inspect.signature(nnx.quantize_int8) == inspect.Signature(
        parameters=[
            inspect.Parameter(
                "model",
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation="NNModel",
            )
        ],
        return_annotation="NNModel",
    )

    qat_sig = inspect.signature(nnx.qat_train_step_factory)
    assert list(qat_sig.parameters) == ["base_step", "qat_config"]
    assert qat_sig.parameters["base_step"].default is None
    assert qat_sig.parameters["qat_config"].default == "8da4w"

    cb_sig = inspect.signature(nnx.QATLifecycleCallback)
    assert list(cb_sig.parameters) == ["qat_config", "groupsize"]
    assert cb_sig.parameters["qat_config"].default == "8da4w"
    assert cb_sig.parameters["groupsize"].default == 32


def test_quantize_int8_predicts_with_same_output_shape(tiny_image_batch):
    import torchao

    model = NNModel(
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
        net_params=NNParams(
            dropout_prob=0.0,
            hidden_dims=[32],
            input_dim=28 * 28,
            output_dim=10,
        ),
    )

    quantized = nnx.quantize_int8(model)

    logits, classes = quantized.predict(X=tiny_image_batch.X)
    quantized_weights = {
        type(module.weight).__name__
        for module in quantized.net.modules()
        if isinstance(module, torch.nn.Linear)
    }
    assert torchao is not None
    assert quantized is not model
    assert quantized_weights == {"Int8Tensor"}
    assert len(pickle.dumps(quantized.net.state_dict())) < len(
        pickle.dumps(model.net.state_dict())
    )
    assert logits.shape == (4, 10)
    assert classes.shape == (4,)
    assert np.issubdtype(classes.dtype, np.integer)


def test_qat_prepare_train_convert_and_inference(tiny_image_batch):
    import torchao

    model = NNModel(
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
        net_params=NNParams(
            dropout_prob=0.0,
            hidden_dims=[32],
            input_dim=28 * 28,
            output_dim=10,
        ),
    )
    qat_config = "8da4w"
    callback = nnx.QATLifecycleCallback(qat_config=qat_config)
    train_step = nnx.qat_train_step_factory(qat_config=qat_config)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run = model.train(
            params=(
                NNTrainParams(n_epochs=1)
                .with_train_loader(value=tiny_image_batch.train_loader)
                .with_val_loader(value=tiny_image_batch.val_loader)
            ),
            callbacks=[callback],
            train_step_fn=train_step,
        )
    qat_warning_evidence = _assert_qat_warning_debt(caught, qat_config=qat_config)
    observation_path_text = os.environ.get("ISSUE62_QAT_DEBT_OBSERVATION")
    if observation_path_text is not None:
        final_root = Path(os.environ["FINAL_ROOT"]).resolve(strict=True)
        observation_path = Path(observation_path_text).resolve()
        assert observation_path == final_root / "qat-warning-debt-observation.json"
        final_sha = os.environ["ISSUE62_FINAL_SHA"]
        assert re.fullmatch(r"[0-9a-f]{40}", final_sha)
        observation = {
            "schema_version": 1,
            "final_sha": final_sha,
            "test_nodeid": (
                "tests/nnx_surface/test_quantization_mnist_ffnn_pytorch.py::"
                "test_qat_prepare_train_convert_and_inference"
            ),
            **qat_warning_evidence,
        }
        observation_path.write_text(
            json.dumps(observation, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    logits, classes = model.predict(X=tiny_image_batch.X)
    checkpoint = NNCheckpoint.load(run=run.id, type=Checkpoints.LAST)
    assert checkpoint is not None
    reconstructed = NNModel.from_checkpoint(checkpoint)
    reconstructed_edp = reconstructed.evaluate(tiny_image_batch.val_loader)
    final_val_edp = run.idps[-1].val_edp
    reconstructed_state = reconstructed.net.state_dict()

    assert torchao is not None
    assert run is not None
    assert callback.is_prepared
    assert callback.is_converted
    assert logits.shape == (4, 10)
    assert classes.shape == (4,)
    assert np.issubdtype(classes.dtype, np.integer)
    assert tuple(reconstructed_state) == tuple(checkpoint.net_state)
    assert all(
        torch.equal(reconstructed_state[key], value)
        for key, value in checkpoint.net_state.items()
    )
    assert checkpoint.idp == run.idps[-1]
    assert np.isfinite(reconstructed_edp.loss)
    assert np.isfinite(reconstructed_edp.accuracy)
    assert final_val_edp is not None


def _notebook_source() -> str:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", ()))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


def test_notebook_declares_bounded_deterministic_smoke_contract() -> None:
    source = _notebook_source()

    assert "SMOKE_TEST_EPOCHS = 1" in source
    assert "N_EPOCHS = SMOKE_TEST_EPOCHS if SMOKE_TEST else 3" in source
    assert "nnx.set_seed(0)" in source
    assert '"seed": 0' in source
    assert '"smoke_test": bool(SMOKE_TEST)' in source


def test_notebook_proves_qat_checkpoint_reconstruction_and_conversion() -> None:
    source = _notebook_source()

    assert 'Path("runs") / qat_run.id / "checkpoints" / "last.pt"' in source
    assert "torch.load(qat_checkpoint_path, weights_only=False)" in source
    assert "isinstance(qat_checkpoint, NNCheckpoint)" in source
    assert "NNModel.from_checkpoint(qat_checkpoint)" in source
    assert "qat_reloaded_model.params == qat_checkpoint.model_params" in source
    assert "qat_reloaded_model.net_params == qat_checkpoint.net_params" in source
    assert "checkpoint_state_parity" in source
    assert "checkpoint_metadata_parity" in source
    assert "checkpoint_evaluation_finite" in source
    assert "qat_cb.is_prepared and qat_cb.is_converted" in source
    assert '"Int8DynActInt4WeightLinear" in classes_after_qat' in source
    assert QUANTIZATION_MARKER_PREFIX in source


def test_notebook_proves_ptq_conversion_independently() -> None:
    source = _notebook_source()

    assert 'ptq_weight_types == {"Int8Tensor"}' in source
    assert "ptq_state_size < fp32_state_size" in source
    assert "ptq_quantized_weights" in source
