"""NNx-surface contract for regression targets in tabular pipelines."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from nnx import (
    Devices,
    Losses,
    NNModel,
    NNModelParams,
    NNOptimParams,
    NNParams,
    NNTabularDataset,
    NNTrainParams,
    Nets,
    Optims,
)


_FEATURE_COLS = ["age", "bmi"]
_TARGET_COL = "progression"


def _tiny_regression_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": np.arange(8, dtype=np.float32),
            "bmi": np.arange(1, 9, dtype=np.float32),
            "progression": np.arange(8, dtype=np.float32),
        }
    )


def test_tabular_regression_targets_are_float_column_vectors_and_train():
    frame = _tiny_regression_frame()
    regression = NNTabularDataset(
        df=frame,
        feature_cols=_FEATURE_COLS,
        target_col=_TARGET_COL,
        batch_sizes=(4, 4, 4),
        val_proportion=0.25,
        test_proportion=0.25,
        target_dtype=torch.float32,
        seed=0,
    )
    _, target = next(iter(regression.train_loader))
    assert regression.output_dim == 1
    assert target.dtype == torch.float32
    assert target.shape[1:] == (1,)

    model = NNModel(
        params=NNModelParams(
            net=Nets.FEED_FWD,
            device=Devices.CPU,
            loss=Losses.MEAN_SQUARED_ERROR,
        ),
        net_params=NNParams(
            dropout_prob=0.0,
            hidden_dims=[],
            input_dim=len(_FEATURE_COLS),
            output_dim=regression.output_dim,
        ),
    )
    run = model.train(
        params=(
            NNTrainParams(
                n_epochs=1,
                optim=NNOptimParams(
                    name=Optims.ADAM,
                    max_lr=1e-2,
                    weight_decay=0.0,
                    momentum=(0.9, 0.999),
                ),
                seed=0,
            )
            .with_train_loader(value=regression.train_loader)
            .with_val_loader(value=regression.val_loader)
        )
    )
    assert run is not None
    X = frame[_FEATURE_COLS].to_numpy(dtype=np.float32, copy=True)
    expected = torch.from_numpy(frame[[_TARGET_COL]].to_numpy(dtype=np.float32, copy=True))
    prediction, _ = model.predict(X=X)
    loss = torch.nn.functional.mse_loss(torch.from_numpy(prediction), expected)
    assert torch.isfinite(loss)
    assert prediction.shape == (len(frame), regression.output_dim)


def test_tabular_default_targets_remain_contiguous_long_classes():
    frame = _tiny_regression_frame().assign(progression=[0, 1, 2, 0, 1, 2, 0, 1])
    classification = NNTabularDataset(
        df=frame,
        feature_cols=_FEATURE_COLS,
        target_col=_TARGET_COL,
        batch_sizes=(4, 4, 4),
        val_proportion=0.25,
        test_proportion=0.25,
        seed=0,
    )
    targets = torch.cat(
        [
            batch_target
            for loader in (
                classification.train_loader,
                classification.val_loader,
                classification.test_loader,
            )
            for _, batch_target in loader
        ]
    )
    assert classification.output_dim == 3
    assert targets.dtype == torch.long
    assert torch.equal(torch.unique(targets), torch.arange(classification.output_dim))
