"""Released NNx 0.2.2 run-identity contracts for known notebook comparisons."""

from __future__ import annotations

from nnx import (
    Activations,
    Devices,
    Losses,
    Nets,
    NNModelParams,
    NNOptimParams,
    NNParams,
    NNRun,
    NNTrainParams,
    Optims,
)


def _peft_run_id(salt: str | None = None) -> str:
    return NNRun(
        net=NNParams(
            input_dim=784,
            output_dim=10,
            hidden_dims=[128, 64],
            dropout_prob=0.0,
            activation=Activations.RELU,
        ),
        model=NNModelParams(
            net=Nets.FEED_FWD,
            device=Devices.CPU,
            loss=Losses.CROSS_ENTROPY,
        ),
        train=NNTrainParams(
            n_epochs=1,
            optim=NNOptimParams(
                name=Optims.ADAM,
                max_lr=0.005,
                momentum=(0.9, 0.999),
                weight_decay=0.0,
            ),
        ),
        salt=salt,
    ).id


def _reddit_run_id(net: Nets, *, salt: str | None = None) -> str:
    return NNRun(
        net=NNParams(
            n_heads=4 if net == Nets.GRAPH_ATT else None,
            hidden_dims=[128],
            dropout_prob=0.25,
            input_dim=602,
            output_dim=41,
        ),
        model=NNModelParams(
            net=net,
            device=Devices.CPU,
            loss=Losses.CROSS_ENTROPY,
        ),
        train=NNTrainParams(
            n_epochs=1,
            optim=NNOptimParams(
                name=Optims.ADAM,
                max_lr=0.01,
                weight_decay=5e-4,
                momentum=(0.9, 0.999),
            ),
        ),
        salt=salt,
    ).id


def test_peft_adapter_salts_separate_otherwise_identical_nnx_runs():
    unsalted = [_peft_run_id() for _ in range(3)]
    assert unsalted == ["47e8c3747f83992e76ee271e59c076ab"] * 3

    identities = {
        "full fine-tune": _peft_run_id(),
        "LoRA": _peft_run_id("lora-adaptation"),
        "DoRA": _peft_run_id("dora-adaptation"),
    }
    assert identities == {
        "full fine-tune": "47e8c3747f83992e76ee271e59c076ab",
        "LoRA": "6eb5599e5429e0e7507f840a11db8f30",
        "DoRA": "2ae29ff7f61f3a5d534f2cffde3c6573",
    }
    assert len(set(identities.values())) == 3


def test_reddit_phase_salts_separate_shared_root_smoke_runs():
    expected = {
        Nets.GRAPH_ATT: (
            "7d7e1ec6856a7791524fef54b592927a",
            "75a54c3f60ead4b80042847119655795",
            "d940799aa340abf2e22111cc5620349a",
        ),
        Nets.GRAPH_SAGE: (
            "606d11108b786970cde83533049aface",
            "7e20e3315b6fd822474947179867627d",
            "9a92739036954c853bcdd4f751287ddd",
        ),
        Nets.GRAPH_CONV: (
            "600ee84827ac3a051b27becc7243a556",
            "c4597b45da7b56d5538827f284fc62ee",
            "4a76882dc086bbf0bf0ac6a0749cbbf6",
        ),
        Nets.FEED_FWD: (
            "17254f216b97b2482a4ec50f838e1964",
            "f41ca43f54faff78e9029820d60c303b",
            "8d4d9c640347df9a98f2c2c2d2e98f2d",
        ),
    }
    for net, expected_ids in expected.items():
        unsalted_phase2_notebook1 = _reddit_run_id(net)
        unsalted_phase2_notebook2 = _reddit_run_id(net)
        assert unsalted_phase2_notebook1 == unsalted_phase2_notebook2 == expected_ids[0]

        actual = (
            _reddit_run_id(net),
            _reddit_run_id(net, salt="phase2-model-selection-notebook1"),
            _reddit_run_id(net, salt="phase2-model-selection-notebook2"),
        )
        assert actual == expected_ids
        assert len(set(actual)) == 3
