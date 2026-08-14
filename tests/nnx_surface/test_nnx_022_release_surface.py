"""Consumer-facing public facade additions released in NNx 0.2.2."""
from __future__ import annotations

import torch

from nnx import ConvNN, FeedFwdMoENN, NNConvParams, NNMoEParams, Nets


def test_conv_facade_constructs_and_forwards_smallest_image_batch():
    assert Nets.CONV is not None
    model = ConvNN(
        NNConvParams(
            dropout_prob=0.0,
            input_dim=16,
            output_dim=3,
            conv_channels=[2],
            kernel_size=3,
            pool_size=2,
        )
    )

    logits = model(torch.zeros((2, 1, 4, 4)))

    assert logits.shape == (2, 3)


def test_feed_forward_moe_facade_constructs_and_forwards_smallest_batch():
    assert Nets.FEED_FWD_MOE is not None
    model = FeedFwdMoENN(
        NNMoEParams(
            dropout_prob=0.0,
            input_dim=4,
            output_dim=3,
            hidden_dims=[4],
            num_experts=2,
            top_k=1,
        )
    )

    logits = model(torch.zeros((2, 4)))

    assert logits.shape == (2, 3)
