"""NNx-surface contract test for notebooks/node_classification-reddit-gnn-pyg/*.

Asserts the canonical call shape used by phase2-notebook4 and (after the
Phase C rewrites of this plan) by every other phase2/phase3 notebook.

`GraphSageNN`, `GraphConvNN`, and `GraphAttNN` all share a `GraphNNBase`
and accept an `NNParams` (with `n_heads` only required for the GAT
branch). This file pins that contract.

The tests don't download Reddit2 (which is ~1.5GB). They build a tiny
synthetic PyG `Data` via the `tiny_graph_data` fixture and exercise the
GNN forward pass via a 1-batch `NeighborLoader`.

The canonical stack requires both `pyg-lib` and `torch-sparse`, and every
model path executes a sampled batch rather than accepting import-only evidence.
The `NNParams.state()` round-trip remains a pure-Python contract.
"""
from __future__ import annotations

import pytest

from nnx import (
    Devices,
    Losses,
    NNGraphDataset,
    NNModel,
    NNModelParams,
    NNParams,
    NNTrainParams,
    Nets,
    set_seed,
)


@pytest.fixture
def gnn_loaders(tiny_graph_data):
    """Build train/val NeighborLoaders over the tiny PyG graph."""
    from torch_geometric.loader import NeighborLoader

    data = tiny_graph_data.data
    train_loader = NeighborLoader(
        data,
        num_neighbors=[2, 2],
        batch_size=int(data.train_mask.sum()),
        input_nodes=data.train_mask,
        shuffle=False,
        num_workers=0,
    )
    val_loader = NeighborLoader(
        data,
        num_neighbors=[2, 2],
        batch_size=int(data.val_mask.sum()),
        input_nodes=data.val_mask,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, val_loader


def _assert_sampled_batch_is_executable(loader) -> None:
    batch = next(iter(loader))
    assert int(batch.batch_size) > 0
    assert int(batch.edge_index.numel()) > 0


def test_canonical_sampler_backends_and_batch_are_executable(tiny_graph_data):
    import pyg_lib
    import torch_sparse
    from torch_geometric.loader import NeighborLoader

    batch = next(
        iter(
            NeighborLoader(
                tiny_graph_data.data,
                num_neighbors=[2, 2],
                batch_size=2,
                input_nodes=tiny_graph_data.data.train_mask,
                shuffle=False,
                num_workers=0,
            )
        )
    )
    assert pyg_lib is not None
    assert torch_sparse is not None
    assert int(batch.batch_size) > 0
    assert int(batch.edge_index.numel()) > 0


def test_public_seed_boundary_reproduces_first_graph_dataset_batch(tiny_graph_data):
    """The released NNx 0.2 sampler consumes the global RNG, not a seed kwarg."""
    data = tiny_graph_data.data

    class TinyGraphDataset:
        num_features = tiny_graph_data.num_features
        num_classes = tiny_graph_data.num_classes

        def __init__(self, root, transform=None):
            del root
            self._data = data.clone()
            if transform is not None:
                self._data = transform(self._data)

        def __getitem__(self, index):
            assert index == 0
            return self._data

    def first_batch_state():
        set_seed(0)
        dataset = NNGraphDataset(
            ds_class=TinyGraphDataset,
            n_neighbors=[2, 2],
            n_workers=0,
            batch_sizes=(2, 1, 1),
        )
        batch = next(iter(dataset.train_loader))
        return batch.n_id.clone(), batch.edge_index.clone()

    first_node_ids, first_edges = first_batch_state()
    second_node_ids, second_edges = first_batch_state()

    assert first_node_ids.equal(second_node_ids)
    assert first_edges.equal(second_edges)


@pytest.mark.parametrize("net_enum", [Nets.GRAPH_SAGE, Nets.GRAPH_CONV])
def test_gnn_train_one_batch_sage_or_conv(net_enum, tiny_graph_data, gnn_loaders):
    """GraphSAGE and GraphConv share the no-attention path. Both should construct + 1-epoch-train."""
    train_loader, val_loader = gnn_loaders
    _assert_sampled_batch_is_executable(train_loader)
    model = NNModel(
        params=NNModelParams(net=net_enum, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
        net_params=NNParams(
            dropout_prob=0.25,
            hidden_dims=[8],
            input_dim=tiny_graph_data.num_features,
            output_dim=tiny_graph_data.num_classes,
        ),
    )
    train_params = (
        NNTrainParams(n_epochs=1)
        .with_train_loader(value=train_loader)
        .with_val_loader(value=val_loader)
    )
    run = model.train(params=train_params)
    assert run is not None


def test_gat_consolidates_n_heads_into_nnparams(tiny_graph_data, gnn_loaders):
    """Regression test for the GraphAttNNParams audit miss.

    Pre-extraction, GAT used a distinct GraphAttNNParams class. The
    consolidation merged that into NNParams with an Optional[int] n_heads
    field. This test pins the consolidated shape so any future un-merging
    fails CI rather than the weekly Tier-B/C smoke.
    """
    train_loader, val_loader = gnn_loaders
    _assert_sampled_batch_is_executable(train_loader)
    model = NNModel(
        params=NNModelParams(net=Nets.GRAPH_ATT, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
        net_params=NNParams(
            n_heads=2,
            dropout_prob=0.25,
            hidden_dims=[8],
            input_dim=tiny_graph_data.num_features,
            output_dim=tiny_graph_data.num_classes,
        ),
    )
    train_params = (
        NNTrainParams(n_epochs=1)
        .with_train_loader(value=train_loader)
        .with_val_loader(value=val_loader)
    )
    run = model.train(params=train_params)
    assert run is not None


def test_nnparams_state_round_trips_n_heads():
    """NNParams.state() omits n_heads when None; includes it when set.

    Pins the rule that prevents existing run.id hashes from shifting when
    n_heads-aware models are introduced.
    """
    p_no_heads = NNParams(dropout_prob=0.1, hidden_dims=[8], input_dim=4, output_dim=3)
    p_with_heads = NNParams(n_heads=2, dropout_prob=0.1, hidden_dims=[8], input_dim=4, output_dim=3)
    assert "n_heads" not in p_no_heads.state()
    assert p_with_heads.state()["n_heads"] == 2
