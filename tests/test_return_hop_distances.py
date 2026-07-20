"""End-to-end coverage for the return_hop_distances flag.

When cfg.return_hop_distances is True, generate_batch returns a
(B, actual_max_n, actual_max_n) int32 tensor keyed "hop_distances",
where actual_max_n is the maximum num_vertices across the batch
(bounded above by cfg.max_num_nodes when that is set, but sized
from what was actually sampled -- often much smaller). Slots inside
each row's [0, num_nodes[i]) x [0, num_nodes[i]) region hold either
the BFS hop distance or HOP_UNREACHABLE (-1); slots outside that
region hold HOP_PAD (INT_MIN).

Exercised across task kinds:
    * BFS -- task doesn't touch the matrix; worker triggers a
      post-task fill so the returned tensor is still populated
    * ShortestPath -- task uses its own 2-BFS pattern; same
      post-task fill applies
    * Center -- task calls sg.hop_distances() itself; the fill
      lands directly in the batch tensor with zero copy
"""
from __future__ import annotations

import numpy as np
import pytest

from get_generator_module import get_generator_module


HOP_PAD         = np.iinfo(np.int32).min  # matches HOP_PAD in C++
HOP_UNREACHABLE = -1


@pytest.fixture(scope="module")
def generator():
    get_generator_module()
    import generator as g  # noqa: WPS433
    return g


def _cfg(generator, *, task_kind: str, return_hop: bool, **overrides):
    kwargs = dict(
        graph_kind="erdos_renyi",
        task_kind=task_kind,
        scratchpad_kind="none",
        tokenization_mode="sean",
        min_num_nodes=8,
        max_num_nodes=12,       # variable so we exercise the padding region
        edge_prob=0.4,
        max_attempts=1000,
        batch_size=4,
        return_hop_distances=return_hop)
    if task_kind == "shortest_path":
        kwargs["min_path_length"] = 1
        kwargs["max_path_length"] = 6
    if task_kind == "center":
        kwargs["min_query_size"] = 2
        kwargs["max_query_size"] = 3
        # Center needs every query vertex reachable from a common center_node.
        # Bump edge_prob so ER samples are very likely to be connected.
        kwargs["edge_prob"] = 0.7
    kwargs.update(overrides)
    return generator.GeneratorConfig(**kwargs)


def _worker(generator, seed=0):
    ctx = generator.WorkerSharedContext()
    return generator.Worker(ctx, seed=seed)


# ---- Default behaviour: flag is False -------------------------------------

@pytest.mark.parametrize("task_kind", ["bfs", "shortest_path", "center"])
def test_hop_distances_absent_by_default(generator, task_kind):
    cfg = _cfg(generator, task_kind=task_kind, return_hop=False)
    result = _worker(generator).generate_batch(cfg)
    # No key when the flag isn't set. Callers who don't care never
    # pay the O(B * max_n^2) allocation or fill cost.
    assert "hop_distances" not in result


# ---- Presence + shape + dtype ---------------------------------------------

@pytest.mark.parametrize("task_kind", ["bfs", "shortest_path", "center"])
def test_hop_distances_present_when_flag_set(generator, task_kind):
    cfg = _cfg(generator, task_kind=task_kind, return_hop=True)
    result = _worker(generator).generate_batch(cfg)
    assert "hop_distances" in result
    D = result["hop_distances"]
    assert D.dtype == np.int32
    # Tensor is sized from the ACTUAL max num_vertices in this batch,
    # not cfg.max_num_nodes. That upper bound is still respected.
    actual_max_n = int(result["num_nodes"].max())
    assert D.shape == (cfg.batch_size, actual_max_n, actual_max_n)
    assert actual_max_n <= cfg.max_num_nodes


# ---- Padding sentinel outside (n, n) region -------------------------------

@pytest.mark.parametrize("task_kind", ["bfs", "shortest_path", "center"])
def test_hop_distances_padded_outside_num_nodes(generator, task_kind):
    cfg = _cfg(generator, task_kind=task_kind, return_hop=True)
    result = _worker(generator).generate_batch(cfg)
    D  = result["hop_distances"]
    nn = result["num_nodes"]
    # Padding tail runs to the tensor's actual max_n (== nn.max()),
    # not cfg.max_num_nodes -- rows below max_n but past this item's
    # num_nodes are HOP_PAD.
    max_n = int(nn.max())
    for i in range(cfg.batch_size):
        n = int(nn[i])
        # Rows past n: entirely HOP_PAD.
        if n < max_n:
            assert (D[i, n:, :] == HOP_PAD).all(), (
                f"row {i}: rows past num_nodes not HOP_PAD"
            )
            # Columns past n on the valid rows: HOP_PAD.
            assert (D[i, :n, n:] == HOP_PAD).all(), (
                f"row {i}: columns past num_nodes not HOP_PAD"
            )
        # Nothing inside the (n, n) region should be HOP_PAD.
        assert (D[i, :n, :n] != HOP_PAD).all(), (
            f"row {i}: HOP_PAD leaked into valid region"
        )


# ---- Diagonal is zero (distance from a vertex to itself) -----------------

@pytest.mark.parametrize("task_kind", ["bfs", "shortest_path", "center"])
def test_hop_distances_diagonal_is_zero(generator, task_kind):
    cfg = _cfg(generator, task_kind=task_kind, return_hop=True)
    result = _worker(generator).generate_batch(cfg)
    D  = result["hop_distances"]
    nn = result["num_nodes"]
    for i in range(cfg.batch_size):
        n = int(nn[i])
        for v in range(n):
            assert D[i, v, v] == 0


# ---- Symmetry on undirected graphs ---------------------------------------

@pytest.mark.parametrize("task_kind", ["bfs", "shortest_path", "center"])
def test_hop_distances_symmetric_when_undirected(generator, task_kind):
    # ER default is undirected; distances must be symmetric.
    cfg = _cfg(generator, task_kind=task_kind, return_hop=True)
    result = _worker(generator).generate_batch(cfg)
    D  = result["hop_distances"]
    nn = result["num_nodes"]
    for i in range(cfg.batch_size):
        n = int(nn[i])
        block = D[i, :n, :n]
        np.testing.assert_array_equal(block, block.T)


# ---- Values are non-negative distances or HOP_UNREACHABLE -----------------

@pytest.mark.parametrize("task_kind", ["bfs", "shortest_path", "center"])
def test_hop_distances_values_are_valid(generator, task_kind):
    cfg = _cfg(generator, task_kind=task_kind, return_hop=True)
    result = _worker(generator).generate_batch(cfg)
    D  = result["hop_distances"]
    nn = result["num_nodes"]
    for i in range(cfg.batch_size):
        n = int(nn[i])
        block = D[i, :n, :n]
        # Every entry is either a non-negative integer (reachable) or
        # exactly HOP_UNREACHABLE. No HOP_PAD or arbitrary junk.
        assert ((block >= 0) | (block == HOP_UNREACHABLE)).all()
        # Bounded above by n-1 (diameter cannot exceed vertex count).
        finite = block[block != HOP_UNREACHABLE]
        assert (finite <= n - 1).all()


# ---- Determinism ---------------------------------------------------------

def test_hop_distances_deterministic_under_fixed_seed(generator):
    cfg = _cfg(generator, task_kind="bfs", return_hop=True, batch_size=3)
    a = _worker(generator, seed=42).generate_batch(cfg)
    b = _worker(generator, seed=42).generate_batch(cfg)
    np.testing.assert_array_equal(a["hop_distances"], b["hop_distances"])


# ---- max_num_nodes no longer required when flag is set -------------------

def test_return_hop_distances_works_without_max_num_nodes(generator):
    # The hop-distance tensor is now sized from the actual max
    # num_vertices in the batch (not cfg.max_num_nodes), so
    # return_hop_distances=True no longer requires max_num_nodes to
    # be set at config time. With min_num_nodes alone the sampler
    # produces fixed-size graphs and the tensor is (B, n, n).
    n = 8
    cfg = generator.GeneratorConfig(
        graph_kind="erdos_renyi",
        task_kind="bfs",
        scratchpad_kind="none",
        tokenization_mode="sean",
        min_num_nodes=n,
        edge_prob=0.4,
        max_attempts=1000,
        batch_size=2,
        return_hop_distances=True,
        # max_num_nodes intentionally omitted -> defaults to -1
    )
    result = _worker(generator).generate_batch(cfg)
    D = result["hop_distances"]
    assert D.shape == (cfg.batch_size, n, n)
