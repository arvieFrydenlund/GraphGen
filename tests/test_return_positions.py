"""End-to-end coverage for the return_positions flag.

When cfg.return_positions is True, generate_batch returns TWO extra
tensors in the batch dict:
  * "node_positions"     -- (B, max_n, dim) float64. Each row's
                            [0, num_nodes[i]) region holds the
                            geometric positions the sampler drew;
                            slots outside that region hold NaN.
  * "internal_to_vocab"  -- (B, max_n) int32 mapping internal vertex
                            id -> vocab token id. Slots outside the
                            valid region hold -1.

Only geometric-graph samplers populate positions today; the flag
is rejected on non-euclidean graph_kind at config-validate time.

Coverage:
  * Presence + shape + dtype under the flag.
  * NaN / -1 sentinels outside the valid region.
  * Positions fall in the unit cube (sample_euclidean's domain).
  * internal_to_vocab agrees with the vocab range [min_vocab,
    max_vocab) and vertex ids are unique per row.
  * Flag rejected on non-euclidean graph_kind.
  * Absent when the flag isn't set.
"""
from __future__ import annotations

import numpy as np
import pytest

from get_generator_module import get_generator_module


@pytest.fixture(scope="module")
def generator():
    get_generator_module()
    import generator as g  # noqa: WPS433
    return g


def _euclidean_cfg(generator, *, n=10, dim=2, batch_size=4,
                   return_positions=True):
    return generator.GeneratorConfig(
        graph_kind="euclidean",
        task_kind="bfs",
        scratchpad_kind="none",
        tokenization_mode="sean",
        min_num_nodes=n,
        max_num_nodes=n,
        dim=dim,
        batch_size=batch_size,
        max_attempts=1000,
        return_positions=return_positions)


def _worker(generator, seed=0):
    ctx = generator.WorkerSharedContext()
    return generator.Worker(ctx, seed=seed)


# ---- Absent by default ---------------------------------------------------

def test_node_positions_absent_when_flag_unset(generator):
    cfg = _euclidean_cfg(generator, return_positions=False)
    result = _worker(generator).generate_batch(cfg)
    assert "node_positions" not in result
    assert "internal_to_vocab" not in result


# ---- Presence, shape, dtype ---------------------------------------------

@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("n",   [5, 15, 30])
def test_node_positions_shape_and_dtype(generator, n, dim):
    cfg = _euclidean_cfg(generator, n=n, dim=dim)
    result = _worker(generator).generate_batch(cfg)
    P = result["node_positions"]
    V = result["internal_to_vocab"]
    # Sized from actual max_n (== n since min==max here).
    assert P.dtype == np.float64
    assert V.dtype == np.int32
    assert P.shape == (cfg.batch_size, n, dim)
    assert V.shape == (cfg.batch_size, n)


# ---- Sentinels outside the valid region ---------------------------------

def test_positions_padded_outside_valid_region(generator):
    # Vary sampled size across the batch so we exercise the padding
    # region on shorter rows.
    n_min, n_max = 5, 12
    cfg = generator.GeneratorConfig(
        graph_kind="euclidean",
        task_kind="bfs",
        scratchpad_kind="none",
        tokenization_mode="sean",
        min_num_nodes=n_min,
        max_num_nodes=n_max,
        dim=2,
        batch_size=8,
        max_attempts=1000,
        return_positions=True)
    result = _worker(generator).generate_batch(cfg)
    P  = result["node_positions"]
    V  = result["internal_to_vocab"]
    nn = result["num_nodes"]
    max_n = int(nn.max())
    for i in range(cfg.batch_size):
        n = int(nn[i])
        # Rows past n: entirely NaN in positions, -1 in vocab map.
        if n < max_n:
            assert np.isnan(P[i, n:, :]).all(), (
                f"row {i}: positions past num_nodes not NaN"
            )
            assert (V[i, n:] == -1).all(), (
                f"row {i}: vocab-map past num_nodes not -1"
            )
        # Nothing inside the valid region should be NaN / -1.
        assert not np.isnan(P[i, :n, :]).any(), (
            f"row {i}: NaN leaked into valid region"
        )
        assert (V[i, :n] != -1).all(), (
            f"row {i}: -1 leaked into valid vocab-map region"
        )


# ---- Position values in the unit cube ------------------------------------

def test_euclidean_positions_in_unit_cube(generator):
    # sample_euclidean draws positions from [0, 1]^dim.
    cfg = _euclidean_cfg(generator, n=15, dim=3, batch_size=4)
    result = _worker(generator).generate_batch(cfg)
    P  = result["node_positions"]
    nn = result["num_nodes"]
    for i in range(cfg.batch_size):
        n = int(nn[i])
        block = P[i, :n, :]
        assert (block >= 0.0).all() and (block <= 1.0).all(), (
            f"row {i}: position outside [0, 1]"
        )


# ---- internal_to_vocab is consistent with the ctx-derived vocab range ---

def test_internal_to_vocab_within_range_and_unique(generator):
    # The vocab range for node ids is derived from the shared context
    # dictionary: [ctx.num_special, ctx.max_vocab). No config-side
    # override -- if the user wants a wider slice they build a bigger
    # dictionary. Here we use the default WorkerSharedContext and
    # check the surfaced tensor stays inside that ctx-derived range.
    ctx = generator.WorkerSharedContext()
    w   = generator.Worker(ctx, seed=0)
    cfg = _euclidean_cfg(generator, n=10, dim=2, batch_size=6)
    result = w.generate_batch(cfg)
    V  = result["internal_to_vocab"]
    nn = result["num_nodes"]
    for i in range(cfg.batch_size):
        n = int(nn[i])
        ids = V[i, :n]
        # In-range for the ctx-derived node-vocab window.
        assert (ids >= ctx.num_special).all()
        assert (ids <  ctx.max_vocab).all()
        # Unique per row (each vertex draws a distinct vocab id).
        assert len(set(ids.tolist())) == n


# ---- Determinism ---------------------------------------------------------

def test_node_positions_deterministic_under_fixed_seed(generator):
    cfg = _euclidean_cfg(generator, n=12, dim=2, batch_size=3)
    a = _worker(generator, seed=42).generate_batch(cfg)
    b = _worker(generator, seed=42).generate_batch(cfg)
    np.testing.assert_array_equal(a["node_positions"], b["node_positions"])
    np.testing.assert_array_equal(a["internal_to_vocab"], b["internal_to_vocab"])


# ---- Validation: flag rejected on non-euclidean graph_kind ---------------

def test_return_positions_rejected_on_non_euclidean(generator):
    with pytest.raises((ValueError, RuntimeError)):
        generator.GeneratorConfig(
            graph_kind="erdos_renyi",
            task_kind="bfs",
            scratchpad_kind="none",
            tokenization_mode="sean",
            min_num_nodes=10, max_num_nodes=10,
            edge_prob=0.3,
            batch_size=2,
            max_attempts=100,
            return_positions=True)
