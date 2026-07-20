"""End-to-end coverage for the return_distance_rank_targets flag.

When cfg.return_distance_rank_targets is True, generate_batch adds a
new tensor to the batch dict:

    "distance_rank_targets"  ->  (B, max_n, max_distance, max_ties) int32

For each source vertex u (in internal-id order along axis 1) it gives
the vocab token ids of every reachable vertex, bucketed by hop
distance along axis 2, ordered by vocab id within each tie along axis
3. The source itself always sits at (u, d=0, k=0). Unreachable pairs
are omitted; empty (d, k) slots hold -1.

Coverage:
  * Absent when flag off.
  * Presence, shape, dtype.
  * (u, 0, 0) is the source vertex itself, (u, 0, k>0) is -1.
  * All non-pad entries at (u, d, :) share hop distance d from u
    (verified against batch['hop_distances']).
  * Ties are ordered by vocab id ascending.
  * Symmetric on undirected graphs: v appears at rank d from u iff
    u appears at rank d from v.
  * Determinism under fixed seed.
  * Composes with return_hop_distances (asking for both isn't more
    work than asking for either).
  * Padding tail (u >= num_nodes[i], or beyond real ties) is -1.
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


def _cfg(generator, *, task_kind="bfs",
         return_drt=True, return_hop=False,
         batch_size=4, n_min=6, n_max=10):
    return generator.GeneratorConfig(
        graph_kind="erdos_renyi",
        task_kind=task_kind,
        scratchpad_kind="none",
        tokenization_mode="sean",
        min_num_nodes=n_min,
        max_num_nodes=n_max,
        edge_prob=0.5,
        max_attempts=1000,
        batch_size=batch_size,
        return_hop_distances=return_hop,
        return_distance_rank_targets=return_drt,
    )


def _worker(generator, seed=0):
    ctx = generator.WorkerSharedContext()
    return generator.Worker(ctx, seed=seed)


# ---- Absent when off ----------------------------------------------------

def test_distance_rank_targets_absent_when_flag_unset(generator):
    cfg = _cfg(generator, return_drt=False)
    result = _worker(generator).generate_batch(cfg)
    assert "distance_rank_targets" not in result


# ---- Shape + dtype ------------------------------------------------------

def test_distance_rank_targets_shape_and_dtype(generator):
    cfg = _cfg(generator, return_drt=True, return_hop=True, batch_size=6)
    result = _worker(generator).generate_batch(cfg)
    T = result["distance_rank_targets"]
    assert T.dtype == np.int32
    # Shape: (B, max_n, max_distance, max_ties).
    assert T.ndim == 4
    B, max_n, max_dist, max_ties = T.shape
    assert B == cfg.batch_size
    # max_n matches whichever tensor sizing we get elsewhere.
    assert max_n == int(result["num_nodes"].max())
    # At least the d=0 self slot per source and at least 1 tie there.
    assert max_dist >= 1
    assert max_ties >= 1


# ---- Semantics: self at (u, 0, 0), pad after --------------------------

def test_distance_rank_source_self_at_d0_k0(generator):
    cfg = _cfg(generator, return_drt=True, batch_size=4)
    ctx = generator.WorkerSharedContext()
    w   = generator.Worker(ctx, seed=1)
    result = w.generate_batch(cfg)
    T  = result["distance_rank_targets"]
    nn = result["num_nodes"]
    # We need internal_to_vocab to verify identity -- ask for positions
    # to surface it (or ask directly via return_positions).
    # Simpler: rerun with return_positions on top so we get the map.
    cfg2 = generator.GeneratorConfig(
        graph_kind="euclidean",  # euclidean surfaces internal_to_vocab
        task_kind="bfs",
        scratchpad_kind="none",
        tokenization_mode="sean",
        min_num_nodes=8, max_num_nodes=8, dim=2,
        max_attempts=1000, batch_size=4,
        return_distance_rank_targets=True,
        return_positions=True,
    )
    result2 = generator.Worker(generator.WorkerSharedContext(),
                               seed=42).generate_batch(cfg2)
    T2  = result2["distance_rank_targets"]
    V2  = result2["internal_to_vocab"]
    nn2 = result2["num_nodes"]
    for b in range(cfg2.batch_size):
        n = int(nn2[b])
        for u in range(n):
            # (u, 0, 0) == internal_to_vocab[u]
            assert T2[b, u, 0, 0] == V2[b, u], (
                f"batch {b} source {u}: T[0,0]={T2[b,u,0,0]} "
                f"vs vocab={V2[b,u]}"
            )
            # (u, 0, k>0) is pad -- source is unique at distance 0.
            assert (T2[b, u, 0, 1:] == -1).all()


# ---- Semantics: all entries at (u, d, :) have hop distance d ----------

def test_distance_rank_entries_agree_with_hop_distances(generator):
    # Request both tensors so we can cross-check DRT against the
    # hop_distances ground truth vertex-by-vertex.
    cfg = generator.GeneratorConfig(
        graph_kind="euclidean",
        task_kind="bfs",
        scratchpad_kind="none",
        tokenization_mode="sean",
        min_num_nodes=10, max_num_nodes=10, dim=2,
        max_attempts=1000, batch_size=3,
        return_hop_distances=True,
        return_positions=True,      # for internal_to_vocab surface
        return_distance_rank_targets=True,
    )
    result = generator.Worker(generator.WorkerSharedContext(),
                              seed=7).generate_batch(cfg)
    T  = result["distance_rank_targets"]
    D  = result["hop_distances"]
    V  = result["internal_to_vocab"]      # (B, n) vocab ids per internal id
    nn = result["num_nodes"]
    _, max_n, max_dist, max_ties = T.shape
    for b in range(cfg.batch_size):
        n = int(nn[b])
        # vocab -> internal-id lookup for this batch row
        vocab_to_internal = {int(V[b, i]): i for i in range(n)}
        for u in range(n):
            for d in range(max_dist):
                for k in range(max_ties):
                    val = int(T[b, u, d, k])
                    if val == -1:
                        continue
                    # `val` is a vocab id. Map back to internal id, then
                    # check hop_distances[u, that_internal] == d.
                    assert val in vocab_to_internal, (
                        f"batch {b} (u,d,k)=({u},{d},{k}): DRT holds "
                        f"vocab id {val} that isn't a vertex of this graph"
                    )
                    v_internal = vocab_to_internal[val]
                    assert D[b, u, v_internal] == d, (
                        f"batch {b} (u,d,k)=({u},{d},{k}): DRT says "
                        f"distance {d} but hop_distances says "
                        f"{D[b, u, v_internal]}"
                    )


# ---- Ties are ordered by vocab id ascending --------------------------

def test_distance_rank_tie_order_is_vocab_id_ascending(generator):
    cfg = generator.GeneratorConfig(
        graph_kind="erdos_renyi",
        task_kind="bfs",
        scratchpad_kind="none",
        tokenization_mode="sean",
        min_num_nodes=12, max_num_nodes=12,
        edge_prob=0.6,
        max_attempts=1000, batch_size=4,
        return_distance_rank_targets=True,
    )
    result = _worker(generator).generate_batch(cfg)
    T  = result["distance_rank_targets"]
    nn = result["num_nodes"]
    _, _, max_dist, _ = T.shape
    for b in range(cfg.batch_size):
        n = int(nn[b])
        for u in range(n):
            for d in range(max_dist):
                row = T[b, u, d, :]
                real = row[row != -1]
                if len(real) < 2:
                    continue
                # Ascending vocab id order within the tie group.
                assert (np.diff(real) > 0).all(), (
                    f"batch {b} source {u} d={d}: tie order not "
                    f"strictly ascending vocab ids: {real.tolist()}"
                )


# ---- Symmetry on undirected graphs ------------------------------------

def test_distance_rank_symmetric_when_undirected(generator):
    # If v appears in the (u, d, :) rank of an undirected graph then
    # u must appear in the (v, d, :) rank -- their hop distance is
    # symmetric.
    cfg = generator.GeneratorConfig(
        graph_kind="euclidean",
        task_kind="bfs",
        scratchpad_kind="none",
        tokenization_mode="sean",
        min_num_nodes=8, max_num_nodes=8, dim=2,
        max_attempts=1000, batch_size=3,
        return_positions=True,
        return_distance_rank_targets=True,
    )
    result = generator.Worker(generator.WorkerSharedContext(),
                              seed=3).generate_batch(cfg)
    T  = result["distance_rank_targets"]
    V  = result["internal_to_vocab"]
    nn = result["num_nodes"]
    _, _, max_dist, _ = T.shape
    for b in range(cfg.batch_size):
        n = int(nn[b])
        vocab_to_internal = {int(V[b, i]): i for i in range(n)}
        # Distance of v from u (via DRT); -1 if not present.
        def d_from_uv(u, v_vocab):
            for d in range(max_dist):
                row = T[b, u, d, :]
                if (row == v_vocab).any():
                    return d
            return -1
        for u in range(n):
            for v_internal in range(n):
                v_vocab = int(V[b, v_internal])
                d_uv = d_from_uv(u, v_vocab)
                u_vocab = int(V[b, u])
                d_vu = d_from_uv(v_internal, u_vocab)
                assert d_uv == d_vu, (
                    f"batch {b}: asymmetric ranks u={u} v={v_internal} "
                    f"(d_uv={d_uv} vs d_vu={d_vu})"
                )


# ---- Padding tail beyond num_nodes ------------------------------------

def test_distance_rank_padding_beyond_num_nodes(generator):
    # Sources past this item's num_nodes must be entirely -1.
    cfg = _cfg(generator, return_drt=True, batch_size=6, n_min=5, n_max=10)
    result = _worker(generator).generate_batch(cfg)
    T  = result["distance_rank_targets"]
    nn = result["num_nodes"]
    _, max_n, _, _ = T.shape
    for b in range(cfg.batch_size):
        n = int(nn[b])
        if n < max_n:
            assert (T[b, n:, :, :] == -1).all(), (
                f"batch {b}: source-axis rows past num_nodes not -1"
            )


# ---- Determinism ------------------------------------------------------

def test_distance_rank_deterministic_under_fixed_seed(generator):
    cfg = _cfg(generator, return_drt=True, batch_size=4)
    a = _worker(generator, seed=42).generate_batch(cfg)
    b = _worker(generator, seed=42).generate_batch(cfg)
    np.testing.assert_array_equal(a["distance_rank_targets"],
                                  b["distance_rank_targets"])


# ---- Composes with return_hop_distances -------------------------------

def test_distance_rank_and_hop_distances_together(generator):
    # Both flags on: both keys present, both consistent (already
    # verified point-wise by test_..._agree_with_hop_distances above,
    # this one just confirms co-existence).
    cfg = _cfg(generator, return_drt=True, return_hop=True, batch_size=3)
    result = _worker(generator, seed=5).generate_batch(cfg)
    assert "hop_distances" in result
    assert "distance_rank_targets" in result


# ---- pprint_distance_ranks ---------------------------------------------

def test_pprint_distance_ranks_runs_and_covers_every_source(generator, capsys):
    # Smoke-test the pretty printer: it must run without raising,
    # emit one "Source v{u}" block per source in [0, num_nodes[b]),
    # and label each block with the source's own symbol (which lives
    # at the tensor's (u, d=0, k=0) slot).
    ctx = generator.WorkerSharedContext()
    w   = generator.Worker(ctx, seed=7)
    cfg = _cfg(generator, return_drt=True, batch_size=2, n_min=5, n_max=8)
    result = w.generate_batch(cfg)

    generator.pprint_distance_ranks(result, ctx, indices=[0])
    out = capsys.readouterr().out
    # Header for the batch row we selected.
    assert 'BATCH INDEX 0' in out
    assert 'distance_rank_targets' in out
    # One "Source v{u}" block per source in row 0.
    n0 = int(result['num_nodes'][0])
    for u in range(n0):
        assert f'Source v{u}' in out, (
            f"missing block for source v{u} in printer output"
        )
    # Every source must emit at least a d=0 line (source-self is
    # always the closest by construction).
    assert out.count('d=0:') >= n0


def test_pprint_distance_ranks_missing_key_raises(generator):
    # No return_distance_rank_targets flag -> the key isn't in the
    # dict; printer must surface that as a clear assertion rather
    # than silently mis-rendering.
    ctx = generator.WorkerSharedContext()
    w   = generator.Worker(ctx, seed=0)
    cfg = _cfg(generator, return_drt=False, batch_size=1)
    result = w.generate_batch(cfg)
    with pytest.raises(AssertionError, match="distance_rank_targets"):
        generator.pprint_distance_ranks(result, ctx)
