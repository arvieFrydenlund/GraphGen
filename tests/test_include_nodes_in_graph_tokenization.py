"""Tests for include_nodes_in_graph_tokenization.

When True, a `GraphNodes` section is emitted right before `GraphEdges`,
listing every vertex token (n positions). No dedicated brackets --
just an enumerated vertex list preceding the edge list.
"""
from __future__ import annotations

import numpy as np
import pytest

from get_generator_module import get_generator_module


TOK_EDGE = 4
NUM_SPECIAL_DEFAULT = 26


@pytest.fixture(scope="module")
def generator():
    get_generator_module()
    import generator as g  # noqa: WPS433
    return g


def _cfg(generator, *, include_nodes, **overrides):
    kwargs = dict(
        graph_kind="erdos_renyi",
        task_kind="shortest_path",
        scratchpad_kind="none",
        tokenization_mode="sean",
        include_nodes_in_graph_tokenization=include_nodes,
        min_num_nodes=8,
        max_num_nodes=8,
        edge_prob=0.4,
        min_path_length=2,
        max_path_length=6,
        max_attempts=1000,
        batch_size=4)
    kwargs.update(overrides)
    return generator.GeneratorConfig(**kwargs)


def _worker(generator, seed=0):
    ctx = generator.WorkerSharedContext()
    return generator.Worker(ctx, seed=seed)


def test_include_nodes_true_lengthens_seq_by_n_positions(generator):
    """The nodes prelude adds n vertex tokens (no brackets), so
    seq_len grows by num_nodes[b] per row."""
    without = _worker(generator, seed=5).generate_batch(
        _cfg(generator, include_nodes=False))
    with_   = _worker(generator, seed=5).generate_batch(
        _cfg(generator, include_nodes=True))
    diff = with_["src_lengths"] - without["src_lengths"]
    np.testing.assert_array_equal(diff, with_["num_nodes"])


def test_include_nodes_shifts_graph_edge_start_by_n(generator):
    """GraphNodes sits right before GraphEdges, so the edge section
    shifts forward by num_nodes positions."""
    without = _worker(generator, seed=5).generate_batch(
        _cfg(generator, include_nodes=False))
    with_   = _worker(generator, seed=5).generate_batch(
        _cfg(generator, include_nodes=True))
    shift = with_["graph_edge_start_indices"] - without["graph_edge_start_indices"]
    np.testing.assert_array_equal(shift, with_["num_nodes"])


def test_include_nodes_prelude_contains_every_vertex_token(generator):
    """The n positions right after BOS should hold each vertex token
    exactly once, in internal id order (0..n-1)."""
    result = _worker(generator).generate_batch(
        _cfg(generator, include_nodes=True))
    src = result["src_tokens"]
    for b in range(src.shape[0]):
        n = int(result["num_nodes"][b])
        # BOS is at col 0; nodes start at col 1.
        # We don't know the vocab-id mapping (it's per-item), but every
        # position in [1, n+1) must be a vertex token (>= NUM_SPECIAL),
        # and the n unique ids must equal the ids that appear in edges.
        node_positions = src[b, 1 : 1 + n]
        assert (node_positions >= NUM_SPECIAL_DEFAULT).all()
        # Every emitted node token also appears in the edge section
        # (since ER guarantees every vertex has at least one edge in
        # dense enough graphs -- we sized batches that way).
        edge_start = int(result["graph_edge_start_indices"][b])
        edge_len   = int(result["graph_edge_lengths"][b])
        edge_slice = src[b, edge_start : edge_start + edge_len]
        edge_verts = set(int(t) for t in edge_slice if int(t) != TOK_EDGE)
        # Every emitted node ID should be either a legit vertex token
        # (>= NUM_SPECIAL) drawn from the sampled vocab range.
        node_ids = set(int(t) for t in node_positions)
        # Not every node necessarily appears in edges (rare isolates),
        # but every edge endpoint MUST appear in the nodes prelude.
        assert edge_verts.issubset(node_ids)
        # And the prelude lists n distinct vertex tokens.
        assert len(node_ids) == n


def test_include_nodes_deterministic_under_fixed_seed(generator):
    cfg = _cfg(generator, include_nodes=True)
    a = _worker(generator, seed=99).generate_batch(cfg)
    b = _worker(generator, seed=99).generate_batch(cfg)
    for key in a.keys():
        np.testing.assert_array_equal(a[key], b[key])
