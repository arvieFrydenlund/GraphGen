"""End-to-end tests for include_duplicate_edges_in_graph_tokenization.

When True, each undirected edge is emitted twice — once as `u v EDGE`
and again as `v u EDGE`, doubling the length of the GraphEdges section.
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


def _cfg(generator, *, duplicate, **overrides):
    kwargs = dict(
        graph_kind="erdos_renyi",
        task_kind="shortest_path",
        scratchpad_kind="none",
        tokenization_mode="sean",
        include_duplicate_edges_in_graph_tokenization=duplicate,
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


def test_duplicate_edges_doubles_graph_edge_length_sean(generator):
    """SEAN mode: single edge = 3 positions, duplicated = 6 positions.
    graph_edge_lengths reports position count, so it doubles."""
    single = _worker(generator, seed=5).generate_batch(
        _cfg(generator, duplicate=False))
    double = _worker(generator, seed=5).generate_batch(
        _cfg(generator, duplicate=True))
    np.testing.assert_array_equal(
        double["graph_edge_lengths"], 2 * single["graph_edge_lengths"])
    # num_edges reports the graph's actual edge count (unchanged).
    np.testing.assert_array_equal(double["num_edges"], single["num_edges"])


def test_duplicate_edges_grows_seq_len_by_3m(generator):
    single = _worker(generator, seed=5).generate_batch(
        _cfg(generator, duplicate=False))
    double = _worker(generator, seed=5).generate_batch(
        _cfg(generator, duplicate=True))
    diff = double["src_lengths"] - single["src_lengths"]
    np.testing.assert_array_equal(diff, 3 * single["num_edges"])


def test_duplicate_edges_emits_two_identical_passes(generator):
    """The edge list is emitted TWICE, identically -- pass 2 is a
    byte-for-byte repeat of pass 1 (same order, same endpoints, no
    swap). Motivation is training-side: under causal attention pass 2
    can attend to every token in pass 1."""
    result = _worker(generator).generate_batch(_cfg(generator, duplicate=True))
    src = result["src_tokens"]
    for b in range(src.shape[0]):
        start = int(result["graph_edge_start_indices"][b])
        m = int(result["num_edges"][b])
        pass1 = src[b, start           : start + 3 * m]
        pass2 = src[b, start + 3 * m   : start + 6 * m]
        np.testing.assert_array_equal(pass1, pass2)


def test_duplicate_edges_deterministic_under_fixed_seed(generator):
    cfg = _cfg(generator, duplicate=True)
    a = _worker(generator, seed=99).generate_batch(cfg)
    b = _worker(generator, seed=99).generate_batch(cfg)
    for key in a.keys():
        np.testing.assert_array_equal(a[key], b[key])


def test_duplicate_edges_under_stan_doubles_edge_positions(generator):
    """STAN packs each edge into 1 position × 3 cols. Duplicating
    the edge takes 2 positions × 3 cols; edge count doubles."""
    single = _worker(generator, seed=5).generate_batch(
        _cfg(generator, duplicate=False, tokenization_mode="stan"))
    double = _worker(generator, seed=5).generate_batch(
        _cfg(generator, duplicate=True,  tokenization_mode="stan"))
    np.testing.assert_array_equal(
        double["graph_edge_lengths"], 2 * single["graph_edge_lengths"])
    # Under STAN, graph_edge_length under duplication equals 2 * num_edges.
    np.testing.assert_array_equal(
        double["graph_edge_lengths"], 2 * double["num_edges"])
