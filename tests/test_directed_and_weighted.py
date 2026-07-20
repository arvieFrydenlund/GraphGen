"""Tests for the top-level directed / weighted graph-structure flags.

directed: default False (undirected). When True the sampler considers
  every ordered pair (u, v) with u != v as an independent Bernoulli
  trial, so both u->v AND v->u can coexist. Currently only wired for
  erdos_renyi (other graph_kinds are themselves unimplemented).

weighted: STUB. Setting True is rejected at construction time until
  weighted tasks and tokenization land.
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


def _cfg(generator, **overrides):
    kwargs = dict(
        graph_kind="erdos_renyi",
        # BFS task by default -- shortest_path doesn't yet support
        # directed graphs (rejected at validation time), and the BFS
        # task's forward-BFS visit-order semantic works uniformly
        # over both directedness modes.
        task_kind="bfs",
        scratchpad_kind="none",
        tokenization_mode="sean",
        min_num_nodes=10,
        max_num_nodes=10,
        edge_prob=0.4,
        max_attempts=1000,
        batch_size=4)
    kwargs.update(overrides)
    return generator.GeneratorConfig(**kwargs)


def _worker(generator, seed=0):
    ctx = generator.WorkerSharedContext()
    return generator.Worker(ctx, seed=seed)


# ---------------------------------------------------------------------------
# weighted: rejection stub
# ---------------------------------------------------------------------------

def test_weighted_true_is_rejected_at_construction(generator):
    with pytest.raises(ValueError) as exc_info:
        generator.GeneratorConfig(weighted=True, graph_kind="erdos_renyi",
                                  min_num_nodes=5, edge_prob=0.5)
    msg = str(exc_info.value)
    assert "weighted" in msg
    assert "not implemented" in msg


def test_weighted_default_is_false(generator):
    cfg = _cfg(generator)
    assert cfg.weighted is False
    assert cfg.to_dict()["weighted"] is False


# ---------------------------------------------------------------------------
# directed
# ---------------------------------------------------------------------------

def test_directed_default_is_false(generator):
    cfg = _cfg(generator)
    assert cfg.directed is False


def test_directed_true_produces_more_edges_on_average(generator):
    """Directed ER samples each of the n*(n-1) ordered pairs
    independently; undirected samples the n*(n-1)/2 unordered pairs.
    Same edge_prob => directed expects roughly 2x as many edges."""
    # Average over a decent batch to smooth out variance.
    und = _worker(generator, seed=7).generate_batch(
        _cfg(generator, batch_size=64, directed=False))
    dir_ = _worker(generator, seed=7).generate_batch(
        _cfg(generator, batch_size=64, directed=True))
    und_avg = np.mean(und["num_edges"])
    dir_avg = np.mean(dir_["num_edges"])
    # Directed is 2x in expectation; allow a wide window for sample variance.
    assert dir_avg > 1.4 * und_avg
    assert dir_avg < 2.6 * und_avg


def test_directed_true_can_produce_both_uv_and_vu_edges(generator):
    """Directed graphs can store both a->b AND b->a as separate edges;
    undirected canonicalises to store each edge once, so among the
    emitted vocab-id tuples an undirected batch NEVER contains both
    (a, b) and (b, a) for any (a, b) with a != b.  A directed batch
    over enough graphs eventually does."""
    def has_reverse_pair(result):
        for b in range(result["src_tokens"].shape[0]):
            start = int(result["graph_edge_start_indices"][b])
            m     = int(result["num_edges"][b])
            pairs = set()
            for i in range(m):
                u = int(result["src_tokens"][b, start + i * 3 + 0])
                v = int(result["src_tokens"][b, start + i * 3 + 1])
                if (v, u) in pairs:
                    return True
                pairs.add((u, v))
        return False

    und = _worker(generator, seed=13).generate_batch(
        _cfg(generator, batch_size=32, directed=False))
    assert not has_reverse_pair(und), \
        "undirected batch emitted an edge and its reverse in the same row"

    dir_ = _worker(generator, seed=13).generate_batch(
        _cfg(generator, batch_size=64, directed=True))
    assert has_reverse_pair(dir_), \
        "directed batch never emitted an a->b and b->a pair"


def test_directed_deterministic_under_fixed_seed(generator):
    cfg = _cfg(generator, directed=True)
    a = _worker(generator, seed=99).generate_batch(cfg)
    b = _worker(generator, seed=99).generate_batch(cfg)
    for key in a.keys():
        np.testing.assert_array_equal(a[key], b[key])


# ---------------------------------------------------------------------------
# directed + duplicate_edges: rejected combination
# ---------------------------------------------------------------------------

def test_directed_composes_with_duplicate_edges(generator):
    """duplicate_edges emits an EXACT repeat of the edge list (no
    endpoint swap), so it composes cleanly with directed=True: the
    graph is unchanged, we just show the model the edge list twice
    to defeat causal attention's forward-only view."""
    cfg = _cfg(generator, directed=True,
               include_duplicate_edges_in_graph_tokenization=True,
               batch_size=4)
    result = _worker(generator, seed=51).generate_batch(cfg)
    src = result["src_tokens"]
    for b in range(src.shape[0]):
        start = int(result["graph_edge_start_indices"][b])
        m = int(result["num_edges"][b])
        # Length is 2 * 3m = 6m positions.
        assert int(result["graph_edge_lengths"][b]) == 6 * m
        pass1 = src[b, start           : start + 3 * m]
        pass2 = src[b, start + 3 * m   : start + 6 * m]
        np.testing.assert_array_equal(pass1, pass2)


def test_directed_composes_with_shortest_path_task(generator):
    """shortest_path on directed graphs uses reverse BFS from `end` to
    build the DAG. Every emitted path must be a valid directed path:
    each consecutive pair (path[i], path[i+1]) must exist as an edge
    in the emitted edge list."""
    cfg = _cfg(generator,
               directed=True,
               task_kind="shortest_path",
               min_path_length=2,
               max_path_length=5,
               edge_prob=0.6,      # dense enough that most pairs are reachable
               batch_size=32,
               max_attempts=5000)
    result = _worker(generator, seed=61).generate_batch(cfg)
    src = result["src_tokens"]
    for b in range(src.shape[0]):
        # Build the set of ordered edges from the graph section.
        e_start = int(result["graph_edge_start_indices"][b])
        m       = int(result["num_edges"][b])
        edges = set()
        for i in range(m):
            u = int(src[b, e_start + i * 3 + 0])
            v = int(src[b, e_start + i * 3 + 1])
            edges.add((u, v))

        # Read the target path and verify each hop is a directed edge.
        t_start = int(result["task_start_indices"][b])
        t_len   = int(result["task_lengths"][b])
        path = [int(src[b, t_start + i]) for i in range(t_len)]
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            assert (u, v) in edges, (
                f"row {b}: path hop ({u}->{v}) is not a directed edge; "
                f"this would only happen if sample_shortest_path assumed "
                f"symmetric distances on a directed graph")


def test_directed_shortest_path_deterministic_under_fixed_seed(generator):
    cfg = _cfg(generator,
               directed=True,
               task_kind="shortest_path",
               min_path_length=2, max_path_length=5,
               edge_prob=0.6, batch_size=4, max_attempts=5000)
    a = _worker(generator, seed=71).generate_batch(cfg)
    b = _worker(generator, seed=71).generate_batch(cfg)
    for key in a.keys():
        np.testing.assert_array_equal(a[key], b[key])
