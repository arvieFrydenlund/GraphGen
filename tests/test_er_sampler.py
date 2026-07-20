"""Erdős-Rényi sampler invariant tests.

Verifies the freshly-ported sample_erdos_renyi obeys the properties every
G(n, p) draw is supposed to: vertex count matches config, edge count is
close to the expected n(n-1)p/2, component count is in [1, n], and the
draw is deterministic under a fixed seed. Uses the Worker.sample_graph_stats
Step-5 debug hook to peek at what the sampler produced (this hook goes
away when generate_batch grows a real return shape).
"""
from __future__ import annotations

import pytest

from get_generator_module import get_generator_module


@pytest.fixture(scope="module")
def generator():
    get_generator_module()
    import generator as g  # noqa: WPS433
    return g


def _make_cfg(generator, *, n, p, seed_note=""):
    return generator.GeneratorConfig(
        graph_kind="erdos_renyi",
        min_num_nodes=n,
        max_num_nodes=n,
        edge_prob=p,
        batch_size=1,
    )


def _make_worker(generator, seed):
    ctx = generator.WorkerSharedContext()
    return generator.Worker(ctx, seed=seed)


@pytest.mark.parametrize("n", [10, 50])
@pytest.mark.parametrize("p", [0.05, 0.3, 0.7])
def test_er_vertex_count_matches_config(generator, n, p):
    w = _make_worker(generator, seed=1)
    stats = w.sample_graph_stats(_make_cfg(generator, n=n, p=p))
    assert stats["num_vertices"] == n


@pytest.mark.parametrize("n", [10, 50])
@pytest.mark.parametrize("p", [0.05, 0.3, 0.7])
def test_er_edge_count_close_to_expectation(generator, n, p):
    # Draw many graphs; the mean edge count should be within a loose
    # tolerance of the theoretical expectation E[m] = C(n, 2) * p. A
    # single draw is too noisy to check, but averaged over enough
    # samples the estimator is tight.
    w = _make_worker(generator, seed=42)
    samples = 200
    total = 0
    max_possible = n * (n - 1) // 2
    for _ in range(samples):
        s = w.sample_graph_stats(_make_cfg(generator, n=n, p=p))
        assert 0 <= s["num_edges"] <= max_possible
        total += s["num_edges"]
    mean = total / samples
    expected = max_possible * p
    # Loose tolerance -- variance is p(1-p) * max_possible per draw, so
    # std of the mean is sqrt(p(1-p) * max_possible / samples).
    # A window of expected * 0.2 is plenty for these sizes.
    tolerance = max(expected * 0.2, 2.0)
    assert abs(mean - expected) < tolerance, (
        f"n={n} p={p}: mean edges {mean:.2f}, expected {expected:.2f}, "
        f"tolerance {tolerance:.2f}"
    )


@pytest.mark.parametrize("n", [10, 50])
@pytest.mark.parametrize("p", [0.05, 0.3, 0.7])
def test_er_num_components_in_valid_range(generator, n, p):
    w = _make_worker(generator, seed=7)
    for _ in range(20):
        s = w.sample_graph_stats(_make_cfg(generator, n=n, p=p))
        assert 1 <= s["num_components"] <= n


def test_er_deterministic_under_fixed_seed(generator):
    cfg = _make_cfg(generator, n=25, p=0.3)
    a = _make_worker(generator, seed=1234).sample_graph_stats(cfg)
    b = _make_worker(generator, seed=1234).sample_graph_stats(cfg)
    assert a["num_vertices"] == b["num_vertices"]
    assert a["num_edges"] == b["num_edges"]
    assert a["num_components"] == b["num_components"]
    assert a["vocab_ids"] == b["vocab_ids"]


def test_er_different_seeds_give_different_graphs(generator):
    cfg = _make_cfg(generator, n=30, p=0.3)
    a = _make_worker(generator, seed=1).sample_graph_stats(cfg)
    b = _make_worker(generator, seed=2).sample_graph_stats(cfg)
    # It's astronomically unlikely two different seeds produce identical
    # edge counts AND identical vocab id vectors.
    same_edges = a["num_edges"] == b["num_edges"]
    same_vocab = a["vocab_ids"] == b["vocab_ids"]
    assert not (same_edges and same_vocab)


def test_er_p_zero_gives_empty_graph(generator):
    w = _make_worker(generator, seed=99)
    s = w.sample_graph_stats(_make_cfg(generator, n=15, p=0.0))
    assert s["num_edges"] == 0
    assert s["num_components"] == 15  # every vertex is isolated


def test_er_p_one_gives_complete_graph(generator):
    w = _make_worker(generator, seed=99)
    n = 12
    s = w.sample_graph_stats(_make_cfg(generator, n=n, p=1.0))
    assert s["num_edges"] == n * (n - 1) // 2
    assert s["num_components"] == 1


def test_er_vocab_ids_have_correct_shape(generator):
    w = _make_worker(generator, seed=0)
    n = 20
    s = w.sample_graph_stats(_make_cfg(generator, n=n, p=0.5))
    assert len(s["vocab_ids"]) == n
    assert len(set(s["vocab_ids"])) == n  # all distinct


def test_er_missing_edge_prob_raises(generator):
    w = _make_worker(generator, seed=0)
    cfg = generator.GeneratorConfig(
        graph_kind="erdos_renyi",
        min_num_nodes=10,
        max_num_nodes=10,
        batch_size=1,
        # edge_prob deliberately omitted
    )
    with pytest.raises(RuntimeError) as exc_info:
        w.sample_graph_stats(cfg)
    assert "edge_prob" in str(exc_info.value)
