"""Euclidean (random geometric graph) sampler invariant tests.

sample_euclidean draws n vertex positions in [0, 1]^dim and adds an
edge between every pair whose Euclidean distance falls in
[min_edge_length, max_edge_length]. Defaults: dim=2, min=0,
max=1/sqrt(n) (the classic connectivity-threshold radius).

Positions aren't currently surfaced to Python (see PLAN.md's
"return_positions" future note), so these tests focus on structural
invariants: vertex count, edge-count bounds, monotonicity of edge
count in max_edge_length, determinism, and expected behaviour at the
degenerate boundaries (radius=0 -> empty; radius >= sqrt(dim) -> K_n).
"""
from __future__ import annotations

import math

import pytest

from get_generator_module import get_generator_module


@pytest.fixture(scope="module")
def generator():
    get_generator_module()
    import generator as g  # noqa: WPS433
    return g


def _make_cfg(generator, *, n, dim=2, min_edge=None, max_edge=None):
    kwargs = dict(
        graph_kind="euclidean",
        min_num_nodes=n,
        max_num_nodes=n,
        dim=dim,
        batch_size=1,
    )
    if min_edge is not None:
        kwargs["min_edge_length"] = min_edge
    if max_edge is not None:
        kwargs["max_edge_length"] = max_edge
    return generator.GeneratorConfig(**kwargs)


def _make_worker(generator, seed):
    ctx = generator.WorkerSharedContext()
    return generator.Worker(ctx, seed=seed)


# ---- Vertex count -------------------------------------------------------

@pytest.mark.parametrize("n", [5, 20, 50])
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_euclidean_vertex_count_matches_config(generator, n, dim):
    w = _make_worker(generator, seed=0)
    stats = w.sample_graph_stats(_make_cfg(generator, n=n, dim=dim))
    assert stats["num_vertices"] == n


# ---- Edge-count bounds and monotonicity ---------------------------------

@pytest.mark.parametrize("n", [10, 30])
def test_euclidean_edge_count_within_max_possible(generator, n):
    w = _make_worker(generator, seed=1)
    for _ in range(10):
        s = w.sample_graph_stats(_make_cfg(generator, n=n))
        assert 0 <= s["num_edges"] <= n * (n - 1) // 2


@pytest.mark.parametrize("n", [15, 40])
def test_euclidean_edge_count_monotone_in_max_edge_length(generator, n):
    # For a FIXED position layout (same rng seed), raising the max
    # edge length can only accept more pairs -- edge count is
    # monotone non-decreasing in max_edge_length.
    dim = 2
    diameter = math.sqrt(dim)
    radii = [0.0, 0.1, 0.3, diameter + 1e-6]
    counts = []
    for r in radii:
        # Same seed for every point -> same positions; only the
        # accept threshold changes.
        w = _make_worker(generator, seed=123)
        s = w.sample_graph_stats(_make_cfg(generator, n=n, dim=dim, max_edge=r))
        counts.append(s["num_edges"])
    assert counts == sorted(counts), (
        f"expected non-decreasing edge counts, got {counts} for radii {radii}"
    )


# ---- Degenerate cases ---------------------------------------------------

@pytest.mark.parametrize("n", [8, 16])
def test_euclidean_radius_zero_gives_empty_graph(generator, n):
    # Distance = 0 has measure zero on a continuous distribution.
    w = _make_worker(generator, seed=0)
    s = w.sample_graph_stats(_make_cfg(generator, n=n, max_edge=0.0))
    assert s["num_edges"] == 0
    assert s["num_components"] == n     # every vertex is isolated


@pytest.mark.parametrize("n,dim", [(6, 2), (5, 3), (7, 1)])
def test_euclidean_radius_geq_diameter_gives_complete_graph(generator, n, dim):
    # Unit dim-cube diameter = sqrt(dim); any radius >= diameter
    # accepts every pair -> K_n with n*(n-1)/2 edges, one component.
    diameter = math.sqrt(dim)
    w = _make_worker(generator, seed=0)
    s = w.sample_graph_stats(
        _make_cfg(generator, n=n, dim=dim, max_edge=diameter + 1e-6)
    )
    assert s["num_edges"] == n * (n - 1) // 2
    assert s["num_components"] == 1


# ---- Component count is in a valid range --------------------------------

@pytest.mark.parametrize("n", [10, 25])
def test_euclidean_num_components_in_valid_range(generator, n):
    w = _make_worker(generator, seed=7)
    for _ in range(10):
        s = w.sample_graph_stats(_make_cfg(generator, n=n))
        assert 1 <= s["num_components"] <= n


# ---- Determinism --------------------------------------------------------

def test_euclidean_deterministic_under_fixed_seed(generator):
    cfg = _make_cfg(generator, n=20)
    a = _make_worker(generator, seed=1234).sample_graph_stats(cfg)
    b = _make_worker(generator, seed=1234).sample_graph_stats(cfg)
    assert a["num_vertices"]   == b["num_vertices"]
    assert a["num_edges"]      == b["num_edges"]
    assert a["num_components"] == b["num_components"]
    assert a["vocab_ids"]      == b["vocab_ids"]


def test_euclidean_different_seeds_give_different_graphs(generator):
    cfg = _make_cfg(generator, n=30)
    a = _make_worker(generator, seed=1).sample_graph_stats(cfg)
    b = _make_worker(generator, seed=2).sample_graph_stats(cfg)
    # Distinct seeds -> distinct positions -> almost surely different
    # edge sets AND different vocab id vectors.
    assert not (a["num_edges"] == b["num_edges"] and
                a["vocab_ids"]   == b["vocab_ids"])


# ---- Vocab shape --------------------------------------------------------

def test_euclidean_vocab_ids_have_correct_shape(generator):
    w = _make_worker(generator, seed=0)
    n = 15
    s = w.sample_graph_stats(_make_cfg(generator, n=n))
    assert len(s["vocab_ids"]) == n
    assert len(set(s["vocab_ids"])) == n


# ---- Config validation errors -------------------------------------------

def test_euclidean_rejects_directed_true(generator):
    with pytest.raises((ValueError, RuntimeError)):
        generator.GeneratorConfig(
            graph_kind="euclidean",
            min_num_nodes=10,
            max_num_nodes=10,
            batch_size=1,
            directed=True)


def test_euclidean_rejects_dims_kwarg(generator):
    # cfg.dims is reserved for a future per-dimension extent API and
    # is currently not consumed by sample_euclidean; use cfg.dim.
    with pytest.raises((ValueError, RuntimeError)):
        generator.GeneratorConfig(
            graph_kind="euclidean",
            min_num_nodes=10,
            max_num_nodes=10,
            batch_size=1,
            dims=[5, 5])


def test_euclidean_rejects_dim_less_than_one(generator):
    with pytest.raises((ValueError, RuntimeError)):
        generator.GeneratorConfig(
            graph_kind="euclidean",
            min_num_nodes=10,
            max_num_nodes=10,
            batch_size=1,
            dim=0)


def test_euclidean_rejects_min_greater_than_max_edge_length(generator):
    with pytest.raises((ValueError, RuntimeError)):
        generator.GeneratorConfig(
            graph_kind="euclidean",
            min_num_nodes=10,
            max_num_nodes=10,
            batch_size=1,
            min_edge_length=0.5,
            max_edge_length=0.1)


def test_euclidean_rejects_negative_min_edge_length(generator):
    with pytest.raises((ValueError, RuntimeError)):
        generator.GeneratorConfig(
            graph_kind="euclidean",
            min_num_nodes=10,
            max_num_nodes=10,
            batch_size=1,
            min_edge_length=-0.1)


# ---- End-to-end with a task on top --------------------------------------

def test_euclidean_end_to_end_bfs_task(generator):
    # Spot-check that the euclidean sampler composes with the rest of
    # the pipeline (BFS task, tokenizer). This is not exhaustive; the
    # er_*_end_to_end tests cover the surrounding stages in depth.
    cfg = generator.GeneratorConfig(
        graph_kind="euclidean",
        task_kind="bfs",
        scratchpad_kind="none",
        tokenization_mode="sean",
        min_num_nodes=12,
        max_num_nodes=12,
        dim=2,
        max_edge_length=0.6,   # ensure connected enough for BFS
        batch_size=4,
        max_attempts=100)
    w = _make_worker(generator, seed=0)
    result = w.generate_batch(cfg)
    assert result["src_tokens"].shape[0] == 4
    assert (result["num_nodes"] == 12).all()
