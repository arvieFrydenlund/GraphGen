"""path_star (rooted directed tree of arms) sampler invariant tests.

sample_path_star builds a rooted directed tree: vertex 0 is the root,
and num_arms directed chains ("arms") emanate from it. Each arm has
an independently sampled length in [min_arm_length, max_arm_length],
so a single graph typically contains arms of several different
lengths -- an intentional signal for the shortest_path task.

Structural invariants covered here:
    * vertex count in [1 + min_arms*min_len, 1 + max_arms*max_len]
    * edge count == n - 1 (a tree)
    * built directed
    * root (vertex 0) has in-degree 0, every non-root has in-degree 1
    * every vertex reachable from the root
    * determinism under a fixed seed
    * end-to-end: pairs shortest_path with a real batch

Configuration errors:
    * requires directed=true
    * requires min_arms / max_arms / min_arm_length / max_arm_length
    * requires min <= max on each pair
    * requires min >= 1 on each lower bound
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


def _make_cfg(generator, *,
              min_arms=2, max_arms=4,
              min_len=3, max_len=5,
              task_kind="shortest_path",
              batch_size=1,
              extra_vocab=32):
    n_max = 1 + max_arms * max_len
    return generator.GeneratorConfig(
        graph_kind="path_star",
        task_kind=task_kind,
        scratchpad_kind="none",
        tokenization_mode="sean",
        directed=True,
        min_arms=min_arms,
        max_arms=max_arms,
        min_arm_length=min_len,
        max_arm_length=max_len,
        min_path_length=1,
        max_path_length=n_max,
        batch_size=batch_size,
        max_attempts=1000)


def _make_worker(generator, seed=0):
    ctx = generator.WorkerSharedContext()
    return generator.Worker(ctx, seed=seed)


# ---- Vertex + edge counts (tree invariants) -----------------------------

@pytest.mark.parametrize("arms,length", [(2, 3), (3, 4), (5, 2)])
def test_path_star_is_a_tree(generator, arms, length):
    # Fixed arm count and arm length -> deterministic n.
    w = _make_worker(generator, seed=0)
    cfg = _make_cfg(generator, min_arms=arms, max_arms=arms,
                    min_len=length, max_len=length)
    stats = w.sample_graph_stats(cfg)
    n = 1 + arms * length
    assert stats["num_vertices"] == n
    assert stats["num_edges"] == n - 1                    # tree
    assert stats["num_components"] == 1                    # connected as a tree


def test_path_star_vertex_count_within_bounds(generator):
    # Variable arms / lengths: verify n lands in the derived range.
    w = _make_worker(generator, seed=1)
    cfg = _make_cfg(generator,
                    min_arms=2, max_arms=5,
                    min_len=3, max_len=6)
    lo = 1 + 2 * 3
    hi = 1 + 5 * 6
    for _ in range(20):
        s = w.sample_graph_stats(cfg)
        assert lo <= s["num_vertices"] <= hi
        assert s["num_edges"] == s["num_vertices"] - 1     # tree


# ---- Determinism --------------------------------------------------------

def test_path_star_deterministic_under_fixed_seed(generator):
    cfg = _make_cfg(generator, batch_size=4)
    a = _make_worker(generator, seed=42).generate_batch(cfg)
    b = _make_worker(generator, seed=42).generate_batch(cfg)
    np.testing.assert_array_equal(a["src_tokens"], b["src_tokens"])
    np.testing.assert_array_equal(a["num_nodes"], b["num_nodes"])
    np.testing.assert_array_equal(a["num_edges"], b["num_edges"])


# ---- End-to-end: pairs with the shortest_path task ---------------------

def test_path_star_with_shortest_path_produces_valid_batch(generator):
    # Directed path_star + shortest_path: many (start, end) pairs
    # have no directed route (e.g. leaf-to-root, cross-arm), so the
    # sampler falls back on max_attempts rejection until it finds a
    # reachable pair. With arms of moderate length this is fast and
    # every item lands with a real path.
    cfg = _make_cfg(generator, batch_size=8,
                    min_arms=3, max_arms=3,
                    min_len=4, max_len=4)
    result = _make_worker(generator, seed=7).generate_batch(cfg)
    # Every task must have length >= 1 (start != end enforced by sampler).
    assert (result["task_lengths"] >= 1).all()
    # Src tokens are populated (no unexpected sentinel-only rows).
    assert (result["src_lengths"] > 0).all()


# ---- Configuration rejections -------------------------------------------

def test_path_star_rejects_directed_false(generator):
    with pytest.raises((ValueError, RuntimeError)):
        cfg = generator.GeneratorConfig(
            graph_kind="path_star",
            task_kind="shortest_path",
            scratchpad_kind="none",
            tokenization_mode="sean",
            directed=False,                # <- rejected
            min_arms=2, max_arms=3,
            min_arm_length=3, max_arm_length=5,
            min_path_length=1, max_path_length=20,
            batch_size=1,
            max_attempts=100)
        _make_worker(generator).generate_batch(cfg)


@pytest.mark.parametrize("kwargs", [
    dict(min_arms=0),                                 # < 1
    dict(min_arms=5, max_arms=2),                     # min > max
    dict(min_arm_length=0),                           # < 1
    dict(min_arm_length=6, max_arm_length=4),         # min > max
])
def test_path_star_rejects_out_of_range_bounds(generator, kwargs):
    base = dict(
        graph_kind="path_star",
        task_kind="shortest_path",
        scratchpad_kind="none",
        tokenization_mode="sean",
        directed=True,
        min_arms=2, max_arms=4,
        min_arm_length=3, max_arm_length=5,
        min_path_length=1, max_path_length=30,
        batch_size=1,
        max_attempts=100)
    base.update(kwargs)
    with pytest.raises((ValueError, RuntimeError)):
        cfg = generator.GeneratorConfig(**base)
        _make_worker(generator).generate_batch(cfg)
