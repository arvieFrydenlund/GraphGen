"""End-to-end token-content invariants for (ER, ShortestPath, BFS/Plain).

The BFS scratchpad is the CoT trace for a shortest-path task: run BFS
from the start node, emit `v [ neighbours ]` per visited vertex, then
the target is the actual shortest path. Verifies:
  * scratchpad_start_indices / scratchpad_lengths are populated
  * SCRATCH_START / SCRATCH_END bracket the scratchpad content
  * Each per-vertex block has shape `v [ n_1 ... n_deg(v) ]`
  * TASK_START / TASK_END bracket the shortest-path target
  * Determinism under fixed seed
"""
from __future__ import annotations

import numpy as np
import pytest

from get_generator_module import get_generator_module


# Token constants -- mirror WorkerSharedContext static constexpr values.
TOK_BOS            = 0
TOK_PAD            = 1
TOK_EOS            = 2
TOK_EDGE           = 4
TOK_TASK_START     = 6
TOK_TASK_END       = 7
TOK_QUERY_START    = 13
TOK_QUERY_END      = 14
TOK_SCRATCH_START  = 16
TOK_SCRATCH_END    = 17
TOK_BFS_ADJ_START  = 18
TOK_BFS_ADJ_END    = 19
NUM_SPECIAL_DEFAULT = 26


@pytest.fixture(scope="module")
def generator():
    get_generator_module()
    import generator as g  # noqa: WPS433
    return g


def _cfg(generator, **overrides):
    kwargs = dict(
        graph_kind="erdos_renyi",
        task_kind="shortest_path",
        scratchpad_kind="bfs",
        tokenization_mode="sean",
        min_num_nodes=10,
        max_num_nodes=10,
        edge_prob=0.45,
        min_path_length=2,
        max_path_length=6,
        max_attempts=1000,
        batch_size=6)
    kwargs.update(overrides)
    return generator.GeneratorConfig(**kwargs)


def _worker(generator, seed=0):
    ctx = generator.WorkerSharedContext()
    return generator.Worker(ctx, seed=seed)


def test_scratchpad_indices_populated(generator):
    result = _worker(generator).generate_batch(_cfg(generator))
    starts  = result["scratchpad_start_indices"]
    lengths = result["scratchpad_lengths"]
    for b in range(len(starts)):
        assert int(starts[b])  > 0
        assert int(lengths[b]) > 0


def test_scratchpad_bracketed_by_scratch_start_and_scratch_end(generator):
    result = _worker(generator).generate_batch(_cfg(generator))
    src     = result["src_tokens"]
    starts  = result["scratchpad_start_indices"]
    lengths = result["scratchpad_lengths"]
    for b in range(src.shape[0]):
        s = int(starts[b])
        L = int(lengths[b])
        # SCRATCH_START sits at s-1; SCRATCH_END right after content.
        assert src[b, s - 1] == TOK_SCRATCH_START
        assert src[b, s + L] == TOK_SCRATCH_END


def test_scratchpad_end_precedes_task_start(generator):
    # Symmetric bracketing means TASK_START comes AFTER SCRATCH_END,
    # not merged with it.
    result = _worker(generator).generate_batch(_cfg(generator))
    src         = result["src_tokens"]
    starts      = result["scratchpad_start_indices"]
    lengths     = result["scratchpad_lengths"]
    task_starts = result["task_start_indices"]
    for b in range(src.shape[0]):
        s   = int(starts[b])
        L   = int(lengths[b])
        end = s + L
        assert src[b, end]      == TOK_SCRATCH_END
        assert src[b, end + 1]  == TOK_TASK_START
        assert int(task_starts[b]) == end + 2  # first content token after TASK_START


def test_query_carries_start_and_end_vertices(generator):
    result = _worker(generator).generate_batch(_cfg(generator))
    src           = result["src_tokens"]
    query_starts  = result["query_start_indices"]
    query_lengths = result["query_lengths"]
    for b in range(src.shape[0]):
        # SP query is (start, end) -- 2 vertex tokens.
        assert int(query_lengths[b]) == 2
        q = int(query_starts[b])
        assert int(src[b, q])     >= NUM_SPECIAL_DEFAULT
        assert int(src[b, q + 1]) >= NUM_SPECIAL_DEFAULT


def test_scratchpad_content_is_v_lbracket_neighbours_rbracket(generator):
    # Parse the scratchpad section by scanning for `(v, [, nbrs..., ])`
    # triples. Count them and infer sum_deg from the total vertex-token
    # count.
    result = _worker(generator).generate_batch(_cfg(generator))
    src     = result["src_tokens"]
    starts  = result["scratchpad_start_indices"]
    lengths = result["scratchpad_lengths"]
    for b in range(src.shape[0]):
        s   = int(starts[b])
        L   = int(lengths[b])
        end = s + L
        i = s
        n_visited = 0
        while i < end:
            # vertex head
            assert int(src[b, i]) >= NUM_SPECIAL_DEFAULT
            i += 1
            assert int(src[b, i]) == TOK_BFS_ADJ_START
            i += 1
            # neighbours
            while i < end and int(src[b, i]) != TOK_BFS_ADJ_END:
                assert int(src[b, i]) >= NUM_SPECIAL_DEFAULT
                i += 1
            assert i < end and int(src[b, i]) == TOK_BFS_ADJ_END
            i += 1
            n_visited += 1
        assert n_visited >= 1
        # scratchpad_len == 3 * n_visited + sum_deg
        slice_ = src[b, s : s + L]
        vertex_toks = int(np.sum(slice_ >= NUM_SPECIAL_DEFAULT))
        sum_deg = vertex_toks - n_visited
        assert L == 3 * n_visited + sum_deg


def test_target_is_shortest_path_bracketed_by_task_markers(generator):
    result = _worker(generator).generate_batch(_cfg(generator))
    src         = result["src_tokens"]
    task_starts = result["task_start_indices"]
    task_lens   = result["task_lengths"]
    for b in range(src.shape[0]):
        t = int(task_starts[b])
        k = int(task_lens[b])
        assert src[b, t - 1]     == TOK_TASK_START
        assert src[b, t + k]     == TOK_TASK_END
        # All target tokens are vertex tokens (shortest-path vertices).
        for j in range(k):
            assert int(src[b, t + j]) >= NUM_SPECIAL_DEFAULT


def test_sections_stay_inside_src_lengths(generator):
    result = _worker(generator).generate_batch(_cfg(generator))
    src_lengths = result["src_lengths"]
    for start_key, len_key in [
        ("graph_edge_start_indices", "graph_edge_lengths"),
        ("query_start_indices",      "query_lengths"),
        ("scratchpad_start_indices", "scratchpad_lengths"),
        ("task_start_indices",       "task_lengths"),
    ]:
        ends = result[start_key] + result[len_key]
        assert (ends <= src_lengths).all()


def test_deterministic_under_fixed_seed(generator):
    cfg = _cfg(generator, batch_size=4)
    a = _worker(generator, seed=777).generate_batch(cfg)
    b = _worker(generator, seed=777).generate_batch(cfg)
    for key in a.keys():
        np.testing.assert_array_equal(a[key], b[key])
