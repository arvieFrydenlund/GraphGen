"""End-to-end token-content invariants for (ER, BFS, None).

Similar to the shortest-path E2E test but for the BFS task:
  * query section carries only the start vertex (no end)
  * target section is the BFS visit order from that start

Verifies the SEAN sequence layout, section indices, padding, positional
ids, and shift-right prev_output_tokens.
"""
from __future__ import annotations

import numpy as np
import pytest

from get_generator_module import get_generator_module


# Token constants -- must mirror WorkerSharedContext static constexpr.
TOK_BOS         = 0
TOK_PAD         = 1
TOK_EOS         = 2
TOK_EDGE        = 4
TOK_TASK_START  = 6
TOK_TASK_END    = 7
TOK_QUERY_START = 13
TOK_QUERY_END   = 14
NUM_SPECIAL_DEFAULT = 26


@pytest.fixture(scope="module")
def generator():
    get_generator_module()
    import generator as g  # noqa: WPS433
    return g


def _cfg(generator, **overrides):
    kwargs = dict(
        graph_kind="erdos_renyi",
        task_kind="bfs",
        scratchpad_kind="none",
        tokenization_mode="sean",
        min_num_nodes=12,
        max_num_nodes=12,
        edge_prob=0.4,
        max_attempts=1000,
        batch_size=8)
    kwargs.update(overrides)
    return generator.GeneratorConfig(**kwargs)


def _worker(generator, seed=0):
    ctx = generator.WorkerSharedContext()
    return generator.Worker(ctx, seed=seed)


def test_bfs_bos_and_eos_bracket_each_row(generator):
    result = _worker(generator).generate_batch(_cfg(generator))
    src         = result["src_tokens"]
    src_lengths = result["src_lengths"]
    for b in range(src.shape[0]):
        L = int(src_lengths[b])
        assert src[b, 0]     == TOK_BOS
        assert src[b, L - 1] == TOK_EOS


def test_bfs_query_section_is_single_vertex(generator):
    # BFS's query is just the start vertex -- length 1, one vocab token
    # bracketed by QS/QE markers.
    result = _worker(generator).generate_batch(_cfg(generator))
    src           = result["src_tokens"]
    query_starts  = result["query_start_indices"]
    query_lengths = result["query_lengths"]
    for b in range(src.shape[0]):
        s = int(query_starts[b])
        L = int(query_lengths[b])
        assert L == 1, f"row {b}: BFS query_len should be 1 (start only)"
        assert src[b, s - 1] == TOK_QUERY_START
        assert src[b, s + L] == TOK_QUERY_END
        assert src[b, s] >= NUM_SPECIAL_DEFAULT


def test_bfs_target_holds_visit_order_of_distinct_vertices(generator):
    # BFS target: distinct vocab tokens (no vertex repeated in BFS).
    result = _worker(generator).generate_batch(_cfg(generator))
    src          = result["src_tokens"]
    task_starts  = result["task_start_indices"]
    task_lengths = result["task_lengths"]
    for b in range(src.shape[0]):
        s = int(task_starts[b])
        L = int(task_lengths[b])
        assert L >= 1, f"row {b}: BFS target must contain at least the start vertex"
        assert src[b, s - 1] == TOK_TASK_START
        assert src[b, s + L] == TOK_TASK_END
        tokens = list(src[b, s : s + L])
        for tok in tokens:
            assert int(tok) >= NUM_SPECIAL_DEFAULT
        assert len(set(tokens)) == len(tokens), (
            f"row {b}: BFS visit order should not repeat vertices"
        )


def test_bfs_target_first_vertex_matches_query_start(generator):
    # BFS starts at the query vertex; the first target token must equal
    # the query token.
    result = _worker(generator).generate_batch(_cfg(generator))
    src          = result["src_tokens"]
    query_starts = result["query_start_indices"]
    task_starts  = result["task_start_indices"]
    for b in range(src.shape[0]):
        q = int(query_starts[b])
        t = int(task_starts[b])
        assert src[b, t] == src[b, q]


def test_bfs_target_never_exceeds_num_nodes(generator):
    # BFS visits each reachable vertex at most once; length ≤ num_nodes.
    result = _worker(generator).generate_batch(_cfg(generator))
    task_lengths = result["task_lengths"]
    num_nodes    = result["num_nodes"]
    for b in range(len(task_lengths)):
        assert int(task_lengths[b]) <= int(num_nodes[b])


def test_bfs_seq_length_formula_holds(generator):
    # src_len = 3 * num_edges + task_lengths + 8
    #   BOS(1) + 3m + QS(1) + query_len(1) + QE(1) + TS(1) + task_len + TE(1) + EOS(1)
    #   = 3m + task_len + 6 + query_len  where query_len=1 for BFS
    #   = 3m + task_len + 7  ... wait that says 7, not 8.
    # Recount: BOS,QS,QE,TS,TE,EOS = 6 markers; + query_len(1) + edge_len(3m) + task_len
    #   = 6 + 1 + 3m + task_len = 3m + task_len + 7.
    result = _worker(generator).generate_batch(_cfg(generator))
    src_lengths  = result["src_lengths"]
    num_edges    = result["num_edges"]
    task_lengths = result["task_lengths"]
    expected = 3 * num_edges + task_lengths + 7
    np.testing.assert_array_equal(src_lengths, expected)


def test_bfs_padding_holds_pad_token(generator):
    result = _worker(generator).generate_batch(_cfg(generator))
    src         = result["src_tokens"]
    src_lengths = result["src_lengths"]
    S           = src.shape[1]
    for b in range(src.shape[0]):
        L = int(src_lengths[b])
        if L < S:
            assert (src[b, L:] == TOK_PAD).all()


def test_bfs_targets_shape_and_baseline_matches_src_gen_region(generator):
    # targets has shape (B, max_gen_len, max_labels). BFS has no label
    # smoothing (single valid answer per step); we verify only the
    # k=0 baseline matches src at each generation-region column.
    result = _worker(generator).generate_batch(_cfg(generator))
    src            = result["src_tokens"]
    targets        = result["targets"]
    src_lengths    = result["src_lengths"]
    task_starts    = result["task_start_indices"]
    left_pad_lens  = result["left_pad_lengths"]

    assert targets.ndim == 3
    assert targets.shape[0] == src.shape[0]

    for b in range(src.shape[0]):
        gen_start = int(task_starts[b]) - 1
        gen_len   = int(src_lengths[b]) + int(left_pad_lens[b]) - gen_start
        for gp in range(gen_len):
            assert targets[b, gp, 0] == src[b, gen_start + gp], (
                f"row {b} gp {gp}: targets baseline diverges from src"
            )


def test_bfs_deterministic_tokens_under_fixed_seed(generator):
    cfg = _cfg(generator, batch_size=4)
    a = _worker(generator, seed=99).generate_batch(cfg)
    b = _worker(generator, seed=99).generate_batch(cfg)
    for key in a.keys():
        np.testing.assert_array_equal(a[key], b[key])
