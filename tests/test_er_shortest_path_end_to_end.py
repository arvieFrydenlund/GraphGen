"""End-to-end token-content invariants for (ER, ShortestPath, None).

Runs Worker.generate_batch for a matrix of seeds and configs, then
verifies the produced src_tokens obey the SEAN layout for shortest
path: boundary markers are in the right positions, the edge section is
a run of (u, v, EDGE) triples, the query section carries two vertex
tokens, the target section is a run of vertex tokens, padding past
src_length holds TOK_PAD, and every non-marker token is either a
special or lives in the graph-vocab slice [num_special, num_special +
max_num_nodes).
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
        task_kind="shortest_path",
        scratchpad_kind="none",
        tokenization_mode="sean",
        min_num_nodes=15,
        max_num_nodes=15,
        edge_prob=0.35,
        min_path_length=2,
        max_path_length=6,
        max_attempts=1000,
        batch_size=8)
    kwargs.update(overrides)
    return generator.GeneratorConfig(**kwargs)


def _worker(generator, seed=0):
    ctx = generator.WorkerSharedContext()
    return generator.Worker(ctx, seed=seed)


def test_bos_and_eos_bracket_each_row(generator):
    result = _worker(generator).generate_batch(_cfg(generator))
    src         = result["src_tokens"]
    src_lengths = result["src_lengths"]
    for b in range(src.shape[0]):
        L = int(src_lengths[b])
        assert src[b, 0]     == TOK_BOS
        assert src[b, L - 1] == TOK_EOS


def test_edge_section_is_uv_edge_triples(generator):
    result = _worker(generator).generate_batch(_cfg(generator))
    src              = result["src_tokens"]
    edge_starts      = result["graph_edge_start_indices"]
    edge_lengths     = result["graph_edge_lengths"]
    num_edges        = result["num_edges"]

    for b in range(src.shape[0]):
        s = int(edge_starts[b])
        L = int(edge_lengths[b])
        m = int(num_edges[b])
        assert L == 3 * m, f"row {b}: edge_len should be 3*num_edges"
        # Every third slot is TOK_EDGE; the other two are graph-vocab.
        for i in range(m):
            u = src[b, s + 3 * i + 0]
            v = src[b, s + 3 * i + 1]
            e = src[b, s + 3 * i + 2]
            assert e == TOK_EDGE, f"row {b}, edge {i}: expected EDGE at slot 2"
            assert u >= NUM_SPECIAL_DEFAULT, f"row {b}, edge {i}: u is a special token"
            assert v >= NUM_SPECIAL_DEFAULT, f"row {b}, edge {i}: v is a special token"


def test_query_section_bracketed_by_markers_holds_two_vocab_tokens(generator):
    result = _worker(generator).generate_batch(_cfg(generator))
    src           = result["src_tokens"]
    query_starts  = result["query_start_indices"]
    query_lengths = result["query_lengths"]

    for b in range(src.shape[0]):
        s = int(query_starts[b])
        L = int(query_lengths[b])
        assert L == 2, f"row {b}: query_len should be 2 (start,end)"
        # QUERY_START marker sits at s-1; QUERY_END at s+L.
        assert src[b, s - 1]     == TOK_QUERY_START
        assert src[b, s + L]     == TOK_QUERY_END
        assert src[b, s]         >= NUM_SPECIAL_DEFAULT
        assert src[b, s + 1]     >= NUM_SPECIAL_DEFAULT


def test_task_target_section_is_vocab_tokens_bracketed_by_markers(generator):
    result = _worker(generator).generate_batch(_cfg(generator))
    src           = result["src_tokens"]
    task_starts   = result["task_start_indices"]
    task_lengths  = result["task_lengths"]

    for b in range(src.shape[0]):
        s = int(task_starts[b])
        L = int(task_lengths[b])
        assert L >= 2, f"row {b}: task_len should be >= 2 (min_path_length=2 → k+1=3)"
        assert src[b, s - 1]  == TOK_TASK_START
        assert src[b, s + L]  == TOK_TASK_END
        for i in range(L):
            token = int(src[b, s + i])
            assert token >= NUM_SPECIAL_DEFAULT, (
                f"row {b} task pos {i}: token {token} is not a vocab id"
            )


def test_padding_holds_pad_token(generator):
    result = _worker(generator).generate_batch(_cfg(generator))
    src         = result["src_tokens"]
    src_lengths = result["src_lengths"]
    S           = src.shape[1]
    for b in range(src.shape[0]):
        L = int(src_lengths[b])
        if L < S:
            assert (src[b, L:] == TOK_PAD).all()


def test_positional_ids_are_zero_indexed_within_each_row(generator):
    result = _worker(generator).generate_batch(_cfg(generator))
    positions   = result["positions"]
    src_lengths = result["src_lengths"]
    for b in range(positions.shape[0]):
        L = int(src_lengths[b])
        # Real positions carry 0..L-1; padding stays at 0 (least-surprising fill).
        expected = np.arange(L, dtype=positions.dtype)
        np.testing.assert_array_equal(positions[b, :L], expected)


def test_targets_shape_and_baseline_matches_src_gen_region(generator):
    # targets has shape (B, max_gen_len, max_labels). The k=0 slot of
    # each generation-region position must match the src token at the
    # corresponding column (prefix_len_b + gp). Alternatives at k>0
    # are for label smoothing at SP intermediate hops.
    result = _worker(generator).generate_batch(_cfg(generator))
    src            = result["src_tokens"]
    targets        = result["targets"]
    src_lengths    = result["src_lengths"]
    task_starts    = result["task_start_indices"]   # start of target section
    task_lengths   = result["task_lengths"]
    left_pad_lens  = result["left_pad_lengths"]

    assert targets.ndim == 3
    assert targets.shape[0] == src.shape[0]

    for b in range(src.shape[0]):
        # Generation region begins at TaskStart marker (no thinking/
        # scratchpad in this fixture); that column is task_start - 1
        # in src-row coords, but left_pad_lengths offsets everything.
        gen_start = int(task_starts[b]) - 1
        gen_len   = int(src_lengths[b]) + int(left_pad_lens[b]) - gen_start
        for gp in range(gen_len):
            assert targets[b, gp, 0] == src[b, gen_start + gp], (
                f"row {b} gp {gp}: targets baseline diverges from src"
            )


def test_targets_alternatives_are_valid_next_hops(generator):
    # For SP intermediate hops (path positions 1..L), the targets
    # label-smoothing dim collects the valid_next_hops at that step.
    # All alternatives must be legitimate vocab tokens (>= NUM_SPECIAL_DEFAULT).
    # k=0 is the chosen hop (already checked by the baseline test);
    # k>=1 are the OTHER equally-valid choices (or TOK_PAD when a step
    # has fewer alternatives than the batch's max_labels).
    result = _worker(generator).generate_batch(_cfg(generator))
    src            = result["src_tokens"]
    targets        = result["targets"]
    task_starts    = result["task_start_indices"]
    task_lengths   = result["task_lengths"]
    left_pad_lens  = result["left_pad_lengths"]

    max_labels = targets.shape[2]
    for b in range(src.shape[0]):
        # Skip if no smoothing was needed (max_labels can be 1 across
        # the whole batch when every SP has unique next hops).
        if max_labels == 1:
            continue
        s = int(task_starts[b])
        L = int(task_lengths[b])
        gen_start = s - 1
        # For path position i in [1, L), gp = (s + i) - gen_start = i + 1
        for i in range(1, L):
            gp = i + 1
            for k in range(max_labels):
                tok = int(targets[b, gp, k])
                assert tok == TOK_PAD or tok >= NUM_SPECIAL_DEFAULT, (
                    f"row {b} gp {gp} k {k}: unexpected target token {tok}"
                )
            # k=0 must not be TOK_PAD (there's always a chosen hop).
            assert int(targets[b, gp, 0]) != TOK_PAD


def test_deterministic_tokens_under_fixed_seed(generator):
    cfg = _cfg(generator, batch_size=4)
    a = _worker(generator, seed=123).generate_batch(cfg)
    b = _worker(generator, seed=123).generate_batch(cfg)
    for key in a.keys():
        np.testing.assert_array_equal(a[key], b[key])
