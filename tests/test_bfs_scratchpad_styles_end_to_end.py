"""Exercises all four bfs_scratchpad_style variants end-to-end through
the pybind pipeline. Each style must produce a self-consistent batch:
scratchpad_len matches the section content, positions stay within
seq_len, and switching styles actually changes the output.
"""
from __future__ import annotations

import numpy as np
import pytest

from get_generator_module import get_generator_module


TOK_BFS_ADJ_START  = 18
TOK_BFS_ADJ_END    = 19
TOK_CURLY_START    = 20
TOK_CURLY_END      = 21
NUM_SPECIAL_DEFAULT = 26


@pytest.fixture(scope="module")
def generator():
    get_generator_module()
    import generator as g  # noqa: WPS433
    return g


def _cfg(generator, style, **overrides):
    kwargs = dict(
        graph_kind="erdos_renyi",
        task_kind="shortest_path",
        scratchpad_kind="bfs",
        bfs_scratchpad_style=style,
        tokenization_mode="sean",
        min_num_nodes=10,
        max_num_nodes=10,
        edge_prob=0.45,
        min_path_length=2,
        max_path_length=6,
        max_attempts=1000,
        batch_size=4)
    kwargs.update(overrides)
    return generator.GeneratorConfig(**kwargs)


def _worker(generator, seed=42):
    ctx = generator.WorkerSharedContext()
    return generator.Worker(ctx, seed=seed)


STYLES = ["plain", "with_queue", "reverse_adjacency", "duplicate_adjacency"]


@pytest.mark.parametrize("style", STYLES)
def test_style_produces_valid_batch(generator, style):
    """Every style must produce a batch where scratchpad indices are
    populated and the section fits inside src_lengths."""
    result = _worker(generator).generate_batch(_cfg(generator, style))
    src_lengths = result["src_lengths"]
    starts  = result["scratchpad_start_indices"]
    lengths = result["scratchpad_lengths"]
    for b in range(len(starts)):
        assert int(lengths[b]) > 0
        assert int(starts[b]) + int(lengths[b]) <= int(src_lengths[b])


@pytest.mark.parametrize("style", STYLES)
def test_style_deterministic_under_fixed_seed(generator, style):
    cfg = _cfg(generator, style)
    a = _worker(generator, seed=7).generate_batch(cfg)
    b = _worker(generator, seed=7).generate_batch(cfg)
    for key in a.keys():
        np.testing.assert_array_equal(a[key], b[key])


def test_style_variants_produce_distinct_scratchpad_lengths(generator):
    """Style choice must actually change the tokenized output. Plain
    and ReverseAdjacency have the SAME length (only order differs);
    the other two are strictly longer."""
    lens = {}
    for style in STYLES:
        r = _worker(generator, seed=13).generate_batch(_cfg(generator, style))
        lens[style] = int(r["scratchpad_lengths"][0])
    assert lens["plain"] == lens["reverse_adjacency"]
    assert lens["duplicate_adjacency"] > lens["plain"]
    assert lens["with_queue"] > lens["plain"]


def test_reverse_adjacency_reverses_neighbour_order(generator):
    """Plain vs Reverse: same length, but the neighbour lists inside
    each `[ ... ]` block are the reverse of each other."""
    cfg_plain   = _cfg(generator, "plain",              batch_size=1)
    cfg_reverse = _cfg(generator, "reverse_adjacency",  batch_size=1)
    plain   = _worker(generator, seed=99).generate_batch(cfg_plain)
    reverse = _worker(generator, seed=99).generate_batch(cfg_reverse)

    s = int(plain["scratchpad_start_indices"][0])
    L = int(plain["scratchpad_lengths"][0])
    assert L == int(reverse["scratchpad_lengths"][0])

    # Extract each `[ nbrs ]` block from Plain and check the same
    # positions in Reverse have the reversed vertex sequence.
    ps = plain["src_tokens"][0]
    rs = reverse["src_tokens"][0]
    i = s
    end = s + L
    while i < end:
        # skip vertex head (identical between plain and reverse)
        assert int(ps[i]) == int(rs[i])
        assert int(ps[i]) >= NUM_SPECIAL_DEFAULT
        i += 1
        assert int(ps[i]) == TOK_BFS_ADJ_START
        assert int(rs[i]) == TOK_BFS_ADJ_START
        i += 1
        # collect neighbour ids in each ordering
        p_nbrs = []
        r_nbrs = []
        while int(ps[i]) != TOK_BFS_ADJ_END:
            p_nbrs.append(int(ps[i]))
            r_nbrs.append(int(rs[i]))
            i += 1
        assert p_nbrs == list(reversed(r_nbrs))
        assert int(ps[i]) == TOK_BFS_ADJ_END
        i += 1


def test_duplicate_adjacency_has_two_bracket_kinds_per_vertex(generator):
    """DuplicateAdjacency emits neighbours in `[ ]` then again in `{ }`;
    every vertex head is followed by exactly one of each bracket pair."""
    cfg = _cfg(generator, "duplicate_adjacency", batch_size=1)
    r = _worker(generator).generate_batch(cfg)
    s = int(r["scratchpad_start_indices"][0])
    L = int(r["scratchpad_lengths"][0])
    slice_ = r["src_tokens"][0, s : s + L]
    adj_starts   = int(np.sum(slice_ == TOK_BFS_ADJ_START))
    adj_ends     = int(np.sum(slice_ == TOK_BFS_ADJ_END))
    curly_starts = int(np.sum(slice_ == TOK_CURLY_START))
    curly_ends   = int(np.sum(slice_ == TOK_CURLY_END))
    assert adj_starts   == adj_ends
    assert curly_starts == curly_ends
    # One `[` and one `{` per visited vertex.
    assert adj_starts == curly_starts


def test_with_queue_has_curly_prefix_before_each_vertex(generator):
    """WithQueue prefixes each per-vertex block with `{ q } `. So the
    number of `{` markers equals the number of visited vertices."""
    cfg = _cfg(generator, "with_queue", batch_size=1)
    r = _worker(generator).generate_batch(cfg)
    s = int(r["scratchpad_start_indices"][0])
    L = int(r["scratchpad_lengths"][0])
    slice_ = r["src_tokens"][0, s : s + L]
    adj_starts   = int(np.sum(slice_ == TOK_BFS_ADJ_START))
    curly_starts = int(np.sum(slice_ == TOK_CURLY_START))
    assert curly_starts == adj_starts  # one queue block per adj block
