"""End-to-end tests for num_thinking_tokens.

Thinking is a section of `num_thinking_tokens` repeated TOK_THINK
tokens at the START of the generation region (before scratchpad if
present, before target otherwise). Bracketed by TOK_THINK_START /
TOK_THINK_END markers.

When num_thinking_tokens > 0, the "generation start" that
align_prefix_front_pad aligns on shifts to the ThinkStart marker.
"""
from __future__ import annotations

import numpy as np
import pytest

from get_generator_module import get_generator_module


TOK_PAD           = 1
TOK_THINK         = 5
TOK_TASK_START    = 6
TOK_TASK_END      = 7
TOK_QUERY_START   = 13
TOK_QUERY_END     = 14
TOK_SCRATCH_START = 16
TOK_SCRATCH_END   = 17
TOK_THINK_START   = 24
TOK_THINK_END     = 25
NUM_SPECIAL_DEFAULT = 26


@pytest.fixture(scope="module")
def generator():
    get_generator_module()
    import generator as g  # noqa: WPS433
    return g


def _cfg(generator, *, thinking, scratchpad="none", align=False, **overrides):
    kwargs = dict(
        graph_kind="erdos_renyi",
        task_kind="shortest_path",
        scratchpad_kind=scratchpad,
        tokenization_mode="sean",
        num_thinking_tokens=thinking,
        align_prefix_front_pad=align,
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


def test_zero_thinking_leaves_thinking_indices_zero(generator):
    result = _worker(generator).generate_batch(_cfg(generator, thinking=0))
    assert (result["thinking_start_indices"] == 0).all()
    assert (result["thinking_lengths"] == 0).all()


def test_thinking_section_bracketed_by_think_start_and_end(generator):
    N = 5
    result = _worker(generator).generate_batch(_cfg(generator, thinking=N))
    src = result["src_tokens"]
    starts  = result["thinking_start_indices"]
    lengths = result["thinking_lengths"]
    for b in range(src.shape[0]):
        s = int(starts[b])
        L = int(lengths[b])
        assert L == N
        # THINK_START at s-1; content is N TOK_THINK tokens; THINK_END at s+L.
        assert int(src[b, s - 1]) == TOK_THINK_START
        for i in range(L):
            assert int(src[b, s + i]) == TOK_THINK
        assert int(src[b, s + L])     == TOK_THINK_END


def test_thinking_precedes_scratchpad(generator):
    N = 4
    result = _worker(generator).generate_batch(
        _cfg(generator, thinking=N, scratchpad="bfs"))
    # Order: BOS edges QS Q QE THINK_START Thinking THINK_END SS ...
    # So thinking_start_indices < scratchpad_start_indices for every row.
    for b in range(result["src_tokens"].shape[0]):
        t_start = int(result["thinking_start_indices"][b])
        s_start = int(result["scratchpad_start_indices"][b])
        assert t_start < s_start
        # And THINK_END immediately precedes SCRATCH_START.
        src = result["src_tokens"]
        t_end_col = t_start + int(result["thinking_lengths"][b])
        assert int(src[b, t_end_col])     == TOK_THINK_END
        assert int(src[b, t_end_col + 1]) == TOK_SCRATCH_START


def test_thinking_precedes_task_when_no_scratchpad(generator):
    N = 3
    result = _worker(generator).generate_batch(_cfg(generator, thinking=N))
    for b in range(result["src_tokens"].shape[0]):
        t_start = int(result["thinking_start_indices"][b])
        tgt_start = int(result["task_start_indices"][b])
        assert t_start < tgt_start
        src = result["src_tokens"]
        t_end_col = t_start + int(result["thinking_lengths"][b])
        assert int(src[b, t_end_col])     == TOK_THINK_END
        assert int(src[b, t_end_col + 1]) == TOK_TASK_START


def test_thinking_moves_align_pivot_to_think_start(generator):
    """With align mode, the pivot column should now be THINK_START
    (not TASK_START) since thinking is the first thing generated."""
    N = 5
    result = _worker(generator).generate_batch(
        _cfg(generator, thinking=N, align=True))
    # Pivot = thinking_start - 1 (THINK_START marker column). Should
    # be the same for every row.
    pivot = result["thinking_start_indices"] - 1
    assert (pivot == pivot[0]).all()
    src = result["src_tokens"]
    for b in range(src.shape[0]):
        assert int(src[b, int(pivot[b])]) == TOK_THINK_START


def test_thinking_adds_expected_positions_to_seq_len(generator):
    """Enabling N thinking tokens grows every row's seq_len by N + 2
    (N content + THINK_START + THINK_END markers)."""
    N = 7
    no_thinking = _worker(generator, seed=1).generate_batch(
        _cfg(generator, thinking=0))
    with_thinking = _worker(generator, seed=1).generate_batch(
        _cfg(generator, thinking=N))
    delta = with_thinking["src_lengths"] - no_thinking["src_lengths"]
    np.testing.assert_array_equal(delta, N + 2)


def test_thinking_deterministic_under_fixed_seed(generator):
    cfg = _cfg(generator, thinking=6)
    a = _worker(generator, seed=42).generate_batch(cfg)
    b = _worker(generator, seed=42).generate_batch(cfg)
    for key in a.keys():
        np.testing.assert_array_equal(a[key], b[key])
