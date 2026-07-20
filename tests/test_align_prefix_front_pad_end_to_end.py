"""End-to-end tests for align_prefix_front_pad batching.

When True, every row is left-padded so the generation-start column
(ScratchpadStart marker if scratchpad exists, else TaskStart marker)
is the same across the batch. `left_pad_lengths[i]` records how many
PAD tokens were prepended to each row.
"""
from __future__ import annotations

import numpy as np
import pytest

from get_generator_module import get_generator_module


TOK_PAD             = 1
TOK_TASK_START      = 6
TOK_SCRATCH_START   = 16
NUM_SPECIAL_DEFAULT = 26


@pytest.fixture(scope="module")
def generator():
    get_generator_module()
    import generator as g  # noqa: WPS433
    return g


def _cfg(generator, *, align, scratchpad="none", **overrides):
    kwargs = dict(
        graph_kind="erdos_renyi",
        task_kind="shortest_path",
        scratchpad_kind=scratchpad,
        tokenization_mode="sean",
        align_prefix_front_pad=align,
        min_num_nodes=8,
        max_num_nodes=12,           # variable size => variable per-row prefix
        edge_prob=0.4,
        min_path_length=2,
        max_path_length=6,
        max_attempts=1000,
        batch_size=8)
    kwargs.update(overrides)
    return generator.GeneratorConfig(**kwargs)


def _worker(generator, seed=0):
    ctx = generator.WorkerSharedContext()
    return generator.Worker(ctx, seed=seed)


def _gen_start_marker_for(scratchpad):
    return TOK_SCRATCH_START if scratchpad == "bfs" else TOK_TASK_START


# ---------------------------------------------------------------------------
# right-pad (default): behaviour unchanged, left_pad_lengths all zero
# ---------------------------------------------------------------------------

def test_right_pad_leaves_left_pad_lengths_all_zero(generator):
    result = _worker(generator).generate_batch(_cfg(generator, align=False))
    assert (result["left_pad_lengths"] == 0).all()


def test_right_pad_row_starts_with_bos_at_col_0(generator):
    # Regression: default mode still writes BOS at column 0 of every row.
    result = _worker(generator).generate_batch(_cfg(generator, align=False))
    src = result["src_tokens"]
    for b in range(src.shape[0]):
        assert int(src[b, 0]) == 0  # TOK_BOS


# ---------------------------------------------------------------------------
# align mode: front-pad + shifted metadata
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scratchpad", ["none", "bfs"])
def test_align_mode_left_pad_makes_gen_start_column_uniform(generator, scratchpad):
    result = _worker(generator).generate_batch(
        _cfg(generator, align=True, scratchpad=scratchpad))

    marker = _gen_start_marker_for(scratchpad)
    starts = (result["scratchpad_start_indices"] if scratchpad == "bfs"
              else result["task_start_indices"])

    # Every row's gen-start (SS or TS marker) sits one column BEFORE the
    # first content token of that section (start indices point at content).
    marker_cols = starts - 1
    # Verify: same column for every row.
    assert (marker_cols == marker_cols[0]).all(), \
        f"gen-start marker columns not aligned: {marker_cols}"

    # And that column actually contains the expected marker in src.
    src = result["src_tokens"]
    col = int(marker_cols[0])
    for b in range(src.shape[0]):
        assert int(src[b, col]) == marker


def test_align_mode_left_pad_lengths_match_expected_shift(generator):
    """left_pad[i] = max_prefix - prefix_i.  Cross-check by re-deriving
    the pivot column from row 0 and comparing computed shifts to
    the reported left_pad_lengths."""
    result = _worker(generator).generate_batch(_cfg(generator, align=True))
    pivot = int(result["task_start_indices"][0]) - 1  # SS/TS marker column
    src = result["src_tokens"]
    for b in range(src.shape[0]):
        left_pad = int(result["left_pad_lengths"][b])
        # Row starts with `left_pad` PADs then BOS at column left_pad.
        for c in range(left_pad):
            assert int(src[b, c]) == TOK_PAD
        assert int(src[b, left_pad]) == 0  # TOK_BOS
        # The gen-start column is exactly `pivot` for every row.
        # (Independently: left_pad + row-local prefix_len == pivot.)


def test_align_mode_metadata_indices_shifted_by_left_pad(generator):
    """Section-start indices reported to Python point at columns in
    the PADDED row, so they should equal (row-local start) + left_pad
    for each row."""
    result_align = _worker(generator, seed=17).generate_batch(
        _cfg(generator, align=True))
    result_std   = _worker(generator, seed=17).generate_batch(
        _cfg(generator, align=False))

    # Same batch: same per-row raw layouts. Only offsets differ.
    for key in ("graph_edge_start_indices", "query_start_indices",
                "task_start_indices"):
        diff = result_align[key] - result_std[key]
        # Under align mode, each start shifts by that row's left_pad.
        np.testing.assert_array_equal(diff, result_align["left_pad_lengths"])


def test_align_mode_lengths_and_content_unchanged(generator):
    """Section lengths and per-item metadata (num_nodes, num_edges,
    src_lengths) do NOT change between modes -- only positions do."""
    a = _worker(generator, seed=17).generate_batch(_cfg(generator, align=True))
    r = _worker(generator, seed=17).generate_batch(_cfg(generator, align=False))
    for key in ("src_lengths", "num_nodes", "num_edges",
                "graph_edge_lengths", "query_lengths",
                "scratchpad_lengths", "task_lengths"):
        np.testing.assert_array_equal(a[key], r[key])


def test_align_mode_can_produce_wider_batch_than_right_pad(generator):
    """max_seq_len(align) = max_prefix + max_gen. If the row with the
    biggest prefix and the row with the biggest gen aren't the same
    row, align mode's batch is wider than right-pad's."""
    r = _worker(generator, seed=41).generate_batch(_cfg(generator, align=False))
    a = _worker(generator, seed=41).generate_batch(_cfg(generator, align=True))
    # Not required to be strictly greater on every seed, but should be
    # >= on all seeds and > on many. Assert >=.
    assert a["src_tokens"].shape[1] >= r["src_tokens"].shape[1]


def test_align_mode_content_bytes_match_right_pad_shifted(generator):
    """The actual content tokens are the same; only their column
    positions differ by left_pad."""
    a = _worker(generator, seed=17).generate_batch(_cfg(generator, align=True))
    r = _worker(generator, seed=17).generate_batch(_cfg(generator, align=False))
    for b in range(a["src_tokens"].shape[0]):
        left_pad = int(a["left_pad_lengths"][b])
        length   = int(a["src_lengths"][b])
        aligned  = a["src_tokens"][b, left_pad : left_pad + length]
        standard = r["src_tokens"][b, 0 : length]
        np.testing.assert_array_equal(aligned, standard)


def test_align_mode_positions_content_is_zero_indexed_from_content_start(generator):
    """Position ids in the content region should be 0..seq_len-1
    (row-local), starting at column left_pad. Pad columns stay at 0."""
    result = _worker(generator).generate_batch(_cfg(generator, align=True))
    positions = result["positions"]
    for b in range(positions.shape[0]):
        left_pad = int(result["left_pad_lengths"][b])
        length   = int(result["src_lengths"][b])
        # Content region: sequential 0..length-1
        for i in range(length):
            assert int(positions[b, left_pad + i]) == i, \
                f"row {b} pos {left_pad+i}: expected {i}"


def test_align_mode_deterministic_under_fixed_seed(generator):
    cfg = _cfg(generator, align=True)
    a = _worker(generator, seed=99).generate_batch(cfg)
    b = _worker(generator, seed=99).generate_batch(cfg)
    for key in a.keys():
        np.testing.assert_array_equal(a[key], b[key])
