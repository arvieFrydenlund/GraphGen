"""End-to-end shape checks on Worker.generate_batch's output dict.

Verifies the batch dict has the expected key set, tensor shapes match
the batch size and computed max_seq_len, and per-item metadata arrays
agree with the SEAN + (ShortestPath, None) length formula:

    src_len(i) = 3 * num_edges(i) + path_length(i) + 9

Values inside the token tensors are ignored (the tokenizer step comes
later; tensors are zero-filled for now).
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


def _make_cfg(generator, **overrides):
    kwargs = dict(
        graph_kind="erdos_renyi",
        task_kind="shortest_path",
        scratchpad_kind="none",
        tokenization_mode="sean",
        min_num_nodes=15,
        max_num_nodes=15,
        edge_prob=0.4,
        min_path_length=1,
        max_path_length=6,
        max_attempts=1000,
        batch_size=8)
    kwargs.update(overrides)
    return generator.GeneratorConfig(**kwargs)


def _worker(generator, seed=0):
    ctx = generator.WorkerSharedContext()
    return generator.Worker(ctx, seed=seed)


EXPECTED_KEYS = {
    "src_tokens",
    "targets",
    "positions",  # return_pos_ids defaults to True
    "src_lengths",
    "num_nodes",
    "num_edges",
    "graph_edge_start_indices",
    "graph_edge_lengths",
    "query_start_indices",
    "query_lengths",
    "thinking_start_indices",
    "thinking_lengths",
    "scratchpad_start_indices",
    "scratchpad_lengths",
    "task_start_indices",
    "task_lengths",
    "left_pad_lengths",
}


def test_dict_has_expected_key_set(generator):
    result = _worker(generator).generate_batch(_make_cfg(generator))
    assert set(result.keys()) == EXPECTED_KEYS


def test_positions_omitted_when_return_pos_ids_false(generator):
    result = _worker(generator).generate_batch(
        _make_cfg(generator, return_pos_ids=False)
    )
    assert "positions" not in result
    assert set(result.keys()) == EXPECTED_KEYS - {"positions"}


@pytest.mark.parametrize("batch_size", [1, 4, 16])
def test_token_tensors_are_2d_and_batch_first(generator, batch_size):
    result = _worker(generator).generate_batch(
        _make_cfg(generator, batch_size=batch_size)
    )
    src     = result["src_tokens"]
    targets = result["targets"]
    pos     = result["positions"]

    assert src.ndim == 2
    assert src.shape[0] == batch_size
    # Targets have shape (B, max_gen_len, max_labels). Only the batch
    # dim must match src; the other axes describe the generation
    # region, not the full sequence.
    assert targets.ndim == 3
    assert targets.shape[0] == batch_size
    assert pos.shape == src.shape

    # And the width matches the max src_length across the batch.
    src_lengths = result["src_lengths"]
    assert src.shape[1] == int(src_lengths.max())


def test_metadata_arrays_are_1d_batch_size(generator):
    result = _worker(generator).generate_batch(_make_cfg(generator))
    metadata_keys = EXPECTED_KEYS - {"src_tokens", "targets", "positions"}
    for key in metadata_keys:
        arr = result[key]
        assert arr.ndim == 1, f"{key}: expected 1D, got shape {arr.shape}"
        assert arr.shape[0] == 8, f"{key}: expected batch_size=8 rows"


def test_src_length_matches_sean_shortest_path_formula(generator):
    # src_len(i) = 3 * num_edges(i) + path_length(i) + 9
    # We don't have direct access to path_length from the batch output,
    # but task_lengths[i] = path_length + 1 (the path is stored end-inclusive),
    # so src_len = 3 * num_edges + task_lengths - 1 + 9
    #            = 3 * num_edges + task_lengths + 8.
    result = _worker(generator).generate_batch(_make_cfg(generator))
    src_lengths  = result["src_lengths"]
    num_edges    = result["num_edges"]
    task_lengths = result["task_lengths"]
    expected = 3 * num_edges + task_lengths + 8
    np.testing.assert_array_equal(src_lengths, expected)


def test_section_indices_lie_within_src_length(generator):
    # Every reported section must be fully contained inside the row's
    # true length (no section extending into the padding).
    result = _worker(generator).generate_batch(_make_cfg(generator))
    src_lengths  = result["src_lengths"]
    for start_key, len_key in [
        ("graph_edge_start_indices", "graph_edge_lengths"),
        ("query_start_indices",      "query_lengths"),
        ("scratchpad_start_indices", "scratchpad_lengths"),
        ("task_start_indices",       "task_lengths"),
    ]:
        starts  = result[start_key]
        lengths = result[len_key]
        ends = starts + lengths
        assert (starts  >= 0).all()
        assert (ends   <= src_lengths).all(), (
            f"{start_key} + {len_key} runs past src_lengths"
        )


def test_deterministic_shapes_under_fixed_seed(generator):
    cfg = _make_cfg(generator, batch_size=4)
    a = _worker(generator, seed=42).generate_batch(cfg)
    b = _worker(generator, seed=42).generate_batch(cfg)
    for key in a.keys():
        np.testing.assert_array_equal(a[key], b[key])


def test_token_tensor_padding_uses_pad_token(generator):
    # Positions past src_lengths[i] must be TOK_PAD (=1). Real content
    # positions can be anything; this test only asserts the padding tail.
    TOK_PAD = 1
    result = _worker(generator).generate_batch(_make_cfg(generator))
    src         = result["src_tokens"]
    src_lengths = result["src_lengths"]
    max_seq_len = src.shape[1]
    for b in range(src.shape[0]):
        L = int(src_lengths[b])
        if L < max_seq_len:
            assert (src[b, L:] == TOK_PAD).all(), (
                f"row {b}: expected TOK_PAD in tail [{L}:], got {src[b, L:]}"
            )
