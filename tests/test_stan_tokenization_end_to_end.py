"""End-to-end tests for STAN tokenization mode.

STAN packs each edge into ONE sequence position across `struct_dim=3`
content columns: [u, v, EDGE]. Every other section (query, scratchpad,
target, markers) sits in column 0 with columns 1..2 padded. Tensor
shape becomes [B, S, 3] instead of [B, S].
"""
from __future__ import annotations

import numpy as np
import pytest

from get_generator_module import get_generator_module


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


def _cfg(generator, mode, **overrides):
    kwargs = dict(
        graph_kind="erdos_renyi",
        task_kind="shortest_path",
        scratchpad_kind="none",
        tokenization_mode=mode,
        min_num_nodes=8,
        max_num_nodes=8,
        edge_prob=0.5,
        min_path_length=2,
        max_path_length=5,
        max_attempts=1000,
        batch_size=4)
    kwargs.update(overrides)
    return generator.GeneratorConfig(**kwargs)


def _worker(generator, seed=0):
    ctx = generator.WorkerSharedContext()
    return generator.Worker(ctx, seed=seed)


def test_stan_produces_3d_token_tensor(generator):
    result = _worker(generator).generate_batch(_cfg(generator, "stan"))
    # SEAN would give [B, S]; STAN gives [B, S, 3].
    assert result["src_tokens"].ndim == 3
    assert result["src_tokens"].shape[-1] == 3
    assert result["positions"].shape == result["src_tokens"].shape
    # `targets` is always 3D (B, max_gen_len, max_labels); its extents
    # describe the generation region + label smoothing, so they don't
    # match src's structural axis under STAN.
    assert result["targets"].ndim == 3
    assert result["targets"].shape[0] == result["src_tokens"].shape[0]


def test_sean_produces_2d_token_tensor(generator):
    result = _worker(generator).generate_batch(_cfg(generator, "sean"))
    assert result["src_tokens"].ndim == 2
    assert result["positions"].shape == result["src_tokens"].shape
    # `targets` is 3D (B, max_gen_len, max_labels) under SEAN too.
    assert result["targets"].ndim == 3
    assert result["targets"].shape[0] == result["src_tokens"].shape[0]


def test_stan_edges_span_three_columns(generator):
    result = _worker(generator).generate_batch(_cfg(generator, "stan"))
    src   = result["src_tokens"]
    starts  = result["graph_edge_start_indices"]
    lengths = result["graph_edge_lengths"]
    num_edges = result["num_edges"]
    for b in range(src.shape[0]):
        s = int(starts[b])
        L = int(lengths[b])
        # Under STAN, L equals the number of edges (1 position per edge).
        assert L == int(num_edges[b])
        for e in range(L):
            u = int(src[b, s + e, 0])
            v = int(src[b, s + e, 1])
            edge_tok = int(src[b, s + e, 2])
            assert u >= NUM_SPECIAL_DEFAULT
            assert v >= NUM_SPECIAL_DEFAULT
            assert edge_tok == TOK_EDGE


def test_stan_non_edge_positions_have_pad_in_cols_1_and_2(generator):
    result = _worker(generator).generate_batch(_cfg(generator, "stan"))
    src = result["src_tokens"]
    edge_starts  = result["graph_edge_start_indices"]
    edge_lengths = result["graph_edge_lengths"]
    src_lengths  = result["src_lengths"]
    for b in range(src.shape[0]):
        e_start = int(edge_starts[b])
        e_end   = e_start + int(edge_lengths[b])
        s_len   = int(src_lengths[b])
        for p in range(s_len):
            if e_start <= p < e_end:
                continue  # edge position -- checked above
            assert int(src[b, p, 1]) == TOK_PAD, (
                f"row {b} pos {p}: col 1 must be PAD outside edges"
            )
            assert int(src[b, p, 2]) == TOK_PAD, (
                f"row {b} pos {p}: col 2 must be PAD outside edges"
            )


def test_stan_seq_len_shorter_than_sean_by_2m(generator):
    """SEAN spreads each edge over 3 positions; STAN packs into 1.
    So the STAN seq_len is exactly SEAN's minus 2*num_edges."""
    result_sean = _worker(generator, seed=42).generate_batch(_cfg(generator, "sean"))
    result_stan = _worker(generator, seed=42).generate_batch(_cfg(generator, "stan"))
    m = result_sean["num_edges"]
    diff = result_sean["src_lengths"] - result_stan["src_lengths"]
    np.testing.assert_array_equal(diff, 2 * m)


def test_stan_column_0_matches_sean_at_non_edge_positions(generator):
    """Away from edge positions, STAN col 0 carries the same section
    tokens as the SEAN 1D stream (markers, query verts, target verts).
    Verified only for the query and target sections since edges
    complicate direct index mapping."""
    result_sean = _worker(generator, seed=99).generate_batch(_cfg(generator, "sean"))
    result_stan = _worker(generator, seed=99).generate_batch(_cfg(generator, "stan"))
    for b in range(result_sean["src_tokens"].shape[0]):
        q_sean = result_sean["query_start_indices"][b]
        q_stan = result_stan["query_start_indices"][b]
        q_len  = result_sean["query_lengths"][b]
        assert q_len == result_stan["query_lengths"][b]
        for i in range(int(q_len)):
            assert int(result_sean["src_tokens"][b, q_sean + i]) == \
                   int(result_stan["src_tokens"][b, q_stan + i, 0])

        t_sean = result_sean["task_start_indices"][b]
        t_stan = result_stan["task_start_indices"][b]
        t_len  = result_sean["task_lengths"][b]
        assert t_len == result_stan["task_lengths"][b]
        for i in range(int(t_len)):
            assert int(result_sean["src_tokens"][b, t_sean + i]) == \
                   int(result_stan["src_tokens"][b, t_stan + i, 0])


def test_stan_deterministic_under_fixed_seed(generator):
    cfg = _cfg(generator, "stan")
    a = _worker(generator, seed=123).generate_batch(cfg)
    b = _worker(generator, seed=123).generate_batch(cfg)
    for key in a.keys():
        np.testing.assert_array_equal(a[key], b[key])
