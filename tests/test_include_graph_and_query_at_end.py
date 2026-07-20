"""Tests for include_graph_in_graph_tokenization and query_at_end.

include_graph_in_graph_tokenization (default True):
    False -> omit the GraphNodes / GraphEdges sections entirely.

query_at_end (default True):
    True  -> query section sits AFTER the graph (right before generation).
    False -> query section sits BEFORE the graph (right after BOS).
"""
from __future__ import annotations

import numpy as np
import pytest

from get_generator_module import get_generator_module


TOK_PAD           = 1
TOK_TASK_START    = 6
TOK_QUERY_START   = 13
TOK_QUERY_END     = 14
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


# ---------------------------------------------------------------------------
# include_graph_in_graph_tokenization
# ---------------------------------------------------------------------------

def test_omitting_graph_zeroes_out_graph_section_indices(generator):
    result = _worker(generator).generate_batch(
        _cfg(generator, include_graph_in_graph_tokenization=False))
    assert (result["graph_edge_lengths"] == 0).all()
    assert (result["graph_edge_start_indices"] == 0).all()


def test_omitting_graph_shortens_seq_len_by_3m(generator):
    """SEAN mode: dropping the graph section removes 3*num_edges positions."""
    with_graph    = _worker(generator, seed=11).generate_batch(_cfg(generator))
    without_graph = _worker(generator, seed=11).generate_batch(
        _cfg(generator, include_graph_in_graph_tokenization=False))
    m = with_graph["num_edges"]  # same graph sample under fixed seed
    diff = with_graph["src_lengths"] - without_graph["src_lengths"]
    np.testing.assert_array_equal(diff, 3 * m)


def test_omitting_graph_query_still_follows_bos(generator):
    """With no graph, structure collapses to BOS QS Q QE TS T TE EOS."""
    result = _worker(generator).generate_batch(
        _cfg(generator, include_graph_in_graph_tokenization=False))
    src = result["src_tokens"]
    for b in range(src.shape[0]):
        # BOS at col 0. QueryStart marker at col 1. Content starts at 2.
        assert int(src[b, 0]) == 0                     # TOK_BOS
        assert int(src[b, 1]) == TOK_QUERY_START
        assert int(result["query_start_indices"][b]) == 2


def test_omitting_graph_num_edges_still_reported(generator):
    """num_edges is a descriptor of the sampled graph, not of what
    the tokenizer emits, so it stays populated even when the graph
    section is omitted."""
    result = _worker(generator).generate_batch(
        _cfg(generator, include_graph_in_graph_tokenization=False))
    assert (result["num_edges"] > 0).all()
    assert (result["num_nodes"] == 8).all()


def test_include_graph_default_is_true(generator):
    """Default config has the graph section populated."""
    result = _worker(generator).generate_batch(_cfg(generator))
    assert (result["graph_edge_lengths"] > 0).all()


# ---------------------------------------------------------------------------
# query_at_end
# ---------------------------------------------------------------------------

def test_default_query_at_end_true_places_query_after_graph(generator):
    result = _worker(generator).generate_batch(_cfg(generator))
    for b in range(result["src_tokens"].shape[0]):
        graph_start = int(result["graph_edge_start_indices"][b])
        query_start = int(result["query_start_indices"][b])
        assert graph_start < query_start


def test_query_at_end_false_places_query_before_graph(generator):
    result = _worker(generator).generate_batch(
        _cfg(generator, query_at_end=False))
    for b in range(result["src_tokens"].shape[0]):
        graph_start = int(result["graph_edge_start_indices"][b])
        query_start = int(result["query_start_indices"][b])
        assert query_start < graph_start
        # Query still lives in the prompt, so it's before task_start.
        task_start = int(result["task_start_indices"][b])
        assert query_start < task_start


def test_query_at_end_false_query_starts_right_after_bos_qs_marker(generator):
    result = _worker(generator).generate_batch(
        _cfg(generator, query_at_end=False))
    src = result["src_tokens"]
    for b in range(src.shape[0]):
        # BOS at 0, QS at 1, query content at 2.
        assert int(src[b, 0]) == 0                     # TOK_BOS
        assert int(src[b, 1]) == TOK_QUERY_START
        assert int(result["query_start_indices"][b]) == 2


def test_query_at_end_does_not_change_seq_len(generator):
    """Flipping query_at_end reorders sections but doesn't add or
    remove any. seq_len must be identical row-for-row under fixed seed."""
    a = _worker(generator, seed=17).generate_batch(_cfg(generator, query_at_end=True))
    b = _worker(generator, seed=17).generate_batch(_cfg(generator, query_at_end=False))
    np.testing.assert_array_equal(a["src_lengths"], b["src_lengths"])


def test_query_at_end_false_task_start_column_unchanged(generator):
    """The generation region is the same regardless of query placement,
    so task_start_indices should match between the two orderings."""
    a = _worker(generator, seed=17).generate_batch(_cfg(generator, query_at_end=True))
    b = _worker(generator, seed=17).generate_batch(_cfg(generator, query_at_end=False))
    np.testing.assert_array_equal(
        a["task_start_indices"], b["task_start_indices"])


# ---------------------------------------------------------------------------
# include_graph=False + query_at_end interaction
# ---------------------------------------------------------------------------

def test_no_graph_and_query_at_end_are_independent(generator):
    """With no graph AND query at start, structure collapses to
    BOS QS Q QE TS T TE EOS -- the query_at_end flag is inert since
    the two candidate positions (after BOS, after graph) coincide."""
    a = _worker(generator, seed=33).generate_batch(_cfg(
        generator, include_graph_in_graph_tokenization=False, query_at_end=True))
    b = _worker(generator, seed=33).generate_batch(_cfg(
        generator, include_graph_in_graph_tokenization=False, query_at_end=False))
    np.testing.assert_array_equal(a["src_tokens"], b["src_tokens"])
