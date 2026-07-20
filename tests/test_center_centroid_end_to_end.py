"""End-to-end token-content invariants for (ER, Center/Centroid, None).

Center and Centroid share the same section-plan/tokenization shape:
  * Query section carries the |Q| query vertices (min_query_size..max_query_size).
  * Target section carries the set of winning vertices (>= 1).

Both use vocab tokens (>= NUM_SPECIAL_DEFAULT) throughout query/target,
bracketed by QS/QE and TS/TE.
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


def _cfg(generator, task_kind, **overrides):
    kwargs = dict(
        graph_kind="erdos_renyi",
        task_kind=task_kind,
        scratchpad_kind="none",
        tokenization_mode="sean",
        min_num_nodes=10,
        max_num_nodes=10,
        # Dense ER so the graph is (almost) always connected -- avoids
        # the "no vertex reachable from every q in Q" throw path.
        edge_prob=0.7,
        max_attempts=1000,
        batch_size=8,
        min_query_size=2,
        max_query_size=3)
    kwargs.update(overrides)
    return generator.GeneratorConfig(**kwargs)


def _worker(generator, seed=0):
    ctx = generator.WorkerSharedContext()
    return generator.Worker(ctx, seed=seed)


@pytest.mark.parametrize("task_kind", ["center", "centroid"])
def test_bos_and_eos_bracket_each_row(generator, task_kind):
    result = _worker(generator).generate_batch(_cfg(generator, task_kind))
    src         = result["src_tokens"]
    src_lengths = result["src_lengths"]
    for b in range(src.shape[0]):
        L = int(src_lengths[b])
        assert src[b, 0]     == TOK_BOS
        assert src[b, L - 1] == TOK_EOS


@pytest.mark.parametrize("task_kind", ["center", "centroid"])
def test_query_section_holds_query_size_vocab_tokens(generator, task_kind):
    result = _worker(generator).generate_batch(_cfg(generator, task_kind))
    src           = result["src_tokens"]
    query_starts  = result["query_start_indices"]
    query_lengths = result["query_lengths"]
    for b in range(src.shape[0]):
        s = int(query_starts[b])
        L = int(query_lengths[b])
        # min_query_size=2, max_query_size=3 per _cfg above.
        assert 2 <= L <= 3, f"row {b}: query_len must be in [2,3], got {L}"
        assert src[b, s - 1] == TOK_QUERY_START
        assert src[b, s + L] == TOK_QUERY_END
        tokens = [int(t) for t in src[b, s : s + L]]
        # All query entries must be vocab tokens and distinct.
        for tok in tokens:
            assert tok >= NUM_SPECIAL_DEFAULT
        assert len(set(tokens)) == len(tokens), (
            f"row {b}: query vertices must be distinct"
        )


@pytest.mark.parametrize("task_kind", ["center", "centroid"])
def test_target_section_holds_at_least_one_vocab_token(generator, task_kind):
    result = _worker(generator).generate_batch(_cfg(generator, task_kind))
    src          = result["src_tokens"]
    task_starts  = result["task_start_indices"]
    task_lengths = result["task_lengths"]
    for b in range(src.shape[0]):
        s = int(task_starts[b])
        L = int(task_lengths[b])
        assert L >= 1, f"row {b}: target must contain >= 1 center_node"
        assert src[b, s - 1] == TOK_TASK_START
        assert src[b, s + L] == TOK_TASK_END
        tokens = [int(t) for t in src[b, s : s + L]]
        for tok in tokens:
            assert tok >= NUM_SPECIAL_DEFAULT
        # center_nodes are distinct too.
        assert len(set(tokens)) == len(tokens), (
            f"row {b}: center_nodes must be distinct"
        )


@pytest.mark.parametrize("task_kind", ["center", "centroid"])
def test_target_never_exceeds_num_nodes(generator, task_kind):
    result = _worker(generator).generate_batch(_cfg(generator, task_kind))
    task_lengths = result["task_lengths"]
    num_nodes    = result["num_nodes"]
    for b in range(len(task_lengths)):
        assert int(task_lengths[b]) <= int(num_nodes[b])


@pytest.mark.parametrize("task_kind", ["center", "centroid"])
def test_seq_length_formula_holds(generator, task_kind):
    # src_len = BOS + 3*num_edges + QS + query_len + QE + TS + task_len + TE + EOS
    #         = 3*num_edges + query_len + task_len + 6
    result = _worker(generator).generate_batch(_cfg(generator, task_kind))
    src_lengths   = result["src_lengths"]
    num_edges     = result["num_edges"]
    query_lengths = result["query_lengths"]
    task_lengths  = result["task_lengths"]
    expected = 3 * num_edges + query_lengths + task_lengths + 6
    np.testing.assert_array_equal(src_lengths, expected)


@pytest.mark.parametrize("task_kind", ["center", "centroid"])
def test_padding_holds_pad_token(generator, task_kind):
    result = _worker(generator).generate_batch(_cfg(generator, task_kind))
    src         = result["src_tokens"]
    src_lengths = result["src_lengths"]
    S           = src.shape[1]
    for b in range(src.shape[0]):
        L = int(src_lengths[b])
        if L < S:
            assert (src[b, L:] == TOK_PAD).all()


@pytest.mark.parametrize("task_kind", ["center", "centroid"])
def test_targets_apply_cumulative_smoothing_over_center_nodes(generator, task_kind):
    # section_plan gives Center/Centroid |center_nodes| sequential
    # target positions (in the shuffled order stored on the task).
    # The targets tensor applies cumulative label smoothing at each
    # position:
    #   position i (row-local target index):
    #     valid labels = center_nodes[i], center_nodes[i+1], ..., center_nodes[N-1]
    # i.e. any center_node not yet emitted in this row is equally
    # valid. This turns the task into a set-generation objective
    # rather than a fixed sequence.
    result = _worker(generator).generate_batch(_cfg(generator, task_kind))
    src            = result["src_tokens"]
    targets        = result["targets"]
    task_starts    = result["task_start_indices"]
    task_lengths   = result["task_lengths"]

    assert targets.ndim == 3
    for b in range(src.shape[0]):
        s = int(task_starts[b])
        L = int(task_lengths[b])
        assert L >= 1
        # Extract the emitted center_nodes for this row (in shuffled
        # order) from src.
        emitted = [int(src[b, s + i]) for i in range(L)]
        # Every emitted token must appear as a valid vocab token.
        for tok in emitted:
            assert tok >= NUM_SPECIAL_DEFAULT
        # Convert to a set for membership checks.
        emitted_set = set(emitted)

        # Target-section positions in generation-region coords:
        # gp_i = (s + i) - gen_start where gen_start = task_start - 1
        # (assuming no thinking/scratchpad in this fixture).
        gen_start = s - 1
        for i in range(L):
            gp = (s + i) - gen_start
            # k=0 must match src at this position.
            assert targets[b, gp, 0] == src[b, s + i], (
                f"row {b} pos {i}: targets[k=0] doesn't match src"
            )
            # The set of non-PAD labels at this step MUST equal
            # {center_nodes[i], ..., center_nodes[L-1]}: every remaining
            # not-yet-emitted center_node, and only those.
            labels_at_step = {
                int(targets[b, gp, k])
                for k in range(targets.shape[2])
                if int(targets[b, gp, k]) != TOK_PAD
            }
            expected_remaining = set(emitted[i:])
            assert labels_at_step == expected_remaining, (
                f"row {b} pos {i}: expected labels {expected_remaining}, "
                f"got {labels_at_step}"
            )


@pytest.mark.parametrize("task_kind", ["center", "centroid"])
def test_deterministic_tokens_under_fixed_seed(generator, task_kind):
    cfg = _cfg(generator, task_kind, batch_size=4)
    a = _worker(generator, seed=99).generate_batch(cfg)
    b = _worker(generator, seed=99).generate_batch(cfg)
    for key in a.keys():
        np.testing.assert_array_equal(a[key], b[key])
