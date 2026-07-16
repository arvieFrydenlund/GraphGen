"""WorkerSharedContext tests.

Cover the pieces that survived the V1 audit and now live on the shared
context: token dictionary, positional dictionary, and validation/test
hash filters. Plus a shared_ptr lifetime check to guarantee the C++
context outlives its Python reference when a Worker is still holding it.
"""
from __future__ import annotations

import gc

import numpy as np
import pytest

from get_generator_module import get_generator_module


@pytest.fixture(scope="module")
def generator():
    get_generator_module()
    import generator as g  # noqa: WPS433 -- deferred import: needs build first
    return g


# ---------------------------------------------------------------------------
# Default token dictionary
# ---------------------------------------------------------------------------

def test_default_ctor_populates_token_dictionary(generator):
    ctx = generator.WorkerSharedContext()
    d = ctx.token_dict
    # V1 default: 23 special tokens at positions 0..22, then 50 integer
    # vocab tokens "0".."49" at positions 23..72. No extras.
    assert d["<s>"] == 0
    assert d["<pad>"] == 1
    assert d["</s>"] == 2
    assert d["<unk>"] == 3
    assert d["|"] == 4
    assert d["D"] == 22
    assert d["0"] == 23
    assert d["49"] == 72
    assert ctx.num_special == 23
    assert ctx.num_extra == 0
    assert ctx.max_vocab == 73
    assert ctx.extra_after_symbol == "D"


def test_special_token_ids_match_dictionary(generator):
    ctx = generator.WorkerSharedContext()
    # The static constants exposed on the class must line up with the
    # dictionary positions the default ctor materialises.
    assert generator.WorkerSharedContext.TOK_BOS == ctx.token_dict["<s>"]
    assert generator.WorkerSharedContext.TOK_PAD == ctx.token_dict["<pad>"]
    assert generator.WorkerSharedContext.TOK_EOS == ctx.token_dict["</s>"]
    assert generator.WorkerSharedContext.TOK_UNK == ctx.token_dict["<unk>"]
    assert generator.WorkerSharedContext.TOK_EDGE == ctx.token_dict["|"]
    assert generator.WorkerSharedContext.NUM_SPECIAL_DEFAULT == 23


def test_default_pos_dictionary(generator):
    ctx = generator.WorkerSharedContext()
    p = ctx.pos_dict
    # Scalars are 1-wide half-open ranges.
    assert p["pad"] == (0, 1)
    assert p["query_invariance"] == (1, 2)
    assert p["task_invariance"] == (6, 7)
    # Ranges: [start, end) semantics; V1's misc_start=11/misc_end=49 became [11, 50).
    assert p["misc"] == (11, 50)
    assert p["query"] == (50, 200)
    assert p["graph"] == (200, 940)
    assert p["graph_sub"] == (940, 950)
    assert p["thinking"] == (950, 1000)
    assert p["task"] == (1000, 5001)


# ---------------------------------------------------------------------------
# Custom pos_dict: structural validation
# ---------------------------------------------------------------------------

def test_set_pos_dictionary_accepts_gaps(generator):
    ctx = generator.WorkerSharedContext()
    # Intentional gap between [0, 1) and [5, 10) is fine.
    ctx.set_pos_dictionary({"pad": (0, 1), "task": (5, 10)})
    assert ctx.pos_dict == {"pad": (0, 1), "task": (5, 10)}


def test_set_pos_dictionary_rejects_negative_start(generator):
    ctx = generator.WorkerSharedContext()
    with pytest.raises(ValueError, match="negative start"):
        ctx.set_pos_dictionary({"bad": (-1, 5)})


def test_set_pos_dictionary_rejects_empty_or_inverted_range(generator):
    ctx = generator.WorkerSharedContext()
    with pytest.raises(ValueError, match="start >= end"):
        ctx.set_pos_dictionary({"pad": (5, 5)})
    with pytest.raises(ValueError, match="start >= end"):
        ctx.set_pos_dictionary({"pad": (10, 5)})


def test_set_pos_dictionary_rejects_overlap(generator):
    ctx = generator.WorkerSharedContext()
    with pytest.raises(ValueError, match="overlap"):
        ctx.set_pos_dictionary({"a": (0, 10), "b": (5, 15)})


# ---------------------------------------------------------------------------
# validate_for_config -- content requirements per config
# ---------------------------------------------------------------------------

def test_validate_for_config_default_pos_dict_supports_topology_kinds(generator):
    ctx = generator.WorkerSharedContext()  # default pos_dict has 'pad' and 'graph'
    for kind in ("erdos_renyi", "euclidean", "random_tree", "path_star", "balanced"):
        cfg = generator.GeneratorConfig(graph_kind=kind, batch_size=1)
        assert ctx.validate_for_config(cfg) == ""


def test_validate_for_config_khops_does_not_require_graph_range(generator):
    ctx = generator.WorkerSharedContext()
    ctx.set_pos_dictionary({"pad": (0, 1), "task": (1, 100)})  # no 'graph'
    for kind in ("khops", "khops_gen"):
        cfg = generator.GeneratorConfig(graph_kind=kind, batch_size=1)
        assert ctx.validate_for_config(cfg) == ""


def test_validate_for_config_missing_pad_fails(generator):
    ctx = generator.WorkerSharedContext()
    ctx.set_pos_dictionary({"task": (0, 100)})  # no 'pad'
    cfg = generator.GeneratorConfig(graph_kind="khops", batch_size=1)
    msg = ctx.validate_for_config(cfg)
    assert "pad" in msg


def test_validate_for_config_topology_kind_missing_graph_fails(generator):
    ctx = generator.WorkerSharedContext()
    ctx.set_pos_dictionary({"pad": (0, 1), "task": (1, 100)})  # no 'graph'
    cfg = generator.GeneratorConfig(graph_kind="erdos_renyi", batch_size=1)
    msg = ctx.validate_for_config(cfg)
    assert "graph" in msg
    assert "erdos_renyi" in msg


# ---------------------------------------------------------------------------
# set_default_dictionary reconfiguration
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("max_num_nodes,extra_after", [
    (10, 0),
    (100, 5),
    (50, 3),
])
def test_set_default_dictionary_resizes(generator, max_num_nodes, extra_after):
    ctx = generator.WorkerSharedContext()
    ctx.set_default_dictionary(max_num_nodes=max_num_nodes, extra_after=extra_after)
    assert ctx.num_special == 23
    assert ctx.num_extra == extra_after
    assert ctx.max_vocab == 23 + max_num_nodes
    # Extra tokens are D0, D1, ... positioned after the vocab range.
    for i in range(extra_after):
        assert ctx.token_dict[f"D{i}"] == 23 + max_num_nodes + i


def test_set_default_dictionary_custom_extra_symbol(generator):
    ctx = generator.WorkerSharedContext()
    ctx.set_default_dictionary(max_num_nodes=5, extra_after=2, extra_after_symbol="X")
    assert ctx.extra_after_symbol == "X"
    assert "X0" in ctx.token_dict
    assert "X1" in ctx.token_dict
    assert "D0" not in ctx.token_dict


# ---------------------------------------------------------------------------
# set_dictionary custom mapping + validation
# ---------------------------------------------------------------------------

def test_set_dictionary_counts_categories(generator):
    ctx = generator.WorkerSharedContext()
    # 2 special (<pad>, <s>), 3 vocab (0, 1, 2), 1 extra (D0).
    d = {"<pad>": 0, "<s>": 1, "0": 2, "1": 3, "2": 4, "D0": 5}
    ctx.set_dictionary(d, max_num_nodes=3, extra_after=1, extra_after_symbol="D")
    assert ctx.num_special == 2
    assert ctx.num_extra == 1
    assert ctx.max_vocab == 2 + 3


def test_set_dictionary_rejects_vocab_size_mismatch(generator):
    ctx = generator.WorkerSharedContext()
    d = {"<pad>": 0, "0": 1, "1": 2}  # 2 vocab tokens
    with pytest.raises(ValueError, match="does not match max_num_nodes"):
        ctx.set_dictionary(d, max_num_nodes=5)


def test_set_dictionary_rejects_extra_count_mismatch(generator):
    ctx = generator.WorkerSharedContext()
    d = {"<pad>": 0, "0": 1, "D0": 2, "D1": 3}  # 2 extras
    with pytest.raises(ValueError, match="do not match extra_after"):
        ctx.set_dictionary(d, extra_after=5)


# ---------------------------------------------------------------------------
# Hash-based validation/test filters
# ---------------------------------------------------------------------------

def test_hash_filters_start_empty(generator):
    ctx = generator.WorkerSharedContext()
    assert ctx.validation_size == 0
    assert ctx.test_size == 0


def test_extend_and_query_validation_hashes(generator):
    ctx = generator.WorkerSharedContext()
    ctx.extend_validation_hashes([100, 200, 300])
    ctx.extend_validation_hashes([300, 400])  # duplicate 300 collapses
    assert ctx.validation_size == 4
    assert ctx.in_validation(200) is True
    assert ctx.in_validation(999) is False


def test_extend_and_query_test_hashes(generator):
    ctx = generator.WorkerSharedContext()
    ctx.extend_test_hashes([1, 2, 3])
    assert ctx.test_size == 3
    assert ctx.in_test(2) is True
    assert ctx.in_test(4) is False


def test_batch_is_in_validation_returns_bool_array(generator):
    ctx = generator.WorkerSharedContext()
    ctx.extend_validation_hashes([10, 20, 30])
    hashes = np.array([5, 10, 15, 20, 25, 30, 35], dtype=np.uint64)
    result = ctx.is_in_validation(hashes)
    assert result.dtype == bool
    assert result.tolist() == [False, True, False, True, False, True, False]


def test_batch_is_invalid_example_unions_validation_and_test(generator):
    ctx = generator.WorkerSharedContext()
    ctx.extend_validation_hashes([10, 20])
    ctx.extend_test_hashes([30, 40])
    hashes = np.array([10, 30, 50, 20, 40, 60], dtype=np.uint64)
    result = ctx.is_invalid_example(hashes)
    assert result.tolist() == [True, True, False, True, True, False]


# ---------------------------------------------------------------------------
# Lifetime: Worker holds a shared_ptr; dropping the Python reference must
# not free the underlying context while any Worker still references it.
# ---------------------------------------------------------------------------

def test_worker_keeps_shared_context_alive(generator):
    ctx = generator.WorkerSharedContext()
    ctx.extend_validation_hashes([42, 1337])
    worker = generator.Worker(ctx, seed=0)

    # Drop the Python-side reference to the context; the shared_ptr held
    # by the Worker must still keep it alive.
    del ctx
    gc.collect()

    # Trigger a call that reaches into the context indirectly via the
    # dispatcher; even the "not yet ported" throw path shouldn't crash
    # from a use-after-free in ctx_.
    cfg = generator.GeneratorConfig(graph_kind="erdos_renyi", batch_size=1)
    with pytest.raises(RuntimeError):
        worker.generate_batch(cfg)


def test_one_context_shared_by_many_workers(generator):
    ctx = generator.WorkerSharedContext()
    workers = [generator.Worker(ctx, seed=i) for i in range(4)]
    del ctx
    gc.collect()

    cfg = generator.GeneratorConfig(graph_kind="euclidean", batch_size=1)
    for w in workers:
        with pytest.raises(RuntimeError):
            w.generate_batch(cfg)
