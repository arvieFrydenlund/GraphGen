"""End-to-end tests for the khops_gen task.

khops_gen is a synthetic-sequence task (no graph): the sampler emits a
random prefix of raw vocab tokens, embeds a k-hop backtrace inside it,
and the model must produce the k ground-truth hops. These tests check
the generated batch's layout, the partition-sampler primitive, and use
`verify_khop_gens` to confirm the emitted prefixes actually decode to
the ground truths under the documented backtrace algorithm.
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


def _worker(generator, seed=0):
    ctx = generator.WorkerSharedContext()
    return generator.Worker(ctx, seed=seed)


def _cfg(generator, *, k=3, batch_size=8, right_side_connect=True,
         no_repeats=False, partition_method="uniform"):
    # Prefix long enough to make every (k in [k, k], P in [2k, 4k])
    # feasible. batch_size=8 gives a few permutations of segment
    # lengths in a single call.
    P_min = 4 * k          # comfortable slack over the 2*k lower bound
    P_max = 6 * k
    return generator.GeneratorConfig(
        graph_kind="khops_gen",
        task_kind="khops_gen",
        tokenization_mode="sean",
        batch_size=batch_size,
        min_khops=k, max_khops=k,
        min_prefix_length=P_min, max_prefix_length=P_max,
        right_side_connect=right_side_connect,
        khops_no_repeats=no_repeats,
        partition_method=partition_method,
    )


# ---- Partition primitive ------------------------------------------------

def test_partition_uniform_sums_and_positive(generator):
    parts = generator.uniform_random_int_partition(20, 5, seed=123)
    assert len(parts) == 5
    assert sum(parts) == 20
    assert all(p >= 1 for p in parts)


def test_partition_non_uniform_sums_and_positive(generator):
    parts = generator.uniform_random_int_partition(
        30, 4, seed=42, method="non_uniform")
    assert len(parts) == 4
    assert sum(parts) == 30
    assert all(p >= 1 for p in parts)


def test_partition_rejects_infeasible(generator):
    with pytest.raises((ValueError, RuntimeError)):
        generator.uniform_random_int_partition(3, 10, seed=1)


def test_partition_rejects_unknown_method(generator):
    with pytest.raises(ValueError):
        generator.uniform_random_int_partition(10, 3, method="banana")


# ---- Config validation --------------------------------------------------

def test_config_rejects_infeasible_partition(generator):
    with pytest.raises(ValueError, match="2 \\* max_khops"):
        generator.GeneratorConfig(
            graph_kind="khops_gen", task_kind="khops_gen",
            min_khops=3, max_khops=5,
            min_prefix_length=8, max_prefix_length=20)  # need >= 10


def test_config_rejects_bad_partition_method(generator):
    with pytest.raises(ValueError, match="partition_method"):
        generator.GeneratorConfig(
            graph_kind="khops_gen", task_kind="khops_gen",
            min_khops=2, max_khops=2,
            min_prefix_length=10, max_prefix_length=10,
            partition_method="banana")


# ---- generate_batch end-to-end -----------------------------------------

def test_generate_batch_returns_expected_keys(generator):
    cfg = _cfg(generator)
    result = _worker(generator).generate_batch(cfg)
    for key in ("src_tokens", "targets", "src_lengths",
                "num_nodes", "num_edges", "task_start_indices"):
        assert key in result, f"missing key: {key}"


def test_num_nodes_zero_for_khops_gen(generator):
    cfg = _cfg(generator)
    result = _worker(generator).generate_batch(cfg)
    # khops_gen has no graph -- every item's node count is 0.
    assert np.all(result["num_nodes"] == 0)
    assert np.all(result["num_edges"] == 0)


def test_src_lengths_within_expected_range(generator):
    k = 3
    cfg = _cfg(generator, k=k)
    result = _worker(generator).generate_batch(cfg)
    # Layout: BOS + KhopsPrefix(P-1) + QS + Q(1) + QE + TS + Target(k) + TE + EOS
    #         = 1 + (P-1) + 1 + 1 + 1 + 1 + k + 1 + 1 = P + k + 6.
    P_min, P_max = 4 * k, 6 * k
    lo = P_min + k + 6
    hi = P_max + k + 6
    lens = result["src_lengths"]
    assert lens.min() >= lo
    assert lens.max() <= hi


@pytest.mark.parametrize("right_side_connect", [True, False])
def test_batch_verifies_via_verify_khop_gens(generator, right_side_connect):
    """The ground truth of a khops_gen batch must be recoverable by
    walking backward from the cursor per the documented rule; running
    verify_khop_gens over the emitted prefix segment should return all
    ones."""
    k = 3
    cfg = _cfg(generator, k=k, batch_size=16,
               right_side_connect=right_side_connect)
    result = _worker(generator, seed=7).generate_batch(cfg)

    src   = result["src_tokens"].astype(np.int32)       # (B, T[, C])
    if src.ndim == 3:
        src = src[..., 0]                                # SEAN uses col 0
    lens  = result["src_lengths"].astype(np.int32)
    tsi   = result["task_start_indices"].astype(np.int32)

    B = src.shape[0]
    # Prefix as understood by verify_khop_gens is
    # [seg_tokens..., <qs>, cursor, <qe>] -- the slice
    # [BOS+1 : TS] where TS = tsi - 1 (task_start_indices points at
    # the first target token, TS marker sits one slot earlier).
    max_prefix = int((tsi - 2).max())
    prefixes = np.zeros((B, max_prefix), dtype=np.int32)
    prefix_lens = np.zeros(B, dtype=np.int32)
    for b in range(B):
        pfx = src[b, 1:tsi[b] - 1]     # drop BOS and TS; include <qe>
        prefixes[b, :len(pfx)] = pfx
        prefix_lens[b] = len(pfx)

    # Ground truths sit at src[b, tsi[b] : tsi[b] + k] -- tsi already
    # points at the first target token.
    gts = np.stack([src[b, tsi[b] : tsi[b] + k] for b in range(B)])

    ok = generator.verify_khop_gens(
        prefixes, prefix_lens, gts,
        right_side_connect=right_side_connect)
    assert np.all(ok == 1), f"verify_khop_gens failed: {ok}"


def test_determinism_same_seed(generator):
    cfg = _cfg(generator)
    a = _worker(generator, seed=99).generate_batch(cfg)
    b = _worker(generator, seed=99).generate_batch(cfg)
    np.testing.assert_array_equal(a["src_tokens"], b["src_tokens"])
    np.testing.assert_array_equal(a["targets"],    b["targets"])


def test_no_repeats_ground_truths_unique(generator):
    k = 4
    cfg = _cfg(generator, k=k, batch_size=32, no_repeats=True)
    result = _worker(generator, seed=1).generate_batch(cfg)
    src = result["src_tokens"]
    if src.ndim == 3:
        src = src[..., 0]
    tsi = result["task_start_indices"]
    for b in range(src.shape[0]):
        gts = src[b, tsi[b] : tsi[b] + k]
        assert len(set(int(x) for x in gts)) == k, \
            f"khops_no_repeats produced dup ground truths: {gts}"
