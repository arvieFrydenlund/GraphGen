"""End-to-end tests for the per-position khops task.

khops (as opposed to khops_gen) asks the model to emit, at every input
position i within a length-P sequence, the k-hop back-target token
computed from the seq. The Query section holds a single "D<k>" depth
marker; the Target section is the seq itself; per-position hop labels
live in the (B, max_gen_len, max_labels) targets tensor.

These tests validate the layout, the two sampling paths
(permutation_version + standard), the mask flags (mask_to_size,
mask_to_vocab_size), intermediate_labels, and that the per-position
labels actually match the documented back-hop rule applied to the
emitted seq.
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


def _worker(generator, seed=0, *, max_k=3):
    """Build a Worker with a WorkerSharedContext whose dictionary
    includes the D<k> depth-marker extras up to `max_k`. The default
    dictionary has `extra_after=0` so callers must opt in."""
    ctx = generator.WorkerSharedContext()
    ctx.set_default_dictionary(max_num_nodes=50,
                               extra_after=max_k + 1,
                               extra_after_symbol="D")
    return generator.Worker(ctx, seed=seed)


def _cfg(generator, *, k=3, batch_size=4, permutation_version=False,
         right_side_connect=True, intermediate_labels=False,
         mask_to_size=None, mask_to_vocab_size=None,
         P_min=None, P_max=None):
    kwargs = dict(
        graph_kind="khops",
        task_kind="khops",
        tokenization_mode="sean",
        batch_size=batch_size,
        min_khops=k, max_khops=k,
        permutation_version=permutation_version,
        right_side_connect=right_side_connect,
        intermediate_labels=intermediate_labels,
    )
    if not permutation_version:
        kwargs["min_prefix_length"] = P_min or 4 * k
        kwargs["max_prefix_length"] = P_max or 6 * k
    if mask_to_size is not None:
        kwargs["mask_to_size"] = mask_to_size
    if mask_to_vocab_size is not None:
        kwargs["mask_to_vocab_size"] = mask_to_vocab_size
    return generator.GeneratorConfig(**kwargs)


# ---- Config validation --------------------------------------------------

def test_config_khops_perm_does_not_require_prefix_length(generator):
    cfg = generator.GeneratorConfig(
        task_kind="khops", min_khops=2, max_khops=3,
        permutation_version=True)
    assert cfg.validate() == ""


def test_config_khops_standard_requires_prefix_length(generator):
    with pytest.raises(ValueError):
        generator.GeneratorConfig(
            task_kind="khops", min_khops=2, max_khops=3,
            permutation_version=False)


# ---- Layout + shape -----------------------------------------------------

def test_generate_batch_returns_expected_keys(generator):
    cfg = _cfg(generator)
    result = _worker(generator).generate_batch(cfg)
    for key in ("src_tokens", "targets", "src_lengths",
                "task_start_indices", "num_nodes"):
        assert key in result, f"missing key: {key}"


def test_num_nodes_zero_for_khops(generator):
    cfg = _cfg(generator)
    result = _worker(generator).generate_batch(cfg)
    assert (result["num_nodes"] == 0).all()


def test_src_lengths_within_expected_range(generator):
    k = 3
    cfg = _cfg(generator, k=k)
    result = _worker(generator).generate_batch(cfg)
    # Layout: BOS + QS + Q(1=D_k) + QE + TS + Target(P) + TE + EOS = P + 7.
    P_min, P_max = 4 * k, 6 * k
    lens = result["src_lengths"]
    assert lens.min() >= P_min + 7
    assert lens.max() <= P_max + 7


# ---- Query section = D_k marker -----------------------------------------

def test_query_section_holds_dk_marker(generator):
    """The single Query token must be the D<k> extras marker."""
    k = 3
    ctx = generator.WorkerSharedContext()
    ctx.set_default_dictionary(max_num_nodes=50, extra_after=k + 1, extra_after_symbol="D")
    w = generator.Worker(ctx, seed=42)
    cfg = _cfg(generator, k=k, batch_size=4)
    result = w.generate_batch(cfg)
    src = result["src_tokens"]
    if src.ndim == 3:
        src = src[..., 0]
    tsi = result["task_start_indices"]
    dk_id = ctx.token_dict[f"D{k}"]
    # Layout: [BOS, QS, D_k, QE, TS, seq..., TE, EOS]
    # tsi points at first target = seq[0]. TS is at tsi-1, QE at tsi-2,
    # D_k at tsi-3.
    for b in range(src.shape[0]):
        assert int(src[b, tsi[b] - 3]) == dk_id, \
            f"item {b}: expected D{k}={dk_id} at Query slot, got {src[b, tsi[b] - 3]}"


# ---- Per-position hop labels match documented rule ----------------------

def _back_hops_from_seq(seq, k, right_side_connect):
    """Reference implementation of hops[k-1][i] for every seq position."""
    P = len(seq)
    offset = 1 if right_side_connect else -1
    back_pointer = [-1] * P
    hops = [[-1] * P for _ in range(k)]
    for i in range(P - 1, -1, -1):
        for j in range(i - 1, -1, -1):
            if seq[j] == seq[i]:
                ptr = j + offset
                if 0 <= ptr < P:
                    back_pointer[i] = ptr
                    hops[0][i] = seq[ptr]
                break
    cur = list(back_pointer)
    for h in range(1, k):
        for i in range(P):
            if cur[i] == -1:
                continue
            cur[i] = back_pointer[cur[i]]
            if cur[i] == -1:
                continue
            hops[h][i] = seq[cur[i]]
    return hops


@pytest.mark.parametrize("right_side_connect", [True, False])
def test_per_position_labels_match_reference(generator, right_side_connect):
    """The targets tensor at each seq position must equal hops[k-1][i]
    computed from the seq itself."""
    k = 3
    cfg = _cfg(generator, k=k, batch_size=6,
               right_side_connect=right_side_connect,
               permutation_version=False)
    result = _worker(generator, seed=11).generate_batch(cfg)

    src = result["src_tokens"]
    if src.ndim == 3:
        src = src[..., 0]
    tgt = result["targets"]         # (B, max_gen_len, max_labels)
    tsi = result["task_start_indices"]
    lens = result["src_lengths"]
    B = src.shape[0]

    # Same layout as above; seq spans tsi..len-3 (before TE, EOS).
    for b in range(B):
        seq = src[b, tsi[b] : lens[b] - 2].tolist()
        P = len(seq)
        assert P >= 2, f"item {b}: unexpectedly short seq"
        hops = _back_hops_from_seq(seq, k, right_side_connect)
        # gen region starts at prefix_len == tsi - 1 (position of TS).
        # gp for seq[i] is (tsi - (tsi-1)) + i = 1 + i.
        for i in range(P):
            gp = 1 + i
            got = int(tgt[b, gp, 0])
            want = hops[k - 1][i]
            if want < 0:
                # Chain didn't reach k hops -- expect PAD (mask).
                # PAD = 1 per WorkerSharedContext.
                assert got == 1, \
                    f"item {b}, pos {i}: chain incomplete -> want PAD (1), got {got}"
            else:
                assert got == want, \
                    f"item {b}, pos {i}: want {want}, got {got}"


# ---- Permutation version guarantees defined hops for tail positions ----

def test_permutation_version_tail_positions_defined(generator):
    """When permutation_version=True the seq is (k+1) copies of the
    vocab shuffled, so every position at index >= vocab_size must have
    a defined k-hop label (never PAD)."""
    k = 2
    cfg = _cfg(generator, k=k, batch_size=4, permutation_version=True)
    ctx = generator.WorkerSharedContext()
    ctx.set_default_dictionary(max_num_nodes=50, extra_after=k + 1, extra_after_symbol="D")
    w = generator.Worker(ctx, seed=99)
    result = w.generate_batch(cfg)

    src = result["src_tokens"]
    if src.ndim == 3:
        src = src[..., 0]
    tgt = result["targets"]
    tsi = result["task_start_indices"]
    lens = result["src_lengths"]

    # V = node-vocab size = max_vocab - num_special.
    V = ctx.max_vocab - ctx.num_special
    B = src.shape[0]
    for b in range(B):
        seq_len = int(lens[b]) - tsi[b] - 2
        # Under permutation_version total seq = (k+1) * V.
        assert seq_len == (k + 1) * V, \
            f"item {b}: perm seq_len={seq_len}, expected {(k+1)*V}"
        # For positions i >= k * V (last permutation block), hops[k-1][i]
        # must be defined. Sample a handful.
        for i in range(k * V, seq_len):
            gp = 1 + i
            assert int(tgt[b, gp, 0]) != 1, \
                f"item {b}, pos {i}: tail position should have a defined label"


# ---- Mask flags ---------------------------------------------------------

def test_mask_to_size_masks_early_positions(generator):
    """mask_to_size=T means only positions i > T get labels; earlier
    ones stay PAD in the targets tensor."""
    k = 2
    T = 8
    cfg = _cfg(generator, k=k, batch_size=4, mask_to_size=T,
               P_min=4 * k, P_max=6 * k)
    result = _worker(generator, seed=3).generate_batch(cfg)
    src = result["src_tokens"]
    if src.ndim == 3:
        src = src[..., 0]
    tgt = result["targets"]
    tsi = result["task_start_indices"]
    lens = result["src_lengths"]

    for b in range(src.shape[0]):
        P = int(lens[b]) - tsi[b] - 2
        for i in range(min(P, T + 1)):
            gp = 1 + i
            assert int(tgt[b, gp, 0]) == 1, \
                f"item {b}, pos {i} <= mask_to_size ({T}): expected PAD, got {tgt[b, gp, 0]}"


def test_mask_to_vocab_size_keeps_only_tail(generator):
    """mask_to_vocab_size means only positions with i >= P - vocab_size
    get labels."""
    k = 2
    ctx = generator.WorkerSharedContext()
    ctx.set_default_dictionary(max_num_nodes=50, extra_after=k + 1, extra_after_symbol="D")
    V = ctx.max_vocab - ctx.num_special
    # Choose P slightly larger than V so the mask actually clips.
    cfg = _cfg(generator, k=k, batch_size=4,
               mask_to_vocab_size=True,
               P_min=V + 5, P_max=V + 10)
    result = generator.Worker(ctx, seed=7).generate_batch(cfg)
    src = result["src_tokens"]
    if src.ndim == 3:
        src = src[..., 0]
    tgt = result["targets"]
    tsi = result["task_start_indices"]
    lens = result["src_lengths"]

    for b in range(src.shape[0]):
        P = int(lens[b]) - tsi[b] - 2
        # Positions i < P - V: masked.
        for i in range(P - V):
            gp = 1 + i
            assert int(tgt[b, gp, 0]) == 1, \
                f"item {b}, pos {i} < P-V ({P-V}): expected PAD, got {tgt[b, gp, 0]}"


# ---- intermediate_labels emits multiple labels per position ------------

def test_intermediate_labels_widens_targets(generator):
    """With intermediate_labels=True the targets tensor has room for
    up to k labels per position, and at least some later positions
    populate multiple columns."""
    k = 3
    cfg = _cfg(generator, k=k, batch_size=4, intermediate_labels=True,
               permutation_version=True)
    result = _worker(generator, seed=17).generate_batch(cfg)
    tgt = result["targets"]
    # max_labels axis should be >= 2 (some position has >1 label).
    assert tgt.shape[-1] >= 2, f"expected max_labels >= 2, got {tgt.shape}"
    # Confirm at least one position has a non-PAD value in column >= 1.
    has_multi = (tgt[..., 1:] != 1).any()
    assert has_multi, "expected at least one position with intermediate labels"


# ---- Determinism --------------------------------------------------------

def test_determinism_same_seed(generator):
    cfg = _cfg(generator, k=3, batch_size=4)
    a = _worker(generator, seed=42).generate_batch(cfg)
    b = _worker(generator, seed=42).generate_batch(cfg)
    np.testing.assert_array_equal(a["src_tokens"], b["src_tokens"])
    np.testing.assert_array_equal(a["targets"], b["targets"])
