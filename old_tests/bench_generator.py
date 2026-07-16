"""
End-to-end timing baselines for the four graph generators we care about:
``erdos_renyi_n``, ``euclidean_n``, ``random_tree_n``, ``path_star_n``.

We deliberately time the *whole* pipeline -- sample -> task -> tokenize ->
Python dict marshalling -- because end-to-end throughput is what user code
experiences.

Scope (per user request):
  * scratchpad_type = "none" only  (no BFS/DFS scratchpad sweep)
  * balanced_n excluded (not important now)
  * shortest_path task only

The C++ ``*_n`` bindings accept a ``py::kwargs`` sink and pull several
config values from it (``min_path_length``, ``max_path_length``, various
``use_*_structure`` flags, ...). Those defaults live in the argparse
parser, so we start from ``base_kwargs`` (parser defaults, provided by
conftest.py) and just override the axes we're sweeping.

Run:
    bash old_tests/run_baseline.sh
or:
    pytest old_tests/bench_generator.py --benchmark-only \\
        --benchmark-json=old_tests/baselines/<name>.json
"""
from __future__ import annotations

import pytest

# ----------------------------------------------------------------------------
# Parameter matrices
# ----------------------------------------------------------------------------

# Node-count sweep for the three generators parametrized by (min, max) nodes.
# Names carry into pytest IDs so JSON output is easy to grep.
NODE_RANGES = [
    pytest.param((15, 25),   id="small"),    #  ~20 nodes
    pytest.param((50, 75),   id="medium"),   #  ~60 nodes
    pytest.param((150, 200), id="large"),    # ~175 nodes
    pytest.param((400, 500), id="xl"),       # ~450 nodes -- cache-pressure regime
]

# path_star_n takes (min_arms, max_arms, min_arm_length, max_arm_length)
# instead of node counts. Configs below target the same total-node buckets
# as NODE_RANGES so results are broadly comparable. The XL bucket is capped
# so that max_arms * max_arm_length + 1 stays under the 600-token vocab
# installed in conftest.py (matches the ER/euclidean XL of ~500 nodes).
ARM_CONFIGS = [
    pytest.param((2, 4,  4,  8),  id="small"),   # ~2..32 nodes
    pytest.param((3, 5,  8, 15),  id="medium"),  # ~24..75 nodes
    pytest.param((4, 8, 15, 30),  id="large"),   # ~60..240 nodes
    pytest.param((4, 8, 30, 60),  id="xl"),      # ~120..481 nodes
]

BATCH_SIZES = [
    pytest.param(64,   id="b64"),
    pytest.param(256,  id="b256"),
    pytest.param(1024, id="b1024"),
]


def _prepare_kwargs(base_kwargs: dict, batch_size: int, max_nodes_hint: int,
                    **overrides) -> dict:
    """Copy parser defaults, apply generator-common overrides + caller extras.

    ``max_edges`` is scaled with node count so the XL bucket doesn't hit the
    default cap and resample forever.
    """
    kw = dict(base_kwargs)
    kw["batch_size"] = batch_size
    kw["max_edges"] = max(kw.get("max_edges", 512), max_nodes_hint * 4)
    kw["task_type"] = "shortest_path"
    kw["scratchpad_type"] = "none"
    kw.update(overrides)
    return kw


def _assert_nonempty_batch(result, expected_batch: int):
    """Cheap correctness check so silent regressions don't hide behind fast times."""
    assert isinstance(result, dict), f"expected dict, got {type(result)}"
    assert result, "generator returned empty dict"
    # First array-valued entry should have batch_size rows.
    for v in result.values():
        try:
            first_dim = v.shape[0]
        except AttributeError:
            continue
        assert first_dim == expected_batch, (
            f"batch dim mismatch: got {first_dim}, expected {expected_batch}"
        )
        return
    pytest.fail("no array-valued entries in generator output")


# ----------------------------------------------------------------------------
# Benchmarks
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("node_range", NODE_RANGES)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_bench_erdos_renyi_n(benchmark, generator_module, base_kwargs, seed,
                             node_range, batch_size):
    min_n, max_n = node_range
    kwargs = _prepare_kwargs(
        base_kwargs, batch_size, max_n,
        min_num_nodes=min_n, max_num_nodes=max_n,
        p=-1.0, c_min=75, c_max=125,
    )
    result = benchmark(lambda: generator_module.erdos_renyi_n(**kwargs))
    _assert_nonempty_batch(result, batch_size)


@pytest.mark.parametrize("node_range", NODE_RANGES)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_bench_euclidean_n(benchmark, generator_module, base_kwargs, seed,
                           node_range, batch_size):
    min_n, max_n = node_range
    kwargs = _prepare_kwargs(
        base_kwargs, batch_size, max_n,
        min_num_nodes=min_n, max_num_nodes=max_n,
        dim=2, radius=-1.0, c_min=75, c_max=125,
    )
    # ``dims`` from the parser conflicts with the C++ arg name ``dim`` --
    # remove the parser-only spelling so we don't get "multiple values".
    kwargs.pop("dims", None)
    result = benchmark(lambda: generator_module.euclidean_n(**kwargs))
    _assert_nonempty_batch(result, batch_size)


@pytest.mark.parametrize("node_range", NODE_RANGES)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_bench_random_tree_n(benchmark, generator_module, base_kwargs, seed,
                             node_range, batch_size):
    min_n, max_n = node_range
    kwargs = _prepare_kwargs(
        base_kwargs, batch_size, max_n,
        min_num_nodes=min_n, max_num_nodes=max_n,
        max_degree=3, max_depth=7, bernoulli_p=0.5,
    )
    result = benchmark(lambda: generator_module.random_tree_n(**kwargs))
    _assert_nonempty_batch(result, batch_size)


@pytest.mark.parametrize("arm_config", ARM_CONFIGS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_bench_path_star_n(benchmark, generator_module, base_kwargs, seed,
                           arm_config, batch_size):
    min_arms, max_arms, min_arm_len, max_arm_len = arm_config
    # Rough upper bound on node count: max_arms * max_arm_len + 1 (center).
    max_n_hint = max_arms * max_arm_len + 1
    kwargs = _prepare_kwargs(
        base_kwargs, batch_size, max_n_hint,
        min_num_arms=min_arms, max_num_arms=max_arms,
        min_arm_length=min_arm_len, max_arm_length=max_arm_len,
    )
    result = benchmark(lambda: generator_module.path_star_n(**kwargs))
    _assert_nonempty_batch(result, batch_size)
