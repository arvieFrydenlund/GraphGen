# V1 pre-refactor timing baselines

Frozen end-to-end benchmarks of the V1 (pre-refactor) generator, run once
before each step of PLAN.md so we can detect regressions and prove
speedups from Steps 3-12.

## What is timed

The four graph generators that will survive the refactor:

* `erdos_renyi_n`
* `euclidean_n`
* `random_tree_n`
* `path_star_n`

Whole pipeline per call: sample -> task (`shortest_path`) -> tokenize ->
numpy dict marshalling. That is exactly the hot path Steps 3-8 rewrite,
so end-to-end throughput here is the primary metric.

## Sweep matrix

| Axis          | Values                                                       |
| ------------- | ------------------------------------------------------------ |
| Node buckets  | small (15-25), medium (50-75), large (150-200), xl (400-500) |
| Batch sizes   | 64, 256, 1024                                                |
| Task          | `shortest_path`                                              |
| Scratchpad    | `none`                                                       |
| Seed          | fixed (`11723`) per iteration                                |

`path_star_n` uses arm/arm-length ranges chosen to hit the same total-node
buckets as the other three -- see `ARM_CONFIGS` in
[bench_generator.py](bench_generator.py).

## What is *not* timed

* `balanced_n` -- out of scope for the refactor.
* BFS / DFS scratchpads -- can add later if useful.
* Per-phase (sample / task / tokenize) breakdown -- deferred until Steps
  3-8, where each phase will be its own C++ function object and can
  carry a scoped timer without touching V1 headers we're about to delete.
* Memory / RSS -- not on the refactor's critical path.
* Multi-worker throughput -- comes with Steps 11-12 (GIL release +
  workers).

## Install

```bash
uv pip install -e ".[dev]"      # or:  pip install -e ".[dev]"
```

Adds `pytest` and `pytest-benchmark` to the venv on top of the
already-pinned build stack.

## Run once, save a baseline

```bash
bash old_tests/run_baseline.sh                 # tags file by date + git SHA
# or, with an explicit label:
bash old_tests/run_baseline.sh step1a_v1
```

JSON drops into `old_tests/baselines/<label>.json`.

## Compare a new run against a saved baseline

```bash
pytest old_tests/bench_generator.py \
    --benchmark-only \
    --benchmark-compare=old_tests/baselines/step1a_v1.json \
    --benchmark-compare-fail=mean:10%   # optional: fail on >10% regression
```

`pytest-benchmark` prints a side-by-side table with mean / median / p95 /
op/s and highlights improvements vs. regressions.

## Ad-hoc: run only one generator

```bash
pytest old_tests/bench_generator.py::test_bench_erdos_renyi_n \
    --benchmark-only
```
