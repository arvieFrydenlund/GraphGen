"""
Session setup for the V1 (pre-refactor) timing baselines.

Boots the `generator` C++ module once per session, installs a default
dictionary/pos-dictionary (required by all `*_n` graph generators), and
provides a `seed` fixture so every benchmark iteration starts from a known
random state.

Kept intentionally small: no monkeypatching, no mocking. Just the minimum
needed for the parametrized benchmarks in bench_generator.py to run.
"""
from __future__ import annotations

import os
import sys

import pytest

# Make the repo root importable so `from get_generator_module import ...` works
# whether pytest is invoked from the repo root or from old_tests/.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


@pytest.fixture(scope="session")
def generator_module():
    """Import & initialize the C++ generator once per session.

    Sets up the default token dictionary sized for our largest benchmark
    (up to ~500 nodes) and a custom position dictionary sized for the XL
    bucket (default's ``graph_end - graph_start = 739`` is too small for
    500-node graphs with edge-rich tokenizations). Individual benchmarks
    re-seed inside each iteration for determinism.
    """
    from get_generator_module import get_generator_module

    get_generator_module()
    import generator

    # Vocab has to cover the largest max_num_nodes we benchmark (~500) plus a
    # margin for special tokens and shuffling headroom.
    generator.set_default_dictionary(
        max_num_nodes=600,
        extra_after=0,
        extra_after_symbol="D",
    )

    # Custom position dictionary. Same keys/ordering as
    # ``set_default_pos_dictionary`` in dictionaries.h, but with the graph
    # range expanded from 739 to 3000 slots to cover XL configs.
    pos_dict = {
        "pad": 0,
        "query_invariance": 1,
        "edge_invariance": 2,
        "node_invariance": 3,
        "graph_invariance": 4,
        "scratchpad_invariance": 5,
        "task_invariance": 6,
        "misc_start": 11,
        "misc_end": 49,
        "query_start": 50,
        "query_end": 199,
        "graph_start": 200,
        "graph_end": 3199,        # was 939 -- widened for XL bucket
        "graph_sub_start": 3200,
        "graph_sub_end": 3209,
        "thinking_start": 3210,
        "thinking_end": 3259,
        "task_start": 3260,
        "task_end": 7259,
    }
    generator.set_pos_dictionary(pos_dict, verbose=False)
    return generator


@pytest.fixture(scope="session")
def base_kwargs(generator_module) -> dict:
    """Full parser-default kwargs dict, sanitized for direct forwarding.

    Several C++ generator params (``min_path_length``, ``max_path_length``,
    various ``use_*_structure`` flags, ...) don't appear in the pybind11
    signatures but *are* required -- they flow through a ``py::kwargs`` sink
    into ``Args``. The parser is the source of truth for their defaults, so
    we build the base kwargs from it once and let each benchmark override.
    """
    parser = generator_module.get_args_parser()
    args = parser.parse_args([])
    kwargs = vars(args)
    # C++ expects an empty list, not None, for the "no task sampling" case.
    if kwargs.get("task_sample_dist") is None:
        kwargs["task_sample_dist"] = []
    # ``graph_type`` is a Python-side dispatcher key only -- the C++
    # ``*_n`` functions don't take it (they *are* the dispatch), and
    # passing it as an unknown kwarg is fine (it goes into the kwargs sink),
    # but drop it for tidiness.
    kwargs.pop("graph_type", None)
    return kwargs


@pytest.fixture
def seed(generator_module):
    """Set a deterministic seed at the start of every benchmark iteration."""
    generator_module.set_seed(11723)
    return 11723
