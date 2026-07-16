"""Worker skeleton tests.

At this stage the Worker parses ``cfg.graph_kind`` into the C++ GraphKind
enum and hands off to GraphSampler. The per-kind sampler bodies are still
stubs, so ``generate_batch`` is expected to raise ``RuntimeError`` naming
the sampler that isn't implemented yet. An unrecognised ``graph_kind``
short-circuits earlier (in ``graph_kind_from_string``) and surfaces as a
``ValueError``.

They serve as a regression fence -- if a topology gets partially wired up
but silently no-ops, this suite catches it.
"""
from __future__ import annotations

import pytest

from get_generator_module import get_generator_module


@pytest.fixture(scope="module")
def generator():
    get_generator_module()
    import generator as g  # noqa: WPS433 -- deferred import: needs build first
    return g


@pytest.fixture
def worker(generator):
    ctx = generator.WorkerSharedContext()
    return generator.Worker(ctx, seed=42)


RECOGNISED_KINDS = [
    "erdos_renyi",
    "euclidean",
    "random_tree",
    "path_star",
    "balanced",
    "khops",
    "khops_gen",
]


def test_shared_context_construction(generator):
    ctx = generator.WorkerSharedContext()
    assert ctx is not None


def test_worker_construction(generator):
    ctx = generator.WorkerSharedContext()
    w = generator.Worker(ctx, seed=0)
    assert w is not None


@pytest.mark.parametrize("kind", RECOGNISED_KINDS)
def test_generate_batch_stub_raises_per_kind(generator, worker, kind):
    cfg = generator.GeneratorConfig(graph_kind=kind, batch_size=1)
    with pytest.raises(RuntimeError) as exc_info:
        worker.generate_batch(cfg)

    msg = str(exc_info.value)
    assert f"sample_{kind}" in msg, f"expected 'sample_{kind}' in message, got: {msg}"
    assert "not implemented" in msg, f"expected 'not implemented' in message, got: {msg}"


def test_generate_batch_unknown_kind_raises(generator, worker):
    cfg = generator.GeneratorConfig(graph_kind="not_a_real_kind", batch_size=1)
    # graph_kind_from_string throws std::invalid_argument, which pybind11
    # translates to ValueError -- distinct from the RuntimeError sampler
    # stubs raise, so callers can tell "typo" from "not yet implemented".
    with pytest.raises(ValueError) as exc_info:
        worker.generate_batch(cfg)

    msg = str(exc_info.value)
    assert "unknown graph_kind" in msg, msg
    assert "not_a_real_kind" in msg, msg
