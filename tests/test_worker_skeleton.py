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
    "random_tree",
    "balanced",
    # khops / khops_gen are now implemented -- see IMPLEMENTED_KINDS
    # in the paired test below (they use empty graphs so they don't
    # fit the "generate_batch on a stub graph_kind throws" pattern
    # this suite covers).
]

# Kinds whose per-topology sampler is fully wired up; generate_batch should
# Kinds whose per-topology sampler is fully wired up; generate_batch should
# succeed (returning a possibly-empty dict at this stage of the pipeline).
IMPLEMENTED_KINDS = ["erdos_renyi", "euclidean", "path_star"]


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


@pytest.mark.parametrize("kind", IMPLEMENTED_KINDS)
def test_generate_batch_implemented_kind_does_not_throw(generator, worker, kind):
    # Kind-specific kwargs; the shared fields (batch_size, vocab) stay
    # the same. Each sampler validates its own required fields, so we
    # only fill in what that sampler needs.
    kind_kwargs = {
        "erdos_renyi": dict(
            min_num_nodes=10, max_num_nodes=10,
            edge_prob=0.5),
        "euclidean": dict(
            min_num_nodes=10, max_num_nodes=10),
        "path_star": dict(
            directed=True,
            min_arms=2, max_arms=3,
            min_arm_length=2, max_arm_length=4),
    }
    cfg = generator.GeneratorConfig(
        graph_kind=kind,
        batch_size=1,
        **kind_kwargs[kind])
    result = worker.generate_batch(cfg)
    assert isinstance(result, dict)


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
