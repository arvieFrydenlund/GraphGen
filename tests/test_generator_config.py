"""Round-trip and validation tests for ``generator.GeneratorConfig``.

Covers the pybind kwargs constructor, cross-field ``validate()`` rules, and
the ``to_dict()`` round-trip for both plain and optional fields.
"""
from __future__ import annotations

import pytest

from get_generator_module import get_generator_module


@pytest.fixture(scope="module")
def generator():
    get_generator_module()
    import generator as g  # noqa: WPS433 -- deferred import: needs build first
    return g


# ---------------------------------------------------------------------------
# Default construction and defaults match the C++ header
# ---------------------------------------------------------------------------

def test_default_construction(generator):
    cfg = generator.GeneratorConfig()
    assert cfg.validate() == "", "empty config should validate"

    # A representative subset -- if these ever drift from the header we want
    # to notice, but we don't need to re-list every single field.
    assert cfg.batch_size == 256
    assert cfg.max_edges == 512
    assert cfg.max_attempts == 1000
    assert cfg.min_num_nodes == -1
    assert cfg.max_num_nodes == -1
    assert cfg.graph_kind == "none"
    assert cfg.task_kind == "shortest_path"
    assert cfg.scratchpad_kind == "none"
    assert cfg.tokenization_mode == "sean"
    assert cfg.return_pos_ids is True
    assert cfg.num_thinking_tokens == 0

    # Optionals default to None on the Python side.
    assert cfg.min_path_length is None
    assert cfg.max_path_length is None
    assert cfg.edge_prob is None
    assert cfg.min_khops is None
    assert cfg.dims is None


# ---------------------------------------------------------------------------
# Kwargs constructor + round-trip via to_dict()
# ---------------------------------------------------------------------------

def test_kwargs_constructor_populates_all_layers(generator):
    cfg = generator.GeneratorConfig(
        min_num_nodes=50,
        max_num_nodes=100,
        batch_size=64,
        max_edges=1024,
        graph_kind="erdos_renyi",
        task_kind="shortest_path",
        scratchpad_kind="none",
        # tokenization
        num_thinking_tokens=4,
        tokenization_mode="stan",
        # task-specific
        min_path_length=2,
        max_path_length=8,
        task_sample_dist=[0.5, 0.5],
        # graph-specific
        edge_prob=0.01)
    d = cfg.to_dict()

    # Shared / dispatch
    assert d["min_num_nodes"] == 50
    assert d["max_num_nodes"] == 100
    assert d["batch_size"] == 64
    assert d["max_edges"] == 1024
    assert d["graph_kind"] == "erdos_renyi"
    assert d["task_kind"] == "shortest_path"
    assert d["scratchpad_kind"] == "none"
    # Tokenization
    assert d["num_thinking_tokens"] == 4
    assert d["tokenization_mode"] == "stan"
    # Task-specific (Option 1: optionals surface as real values when set)
    assert d["min_path_length"] == 2
    assert d["max_path_length"] == 8
    assert d["task_sample_dist"] == pytest.approx([0.5, 0.5])
    # Graph-specific
    assert d["edge_prob"] == pytest.approx(0.01)

    # Unset optionals stay None
    assert d["min_khops"] is None
    assert d["given_query"] is None
    assert d["dims"] is None


@pytest.mark.parametrize(
    "legacy,replacement",
    [
        ("graph_type",      "graph_kind"),
        ("task_type",       "task_kind"),
        ("scratchpad_type", "scratchpad_kind"),
        # The eight tokenization flags below were absorbed into
        # tokenization_mode; each must be rejected up front.
        ("concat_edges",           "tokenization_mode"),
        ("is_flat_model",          "tokenization_mode"),
        ("use_edges_invariance",   "tokenization_mode"),
        ("use_node_invariance",    "tokenization_mode"),
        ("use_graph_invariance",   "tokenization_mode"),
        ("use_query_invariance",   "tokenization_mode"),
        ("use_graph_structure",    "tokenization_mode"),
        ("use_full_structure",     "tokenization_mode"),
        # The three BFS-scratchpad bools folded into bfs_scratchpad_style
        # (mutually exclusive variants of the per-vertex block).
        ("include_queue",             "bfs_scratchpad_style"),
        ("reverse_adjacency_lists",   "bfs_scratchpad_style"),
        ("duplicate_adjacency_lists", "bfs_scratchpad_style"),
        # Two flags dropped outright with an informative marker.
        ("use_unique_depth_markers", "removed"),
        ("scratchpad_as_prefix",     "removed"),
        ("is_causal",                "removed"),
        ("is_direct_ranking",        "removed"),
        ("sort_adjacency_lists",     "removed"),
        # Renamed to be symmetric with include_nodes_in_graph_tokenization.
        ("no_graph",        "include_graph_in_graph_tokenization"),
        ("duplicate_edges", "include_duplicate_edges_in_graph_tokenization"),
    ])
def test_legacy_renamed_kwargs_are_rejected(generator, legacy, replacement):
    """Renamed/absorbed config fields must be rejected at construction
    time with a message pointing at the new name -- silently ignoring
    them would leave the field at its default and fail much later in the
    pipeline."""
    # Pass a value of a type the field would have accepted historically:
    # boolean for the tokenization/scratchpad flags, string for the 3 *_type ones.
    value: object = "path_star" if legacy in ("graph_type", "task_type", "scratchpad_type") else True
    with pytest.raises(ValueError) as exc_info:
        generator.GeneratorConfig(**{legacy: value})
    msg = str(exc_info.value)
    assert legacy in msg
    assert replacement in msg


def test_tokenization_mode_rejects_unknown_value(generator):
    """The string must name a known TokenizationMode -- validate()
    catches typos rather than deferring the failure to layout time."""
    with pytest.raises(ValueError) as exc_info:
        generator.GeneratorConfig(tokenization_mode="not_a_mode")
    msg = str(exc_info.value)
    assert "tokenization_mode" in msg
    assert "not_a_mode" in msg


def test_tokenization_mode_accepts_sean_and_stan(generator):
    for mode in ("sean", "stan"):
        cfg = generator.GeneratorConfig(tokenization_mode=mode)
        assert cfg.tokenization_mode == mode
        assert cfg.to_dict()["tokenization_mode"] == mode


def test_bfs_task_with_bfs_scratchpad_is_rejected_as_redundant(generator):
    """The BFS task's target IS the BFS visit order; layering a BFS
    scratchpad on top would just duplicate the answer. Validation
    catches this at construction rather than letting the pipeline
    produce nonsense training data."""
    with pytest.raises(ValueError) as exc_info:
        generator.GeneratorConfig(
            task_kind="bfs",
            scratchpad_kind="bfs",
            min_num_nodes=5,
            max_num_nodes=5)
    msg = str(exc_info.value)
    assert "bfs" in msg
    assert "redundant" in msg


def test_empty_list_kwargs_treated_as_unset(generator):
    """Matches V1's TaskArgs behavior: an empty list becomes None rather than
    an empty vector, so downstream code can rely on `has_value()`."""
    cfg = generator.GeneratorConfig(task_sample_dist=[], given_query=[], probs=[])
    d = cfg.to_dict()
    assert d["task_sample_dist"] is None
    assert d["given_query"] is None
    assert d["probs"] is None


def test_unknown_kwarg_is_ignored(generator):
    """Unknown kwargs mirror the V1 kwargs-sink behavior and do not raise."""
    cfg = generator.GeneratorConfig(min_num_nodes=10, definitely_not_a_field=42)
    assert cfg.min_num_nodes == 10


def test_readwrite_fields_are_mutable(generator):
    cfg = generator.GeneratorConfig()
    cfg.min_num_nodes = 25
    cfg.max_num_nodes = 50
    cfg.graph_kind = "euclidean"
    cfg.edge_prob = 0.02
    assert cfg.min_num_nodes == 25
    assert cfg.max_num_nodes == 50
    assert cfg.graph_kind == "euclidean"
    assert cfg.edge_prob == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# validate() -- cross-field constraints
# ---------------------------------------------------------------------------

def test_validate_accepts_typical_er_shortest_path(generator):
    cfg = generator.GeneratorConfig(
        min_num_nodes=50,
        max_num_nodes=100,
        graph_kind="erdos_renyi",
        task_kind="shortest_path",
        edge_prob=0.05,
        min_path_length=1,
        max_path_length=10)
    assert cfg.validate() == ""


@pytest.mark.parametrize(
    "kwargs, expected_fragment",
    [
        ({"batch_size": 0}, "batch_size"),
        ({"max_edges": -1}, "max_edges"),
        ({"num_thinking_tokens": -1}, "num_thinking_tokens"),
        (
            {"min_num_nodes": 100, "max_num_nodes": 50},
            "max_num_nodes < min_num_nodes"),
        (
            {
                "task_kind": "shortest_path",
                "min_path_length": 8,
                "max_path_length": 3,
            },
            "min_path_length > max_path_length"),
        (
            {"graph_kind": "erdos_renyi", "edge_prob": 1.5},
            "edge_prob"),
        (
            {
                "graph_kind": "path_star",
                "min_arms": 5,
                "max_arms": 3,
            },
            "min_arms > max_arms"),
        (
            {"task_kind": "khops"},  # missing min_khops / max_khops
            "requires min_khops"),
    ])
def test_validate_rejects_bad_configs(generator, kwargs, expected_fragment):
    with pytest.raises(ValueError) as exc_info:
        generator.GeneratorConfig(**kwargs)
    assert expected_fragment in str(exc_info.value)


def test_validate_accepts_khops_when_all_required_present(generator):
    cfg = generator.GeneratorConfig(
        task_kind="khops",
        min_khops=2,
        max_khops=5,
        # min_prefix_length must be >= 2 * max_khops for partition feasibility.
        min_prefix_length=10,
        max_prefix_length=20)
    assert cfg.validate() == ""


def test_repr_mentions_key_fields(generator):
    cfg = generator.GeneratorConfig(
        graph_kind="erdos_renyi",
        task_kind="shortest_path",
        min_num_nodes=10,
        max_num_nodes=20)
    r = repr(cfg)
    assert "GeneratorConfig(" in r
    assert "erdos_renyi" in r
    assert "shortest_path" in r
    assert "min_num_nodes=10" in r
