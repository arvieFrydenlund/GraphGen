// Implementation for graphgen::GeneratorConfig. See generator_config.h.

#include "graphgen/generator_config.h"

#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace graphgen {

namespace py = pybind11;

namespace {

// ---- kwargs parsing helpers -------------------------------------------------
// Small local wrappers around py::kwargs lookups. Missing or None-valued keys
// leave `out` at its previous value (typically the struct-declared default).

template <typename T>
void set_from_kwargs(const py::kwargs &kw, const char *name, T &out) {
    if (kw.contains(name) && !kw[name].is_none()) {
        out = kw[name].cast<T>();
    }
}

template <typename T>
void set_optional_from_kwargs(const py::kwargs &kw, const char *name,
                              std::optional<T> &out) {
    if (!kw.contains(name) || kw[name].is_none()) {
        return;
    }
    out = kw[name].cast<T>();
}

// Lists that arrive as an empty Python list are treated as "unset", so
// downstream code can use `has_value()` to distinguish "caller did not
// override" from "caller passed an intentionally empty distribution".
template <typename T>
void set_optional_list_from_kwargs(const py::kwargs &kw, const char *name,
                                   std::optional<std::vector<T>> &out) {
    if (!kw.contains(name) || kw[name].is_none()) {
        return;
    }
    auto lst = kw[name].cast<py::list>();
    if (lst.empty()) {
        return;
    }
    out = kw[name].cast<std::vector<T>>();
}

// ---- to_dict helpers --------------------------------------------------------

template <typename T>
py::object opt_to_py(const std::optional<T> &v) {
    if (!v.has_value()) {
        return py::none();
    }
    return py::cast(*v);
}

}  // namespace

GeneratorConfig::GeneratorConfig(const py::kwargs &kw) {
    // --- Shared -----------------------------------------------------------
    set_from_kwargs(kw, "min_num_nodes", min_num_nodes);
    set_from_kwargs(kw, "max_num_nodes", max_num_nodes);
    set_from_kwargs(kw, "min_vocab",     min_vocab);
    set_from_kwargs(kw, "max_vocab",     max_vocab);
    set_from_kwargs(kw, "batch_size",    batch_size);
    set_from_kwargs(kw, "max_edges",     max_edges);
    set_from_kwargs(kw, "max_attempts",  max_attempts);

    // --- Dispatch ---------------------------------------------------------
    // Accept both the canonical "*_kind" names and the legacy "*_type" names
    // so existing callers that still pass task_type / graph_type keep working.
    set_from_kwargs(kw, "graph_kind",      graph_kind);
    set_from_kwargs(kw, "graph_type",      graph_kind);
    set_from_kwargs(kw, "task_kind",       task_kind);
    set_from_kwargs(kw, "task_type",       task_kind);
    set_from_kwargs(kw, "scratchpad_kind", scratchpad_kind);
    set_from_kwargs(kw, "scratchpad_type", scratchpad_kind);

    // --- Tokenization -----------------------------------------------------
    set_from_kwargs(kw, "is_causal",                           is_causal);
    set_from_kwargs(kw, "is_direct_ranking",                   is_direct_ranking);
    set_from_kwargs(kw, "query_at_end",                        query_at_end);
    set_from_kwargs(kw, "no_graph",                            no_graph);
    set_from_kwargs(kw, "concat_edges",                        concat_edges);
    set_from_kwargs(kw, "duplicate_edges",                     duplicate_edges);
    set_from_kwargs(kw, "include_nodes_in_graph_tokenization", include_nodes_in_graph_tokenization);
    set_from_kwargs(kw, "num_thinking_tokens",                 num_thinking_tokens);
    set_from_kwargs(kw, "scratchpad_as_prefix",                scratchpad_as_prefix);
    set_from_kwargs(kw, "is_flat_model",                       is_flat_model);
    set_from_kwargs(kw, "align_prefix_front_pad",              align_prefix_front_pad);

    // --- Pos ids ----------------------------------------------------------
    set_from_kwargs(kw, "return_pos_ids",       return_pos_ids);
    set_from_kwargs(kw, "use_edges_invariance", use_edges_invariance);
    set_from_kwargs(kw, "use_node_invariance",  use_node_invariance);
    set_from_kwargs(kw, "use_graph_invariance", use_graph_invariance);
    set_from_kwargs(kw, "use_query_invariance", use_query_invariance);
    set_from_kwargs(kw, "use_graph_structure",  use_graph_structure);
    set_from_kwargs(kw, "use_full_structure",   use_full_structure);

    // --- Task-specific ----------------------------------------------------
    set_optional_from_kwargs(kw, "min_path_length", min_path_length);
    set_optional_from_kwargs(kw, "max_path_length", max_path_length);

    set_optional_list_from_kwargs(kw, "task_sample_dist", task_sample_dist);
    set_optional_from_kwargs     (kw, "start_at_root",    start_at_root);
    set_optional_from_kwargs     (kw, "end_at_leaf",      end_at_leaf);
    set_optional_list_from_kwargs(kw, "probs",            probs);

    set_optional_from_kwargs     (kw, "min_query_size", min_query_size);
    set_optional_from_kwargs     (kw, "max_query_size", max_query_size);
    set_optional_list_from_kwargs(kw, "given_query",    given_query);

    set_optional_from_kwargs(kw, "min_khops",           min_khops);
    set_optional_from_kwargs(kw, "max_khops",           max_khops);
    set_optional_from_kwargs(kw, "min_prefix_length",   min_prefix_length);
    set_optional_from_kwargs(kw, "max_prefix_length",   max_prefix_length);
    set_optional_from_kwargs(kw, "right_side_connect",  right_side_connect);
    set_optional_from_kwargs(kw, "khops_no_repeats",    khops_no_repeats);
    set_optional_from_kwargs(kw, "permutation_version", permutation_version);
    set_optional_from_kwargs(kw, "mask_to_vocab_size",  mask_to_vocab_size);
    set_optional_from_kwargs(kw, "mask_to_size",        mask_to_size);
    set_optional_from_kwargs(kw, "intermediate_labels", intermediate_labels);
    set_optional_from_kwargs(kw, "partition_method",    partition_method);

    // --- Scratchpad -------------------------------------------------------
    set_optional_from_kwargs(kw, "sort_adjacency_lists",      sort_adjacency_lists);
    set_optional_from_kwargs(kw, "use_unique_depth_markers",  use_unique_depth_markers);
    set_optional_from_kwargs(kw, "stop_once_found",           stop_once_found);
    set_optional_from_kwargs(kw, "include_queue",             include_queue);
    set_optional_from_kwargs(kw, "reverse_adjacency_lists",   reverse_adjacency_lists);
    set_optional_from_kwargs(kw, "duplicate_adjacency_lists", duplicate_adjacency_lists);

    // --- Graph-kind-specific ---------------------------------------------
    set_optional_from_kwargs     (kw, "edge_prob",       edge_prob);

    set_optional_from_kwargs     (kw, "dim",             dim);
    set_optional_from_kwargs     (kw, "min_edge_length", min_edge_length);
    set_optional_from_kwargs     (kw, "max_edge_length", max_edge_length);
    set_optional_list_from_kwargs(kw, "dims",            dims);

    set_optional_from_kwargs(kw, "min_arms",       min_arms);
    set_optional_from_kwargs(kw, "max_arms",       max_arms);
    set_optional_from_kwargs(kw, "min_arm_length", min_arm_length);
    set_optional_from_kwargs(kw, "max_arm_length", max_arm_length);

    set_optional_from_kwargs(kw, "min_lookahead",     min_lookahead);
    set_optional_from_kwargs(kw, "max_lookahead",     max_lookahead);
    set_optional_from_kwargs(kw, "min_noise_reserve", min_noise_reserve);
    set_optional_from_kwargs(kw, "max_num_parents",   max_num_parents);
    set_optional_from_kwargs(kw, "max_noise",         max_noise);

    // --- Debug ------------------------------------------------------------
    set_from_kwargs(kw, "print_cpp_args", print_cpp_args);

    // --- Validate on construction ----------------------------------------
    if (auto err = validate(); !err.empty()) {
        throw std::invalid_argument("GeneratorConfig: " + err);
    }
}

std::string GeneratorConfig::validate() const {
    // Batch sizing sanity.
    if (batch_size <= 0) {
        return "batch_size must be > 0";
    }
    if (max_edges <= 0) {
        return "max_edges must be > 0";
    }
    if (max_attempts <= 0) {
        return "max_attempts must be > 0";
    }

    // Node-count sanity. A max of -1 means "use min only".
    if (min_num_nodes < 0 && max_num_nodes < 0) {
        // Allowed: default construction, will be filled in later by caller.
    } else {
        if (min_num_nodes < 0) {
            return "min_num_nodes must be >= 0 when set";
        }
        if (max_num_nodes >= 0 && max_num_nodes < min_num_nodes) {
            return "max_num_nodes < min_num_nodes";
        }
    }

    // Vocab sanity: the [min_vocab, max_vocab] slice has to be wide enough
    // to host distinct token ids for every possible node.
    if (min_vocab >= 0 && max_vocab >= 0) {
        if (max_vocab < min_vocab) {
            return "max_vocab < min_vocab";
        }
        const int span = max_vocab - min_vocab;
        const int required = (max_num_nodes >= 0 ? max_num_nodes : min_num_nodes);
        if (required > 0 && span < required) {
            return "max_vocab - min_vocab < max_num_nodes";
        }
    }

    // Tokenization sanity.
    if (num_thinking_tokens < 0) {
        return "num_thinking_tokens must be >= 0";
    }

    // Task-specific cross-field constraints.
    if (task_kind == "shortest_path" || task_kind == "bfs") {
        if (min_path_length.has_value() && max_path_length.has_value()) {
            if (*min_path_length > 0 && *min_path_length > *max_path_length) {
                return "min_path_length > max_path_length";
            }
        }
    } else if (task_kind == "center" || task_kind == "centroid") {
        if (min_query_size.has_value() && max_query_size.has_value()) {
            if (*min_query_size > 0 && *max_query_size > 0 &&
                *min_query_size > *max_query_size) {
                return "min_query_size > max_query_size";
            }
        }
    } else if (task_kind == "khops" || task_kind == "khops_gen") {
        if (!min_khops.has_value() || !max_khops.has_value()) {
            return "task_kind='" + task_kind + "' requires min_khops and max_khops";
        }
        if (*min_khops > *max_khops) {
            return "min_khops > max_khops";
        }
        if (!min_prefix_length.has_value() || !max_prefix_length.has_value()) {
            return "task_kind='" + task_kind + "' requires min_prefix_length and max_prefix_length";
        }
        if (*min_prefix_length > *max_prefix_length) {
            return "min_prefix_length > max_prefix_length";
        }
    } else if (task_kind == "none" || task_kind == "None") {
        // Permitted: caller has no task, e.g. sampling graphs for inspection.
    } else {
        // Unknown task_kind is left to the dispatcher that consumes this
        // config to reject with a more contextual error.
    }

    // Scratchpad-specific cross-field constraints. "bfs" is currently the
    // only non-empty scratchpad kind understood by the generator.
    if (scratchpad_kind != "none" && scratchpad_kind != "bfs" &&
        scratchpad_kind != "None") {
        return "unknown scratchpad_kind='" + scratchpad_kind + "'";
    }

    // Graph-kind-specific cross-field constraints.
    if (graph_kind == "erdos_renyi") {
        if (edge_prob.has_value() && (*edge_prob < 0.0 || *edge_prob > 1.0)) {
            return "edge_prob must be in [0, 1]";
        }
    } else if (graph_kind == "euclidean") {
        if (min_edge_length.has_value() && max_edge_length.has_value()) {
            if (*min_edge_length > *max_edge_length) {
                return "min_edge_length > max_edge_length";
            }
        }
    } else if (graph_kind == "path_star") {
        if (min_arms.has_value() && max_arms.has_value() &&
            *min_arms > *max_arms) {
            return "min_arms > max_arms";
        }
        if (min_arm_length.has_value() && max_arm_length.has_value() &&
            *min_arm_length > *max_arm_length) {
            return "min_arm_length > max_arm_length";
        }
    } else if (graph_kind == "balanced") {
        if (min_lookahead.has_value() && max_lookahead.has_value() &&
            *min_lookahead > *max_lookahead) {
            return "min_lookahead > max_lookahead";
        }
    }

    return {};
}

py::dict GeneratorConfig::to_dict() const {
    py::dict d;

    // Shared
    d["min_num_nodes"] = min_num_nodes;
    d["max_num_nodes"] = max_num_nodes;
    d["min_vocab"]     = min_vocab;
    d["max_vocab"]     = max_vocab;
    d["batch_size"]    = batch_size;
    d["max_edges"]     = max_edges;
    d["max_attempts"]  = max_attempts;

    // Dispatch
    d["graph_kind"]      = graph_kind;
    d["task_kind"]       = task_kind;
    d["scratchpad_kind"] = scratchpad_kind;

    // Tokenization
    d["is_causal"]                           = is_causal;
    d["is_direct_ranking"]                   = is_direct_ranking;
    d["query_at_end"]                        = query_at_end;
    d["no_graph"]                            = no_graph;
    d["concat_edges"]                        = concat_edges;
    d["duplicate_edges"]                     = duplicate_edges;
    d["include_nodes_in_graph_tokenization"] = include_nodes_in_graph_tokenization;
    d["num_thinking_tokens"]                 = num_thinking_tokens;
    d["scratchpad_as_prefix"]                = scratchpad_as_prefix;
    d["is_flat_model"]                       = is_flat_model;
    d["align_prefix_front_pad"]              = align_prefix_front_pad;

    // Pos ids
    d["return_pos_ids"]       = return_pos_ids;
    d["use_edges_invariance"] = use_edges_invariance;
    d["use_node_invariance"]  = use_node_invariance;
    d["use_graph_invariance"] = use_graph_invariance;
    d["use_query_invariance"] = use_query_invariance;
    d["use_graph_structure"]  = use_graph_structure;
    d["use_full_structure"]   = use_full_structure;

    // Task-specific
    d["min_path_length"]     = opt_to_py(min_path_length);
    d["max_path_length"]     = opt_to_py(max_path_length);
    d["task_sample_dist"]    = opt_to_py(task_sample_dist);
    d["start_at_root"]       = opt_to_py(start_at_root);
    d["end_at_leaf"]         = opt_to_py(end_at_leaf);
    d["probs"]               = opt_to_py(probs);
    d["min_query_size"]      = opt_to_py(min_query_size);
    d["max_query_size"]      = opt_to_py(max_query_size);
    d["given_query"]         = opt_to_py(given_query);
    d["min_khops"]           = opt_to_py(min_khops);
    d["max_khops"]           = opt_to_py(max_khops);
    d["min_prefix_length"]   = opt_to_py(min_prefix_length);
    d["max_prefix_length"]   = opt_to_py(max_prefix_length);
    d["right_side_connect"]  = opt_to_py(right_side_connect);
    d["khops_no_repeats"]    = opt_to_py(khops_no_repeats);
    d["permutation_version"] = opt_to_py(permutation_version);
    d["mask_to_vocab_size"]  = opt_to_py(mask_to_vocab_size);
    d["mask_to_size"]        = opt_to_py(mask_to_size);
    d["intermediate_labels"] = opt_to_py(intermediate_labels);
    d["partition_method"]    = opt_to_py(partition_method);

    // Scratchpad
    d["sort_adjacency_lists"]      = opt_to_py(sort_adjacency_lists);
    d["use_unique_depth_markers"]  = opt_to_py(use_unique_depth_markers);
    d["stop_once_found"]           = opt_to_py(stop_once_found);
    d["include_queue"]             = opt_to_py(include_queue);
    d["reverse_adjacency_lists"]   = opt_to_py(reverse_adjacency_lists);
    d["duplicate_adjacency_lists"] = opt_to_py(duplicate_adjacency_lists);

    // Graph-kind-specific
    d["edge_prob"]       = opt_to_py(edge_prob);
    d["dim"]             = opt_to_py(dim);
    d["min_edge_length"] = opt_to_py(min_edge_length);
    d["max_edge_length"] = opt_to_py(max_edge_length);
    d["dims"]            = opt_to_py(dims);
    d["min_arms"]        = opt_to_py(min_arms);
    d["max_arms"]        = opt_to_py(max_arms);
    d["min_arm_length"]  = opt_to_py(min_arm_length);
    d["max_arm_length"]  = opt_to_py(max_arm_length);
    d["min_lookahead"]     = opt_to_py(min_lookahead);
    d["max_lookahead"]     = opt_to_py(max_lookahead);
    d["min_noise_reserve"] = opt_to_py(min_noise_reserve);
    d["max_num_parents"]   = opt_to_py(max_num_parents);
    d["max_noise"]         = opt_to_py(max_noise);

    // Debug
    d["print_cpp_args"] = print_cpp_args;

    return d;
}

std::string GeneratorConfig::to_string() const {
    std::ostringstream oss;
    oss << "GeneratorConfig("
        << "graph_kind='" << graph_kind << "'"
        << ", task_kind='" << task_kind << "'"
        << ", scratchpad_kind='" << scratchpad_kind << "'"
        << ", min_num_nodes=" << min_num_nodes
        << ", max_num_nodes=" << max_num_nodes
        << ", min_vocab=" << min_vocab
        << ", max_vocab=" << max_vocab
        << ", batch_size=" << batch_size
        << ", max_edges=" << max_edges
        << ")";
    return oss.str();
}

}  // namespace graphgen
