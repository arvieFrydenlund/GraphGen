// Implementation for graphgen::GeneratorConfig. See generator_config.h.

#include "graphgen/generator_config.h"

#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "graphgen/bfs_scratchpad_style.h"

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

// ---- Kwarg validation -------------------------------------------------------

// Reject caller kwargs that used the pre-rename spelling. Silently
// ignoring them would leave the corresponding field at its "none"
// default and blow up much later in sampler dispatch; catch it here
// with a clear message pointing at the new spelling.
void reject_legacy_kwargs(const py::kwargs &kw) {
    static constexpr const char *kRenamed[][2] = {
        {"graph_type",      "graph_kind"},
        {"task_type",       "task_kind"},
        {"scratchpad_type", "scratchpad_kind"},
        // The following 8 fields were absorbed into `tokenization_mode`
        // (an enum-valued config field, currently exposed as a string).
        // Each combination of these flags used to configure part of the
        // tokenization strategy; the enum now names each valid recipe.
        {"concat_edges",           "tokenization_mode"},
        {"is_flat_model",          "tokenization_mode"},
        {"use_edges_invariance",   "tokenization_mode"},
        {"use_node_invariance",    "tokenization_mode"},
        {"use_graph_invariance",   "tokenization_mode"},
        {"use_query_invariance",   "tokenization_mode"},
        {"use_graph_structure",    "tokenization_mode"},
        {"use_full_structure",     "tokenization_mode"},
        // The three BFS-scratchpad bools were folded into
        // `bfs_scratchpad_style` (mutually exclusive variants).
        {"include_queue",             "bfs_scratchpad_style"},
        {"reverse_adjacency_lists",   "bfs_scratchpad_style"},
        {"duplicate_adjacency_lists", "bfs_scratchpad_style"},
        // Dropped outright: depth markers never worked, and scratchpad
        // ordering is fixed (the model needs the query before it can
        // build the scratchpad, so "prefix" is not a real axis).
        {"use_unique_depth_markers", "<removed: never functional>"},
        {"scratchpad_as_prefix",     "<removed: query must precede scratchpad>"},
        {"is_causal",                "<removed: dead flag>"},
        {"is_direct_ranking",        "<removed: dead flag>"},
        // sort_adjacency_lists is now always on for the BFS scratchpad
        // (matches the legacy default), so the flag is unnecessary.
        {"sort_adjacency_lists",     "<removed: always on for BFS scratchpad>"},
        // Renamed for naming symmetry with include_nodes_in_graph_tokenization.
        // Note: include_graph_in_graph_tokenization has INVERTED polarity
        // (default true = include, false = omit).
        {"no_graph",        "include_graph_in_graph_tokenization (default true; invert polarity)"},
        {"duplicate_edges", "include_duplicate_edges_in_graph_tokenization"},
    };
    for (const auto &pair : kRenamed) {
        if (kw.contains(pair[0])) {
            throw std::invalid_argument(
                std::string("GeneratorConfig: '") + pair[0] +
                "' has been renamed; use '" + pair[1] + "' instead.");
        }
    }
}

}  // namespace

GeneratorConfig::GeneratorConfig(const py::kwargs &kw) {
    reject_legacy_kwargs(kw);

    // --- Shared -----------------------------------------------------------
    set_from_kwargs(kw, "min_num_nodes", min_num_nodes);
    set_from_kwargs(kw, "max_num_nodes", max_num_nodes);
    set_from_kwargs(kw, "batch_size",    batch_size);
    set_from_kwargs(kw, "max_edges",     max_edges);
    set_from_kwargs(kw, "max_attempts",  max_attempts);

    // --- Dispatch ---------------------------------------------------------
    set_from_kwargs(kw, "graph_kind",      graph_kind);
    set_from_kwargs(kw, "task_kind",       task_kind);
    set_from_kwargs(kw, "scratchpad_kind", scratchpad_kind);

    // --- Graph structure --------------------------------------------------
    set_from_kwargs(kw, "directed", directed);
    set_from_kwargs(kw, "weighted", weighted);

    // --- Tokenization -----------------------------------------------------
    set_from_kwargs(kw, "tokenization_mode",                   tokenization_mode);

    set_from_kwargs(kw, "query_at_end",                                 query_at_end);
    set_from_kwargs(kw, "include_graph_in_graph_tokenization",          include_graph_in_graph_tokenization);
    set_from_kwargs(kw, "include_duplicate_edges_in_graph_tokenization", include_duplicate_edges_in_graph_tokenization);
    set_from_kwargs(kw, "include_nodes_in_graph_tokenization",          include_nodes_in_graph_tokenization);
    set_from_kwargs(kw, "num_thinking_tokens",                          num_thinking_tokens);
    set_from_kwargs(kw, "align_prefix_front_pad",                       align_prefix_front_pad);

    // --- Pos ids ----------------------------------------------------------
    set_from_kwargs(kw, "return_pos_ids",       return_pos_ids);

    // --- Distance matrix returns ------------------------------------------
    set_from_kwargs(kw, "return_hop_distances", return_hop_distances);
    set_from_kwargs(kw, "return_positions",     return_positions);
    set_from_kwargs(kw, "return_distance_rank_targets",
                    return_distance_rank_targets);

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
    set_from_kwargs         (kw, "bfs_scratchpad_style",      bfs_scratchpad_style);
    set_optional_from_kwargs(kw, "stop_once_found",           stop_once_found);

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

    // Node-vocab range is derived from the shared context's
    // dictionary at sample time (Worker draws vertex ids from
    // [ctx.num_special, ctx.max_vocab)). No config-level vocab
    // sanity check to run here -- the dictionary owns that.

    // Tokenization sanity.
    if (num_thinking_tokens < 0) {
        return "num_thinking_tokens must be >= 0";
    }
    // Value must name a known TokenizationMode; downstream layout code
    // will convert this string to the enum via tokenization_mode_from_string.
    if (tokenization_mode != "sean" && tokenization_mode != "stan") {
        return "tokenization_mode='" + tokenization_mode +
               "' is not a known mode (expected 'sean' or 'stan')";
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
        if (*min_khops < 1) {
            return "min_khops must be >= 1";
        }
        // Prefix-length requirements diverge by task variant:
        //   * khops_gen ALWAYS needs [min, max]_prefix_length (the
        //     backtrace layout scales with prefix length).
        //   * khops with permutation_version generates its own
        //     length from vocab_size * (k + 1), so prefix_length is
        //     optional. Otherwise (uniform seq) it's required.
        const bool khops_perm =
            task_kind == "khops" && permutation_version.value_or(false);
        if (!khops_perm) {
            if (!min_prefix_length.has_value() || !max_prefix_length.has_value()) {
                return "task_kind='" + task_kind + "' requires min_prefix_length and max_prefix_length";
            }
            if (*min_prefix_length > *max_prefix_length) {
                return "min_prefix_length > max_prefix_length";
            }
        }
        // khops_gen also needs Q >= N (partition feasibility): every
        // (k, P) draw with k in [min_k, max_k] and P in
        // [min_prefix_length, max_prefix_length] must satisfy P >= 2k.
        // The tightest case is the largest k paired with the smallest
        // P.
        if (task_kind == "khops_gen") {
            if (*min_prefix_length < 2 * (*max_khops)) {
                return "min_prefix_length must be >= 2 * max_khops for a feasible partition";
            }
        }
        // partition_method: only used by khops_gen right now; validate
        // if set so misspellings fail loudly at construction.
        if (partition_method.has_value()) {
            const std::string& pm = *partition_method;
            if (pm != "uniform" && pm != "non_uniform") {
                return "partition_method must be 'uniform' or 'non_uniform' (got '" + pm + "')";
            }
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
    // (bfs, bfs) is a redundant combination: the BFS task's target
    // already IS the BFS visit order, so emitting a BFS-walk
    // scratchpad in front of it would just duplicate the answer. BFS
    // scratchpads exist to reason toward a *different* target (e.g.
    // shortest_path via a BFS trace).
    if (task_kind == "bfs" && scratchpad_kind == "bfs") {
        return "(task_kind='bfs', scratchpad_kind='bfs') is redundant: "
               "the bfs task already outputs the BFS visit order; use "
               "scratchpad_kind='none', or pair a bfs scratchpad with a "
               "different task (e.g. shortest_path)";
    }
    // bfs_scratchpad_style is inert when scratchpad_kind != "bfs", but
    // validating it always keeps the failure surface consistent.
    try {
        (void)bfs_scratchpad_style_from_string(bfs_scratchpad_style);
    } catch (const std::invalid_argument &e) {
        return e.what();
    }

    // Graph-structure constraints.
    if (weighted) {
        return "weighted=true is not implemented; edge weights are not yet "
               "wired through the tokenizer or into distance-based tasks. "
               "Leave weighted=false for now.";
    }

    // hop_distances is now sized from the ACTUAL max num_vertices
    // observed in the batch (see worker.cpp), so it no longer
    // requires cfg.max_num_nodes to be pre-set. Any batch with at
    // least one non-empty sampled graph produces a valid tensor.

    // Positions are only produced by geometric graph samplers;
    // other kinds have no natural coordinate system to surface.
    // Reject the flag on non-euclidean configs rather than silently
    // returning a tensor full of NaN.
    if (return_positions && graph_kind != "euclidean") {
        return "return_positions=true is only supported for "
               "graph_kind='euclidean' (got graph_kind='"
               + graph_kind + "')";
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
        if (min_edge_length.has_value() && *min_edge_length < 0.0) {
            return "min_edge_length must be >= 0";
        }
        if (dim.has_value() && *dim < 1) {
            return "dim must be >= 1";
        }
        if (dims.has_value()) {
            return "cfg.dims is reserved for a future per-dimension "
                   "extent API and is not consumed by sample_euclidean; "
                   "use cfg.dim for the dimension count";
        }
        if (directed) {
            return "graph_kind='euclidean' does not support directed=true "
                   "(Euclidean distance is symmetric)";
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
    d["batch_size"]    = batch_size;
    d["max_edges"]     = max_edges;
    d["max_attempts"]  = max_attempts;

    // Dispatch
    d["graph_kind"]      = graph_kind;
    d["task_kind"]       = task_kind;
    d["scratchpad_kind"] = scratchpad_kind;

    // Graph structure
    d["directed"] = directed;
    d["weighted"] = weighted;

    // Tokenization
    d["tokenization_mode"]                   = tokenization_mode;

    d["query_at_end"]                                 = query_at_end;
    d["include_graph_in_graph_tokenization"]          = include_graph_in_graph_tokenization;
    d["include_duplicate_edges_in_graph_tokenization"] = include_duplicate_edges_in_graph_tokenization;
    d["include_nodes_in_graph_tokenization"]          = include_nodes_in_graph_tokenization;
    d["num_thinking_tokens"]                          = num_thinking_tokens;
    d["align_prefix_front_pad"]                       = align_prefix_front_pad;

    // Pos ids
    d["return_pos_ids"]       = return_pos_ids;

    // Distance matrix returns
    d["return_hop_distances"] = return_hop_distances;
    d["return_positions"]     = return_positions;
    d["return_distance_rank_targets"] = return_distance_rank_targets;

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
    d["bfs_scratchpad_style"]      = bfs_scratchpad_style;
    d["stop_once_found"]           = opt_to_py(stop_once_found);

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
        << ", batch_size=" << batch_size
        << ", max_edges=" << max_edges
        << ")";
    return oss.str();
}

}  // namespace graphgen
