// graphgen::GeneratorConfig
//
// Flat, single-source-of-truth between python and c++ configuration struct passed across the pybind
// boundary. Constructed from Python kwargs; carries every knob the generator
// exposes.
//
// Layout:
//   * Shared fields (batch, vocab, tokenization, positional ids) are plain
//     values with sensible defaults.
//   * Task-, scratchpad-, and graph-kind-specific fields are std::optional<T>
//     so a config only carries the values relevant to the selected kind.
//   * Cross-field constraints (e.g. min <= max, task-kind-specific required
//     fields) are enforced by validate(); the constructor invokes it and
//     throws std::invalid_argument on failure.

#ifndef GRAPHGEN_GENERATOR_CONFIG_H
#define GRAPHGEN_GENERATOR_CONFIG_H

#include <optional>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>

namespace graphgen {

namespace py = pybind11;

struct GeneratorConfig {
    // ---- Shared batch / vocab controls ------------------------------------
    int min_num_nodes = -1;
    int max_num_nodes = -1;
    int min_vocab     = -1;
    int max_vocab     = -1;
    int batch_size    = 256;
    int max_edges     = 512;
    int max_attempts  = 1000;

    // Node and edge shuffling used to be user-controlled flags. They now
    // happen unconditionally: the sampler assigns random vocab IDs at
    // construction time (populating SampledGraph::internal_to_vocab) and
    // the edge emitter permutes arc order when it packs a batch.

    // ---- Dispatch ---------------------------------------------------------
    std::string graph_kind      = "none";
    std::string task_kind       = "shortest_path";
    std::string scratchpad_kind = "none";

    // ---- Tokenization -----------------------------------------------------
    bool is_causal                            = false;
    bool is_direct_ranking                    = false;
    bool query_at_end                         = true;
    bool no_graph                             = false;
    bool concat_edges                         = true;
    bool duplicate_edges                      = false;
    bool include_nodes_in_graph_tokenization  = false;
    int  num_thinking_tokens                  = 0;
    bool scratchpad_as_prefix                 = false;
    bool is_flat_model                        = true;
    bool align_prefix_front_pad               = false;

    // ---- Positional ids ---------------------------------------------------
    bool return_pos_ids        = true;
    bool use_edges_invariance  = false;
    bool use_node_invariance   = false;
    bool use_graph_invariance  = false;
    bool use_query_invariance  = false;
    bool use_graph_structure   = false;
    bool use_full_structure    = false;

    // ---- Task-specific ----------------------------------------------------
    // shortest_path / bfs
    std::optional<int> min_path_length;
    std::optional<int> max_path_length;

    // shared task extras
    std::optional<std::vector<float>> task_sample_dist;
    std::optional<bool>               start_at_root;
    std::optional<bool>               end_at_leaf;
    std::optional<std::vector<float>> probs;

    // center / centroid
    std::optional<int>              min_query_size;
    std::optional<int>              max_query_size;
    std::optional<std::vector<int>> given_query;

    // khops / khops_gen
    std::optional<int>         min_khops;
    std::optional<int>         max_khops;
    std::optional<int>         min_prefix_length;
    std::optional<int>         max_prefix_length;
    std::optional<bool>        right_side_connect;
    std::optional<bool>        khops_no_repeats;
    std::optional<bool>        permutation_version;
    std::optional<bool>        mask_to_vocab_size;
    std::optional<int>         mask_to_size;
    std::optional<bool>        intermediate_labels;
    std::optional<std::string> partition_method;

    // ---- Scratchpad (BFS-specific) ----------------------------------------
    std::optional<bool> sort_adjacency_lists;
    std::optional<bool> use_unique_depth_markers;
    std::optional<bool> stop_once_found;
    std::optional<bool> include_queue;
    std::optional<bool> reverse_adjacency_lists;
    std::optional<bool> duplicate_adjacency_lists;

    // ---- Graph-kind-specific ---------------------------------------------
    // erdos_renyi
    std::optional<double> edge_prob;

    // euclidean
    std::optional<int>              dim;
    std::optional<double>           min_edge_length;
    std::optional<double>           max_edge_length;
    std::optional<std::vector<int>> dims;

    // path_star
    std::optional<int> min_arms;
    std::optional<int> max_arms;
    std::optional<int> min_arm_length;
    std::optional<int> max_arm_length;

    // balanced
    std::optional<int> min_lookahead;
    std::optional<int> max_lookahead;
    std::optional<int> min_noise_reserve;
    std::optional<int> max_num_parents;
    std::optional<int> max_noise;

    // ---- Debug ------------------------------------------------------------
    bool print_cpp_args = false;

    // ---- Construction -----------------------------------------------------
    GeneratorConfig() = default;

    // Parse from Python kwargs; unknown keys are ignored (matches V1 kwargs
    // sink behavior). Runs validate() and throws std::invalid_argument on any
    // cross-field constraint violation.
    explicit GeneratorConfig(const py::kwargs &kwargs);

    // Returns empty string on success, else a human-readable error message.
    // [[nodiscard]] because ignoring the return value would silently accept
    // invalid configs.
    [[nodiscard]] std::string validate() const;

    // Round-trip back to a Python dict. Optional fields that are unset appear
    // as None. Useful for reproducing runs and for the pybind __repr__.
    py::dict to_dict() const;

    std::string to_string() const;
};

}  // namespace graphgen

#endif  // GRAPHGEN_GENERATOR_CONFIG_H
