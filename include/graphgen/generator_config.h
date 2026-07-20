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
    // The node-vocab range is derived entirely from the shared
    // context's token dictionary at sample time
    // ([ctx.num_special, ctx.max_vocab) covers exactly the node
    // slots; extras such as BFS depth markers live above
    // ctx.max_vocab). Callers who want a different range build a
    // different dictionary -- one source of truth.
    int batch_size    = 256;
    int max_edges     = 512;
    int max_attempts  = 1000;

    // Node and edge shuffling happen unconditionally at sample time:
    // the sampler assigns random vocab ids for each internal vertex
    // (populating SampledGraph::internal_to_vocab), and shuffles the
    // edge list before handing it to CsrGraph so each vertex's
    // adjacency list ends up in a randomised order. Both are done
    // once per item during graph construction and become part of the
    // deterministic representation of that item.

    // ---- Dispatch ---------------------------------------------------------
    std::string graph_kind      = "none";
    std::string task_kind       = "shortest_path";
    std::string scratchpad_kind = "none";

    // ---- Graph structure --------------------------------------------------
    // Applies to every graph_kind. Sits above graph-kind-specific
    // fields because these are properties of the graph as a whole
    // (its type of relation), not of any particular sampler.
    //
    // directed: sample undirected (default) or directed graphs. CsrGraph
    //   stores each edge in its native direction; edges() and
    //   num_edges() honour the flag automatically. Currently
    //   implemented only by sample_erdos_renyi (other graph kinds are
    //   themselves unimplemented).
    //
    // weighted: STUB. When true, edges would carry float weights and
    //   distance-based tasks (shortest_path) would use Dijkstra
    //   instead of BFS. Not yet wired end-to-end -- validation
    //   rejects `weighted=true` for now.
    bool directed = false;
    bool weighted = false;

    // ---- Tokenization -----------------------------------------------------
    // tokenization_mode is a named recipe that fixes content columns,
    // positional columns, and invariant/pos-id semantics all at once.
    // Sean = 2D output, edges as `u v EDGE` on the sequence axis. Stan =
    // 3D output, edges packed as `(u, v, EDGE)` across struct columns.
    // See tokenization_mode.h for the enum and future variants.
    std::string tokenization_mode = "sean";

    bool query_at_end                                       = true;
    // Whether the graph section (nodes/edges) appears in the token
    // stream at all. Default true; set false to omit the graph
    // entirely (e.g. when training a pure-query model that only sees
    // the query and target).
    bool include_graph_in_graph_tokenization                = true;
    // Whether the whole edge list is emitted twice, identically.
    // Pass 2 is an EXACT REPEAT of pass 1 -- same order, same
    // endpoints, no swap.
    //
    //   pass 1:  u_1 v_1 EDGE   u_2 v_2 EDGE   ...   u_m v_m EDGE
    //   pass 2:  u_1 v_1 EDGE   u_2 v_2 EDGE   ...   u_m v_m EDGE
    //
    // Motivation is a training-side trick, not a graph change: under
    // causal attention any token in pass 2 can attend to every token
    // in pass 1, so the model gets a full view of the edge list
    // before predicting anything inside it. Never modifies the graph
    // (endpoint order stays canonical) and composes cleanly with the
    // `directed` flag.
    //
    // Undirected mirror-edge note: CsrGraph stores each undirected
    // edge in both slots (u->v AND v->u) so neighbour lookups work
    // from either endpoint and BFS-based tasks are correct. The
    // tokenizer ALWAYS emits each undirected edge ONCE in canonical
    // (u < v) form -- the mirror is a representation detail, not a
    // tokenization one. If we want the model to see each edge twice,
    // this flag (identical repeat) is the way; a separate
    // mirror-emit toggle would combine multiplicatively and 4x the
    // edge section length without adding information.
    //
    // Canonicalisation ordering: the (u < v) comparison uses INTERNAL
    // vertex ids (0..n-1, the construction indices), applied BEFORE
    // the vocab-id shuffle in `sg.internal_to_vocab` is used. This is
    // deliberate -- vocab ids are just labels drawn uniformly at
    // random from the vocab range, so they carry no semantic meaning
    // and no natural ordering. Canonicalising on vocab ids would make
    // the emitted edge order (`vocab_a < vocab_b`) leak the arbitrary
    // shuffle into the token stream, and the model would implicitly
    // learn a spurious ordering between tokens that are supposed to
    // be semantically unrelated. Internal-id canonicalisation keeps
    // the shuffle behind the tokenizer boundary.
    bool include_duplicate_edges_in_graph_tokenization      = false;
    // Whether an enumerated vertex list precedes the edge list.
    bool include_nodes_in_graph_tokenization                = false;
    int  num_thinking_tokens                                = 0;
    bool align_prefix_front_pad                             = false;

    // ---- Positional ids ---------------------------------------------------
    // Whether to include the positional-ids tensor in the batch output.
    // The *content* of that tensor (raw ids vs role invariants, 1D vs
    // multi-dim) is determined by tokenization_mode.
    bool return_pos_ids        = true;

    // ---- Distance matrix returns ------------------------------------------
    // Hop distances (int32, BFS-derived, unweighted). When true the
    // batch dict includes a padded (B, actual_max_n, actual_max_n)
    // int32 tensor keyed "hop_distances", where actual_max_n is the
    // maximum num_vertices across the batch (bounded above by
    // cfg.max_num_nodes when set, but sized from what was actually
    // sampled). Padded slots outside each item's
    // [0, num_nodes[i]) x [0, num_nodes[i]) region hold HOP_PAD
    // (INT_MIN); disconnected pairs inside the region hold
    // HOP_UNREACHABLE (-1). See include/graphgen/hop_distance_matrix.h.
    //
    // Naming is intentionally semantics-specific ("hop_distances" not
    // "distances") because a future return_weighted_distances flag
    // will expose Dijkstra-derived edge-weight-sum distances alongside
    // it -- the two matrices are independent, not either/or.
    bool return_hop_distances  = false;

    // Vertex positions (float64, geometric-graph-derived). When true
    // the batch dict includes:
    //   * "node_positions"     -- (B, max_n, dim) float64, NaN
    //                             outside each row's [0, num_nodes[i])
    //                             region.
    //   * "internal_to_vocab"  -- (B, max_n) int32 mapping internal
    //                             vertex id -> vocab token id (-1
    //                             outside the valid region). Included
    //                             here because callers who want to
    //                             plot with symbol ids need this
    //                             mapping to turn internal-space
    //                             positions into a symbol->xy dict.
    //
    // Currently only sample_euclidean populates positions; other
    // graph kinds have no natural coordinate system, so
    // `return_positions=true` on a non-euclidean config is rejected at
    // validate time.
    bool return_positions      = false;

    // Distance-rank targets (int32). When true the batch dict
    // includes a padded
    //   (B, max_n, max_distance, max_ties) int32
    // tensor keyed "distance_rank_targets". For each source vertex
    // u (in internal-id order along axis 1) it gives the vocab
    // token ids of every reachable vertex, bucketed by hop distance
    // along axis 2 and ordered by vocab id within each tie along
    // axis 3. The source itself always sits at (u, d=0, k=0).
    // Unreachable pairs are omitted; empty (d, k) slots hold -1.
    //
    // Computed with hop-count semantics (BFS-derived); shares the
    // per-item hop matrix with the return_hop_distances path, so
    // asking for both is not more work than asking for either.
    // Only defined for hop-semantic graphs (all current graph kinds
    // qualify; a future weighted graph kind would need its own
    // weighted_rank targets).
    bool return_distance_rank_targets = false;

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
    // bfs_scratchpad_style names one of a small set of mutually
    // exclusive per-vertex block layouts (plain, with_queue,
    // reverse_adjacency, duplicate_adjacency). Legacy code exposed
    // three separate bools for the alternatives; they combined
    // ambiguously so we collapse them into a single named recipe.
    // See bfs_scratchpad_style.h.
    std::string bfs_scratchpad_style = "plain";

    std::optional<bool> stop_once_found;

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
