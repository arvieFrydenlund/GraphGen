// Batch output tensors packaged as numpy arrays.
//
// Constructor:
//   * allocates one numpy tensor per output field, sized from the
//     precomputed BatchLayout so all rows share a single width;
//   * zero-fills the token tensors (real values arrive from the
//     tokenizer in a later step);
//   * writes per-item metadata (section indices, lengths, node/edge
//     counts) into the [B]-shaped arrays from the Layout list.
//
// The class owns the numpy arrays; `to_dict()` hands out reference-
// counted handles to Python without copying data.

#ifndef GRAPHGEN_BATCH_OUTPUT_ARRAYS_H
#define GRAPHGEN_BATCH_OUTPUT_ARRAYS_H

#include <cstdint>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "graphgen/hop_distance_matrix.h"

namespace graphgen {

namespace py = pybind11;

struct BatchLayout;      // fwd
struct GeneratorConfig;  // fwd

class BatchOutputArrays {
public:
    BatchOutputArrays(const BatchLayout& layout, const GeneratorConfig& cfg,
                      int max_labels);

    // Reference-counted copy of each py::array_t handle; no data copy.
    py::dict to_dict() const;

    // Direct row accessors for the tokenizer step to write into each
    // row's slice. Returns a pointer to the start of the row; caller
    // indexes with `dst[pos * struct_dim + col]`. Under SEAN this
    // collapses to `dst[pos]` since struct_dim == 1.
    int32_t* src_tokens_row(int row);
    int32_t* positions_row(int row);       // nullptr if !return_pos_ids
    int      row_stride() const;           // int32 elements per row (== max_seq_len * struct_dim)

    // Targets tensor row: shape (max_gen_len, max_labels). Caller
    // indexes `targets[pos * max_labels + label_alt]`. Pre-filled with
    // TOK_PAD; the tokenizer writes labels for each generation-region
    // position and leaves unused label alternatives at TOK_PAD.
    int32_t* targets_row(int row);
    int      targets_row_stride() const;   // max_gen_len * max_labels
    int      max_labels() const { return max_labels_; }
    int      max_gen_len() const { return max_gen_len_; }

    // Adopt an already-allocated hop-distances tensor (shape [B,
    // max_n, max_n], int32, pre-filled with HOP_PAD by the caller).
    // Ownership transferred; to_dict() will surface it under key
    // "hop_distances". Called by the worker once the tensor's per-item
    // views have been attached to each SampledGraph and tasks have
    // filled them.
    void adopt_hop_distances(py::array_t<int32_t> tensor);

    // Adopt an already-allocated node-positions tensor (shape [B,
    // max_n, dim], float64, NaN outside each row's valid region)
    // and its companion internal_to_vocab mapping (shape [B, max_n],
    // int32, -1 outside). Ownership transferred; to_dict() will
    // surface both under keys "node_positions" and
    // "internal_to_vocab" respectively. Positions come from geometric
    // graph samplers (euclidean, ...); other kinds don't call this.
    void adopt_node_positions(py::array_t<double>  positions,
                              py::array_t<int32_t> internal_to_vocab);

    // Adopt an already-allocated distance-rank-targets tensor (shape
    // [B, max_n, max_distance, max_ties], int32, -1 outside the
    // valid region). Ownership transferred; to_dict() will surface
    // it under key "distance_rank_targets". See
    // GeneratorConfig::return_distance_rank_targets for semantics.
    void adopt_distance_rank_targets(py::array_t<int32_t> tensor);

private:
    // Source tokens: SEAN [B, max_seq_len] (2D); STAN [B, max_seq_len,
    // struct_dim] (3D). int32_t (see PLAN.md's dtype note). Consumers
    // feeding this into nn.Embedding need a one-time `.long()` cast.
    py::array_t<int32_t> src_tokens_;

    // Positional ids, same shape as src_tokens_. Only allocated when
    // cfg.return_pos_ids; the dict omits the key otherwise.
    py::array_t<int32_t> positions_;
    bool                 return_pos_ids_ = false;

    // Targets tensor: shape [B, max_gen_len, max_labels], int32,
    // pre-filled with TOK_PAD. For each row `b` and each generation-
    // region position `i` in [0, gen_len_b):
    //   targets[b, i, 0]   = the "chosen" token at that position
    //   targets[b, i, k>0] = other equally-valid tokens (label
    //                        smoothing) -- e.g. valid_next_hops for
    //                        shortest_path intermediate steps, or all
    //                        center_nodes for center/centroid.
    //   targets[b, i, k+] = TOK_PAD  (loss mask)
    //   targets[b, i>=gen_len_b, *] = TOK_PAD
    // Named "targets" (rather than fairseq's "prev_output_tokens") to
    // stop pretending this is the decoder-input tensor -- it's the
    // labels tensor for a label-smoothed CE loss.
    py::array_t<int32_t> targets_;
    int                  max_gen_len_ = 0;
    int                  max_labels_  = 0;

    // Per-item metadata, shape [B].
    py::array_t<int32_t> src_lengths_;
    py::array_t<int32_t> num_nodes_;
    py::array_t<int32_t> num_edges_;
    py::array_t<int32_t> graph_edge_start_indices_;
    py::array_t<int32_t> graph_edge_lengths_;
    py::array_t<int32_t> query_start_indices_;
    py::array_t<int32_t> query_lengths_;
    // Thinking indices are always allocated; entries are 0 when the
    // row's num_thinking_tokens is 0.
    py::array_t<int32_t> thinking_start_indices_;
    py::array_t<int32_t> thinking_lengths_;
    // Scratchpad indices are always allocated; entries are 0 when the
    // row's scratchpad_kind is None.
    py::array_t<int32_t> scratchpad_start_indices_;
    py::array_t<int32_t> scratchpad_lengths_;
    py::array_t<int32_t> task_start_indices_;
    py::array_t<int32_t> task_lengths_;
    // Number of PAD tokens prepended to each row. Nonzero only when
    // align_prefix_front_pad is true. Downstream code can read
    // content-start with `left_pad_lengths[i]` and content-end with
    // `left_pad_lengths[i] + src_lengths[i]`.
    py::array_t<int32_t> left_pad_lengths_;

    // Hop-distances tensor, adopted from the worker via
    // adopt_hop_distances(). Empty (has_hop_distances_ == false) when
    // the caller did not request the return. See adopt_hop_distances
    // for shape/dtype/sentinel conventions.
    py::array_t<int32_t> hop_distances_;
    bool                 has_hop_distances_ = false;

    // Node-positions tensor + internal_to_vocab mapping, adopted via
    // adopt_node_positions(). Empty when the caller did not request
    // geometry. See adopt_node_positions for shape/dtype/sentinel
    // conventions.
    py::array_t<double>  node_positions_;
    py::array_t<int32_t> internal_to_vocab_;
    bool                 has_node_positions_ = false;

    // Distance-rank-targets tensor, adopted via
    // adopt_distance_rank_targets(). Empty when the caller did not
    // request it. See adopt_distance_rank_targets for shape/dtype/
    // sentinel conventions.
    py::array_t<int32_t> distance_rank_targets_;
    bool                 has_distance_rank_targets_ = false;

    // Per-row left-pad in sequence positions. Cached so row accessors
    // can offset writes without re-scanning the layout.
    std::vector<int32_t> left_pad_by_row_;

    int max_seq_len_ = 0;
    int struct_dim_  = 1;
};

}  // namespace graphgen

#endif  // GRAPHGEN_BATCH_OUTPUT_ARRAYS_H
