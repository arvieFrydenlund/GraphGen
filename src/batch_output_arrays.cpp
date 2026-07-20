#include "graphgen/batch_output_arrays.h"

#include <cstring>
#include <utility>

#include "graphgen/check.h"
#include "graphgen/generator_config.h"
#include "graphgen/layout.h"
#include "graphgen/worker_shared_context.h"

namespace graphgen {

namespace {

// Fill an arithmetic-typed contiguous array with `value`, cast to the
// array's element type. Used to pre-fill token tensors with TOK_PAD so
// the tokenizer only writes the seq_len prefix and the tail stays
// valid.
template <typename T>
void fill_array(py::array_t<T>& arr, int64_t value) {
    if (arr.size() == 0) return;
    T* data = arr.mutable_data();
    const T v = static_cast<T>(value);
    for (std::size_t i = 0; i < static_cast<std::size_t>(arr.size()); ++i) {
        data[i] = v;
    }
}

// Small helper: allocate a 2D py::array_t of shape [rows, cols] and
// pre-fill with pad_value.
template <typename T>
py::array_t<T> alloc_filled_2d(int rows, int cols, int64_t pad_value) {
    py::array_t<T> arr({rows, cols});
    fill_array(arr, pad_value);
    return arr;
}

// 3D variant for STAN's [B, S, struct_dim] token tensors.
template <typename T>
py::array_t<T> alloc_filled_3d(int d0, int d1, int d2, int64_t pad_value) {
    py::array_t<T> arr({d0, d1, d2});
    fill_array(arr, pad_value);
    return arr;
}

// 1D variant. Metadata arrays get zero-filled and then per-item values
// written into them; explicit init keeps behaviour deterministic.
template <typename T>
py::array_t<T> alloc_zeroed_1d(int n) {
    py::array_t<T> arr(n);
    if (arr.size() > 0) {
        std::memset(arr.mutable_data(), 0, arr.size() * sizeof(T));
    }
    return arr;
}

// Allocate a token tensor with the mode's natural rank: 2D for
// struct_dim==1 (SEAN), 3D for struct_dim>1 (STAN). Pre-fill with
// pad_value so tokenizer writes only need to overwrite what changes.
template <typename T>
py::array_t<T> alloc_token_tensor(int B, int S, int D, int64_t pad_value) {
    return D == 1
        ? alloc_filled_2d<T>(B, S, pad_value)
        : alloc_filled_3d<T>(B, S, D, pad_value);
}

}  // namespace

BatchOutputArrays::BatchOutputArrays(const BatchLayout& layout,
                                     const GeneratorConfig& cfg,
                                     int max_labels)
    : max_gen_len_(layout.max_gen_len),
      max_labels_(max_labels),
      max_seq_len_(layout.max_seq_len),
      struct_dim_(layout.struct_dim) {
    const int B = layout.batch_size;
    const int S = layout.max_seq_len;
    const int D = layout.struct_dim;

    GG_CHECK(D >= 1, "BatchOutputArrays: struct_dim must be >= 1 (got "
                     << D << ")");
    GG_CHECK(B >= 0, "BatchOutputArrays: batch_size must be >= 0");
    GG_CHECK(max_labels >= 1,
             "BatchOutputArrays: max_labels must be >= 1 (got "
                 << max_labels << ")");

    // ---- Source tokens ---------------------------------------------------
    // Shape [B, S] for SEAN (D=1), [B, S, D] for STAN. Pre-fill with
    // TOK_PAD so the tokenizer only writes the seq_len prefix of each
    // row (and, under STAN, only the columns each section actually
    // fills -- unfilled columns stay TOK_PAD).
    src_tokens_ = alloc_token_tensor<int32_t>(B, S, D, WorkerSharedContext::TOK_PAD);

    return_pos_ids_ = cfg.return_pos_ids;
    if (return_pos_ids_) {
        // Positional ids share the source token tensor shape; column
        // 0 gets the sequence position and columns 1..D-1 stay at 0.
        positions_ = alloc_token_tensor<int32_t>(B, S, D, /*fill=*/0);
    }

    // ---- Targets tensor [B, max_gen_len, max_labels] ---------------------
    // Pre-fill with TOK_PAD. The tokenizer writes labels for each
    // generation-region position; unused label alternatives (indices
    // >= actual number of valid labels at that step) stay at TOK_PAD
    // so the loss can mask them out. Rows past a row's gen_len also
    // stay TOK_PAD.
    targets_ = alloc_filled_3d<int32_t>(B, max_gen_len_, max_labels_,
                                        WorkerSharedContext::TOK_PAD);

    // ---- Metadata tensors [B] --------------------------------------------
    src_lengths_              = alloc_zeroed_1d<int32_t>(B);
    num_nodes_                = alloc_zeroed_1d<int32_t>(B);
    num_edges_                = alloc_zeroed_1d<int32_t>(B);
    graph_edge_start_indices_ = alloc_zeroed_1d<int32_t>(B);
    graph_edge_lengths_       = alloc_zeroed_1d<int32_t>(B);
    query_start_indices_      = alloc_zeroed_1d<int32_t>(B);
    query_lengths_            = alloc_zeroed_1d<int32_t>(B);
    thinking_start_indices_   = alloc_zeroed_1d<int32_t>(B);
    thinking_lengths_         = alloc_zeroed_1d<int32_t>(B);
    scratchpad_start_indices_ = alloc_zeroed_1d<int32_t>(B);
    scratchpad_lengths_       = alloc_zeroed_1d<int32_t>(B);
    task_start_indices_       = alloc_zeroed_1d<int32_t>(B);
    task_lengths_             = alloc_zeroed_1d<int32_t>(B);
    left_pad_lengths_         = alloc_zeroed_1d<int32_t>(B);

    // Cache per-row left-pad so row accessors can offset writes
    // without recomputing. All zeros under right-pad mode.
    left_pad_by_row_.assign(static_cast<std::size_t>(B), 0);

    if (B == 0) return;

    auto* src_lengths_d              = src_lengths_.mutable_data();
    auto* num_nodes_d                = num_nodes_.mutable_data();
    auto* num_edges_d                = num_edges_.mutable_data();
    auto* graph_edge_start_indices_d = graph_edge_start_indices_.mutable_data();
    auto* graph_edge_lengths_d       = graph_edge_lengths_.mutable_data();
    auto* query_start_indices_d      = query_start_indices_.mutable_data();
    auto* query_lengths_d            = query_lengths_.mutable_data();
    auto* thinking_start_indices_d   = thinking_start_indices_.mutable_data();
    auto* thinking_lengths_d         = thinking_lengths_.mutable_data();
    auto* scratchpad_start_indices_d = scratchpad_start_indices_.mutable_data();
    auto* scratchpad_lengths_d       = scratchpad_lengths_.mutable_data();
    auto* task_start_indices_d       = task_start_indices_.mutable_data();
    auto* task_lengths_d             = task_lengths_.mutable_data();
    auto* left_pad_lengths_d         = left_pad_lengths_.mutable_data();

    for (int i = 0; i < B; ++i) {
        const Layout& L = layout.items[static_cast<std::size_t>(i)];
        // Under align mode each row is shifted right so its
        // ScratchpadStart / TaskStart marker (prefix_len from row 0)
        // lands at column max_prefix_len. Section indices below are
        // shifted by that offset so they still point at real content
        // columns in the padded row.
        const int left_pad = layout.align_prefix_front_pad
            ? layout.max_prefix_len - L.prefix_len
            : 0;
        left_pad_by_row_[static_cast<std::size_t>(i)] = left_pad;

        src_lengths_d[i]              = L.seq_len;
        num_nodes_d[i]                = L.num_nodes;
        num_edges_d[i]                = L.num_edges;
        graph_edge_start_indices_d[i] = L.graph_edge_start + left_pad;
        graph_edge_lengths_d[i]       = L.graph_edge_len;
        query_start_indices_d[i]      = L.query_start + left_pad;
        query_lengths_d[i]            = L.query_len;
        // Section-start indices are shifted only when the section
        // actually exists (length > 0); zero-length sections keep
        // their sentinel 0 to distinguish "absent" from "at col 0".
        thinking_start_indices_d[i]   =
            L.thinking_len > 0 ? L.thinking_start + left_pad : 0;
        thinking_lengths_d[i]         = L.thinking_len;
        scratchpad_start_indices_d[i] =
            L.scratchpad_len > 0 ? L.scratchpad_start + left_pad : 0;
        scratchpad_lengths_d[i]       = L.scratchpad_len;
        task_start_indices_d[i]       = L.task_target_start + left_pad;
        task_lengths_d[i]             = L.task_target_len;
        left_pad_lengths_d[i]         = left_pad;
    }
}

py::dict BatchOutputArrays::to_dict() const {
    py::dict d;
    d["src_tokens"]                = src_tokens_;
    d["targets"]                   = targets_;
    if (return_pos_ids_) {
        d["positions"] = positions_;
    }
    d["src_lengths"]               = src_lengths_;
    d["num_nodes"]                 = num_nodes_;
    d["num_edges"]                 = num_edges_;
    d["graph_edge_start_indices"]  = graph_edge_start_indices_;
    d["graph_edge_lengths"]        = graph_edge_lengths_;
    d["query_start_indices"]       = query_start_indices_;
    d["query_lengths"]             = query_lengths_;
    d["thinking_start_indices"]    = thinking_start_indices_;
    d["thinking_lengths"]          = thinking_lengths_;
    d["scratchpad_start_indices"]  = scratchpad_start_indices_;
    d["scratchpad_lengths"]        = scratchpad_lengths_;
    d["task_start_indices"]        = task_start_indices_;
    d["task_lengths"]              = task_lengths_;
    d["left_pad_lengths"]          = left_pad_lengths_;
    if (has_hop_distances_) {
        d["hop_distances"] = hop_distances_;
    }
    if (has_node_positions_) {
        d["node_positions"]     = node_positions_;
        d["internal_to_vocab"]  = internal_to_vocab_;
    }
    if (has_distance_rank_targets_) {
        d["distance_rank_targets"] = distance_rank_targets_;
    }
    return d;
}

// Row accessors return a pointer offset by the row's left_pad, so
// the tokenizer can keep writing at (row-local) column 0 without
// knowing about front-padding. Column stride is struct_dim entries.
int32_t* BatchOutputArrays::src_tokens_row(int row) {
    return src_tokens_.mutable_data()
         + (row * max_seq_len_ + left_pad_by_row_[static_cast<std::size_t>(row)])
             * struct_dim_;
}

int32_t* BatchOutputArrays::positions_row(int row) {
    if (!return_pos_ids_) return nullptr;
    return positions_.mutable_data()
         + (row * max_seq_len_ + left_pad_by_row_[static_cast<std::size_t>(row)])
             * struct_dim_;
}

int BatchOutputArrays::row_stride() const {
    return max_seq_len_ * struct_dim_;
}

// Targets row: pointer to the (max_gen_len, max_labels) block for
// row `row`. Caller writes `dst[pos * max_labels + label_alt] = tok`
// for each generation-region position and each equally-valid label.
// Front-pad does not apply here -- the targets tensor is indexed
// row-locally by generation-region position, not by src column.
int32_t* BatchOutputArrays::targets_row(int row) {
    return targets_.mutable_data() + row * max_gen_len_ * max_labels_;
}

int BatchOutputArrays::targets_row_stride() const {
    return max_gen_len_ * max_labels_;
}

void BatchOutputArrays::adopt_hop_distances(py::array_t<int32_t> tensor) {
    hop_distances_     = std::move(tensor);
    has_hop_distances_ = true;
}

void BatchOutputArrays::adopt_node_positions(
    py::array_t<double>  positions,
    py::array_t<int32_t> internal_to_vocab) {
    node_positions_     = std::move(positions);
    internal_to_vocab_  = std::move(internal_to_vocab);
    has_node_positions_ = true;
}

void BatchOutputArrays::adopt_distance_rank_targets(
    py::array_t<int32_t> tensor) {
    distance_rank_targets_     = std::move(tensor);
    has_distance_rank_targets_ = true;
}

}  // namespace graphgen
