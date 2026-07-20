// Worker implementation.
//
// generate_batch drives the pipeline for one batch:
//   precompute: sample graph + task/scratchpad + Layout per item,
//               storing each tuple for the compute phase
//   allocate  : BatchOutputArrays sized from the per-batch max
//   compute   : tokenizer writes each row's real token ids
//   return    : arrays as a Python dict

#include "graphgen/worker.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "graphgen/batch_output_arrays.h"
#include "graphgen/check.h"
#include "graphgen/graph_kind.h"
#include "graphgen/hop_distance_matrix.h"
#include "graphgen/layout.h"
#include "graphgen/random_utils.h"
#include "graphgen/sampled_graph.h"
#include "graphgen/scratchpad.h"
#include "graphgen/section_plan.h"
#include "graphgen/task.h"
#include "graphgen/tokenization_mode.h"

namespace graphgen {

namespace {

// Precompute-phase output: everything the compute phase needs to
// tokenize one row. `plan` is the sequential-axis description produced
// by make_section_plan; both Layout::from_plan and Tokenizer walk it,
// so precomputing once here avoids doing the walk twice per item.
// Kept local to generate_batch -- callers never see this type.
struct PrecomputedItem {
    SampledGraph sg;
    Task         task;
    Scratchpad   scratchpad;
    SectionPlan  plan;
    Layout       layout;
};

// Build a stride-aware view of row `i` of a (B, max_n, max_n) int32
// tensor. Row stride == max_n, column stride == 1.
HopDistView hop_row_view(std::int32_t* base, int max_n, int row) {
    std::int32_t* row_base = base
        + static_cast<std::size_t>(row)
            * static_cast<std::size_t>(max_n)
            * static_cast<std::size_t>(max_n);
    std::array<int, 2> strides{max_n, 1};
    return HopDistView{
        row_base,
        md::layout_stride::mapping{
            md::dextents<int, 2>{max_n, max_n},
            strides}};
}

}  // namespace

Worker::Worker(std::shared_ptr<WorkerSharedContext> ctx, std::uint64_t seed)
    : ctx_(std::move(ctx)), gen_(seed) {}

py::dict Worker::generate_batch(const GeneratorConfig& cfg) {
    const GraphKind      graph_kind      = graph_kind_from_string(cfg.graph_kind);
    const TaskKind       task_kind       = task_kind_from_string(cfg.task_kind);
    const ScratchpadKind scratchpad_kind =
        scratchpad_kind_from_string(cfg.scratchpad_kind);
    const TokenizationMode mode =
        tokenization_mode_from_string(cfg.tokenization_mode);

    GG_CHECK(cfg.batch_size > 0,
             "generate_batch: batch_size must be > 0 (got "
                 << cfg.batch_size << ")");

    // ---- Pass 1: sample graphs -------------------------------------------
    // Split from the task pass so we can allocate the batch-scoped
    // hop-distances tensor (if requested) and attach per-item views
    // into it BEFORE tasks run. That way tasks that touch
    // sg.hop_distances() fill the batch tensor directly, no copy.
    std::vector<PrecomputedItem> items;
    items.reserve(static_cast<std::size_t>(cfg.batch_size));
    for (int i = 0; i < cfg.batch_size; ++i) {
        SampledGraph sg = graph_sampler_.run(graph_kind, gen_, cfg);
        // Assign vocab token ids to internal vertices. The dictionary
        // is the sole source of truth for the node-vocab range:
        // [ctx.num_special, ctx.max_vocab) covers exactly the node
        // slots (any extras like BFS depth markers live above
        // ctx.max_vocab). Values stored here are ACTUAL token ids;
        // the tokenizer no longer applies any shift.
        sg.internal_to_vocab = sample_distinct_ints(
            sg.graph.num_vertices(),
            ctx_->num_special, ctx_->max_vocab, gen_);
        items.push_back({std::move(sg), Task{}, Scratchpad{}, SectionPlan{}, Layout{}});
    }

    // Batch-wide max num_vertices across all sampled items. Sizes
    // every "per-vertex axis" tensor allocated below (hop distances,
    // node positions, distance-rank targets). Reads much smaller
    // than cfg.max_num_nodes on typical batches, so the tensors are
    // typed to the actual data rather than the config upper bound
    // -- a 2-10x memory / bandwidth win.
    int actual_max_n = 0;
    for (const auto& it : items) {
        actual_max_n = std::max(actual_max_n, it.sg.graph.num_vertices());
    }

    // ---- Optional: allocate hop-distances batch tensor -------------------
    // Shape [B, actual_max_n, actual_max_n] int32, pre-filled with
    // HOP_PAD so slots outside each row's [0, num_nodes[i])^2 region
    // are recognisable as padding. Views into row i are attached to
    // items[i].sg so any consumer that calls sg.hop_distances() during
    // the task pass writes straight into the batch tensor.
    py::array_t<int32_t> hop_dist_tensor;
    if (cfg.return_hop_distances) {
        GG_CHECK(actual_max_n > 0,
                 "generate_batch: return_hop_distances=true but every "
                 "sampled graph is empty (max_n=0)");
        hop_dist_tensor = py::array_t<int32_t>(
            {cfg.batch_size, actual_max_n, actual_max_n});
        std::fill_n(hop_dist_tensor.mutable_data(),
                    static_cast<std::size_t>(hop_dist_tensor.size()),
                    HOP_PAD);
        for (int i = 0; i < cfg.batch_size; ++i) {
            items[static_cast<std::size_t>(i)].sg.attach_hop_distances_view(
                hop_row_view(hop_dist_tensor.mutable_data(), actual_max_n, i));
        }
    }

    // ---- Pass 2: tasks + plans + layouts ---------------------------------
    // Task samplers may consume sg.hop_distances(); when the view was
    // attached above the fill lands in the batch tensor, otherwise in
    // an owned per-graph buffer that gets released below.
    for (int i = 0; i < cfg.batch_size; ++i) {
        PrecomputedItem& it = items[static_cast<std::size_t>(i)];
        auto [task, scratchpad] =
            task_sampler_.run(task_kind, scratchpad_kind, it.sg, gen_, cfg,
                              ctx_->num_special, ctx_->max_vocab);
        it.task       = std::move(task);
        it.scratchpad = std::move(scratchpad);
        it.plan       = make_section_plan(it.sg, it.task, it.scratchpad, cfg);
        it.layout     = Layout::from_plan(it.plan, mode,
                                          it.sg.graph.num_vertices(),
                                          it.sg.graph.num_edges());
    }

    // ---- Optional: release owned hop buffers -----------------------------
    // Any sg that computed hop distances into an owned buffer (i.e.
    // return_hop_distances was false) can drop it now that the task
    // stage is done -- nothing downstream reads it, so we don't want
    // to carry O(sum n_i^2) memory through tokenisation.
    //
    // Exception: any flag that consumes hop distances downstream
    // (return_hop_distances -- needs the batch view filled;
    // return_distance_rank_targets -- needs per-item distances to
    // bucket into ranks) forces the fill instead. sg.hop_distances()
    // is a no-op fast path when already computed, so items whose
    // task did use it don't recompute.
    const bool need_hop_distances = cfg.return_hop_distances
                                 || cfg.return_distance_rank_targets;
    if (!need_hop_distances) {
        for (auto& it : items) it.sg.release_hop_distances();
    } else {
        for (auto& it : items) {
            if (!it.sg.hop_distances_computed()) {
                (void)it.sg.hop_distances();
            }
        }
    }

    // Aggregate: per-batch shape.
    std::vector<Layout> layouts;
    layouts.reserve(items.size());
    for (const auto& it : items) layouts.push_back(it.layout);
    BatchLayout batch_layout =
        BatchLayout::from(std::move(layouts), cfg.align_prefix_front_pad);

    // Compute per-batch max_labels: the maximum number of
    // equally-valid target labels at any single step across all items.
    // Sizes the label-smoothing dim of the (B, max_gen_len,
    // max_labels) `targets` tensor.
    //   * ShortestPath: max over hops of valid_next_hops[k].size()
    //   * Center/Centroid: vertex_set.size() at the single target step
    //   * BFS / other: 1 (no smoothing)
    // Never smaller than 1 (every step has at least the chosen label).
    int max_labels = 1;
    for (const auto& it : items) {
        switch (it.task.kind) {
            case TaskKind::ShortestPath:
                if (it.task.target.valid_next_hops.has_value()) {
                    for (const auto& alts : *it.task.target.valid_next_hops) {
                        max_labels = std::max(
                            max_labels, static_cast<int>(alts.size()));
                    }
                }
                break;
            case TaskKind::Center:
            case TaskKind::Centroid:
                if (it.task.target.vertex_set.has_value()) {
                    max_labels = std::max(
                        max_labels,
                        static_cast<int>(it.task.target.vertex_set->size()));
                }
                break;
            case TaskKind::Khops:
                // intermediate_labels can produce up to k labels per
                // position; without it every un-masked position emits
                // exactly one. Scan the actual per-position labels so
                // the sizing tracks whatever the sampler produced.
                if (it.task.target.per_position_labels.has_value()) {
                    for (const auto& labs : *it.task.target.per_position_labels) {
                        max_labels = std::max(
                            max_labels, static_cast<int>(labs.size()));
                    }
                }
                break;
            case TaskKind::BFS:
            case TaskKind::None:
            case TaskKind::KhopsGen:
                break;   // no smoothing
        }
    }

    // Allocate: pad-filled numpy tensors sized from the batch max,
    // plus per-item metadata written from the layouts.
    BatchOutputArrays arrays(batch_layout, cfg, max_labels);
    if (cfg.return_hop_distances) {
        arrays.adopt_hop_distances(std::move(hop_dist_tensor));
    }

    // ---- Optional: allocate node-positions batch tensor -----------------
    // Shape [B, actual_max_n, dim] float64 (NaN outside each row's
    // [0, num_nodes[i]) rows) plus a companion [B, actual_max_n]
    // int32 internal_to_vocab tensor (-1 outside). Only euclidean
    // (or future geometric-graph samplers) populate sg.positions;
    // items whose graph has no positions leave their row entirely
    // NaN. `dim` is inferred from the FIRST item with a non-empty
    // positions vector -- all items in a batch share graph_kind and
    // thus dim, so a single read suffices.
    if (cfg.return_positions) {
        int dim = 0;
        for (const auto& it : items) {
            if (dim == 0 && it.sg.dim > 0) { dim = it.sg.dim; break; }
        }
        GG_CHECK(actual_max_n > 0,
                 "generate_batch: return_positions=true but every "
                 "sampled graph is empty (max_n=0)");
        GG_CHECK(dim > 0,
                 "generate_batch: return_positions=true but no sampled "
                 "graph carried positions -- did you set the flag on a "
                 "non-geometric graph_kind?");

        py::array_t<double>  pos_tensor(
            {cfg.batch_size, actual_max_n, dim});
        py::array_t<int32_t> vocab_tensor(
            {cfg.batch_size, actual_max_n});
        std::fill_n(pos_tensor.mutable_data(),
                    static_cast<std::size_t>(pos_tensor.size()),
                    std::numeric_limits<double>::quiet_NaN());
        std::fill_n(vocab_tensor.mutable_data(),
                    static_cast<std::size_t>(vocab_tensor.size()),
                    static_cast<int32_t>(-1));

        double*  pos_base   = pos_tensor.mutable_data();
        int32_t* vocab_base = vocab_tensor.mutable_data();
        for (int i = 0; i < cfg.batch_size; ++i) {
            const SampledGraph& sg = items[static_cast<std::size_t>(i)].sg;
            const int n = sg.graph.num_vertices();
            // Copy positions row: sg.positions is flat (n, dim), we
            // write it into the [i, :n, :] slice of pos_tensor.
            if (!sg.positions.empty() && sg.dim == dim) {
                double* row = pos_base
                    + static_cast<std::size_t>(i)
                        * static_cast<std::size_t>(actual_max_n)
                        * static_cast<std::size_t>(dim);
                std::copy(sg.positions.begin(), sg.positions.end(), row);
            }
            // Copy internal_to_vocab: (n,) int -> [i, :n].
            int32_t* vrow = vocab_base
                + static_cast<std::size_t>(i)
                    * static_cast<std::size_t>(actual_max_n);
            for (int u = 0; u < n; ++u) {
                vrow[u] = static_cast<int32_t>(
                    sg.internal_to_vocab[static_cast<std::size_t>(u)]);
            }
        }
        arrays.adopt_node_positions(std::move(pos_tensor),
                                    std::move(vocab_tensor));
    }

    // ---- Optional: allocate distance-rank-targets batch tensor -----------
    // For each source vertex u in each item, produce a per-source
    // "which vertex ids sit at distance d, in tie order k" table.
    // Two passes over the per-item hop matrices:
    //
    //   Pass 1: scan every (item, source) to find the batch-wide
    //           max_distance (deepest reachable radius) and max_ties
    //           (largest tie group at any (source, distance) pair).
    //   Pass 2: for each (item, source) sort the reachable vertices
    //           by (distance, vocab_id) and place into the tensor.
    //
    // Sort key is (distance ASC, vocab_id ASC): ties within a distance
    // group are ordered by vocab id (already randomly shuffled) so
    // the layout doesn't leak internal-id ordering into the target
    // tensor. Cost per graph: O(n^2 log n) worst case (sort dominates
    // the O(n^2) place); comparable to the hop-distance compute itself.
    if (cfg.return_distance_rank_targets) {
        // Pass 1: per-item scan for the batch dimensions.
        int batch_max_dist = 1;   // at least the d=0 slot per source
        int batch_max_ties = 1;   // at least the source itself at d=0
        std::vector<int> dist_counts;   // reused scratch across passes
        for (const auto& it : items) {
            const int n_i = it.sg.graph.num_vertices();
            const HopDistView hd = it.sg.hop_distances();
            for (int u = 0; u < n_i; ++u) {
                dist_counts.assign(1, 0);   // room for d=0 at minimum
                for (int v = 0; v < n_i; ++v) {
                    const int d = hd[u, v];
                    if (d < 0) continue;   // HOP_UNREACHABLE
                    if (d + 1 > static_cast<int>(dist_counts.size())) {
                        dist_counts.resize(static_cast<std::size_t>(d + 1), 0);
                    }
                    dist_counts[static_cast<std::size_t>(d)]++;
                }
                if (static_cast<int>(dist_counts.size()) > batch_max_dist) {
                    batch_max_dist = static_cast<int>(dist_counts.size());
                }
                for (int c : dist_counts) {
                    if (c > batch_max_ties) batch_max_ties = c;
                }
            }
        }

        py::array_t<int32_t> drt(
            {cfg.batch_size, actual_max_n, batch_max_dist, batch_max_ties});
        std::fill_n(drt.mutable_data(),
                    static_cast<std::size_t>(drt.size()),
                    static_cast<int32_t>(-1));

        // Row-major strides. drt is contiguous C-order.
        const std::size_t stride_b = static_cast<std::size_t>(actual_max_n)
                                   * static_cast<std::size_t>(batch_max_dist)
                                   * static_cast<std::size_t>(batch_max_ties);
        const std::size_t stride_u = static_cast<std::size_t>(batch_max_dist)
                                   * static_cast<std::size_t>(batch_max_ties);
        const std::size_t stride_d = static_cast<std::size_t>(batch_max_ties);
        int32_t* drt_base = drt.mutable_data();

        // Pass 2: bucket-sort + fill. Reuse pairs scratch across all
        // (item, source) pairs -- one allocation for the whole batch.
        std::vector<std::pair<int, int32_t>> pairs;
        pairs.reserve(static_cast<std::size_t>(actual_max_n));
        for (int i = 0; i < cfg.batch_size; ++i) {
            const SampledGraph& sg = items[static_cast<std::size_t>(i)].sg;
            const int n_i = sg.graph.num_vertices();
            const HopDistView hd = sg.hop_distances();
            for (int u = 0; u < n_i; ++u) {
                pairs.clear();
                for (int v = 0; v < n_i; ++v) {
                    const int d = hd[u, v];
                    if (d < 0) continue;
                    pairs.emplace_back(
                        d,
                        static_cast<int32_t>(
                            sg.internal_to_vocab[static_cast<std::size_t>(v)]));
                }
                std::sort(pairs.begin(), pairs.end());
                int prev_d = -1;
                int k      = 0;
                for (const auto& [d, vocab_id] : pairs) {
                    if (d != prev_d) { k = 0; prev_d = d; }
                    // Sizing pass guarantees k < batch_max_ties.
                    drt_base[
                        static_cast<std::size_t>(i) * stride_b
                      + static_cast<std::size_t>(u) * stride_u
                      + static_cast<std::size_t>(d) * stride_d
                      + static_cast<std::size_t>(k)
                    ] = vocab_id;
                    ++k;
                }
            }
        }
        arrays.adopt_distance_rank_targets(std::move(drt));
    }

    // Compute phase: write real tokens into each row. Use the
    // plan-explicit tokenize_into_row overload so we don't rebuild
    // the plan we already have.
    for (int i = 0; i < cfg.batch_size; ++i) {
        const auto& it = items[static_cast<std::size_t>(i)];
        tokenizer_.tokenize_into_row(arrays.src_tokens_row(i),
                                     arrays.targets_row(i),
                                     max_labels,
                                     arrays.positions_row(i),
                                     it.plan,
                                     it.sg, it.task, it.scratchpad,
                                     it.layout, *ctx_, cfg);
    }

    return arrays.to_dict();
}

py::dict Worker::sample_graph_stats(const GeneratorConfig& cfg) {
    const GraphKind kind = graph_kind_from_string(cfg.graph_kind);
    SampledGraph sg = graph_sampler_.run(kind, gen_, cfg);
    sg.internal_to_vocab = sample_distinct_ints(
        sg.graph.num_vertices(),
        ctx_->num_special, ctx_->max_vocab, gen_);

    auto cc = sg.graph.connected_components();
    const int num_components =
        cc.empty() ? 0 : *std::max_element(cc.begin(), cc.end()) + 1;

    py::dict d;
    d["num_vertices"]   = sg.graph.num_vertices();
    d["num_edges"]      = sg.graph.num_edges();
    d["num_components"] = num_components;
    d["vocab_ids"]      = py::cast(sg.internal_to_vocab);
    return d;
}

py::dict Worker::sample_shortest_path_stats(const GeneratorConfig& cfg) {
    const GraphKind kind = graph_kind_from_string(cfg.graph_kind);
    const TaskKind task_kind = task_kind_from_string(cfg.task_kind);
    const ScratchpadKind scratchpad_kind =
        scratchpad_kind_from_string(cfg.scratchpad_kind);

    SampledGraph sg = graph_sampler_.run(kind, gen_, cfg);
    sg.internal_to_vocab = sample_distinct_ints(
        sg.graph.num_vertices(),
        ctx_->num_special, ctx_->max_vocab, gen_);
    auto [task, scratchpad] =
        task_sampler_.run(task_kind, scratchpad_kind, sg, gen_, cfg,
                          ctx_->num_special, ctx_->max_vocab);
    (void)scratchpad;

    py::dict d;
    d["num_vertices"] = sg.graph.num_vertices();
    d["num_edges"]    = sg.graph.num_edges();
    d["start"]        = task.query.start_node.value();
    d["end"]          = task.query.end_node.value();
    d["path"]         = py::cast(task.target.path.value());
    d["path_length"]  = static_cast<int>(task.target.path.value().size()) - 1;
    d["valid_next_hops"] = py::cast(task.target.valid_next_hops.value());
    return d;
}

}  // namespace graphgen
