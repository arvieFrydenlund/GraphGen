// Layout -- per-item shape and section indices for one training example,
// derived by folding a SectionPlan under a TokenizationMode.
//
// Layout is a plain data record: seq_len, struct_dim, and the start/len
// span for every named section. The two axes are kept clean:
//   * SectionPlan (section_plan.h) owns the sequential axis -- which
//     sections exist and their semantic counts.
//   * TokenizationMode owns the annotation axis -- struct_dim and how
//     many sequence positions each section actually costs.
//
// Section-index convention: `*_start` points at the *first content
// token* of the section (not the boundary marker before it), and
// `*_len` counts only content tokens.

#ifndef GRAPHGEN_LAYOUT_H
#define GRAPHGEN_LAYOUT_H

#include <utility>
#include <vector>

#include "graphgen/section_plan.h"
#include "graphgen/tokenization_mode.h"

namespace graphgen {

struct GeneratorConfig;  // forward decl
class SampledGraph;
struct Task;
struct Scratchpad;

struct Layout {
    int seq_len          = 0;
    int struct_dim       = 1;  // SEAN=1; STAN=3.

    // Section indices into the sequence axis (0-based).
    int graph_edge_start  = 0;
    int graph_edge_len    = 0;
    int query_start       = 0;
    int query_len         = 0;
    // Thinking section holds `num_thinking_tokens` TOK_THINK tokens
    // at the start of the generation region. Both fields are 0 when
    // num_thinking_tokens == 0.
    int thinking_start    = 0;
    int thinking_len      = 0;
    // Scratchpad section holds the "shown work" trace between
    // thinking (if any) and target. Both fields are 0 when
    // scratchpad_kind == None.
    int scratchpad_start  = 0;
    int scratchpad_len    = 0;
    int task_target_start = 0;
    int task_target_len   = 0;

    // Descriptive metadata surfaced on the batch output so downstream
    // Python code can stratify without re-deriving from the tokens.
    int num_nodes         = 0;
    int num_edges         = 0;

    // Sequence position where generation begins -- i.e. the column
    // that contains the ScratchpadStart marker if a scratchpad
    // exists, else the TaskStart marker. Used by BatchLayout when
    // align_prefix_front_pad is true so all rows in the batch have
    // their generation-start at the same column.
    int prefix_len        = 0;

    // Length of the generation region (from prefix_len through EOS
    // inclusive). Also used for the front-padded batch shape:
    //   max_seq_len = max(prefix_len_i) + max(gen_len_i)
    int gen_len           = 0;

    // Primary constructor: fold a SectionPlan into a Layout under a
    // given TokenizationMode. num_nodes / num_edges are stamped as
    // metadata (the plan doesn't carry them).
    static Layout from_plan(const SectionPlan& plan,
                            TokenizationMode   mode,
                            int                num_nodes,
                            int                num_edges);

    // Convenience wrapper: build the plan from (sg, task, scratchpad,
    // cfg) and fold it. Preferred for external callers and tests where
    // the plan isn't already in hand. Worker precomputes the plan and
    // uses from_plan directly to avoid duplicating the walk.
    static Layout from(const SampledGraph& sg,
                       const Task&         task,
                       const Scratchpad&   scratchpad,
                       const GeneratorConfig& cfg);
};

// Batch-level layout: aggregates a vector of per-item Layouts and
// records the maxima the allocator needs. Kept as a plain struct
// (rather than an accessor-heavy class) because it's only ever built
// once per batch and read a handful of times.
//
// When align_prefix_front_pad is true, the batch shape is
//   max_seq_len = max(prefix_len_i) + max(gen_len_i)
// and every row will be left-padded so its ScratchpadStart / TaskStart
// marker sits at column max_prefix. Otherwise rows are right-padded
// to `max(seq_len_i)` starting at column 0.
struct BatchLayout {
    int batch_size  = 0;
    int max_seq_len = 0;
    int struct_dim  = 1;
    // Left-align pivot when align_prefix_front_pad is true (== max
    // over items of prefix_len). Zero under right-pad mode.
    int max_prefix_len = 0;
    // Maximum generation-region length across items. Independent of
    // align mode -- computed as max over items of `Layout::gen_len`.
    // Used by BatchOutputArrays to size the (B, max_gen_len,
    // max_labels) `targets` tensor.
    int max_gen_len    = 0;
    bool align_prefix_front_pad = false;
    std::vector<Layout> items;

    static BatchLayout from(std::vector<Layout> items,
                            bool align_prefix_front_pad = false);
};

}  // namespace graphgen

#endif  // GRAPHGEN_LAYOUT_H
