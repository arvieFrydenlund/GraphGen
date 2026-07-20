// Layout implementation. See layout.h.
//
// Layout::from_plan is a straightforward fold over the SectionPlan:
// walk each Section, ask num_positions() how big it is under the
// current mode, and record start/len into the matching Layout field
// for named content sections.  All per-task / per-scratchpad shape
// logic lives in section_plan.cpp, not here.

#include "graphgen/layout.h"

#include <algorithm>

#include "graphgen/check.h"
#include "graphgen/csr_graph.h"
#include "graphgen/generator_config.h"
#include "graphgen/sampled_graph.h"
#include "graphgen/scratchpad.h"
#include "graphgen/section_plan.h"
#include "graphgen/task.h"
#include "graphgen/tokenization_mode.h"

namespace graphgen {

Layout Layout::from_plan(const SectionPlan& plan,
                         TokenizationMode   mode,
                         int                num_nodes,
                         int                num_edges) {
    Layout L;
    L.struct_dim = struct_dim_for(mode);
    L.num_nodes  = num_nodes;
    L.num_edges  = num_edges;

    int pos = 0;
    for (const Section& s : plan) {
        const int n = num_positions(s, mode);
        // Only content sections have a matching (start, len) slot on
        // Layout; markers (Bos, QueryStart, ...) advance position but
        // record nothing beyond that. `default` catches all markers.
        switch (s.kind) {
            case SectionKind::GraphEdges:
                L.graph_edge_start = pos;
                L.graph_edge_len   = n;
                break;
            case SectionKind::Query:
                L.query_start = pos;
                L.query_len   = n;
                break;
            case SectionKind::Thinking:
                L.thinking_start = pos;
                L.thinking_len   = n;
                break;
            case SectionKind::Scratchpad:
                L.scratchpad_start = pos;
                L.scratchpad_len   = n;
                break;
            case SectionKind::Target:
                L.task_target_start = pos;
                L.task_target_len   = n;
                break;
            default:
                break;
        }
        pos += n;
    }
    L.seq_len = pos;

    // Generation-region boundary: the model's output starts at the
    // FIRST of the generation-region markers that actually exists,
    // in this order: ThinkStart, ScratchpadStart, TaskStart. `*_start`
    // fields point at the first CONTENT token of a section, so the
    // preceding marker sits one column earlier -- hence the -1.
    if (L.thinking_len > 0) {
        L.prefix_len = L.thinking_start - 1;
    } else if (L.scratchpad_len > 0) {
        L.prefix_len = L.scratchpad_start - 1;
    } else {
        L.prefix_len = L.task_target_start - 1;
    }
    L.gen_len    = L.seq_len - L.prefix_len;
    return L;
}

Layout Layout::from(const SampledGraph& sg,
                    const Task&         task,
                    const Scratchpad&   scratchpad,
                    const GeneratorConfig& cfg) {
    const TokenizationMode mode =
        tokenization_mode_from_string(cfg.tokenization_mode);

    const SectionPlan plan = make_section_plan(sg, task, scratchpad, cfg);
    return Layout::from_plan(plan, mode,
                             sg.graph.num_vertices(),
                             sg.graph.num_edges());
}

BatchLayout BatchLayout::from(std::vector<Layout> items,
                              bool align_prefix_front_pad) {
    BatchLayout bl;
    bl.batch_size              = static_cast<int>(items.size());
    bl.align_prefix_front_pad  = align_prefix_front_pad;
    if (items.empty()) {
        return bl;
    }
    bl.struct_dim  = items.front().struct_dim;

    if (align_prefix_front_pad) {
        // Align mode: batch is sized as max_prefix + max_gen so every
        // row can be left-padded to put its gen-start at column
        // max_prefix. Rows with a big prefix and rows with a big gen
        // both drive the batch width up independently.
        int max_prefix = 0;
        int max_gen    = 0;
        for (const Layout& L : items) {
            GG_CHECK(L.struct_dim == bl.struct_dim,
                     "BatchLayout::from: items disagree on struct_dim ("
                         << L.struct_dim << " vs " << bl.struct_dim << ")");
            max_prefix = std::max(max_prefix, L.prefix_len);
            max_gen    = std::max(max_gen,    L.gen_len);
        }
        bl.max_prefix_len = max_prefix;
        bl.max_gen_len    = max_gen;
        bl.max_seq_len    = max_prefix + max_gen;
    } else {
        // Right-pad mode: rows start at column 0, batch width is the
        // longest row's seq_len. Front-pad slot stays zero. max_gen_len
        // is still tracked -- the targets tensor is sized from it
        // regardless of the src padding strategy.
        bl.max_seq_len = 0;
        int max_gen    = 0;
        for (const Layout& L : items) {
            GG_CHECK(L.struct_dim == bl.struct_dim,
                     "BatchLayout::from: items disagree on struct_dim ("
                         << L.struct_dim << " vs " << bl.struct_dim << ")");
            bl.max_seq_len = std::max(bl.max_seq_len, L.seq_len);
            max_gen        = std::max(max_gen,        L.gen_len);
        }
        bl.max_gen_len = max_gen;
    }
    bl.items = std::move(items);
    return bl;
}

}  // namespace graphgen
