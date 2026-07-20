// Tokenizer implementation.
//
// One walk over the SectionPlan.  Each Section either:
//   * writes a single boundary token (Bos / Eos / QueryStart / ...), or
//   * dispatches to a mode-aware writer for its content
//     (write_graph_edges, write_query_content,
//      write_scratchpad_content, write_target_content).
//
// The mode branch is only meaningful for GraphEdges today (SEAN
// spreads across 3 positions, STAN packs into 1 × 3-column position).
// Every other content section is 1 token per position regardless of
// mode. When more modes land, mode-specific behaviour is added to the
// writers named above -- the plan walk itself stays put.

#include "graphgen/tokenizer.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "graphgen/bfs_scratchpad_style.h"
#include "graphgen/check.h"
#include "graphgen/csr_graph.h"
#include "graphgen/generator_config.h"
#include "graphgen/layout.h"
#include "graphgen/sampled_graph.h"
#include "graphgen/scratchpad.h"
#include "graphgen/section_plan.h"
#include "graphgen/task.h"
#include "graphgen/tokenization_mode.h"
#include "graphgen/worker_shared_context.h"

namespace graphgen {

namespace {

int num_special_from_ctx(const WorkerSharedContext& ctx) {
    return ctx.num_special > 0
        ? ctx.num_special
        : WorkerSharedContext::NUM_SPECIAL_DEFAULT;
}

// Vertex id -> token id. Under the current design `sg.internal_to_vocab`
// is populated by the Worker in TOKEN-ID space (drawn from
// [ctx.num_special, ctx.max_vocab) -- the node-vocab range the
// dictionary defines), so this is a direct passthrough with no shift.
//
// The `num_special` parameter is now dead weight; kept in the
// signature only to avoid touching every call site in one pass.
// TODO: strip `num_special` from vertex_token + every helper that
// threads it through.
inline int32_t vertex_token(int u,
                            const SampledGraph& sg,
                            [[maybe_unused]] int num_special) {
    return static_cast<int32_t>(
        sg.internal_to_vocab[static_cast<std::size_t>(u)]);
}

// Row-write abstraction that hides SEAN/STAN's differing content-column
// counts. The tokenizer uses SEQUENCE POSITIONS as its cursor; this
// view maps (position, column) -> flat offset via `pos * struct_dim + col`.
// Under SEAN (struct_dim == 1) the column dimension is degenerate and
// `set0(pos, tok)` reduces to `dst[pos] = tok`. Under STAN the writers
// that pack multiple values into one position (edges) use `set(pos, col, tok)`.
struct WriteView {
    int32_t* dst;
    int      struct_dim;

    void set0(int pos, int32_t tok) const {
        dst[pos * struct_dim] = tok;
    }
    void set(int pos, int col, int32_t tok) const {
        dst[pos * struct_dim + col] = tok;
    }
};

// For each internal vertex id, the index into the emitted edge list of
// the FIRST edge that mentions it. Isolated vertices (no incident edge)
// stay at -1. Used by the BFS scratchpad to order per-vertex adjacency
// lists so the ordering (a) is derivable from what the model has
// already seen in its context, and (b) doesn't leak the arbitrary
// vocab shuffle or internal-id order into the token stream.
std::vector<int> compute_first_mention_edge_index(const SampledGraph& sg) {
    const int n = sg.graph.num_vertices();
    std::vector<int> first_mention(static_cast<std::size_t>(n), -1);
    int i = 0;
    for (const auto e : sg.graph.edges()) {
        if (first_mention[static_cast<std::size_t>(e.u)] == -1) {
            first_mention[static_cast<std::size_t>(e.u)] = i;
        }
        if (first_mention[static_cast<std::size_t>(e.v)] == -1) {
            first_mention[static_cast<std::size_t>(e.v)] = i;
        }
        ++i;
    }
    return first_mention;
}

// Neighbours of `v` sorted ascending by first_mention[nbr]. Vertices
// with first_mention == -1 (isolated -- unreachable via BFS from
// anywhere, so shouldn't appear in a BFS scratchpad but defensively
// sort them last). std::stable_sort keeps the CSR order among ties;
// with the edge shuffle in place there generally aren't ties.
std::vector<int> sorted_neighbours_by_first_mention(
    int v,
    const CsrGraph& g,
    const std::vector<int>& first_mention) {
    std::vector<int> nbrs;
    for (int n : g.neighbours(v)) nbrs.push_back(n);
    std::stable_sort(nbrs.begin(), nbrs.end(),
                     [&](int a, int b) {
                         int fa = first_mention[static_cast<std::size_t>(a)];
                         int fb = first_mention[static_cast<std::size_t>(b)];
                         if (fa == -1) fa = std::numeric_limits<int>::max();
                         if (fb == -1) fb = std::numeric_limits<int>::max();
                         return fa < fb;
                     });
    return nbrs;
}

// Write one boundary marker for the given SectionKind. Returns the
// number of positions advanced (always 1). If the section isn't a
// marker, returns 0 and writes nothing.
int write_marker(WriteView view, int cursor, SectionKind kind) {
    switch (kind) {
        case SectionKind::BOS:             view.set0(cursor, WorkerSharedContext::TOK_BOS);           return 1;
        case SectionKind::EOS:             view.set0(cursor, WorkerSharedContext::TOK_EOS);           return 1;
        case SectionKind::QueryStart:      view.set0(cursor, WorkerSharedContext::TOK_QUERY_START);   return 1;
        case SectionKind::QueryEnd:        view.set0(cursor, WorkerSharedContext::TOK_QUERY_END);     return 1;
        case SectionKind::ThinkStart:      view.set0(cursor, WorkerSharedContext::TOK_THINK_START);   return 1;
        case SectionKind::ThinkEnd:        view.set0(cursor, WorkerSharedContext::TOK_THINK_END);     return 1;
        case SectionKind::ScratchpadStart: view.set0(cursor, WorkerSharedContext::TOK_SCRATCH_START); return 1;
        case SectionKind::ScratchpadEnd:   view.set0(cursor, WorkerSharedContext::TOK_SCRATCH_END);   return 1;
        case SectionKind::TaskStart:       view.set0(cursor, WorkerSharedContext::TOK_TASK_START);    return 1;
        case SectionKind::TaskEnd:         view.set0(cursor, WorkerSharedContext::TOK_TASK_END);      return 1;
        default:                           return 0;
    }
}

// GraphEdges under a given mode. Returns positions advanced.
// SEAN: 3 positions per edge, each with 1 content column (u, v, EDGE).
// STAN: 1 position per edge, with u, v, EDGE packed across cols 0, 1, 2.
// When cfg.include_duplicate_edges_in_graph_tokenization is true, the
// entire edge list is emitted TWICE, identically -- pass 2 is an exact
// repeat of pass 1 (same order, same endpoints, no swap). The
// repetition is a training-side trick: under causal attention, a token
// in pass 2 can attend to every token in pass 1, so the model gets to
// see the full edge list before predicting anything within it.
int write_graph_edges(WriteView view,
                      int cursor,
                      const SampledGraph& sg,
                      TokenizationMode mode,
                      const GeneratorConfig& cfg,
                      int num_special) {
    const int passes = cfg.include_duplicate_edges_in_graph_tokenization ? 2 : 1;
    int p = cursor;
    if (mode == TokenizationMode::Sean) {
        for (int pass = 0; pass < passes; ++pass) {
            for (auto e : sg.graph.edges()) {
                view.set0(p++, vertex_token(e.u, sg, num_special));
                view.set0(p++, vertex_token(e.v, sg, num_special));
                view.set0(p++, WorkerSharedContext::TOK_EDGE);
            }
        }
    } else {  // Stan
        GG_CHECK(view.struct_dim >= 3,
                 "Tokenizer: STAN requires struct_dim >= 3 (got "
                     << view.struct_dim << ")");
        for (int pass = 0; pass < passes; ++pass) {
            for (auto e : sg.graph.edges()) {
                view.set(p, 0, vertex_token(e.u, sg, num_special));
                view.set(p, 1, vertex_token(e.v, sg, num_special));
                view.set(p, 2, WorkerSharedContext::TOK_EDGE);
                ++p;
            }
        }
    }
    return p - cursor;
}

// GraphNodes: emit every vertex token in internal-id order. Always
// col 0 regardless of mode -- STAN's packed columns are only used by
// GraphEdges.
//
// Named `graph_vertex_list` at the C++/internal layer even though the
// section it feeds (`SectionKind::GraphNodes` / config field
// `include_nodes_in_graph_tokenization`) uses "nodes" at the user
// surface. See the vertex/node convention: internal graph-theory
// code uses `vertex`, user-facing config + batch dict uses `node`.
int write_graph_vertex_list(WriteView view,
                            int cursor,
                            const SampledGraph& sg,
                            int num_special) {
    const int n = sg.graph.num_vertices();
    for (int v = 0; v < n; ++v) {
        view.set0(cursor + v, vertex_token(v, sg, num_special));
    }
    return n;
}

// Query content (the vertex tokens between QUERY_START and QUERY_END).
int write_query_content(WriteView view,
                        int cursor,
                        const SampledGraph& sg,
                        const Task& task,
                        int num_special,
                        const WorkerSharedContext& ctx) {
    int p = cursor;
    switch (task.kind) {
        case TaskKind::ShortestPath:
            view.set0(p++, vertex_token(*task.query.start_node, sg, num_special));
            view.set0(p++, vertex_token(*task.query.end_node,   sg, num_special));
            break;
        case TaskKind::BFS:
            view.set0(p++, vertex_token(*task.query.start_node, sg, num_special));
            break;
        case TaskKind::Center:
        case TaskKind::Centroid:
            for (int q : *task.query.vertex_set) {
                view.set0(p++, vertex_token(q, sg, num_special));
            }
            break;
        case TaskKind::KhopsGen: {
            // Query section for khops_gen is exactly the cursor token:
            // the last element of task.query.prefix. It's already a raw
            // vocab id (sampled directly from the node-vocab range by
            // sample_khops_gen), so we bypass vertex_token entirely.
            GG_CHECK(task.query.prefix.has_value() && !task.query.prefix->empty(),
                     "Tokenizer: khops_gen Task missing / empty query.prefix");
            view.set0(p++, task.query.prefix->back());
            (void)sg; (void)num_special;
            break;
        }
        case TaskKind::Khops: {
            // Query for the per-position khops variant is a single
            // "D<k>" depth marker sourced from ctx.token_dict. If the
            // user forgot to include enough D-extras when building
            // the dictionary this throws with an actionable message.
            GG_CHECK(task.query.khops_k.has_value(),
                     "Tokenizer: khops Task missing query.khops_k");
            const int k = *task.query.khops_k;
            const std::string key = "D" + std::to_string(k);
            auto it = ctx.token_dict.find(key);
            GG_CHECK(it != ctx.token_dict.end(),
                     "Tokenizer: khops depth marker '" << key << "' not found in "
                     "ctx.token_dict; ensure set_default_dictionary was called "
                     "with extra_after >= max_khops (or a custom dict including "
                     "the D<k> extras)");
            view.set0(p++, it->second);
            (void)sg; (void)num_special;
            break;
        }
        case TaskKind::None:
            GG_CHECK(false,
                     "Tokenizer: Query for task_kind='" << to_string(task.kind)
                         << "' not implemented");
    }
    return p - cursor;
}

// KhopsPrefix content: emit prefix[0..N-1) as raw vocab tokens (the
// final prefix token becomes the query cursor and is emitted by
// write_query_content instead). Tokens are already sampled from the
// node-vocab range so no vertex_token lookup is needed here.
int write_khops_prefix_content(WriteView view,
                               int cursor,
                               const Task& task) {
    GG_CHECK(task.kind == TaskKind::KhopsGen,
             "Tokenizer: KhopsPrefix section only valid for khops_gen "
             "(got task_kind='" << to_string(task.kind) << "')");
    GG_CHECK(task.query.prefix.has_value() && task.query.prefix->size() >= 2,
             "Tokenizer: khops_gen Task missing / undersized query.prefix");
    const auto& pfx = *task.query.prefix;
    const int  n    = static_cast<int>(pfx.size()) - 1;
    for (int i = 0; i < n; ++i) {
        view.set0(cursor + i, pfx[static_cast<std::size_t>(i)]);
    }
    return n;
}

// Thinking content: `count` repeated TOK_THINK tokens in column 0.
// No per-vertex variability -- just the marker at every position.
int write_thinking_content(WriteView view, int cursor, int count) {
    for (int i = 0; i < count; ++i) {
        view.set0(cursor + i, WorkerSharedContext::TOK_THINK);
    }
    return count;
}

// Emit one per-vertex BFS scratchpad block under the given style.
// Neighbours are always listed in first-mention edge-list order (see
// compute_first_mention_edge_index above) -- this was `sort_adjacency_lists`
// under a flag in the legacy code; it's now the fixed default because
// (a) it's derivable by the model from context and (b) any alternative
// order would leak either the internal-id or vocab-id ordering.
// Returns positions written.
int write_bfs_block_plain(WriteView view, int cursor,
                          int v,
                          const SampledGraph& sg,
                          const std::vector<int>& first_mention,
                          int num_special) {
    int p = cursor;
    view.set0(p++, vertex_token(v, sg, num_special));
    view.set0(p++, WorkerSharedContext::TOK_BFS_ADJ_START);
    for (int n : sorted_neighbours_by_first_mention(v, sg.graph, first_mention)) {
        view.set0(p++, vertex_token(n, sg, num_special));
    }
    view.set0(p++, WorkerSharedContext::TOK_BFS_ADJ_END);
    return p - cursor;
}

int write_bfs_block_reverse(WriteView view, int cursor,
                            int v,
                            const SampledGraph& sg,
                            const std::vector<int>& first_mention,
                            int num_special) {
    int p = cursor;
    view.set0(p++, vertex_token(v, sg, num_special));
    view.set0(p++, WorkerSharedContext::TOK_BFS_ADJ_START);
    // Reverse of the sorted (first-mention) order.
    auto nbrs = sorted_neighbours_by_first_mention(v, sg.graph, first_mention);
    for (auto it = nbrs.rbegin(); it != nbrs.rend(); ++it) {
        view.set0(p++, vertex_token(*it, sg, num_special));
    }
    view.set0(p++, WorkerSharedContext::TOK_BFS_ADJ_END);
    return p - cursor;
}

int write_bfs_block_duplicate(WriteView view, int cursor,
                              int v,
                              const SampledGraph& sg,
                              const std::vector<int>& first_mention,
                              int num_special) {
    int p = cursor;
    const auto nbrs = sorted_neighbours_by_first_mention(v, sg.graph, first_mention);
    view.set0(p++, vertex_token(v, sg, num_special));
    view.set0(p++, WorkerSharedContext::TOK_BFS_ADJ_START);
    for (int n : nbrs) {
        view.set0(p++, vertex_token(n, sg, num_special));
    }
    view.set0(p++, WorkerSharedContext::TOK_BFS_ADJ_END);
    // Second copy inside `{ ... }` for the alt-list fallback path.
    view.set0(p++, WorkerSharedContext::TOK_CURLY_START);
    for (int n : nbrs) {
        view.set0(p++, vertex_token(n, sg, num_special));
    }
    view.set0(p++, WorkerSharedContext::TOK_CURLY_END);
    return p - cursor;
}

int write_bfs_block_with_queue(WriteView view, int cursor,
                               int v,
                               const std::vector<int>& frontier,
                               const SampledGraph& sg,
                               const std::vector<int>& first_mention,
                               int num_special) {
    int p = cursor;
    // `{ q_0 q_1 ... q_k } v [ nbrs ]` -- q_0 is always v itself,
    // then the rest of the frontier at the moment v was popped.
    view.set0(p++, WorkerSharedContext::TOK_CURLY_START);
    for (int q : frontier) {
        view.set0(p++, vertex_token(q, sg, num_special));
    }
    view.set0(p++, WorkerSharedContext::TOK_CURLY_END);
    view.set0(p++, vertex_token(v, sg, num_special));
    view.set0(p++, WorkerSharedContext::TOK_BFS_ADJ_START);
    for (int n : sorted_neighbours_by_first_mention(v, sg.graph, first_mention)) {
        view.set0(p++, vertex_token(n, sg, num_special));
    }
    view.set0(p++, WorkerSharedContext::TOK_BFS_ADJ_END);
    return p - cursor;
}

// Scratchpad content. Dispatches on scratchpad.kind and, within BFS,
// on bfs_scratchpad_style. All four styles land here.
int write_scratchpad_content(WriteView view,
                             int cursor,
                             const SampledGraph& sg,
                             const Task& task,
                             const Scratchpad& scratchpad,
                             const GeneratorConfig& cfg,
                             int num_special) {
    int p = cursor;
    switch (scratchpad.kind) {
        case ScratchpadKind::BFS: {
            const BFSScratchpadStyle style =
                bfs_scratchpad_style_from_string(cfg.bfs_scratchpad_style);
            GG_CHECK(scratchpad.trace.has_value(),
                     "Tokenizer: bfs scratchpad missing trace");
            // Compute once, share across all per-vertex blocks.
            const auto first_mention = compute_first_mention_edge_index(sg);
            switch (style) {
                case BFSScratchpadStyle::Plain:
                    for (int v : *scratchpad.trace) {
                        p += write_bfs_block_plain(view, p, v, sg,
                                                   first_mention, num_special);
                    }
                    break;
                case BFSScratchpadStyle::ReverseAdjacency:
                    for (int v : *scratchpad.trace) {
                        p += write_bfs_block_reverse(view, p, v, sg,
                                                     first_mention, num_special);
                    }
                    break;
                case BFSScratchpadStyle::DuplicateAdjacency:
                    for (int v : *scratchpad.trace) {
                        p += write_bfs_block_duplicate(view, p, v, sg,
                                                       first_mention, num_special);
                    }
                    break;
                case BFSScratchpadStyle::WithQueue: {
                    // Frontier snapshots were cached on Scratchpad by
                    // sample_bfs, aligned index-for-index with trace.
                    GG_CHECK(scratchpad.frontier_at_pop.has_value(),
                             "Tokenizer: bfs+with_queue scratchpad missing "
                             "frontier_at_pop (sample_bfs should have "
                             "populated it)");
                    const auto& trace     = *scratchpad.trace;
                    const auto& frontiers = *scratchpad.frontier_at_pop;
                    GG_CHECK(trace.size() == frontiers.size(),
                             "Tokenizer: trace and frontier_at_pop size "
                             "mismatch (" << trace.size() << " vs "
                             << frontiers.size() << ")");
                    for (std::size_t i = 0; i < trace.size(); ++i) {
                        p += write_bfs_block_with_queue(view, p, trace[i],
                                                        frontiers[i], sg,
                                                        first_mention, num_special);
                    }
                    break;
                }
            }
            break;
        }
        case ScratchpadKind::DFS:
            GG_CHECK(false,
                     "Tokenizer: scratchpad_kind='dfs' not implemented");
            break;
        case ScratchpadKind::None:
            // Unreachable: make_section_plan omits the Scratchpad
            // section entirely for None. Guard anyway.
            GG_CHECK(false,
                     "Tokenizer: Scratchpad section reached with kind='none'");
            break;
    }
    return p - cursor;
}

// Target content (the tokens between TASK_START and TASK_END).
int write_target_content(WriteView view,
                         int cursor,
                         const SampledGraph& sg,
                         const Task& task,
                         int num_special) {
    int p = cursor;
    switch (task.kind) {
        case TaskKind::ShortestPath:
            GG_CHECK(task.target.path.has_value(),
                     "Tokenizer: shortest_path Task missing target.path");
            for (int v : *task.target.path) {
                view.set0(p++, vertex_token(v, sg, num_special));
            }
            break;
        case TaskKind::BFS:
            GG_CHECK(task.target.path.has_value(),
                     "Tokenizer: bfs Task missing target.path (visit order)");
            for (int v : *task.target.path) {
                view.set0(p++, vertex_token(v, sg, num_special));
            }
            break;
        case TaskKind::Center:
        case TaskKind::Centroid:
            GG_CHECK(task.target.vertex_set.has_value(),
                     "Tokenizer: " << to_string(task.kind)
                         << " Task missing target.vertex_set");
            GG_CHECK(!task.target.vertex_set->empty(),
                     "Tokenizer: " << to_string(task.kind)
                         << " Task has empty target.vertex_set");
            // Emit each center_node as a sequential token in the order
            // stored on the task (already shuffled at sample time by
            // sample_center_centroid). The RIGHT interpretation of
            // "correct output" here is a set, not a sequence -- the
            // targets tensor uses cumulative label smoothing so any
            // permutation minimises the loss equally.
            for (int v : *task.target.vertex_set) {
                view.set0(p++, vertex_token(v, sg, num_special));
            }
            break;
        case TaskKind::KhopsGen:
            // Target for khops_gen is the k ground-truth hop tokens,
            // stored in target.path as RAW vocab ids (not vertex ids).
            // Emit them straight through -- no vertex_token indirection.
            GG_CHECK(task.target.path.has_value(),
                     "Tokenizer: khops_gen Task missing target.path");
            for (int t : *task.target.path) {
                view.set0(p++, t);
            }
            (void)sg; (void)num_special;
            break;
        case TaskKind::Khops:
            // Target for the per-position khops variant is the entire
            // length-P vocab sequence (target.path stores the raw
            // vocab tokens). The per-position hop LABELS live on
            // target.per_position_labels and are patched into the
            // targets tensor by fill_targets; the src stream at each
            // seq position still shows the seq token itself so the
            // model reads seq[i] as INPUT and predicts the hop label
            // as OUTPUT at the same position.
            GG_CHECK(task.target.path.has_value(),
                     "Tokenizer: khops Task missing target.path (seq)");
            for (int t : *task.target.path) {
                view.set0(p++, t);
            }
            (void)sg; (void)num_special;
            break;
        case TaskKind::None:
            GG_CHECK(false,
                     "Tokenizer: Target for task_kind='" << to_string(task.kind)
                         << "' not implemented");
    }
    return p - cursor;
}

// Fill positional-ids for one row. Padding past seq_len is left to
// the caller's pre-fill (0 for positions).
//
// Positions are written at column 0 of each position (columns 1..D-1
// stay at the pad-fill 0). Under SEAN struct_dim==1 so col 0 is the
// only slot; under STAN cols 1..D-1 stay at 0 by design.
void fill_positions(int32_t* pos, int seq_len, int struct_dim) {
    if (pos == nullptr) return;
    for (int i = 0; i < seq_len; ++i) {
        pos[i * struct_dim] = static_cast<int32_t>(i);
    }
}

// Fill the targets tensor row for one item.
//
// `targets` points at the row's (max_gen_len, max_labels) block,
// laid out row-major and pre-filled with TOK_PAD. For each
// generation-region position gp in [0, gen_len):
//
//   targets[gp, 0]        = the "chosen" token (matches what
//                           `src_view` holds at column prefix_len+gp,
//                           col 0)
//   targets[gp, k>0]      = other equally-valid labels at that step
//                           (label smoothing), specific to task kind
//   targets[gp, k>=count] = untouched -> TOK_PAD
//   targets[gp>=gen_len]  = untouched -> TOK_PAD
//
// Alternatives are populated per task kind:
//   * ShortestPath: intermediate hop positions use
//     `task.target.valid_next_hops[step]`. Start and end positions
//     have only the single canonical label.
//   * Center / Centroid: each of the |center_nodes| target positions
//     gets the remaining not-yet-emitted center_nodes as equally
//     valid labels (cumulative label smoothing).
//   * BFS / other: single label per position (no smoothing).
//
// max_labels comes from the batch scan in worker.cpp -- alternatives
// past max_labels are dropped (rare in practice; a warning could be
// added if a task ever needs more than the batch's max).
void fill_targets(WriteView            src_view,
                  int32_t*             targets,
                  int                  max_labels,
                  const Layout&        layout,
                  const SampledGraph&  sg,
                  const Task&          task,
                  int                  struct_dim,
                  int                  num_special) {
    const int prefix  = layout.prefix_len;
    const int gen_len = layout.gen_len;

    // Baseline: every generation-region position gets its src token
    // as the single (k=0) label. Most positions -- markers, thinking,
    // scratchpad content, BFS visit order -- are done here and stay
    // untouched below.
    for (int gp = 0; gp < gen_len; ++gp) {
        const int src_col = prefix + gp;
        targets[gp * max_labels + 0] = src_view.dst[src_col * struct_dim];
    }

    // Patch in per-task alternatives at the target-section positions
    // that have label smoothing.
    const int t_start = layout.task_target_start;
    const int t_gp    = t_start - prefix;   // target-section origin in gen coords

    switch (task.kind) {
        case TaskKind::ShortestPath: {
            // path has L+1 entries; valid_next_hops has L. For each
            // step k in [0, L), path position (k+1) gets alternatives
            // from valid_next_hops[k]. Path position 0 (start) has no
            // alternatives -- it's the fixed query.
            if (!task.target.valid_next_hops.has_value()) break;
            const auto& vnh  = *task.target.valid_next_hops;
            const auto& path = *task.target.path;
            for (int step = 0; step < static_cast<int>(vnh.size()); ++step) {
                const int gp     = t_gp + step + 1;
                const int chosen = path[static_cast<std::size_t>(step + 1)];
                // Chosen at k=0 (already matches src). Fill k>=1 with
                // the OTHER valid_next_hops entries; skip `chosen` to
                // avoid a duplicate label at k=0 and k=?.
                int k = 1;
                for (int v : vnh[static_cast<std::size_t>(step)]) {
                    if (v == chosen) continue;
                    if (k >= max_labels) break;
                    targets[gp * max_labels + k] =
                        vertex_token(v, sg, num_special);
                    ++k;
                }
            }
            break;
        }
        case TaskKind::Center:
        case TaskKind::Centroid: {
            // Cumulative label smoothing across |center_nodes| target
            // positions. At target position i:
            //   remaining = center_nodes[i], center_nodes[i+1], ..., center_nodes[N-1]
            //   targets[i, 0] = center_nodes[i]   (matches src)
            //   targets[i, k>0] = center_nodes[i+k]   (label smoothing:
            //                    "any not-yet-emitted center_node is
            //                    equally valid at this step")
            // The baseline copy above already put center_nodes[i] at
            // (gp=t_gp+i, k=0), so we only need to fill k>=1 here.
            if (!task.target.vertex_set.has_value()) break;
            const auto&   cn  = *task.target.vertex_set;
            const int     N   = static_cast<int>(cn.size());
            for (int i = 0; i < N; ++i) {
                const int gp = t_gp + i;
                // Alternatives at this step: center_nodes[i+1..N-1].
                // Cap at max_labels; leftover slots stay TOK_PAD.
                int k = 1;
                for (int j = i + 1; j < N; ++j) {
                    if (k >= max_labels) break;
                    targets[gp * max_labels + k] =
                        vertex_token(cn[static_cast<std::size_t>(j)],
                                     sg, num_special);
                    ++k;
                }
            }
            break;
        }
        case TaskKind::Khops: {
            // Per-position hop labels. Each seq position i in [0, P)
            // maps to gen position gp = t_gp + i. Sample-time
            // per_position_labels[i] carries either:
            //   * empty vector  -- this position is MASKED, so we
            //                      clobber the baseline single label
            //                      with TOK_PAD across every column
            //                      so the loss ignores it.
            //   * non-empty     -- ordered [chosen, alt_1, alt_2, ...]
            //                      list of vocab tokens. The baseline
            //                      wrote the src token (== seq[i]) at
            //                      column 0; we overwrite [0..M-1]
            //                      with the labels and leave [M..]
            //                      as TOK_PAD.
            //
            // In intermediate_labels mode M can go up to k; otherwise
            // M == 1. Either way we cap at max_labels (batch-wide).
            if (!task.target.per_position_labels.has_value()) break;
            const auto& ppl = *task.target.per_position_labels;
            const int   P   = static_cast<int>(ppl.size());
            for (int i = 0; i < P; ++i) {
                const int gp = t_gp + i;
                const auto& labels = ppl[static_cast<std::size_t>(i)];
                if (labels.empty()) {
                    // Masked. Wipe every label slot at this position.
                    for (int c = 0; c < max_labels; ++c) {
                        targets[gp * max_labels + c] =
                            WorkerSharedContext::TOK_PAD;
                    }
                    continue;
                }
                const int M = std::min(static_cast<int>(labels.size()), max_labels);
                for (int c = 0; c < M; ++c) {
                    targets[gp * max_labels + c] =
                        labels[static_cast<std::size_t>(c)];
                }
            }
            (void)sg; (void)num_special;
            break;
        }
        case TaskKind::BFS:
        case TaskKind::None:
        case TaskKind::KhopsGen:
            // No smoothing; baseline single-label copy above is all
            // that's needed.
            break;
    }
}

}  // namespace

void Tokenizer::tokenize_into_row(int32_t* src,
                                  int32_t* targets,
                                  int      max_labels,
                                  int32_t* pos_row,
                                  const SampledGraph& sg,
                                  const Task& task,
                                  const Scratchpad& scratchpad,
                                  const Layout& layout,
                                  const WorkerSharedContext& ctx,
                                  const GeneratorConfig& cfg) {
    // Convenience path: rebuild the plan and delegate. Worker uses the
    // plan-explicit overload to avoid this rebuild.
    const SectionPlan plan = make_section_plan(sg, task, scratchpad, cfg);
    tokenize_into_row(src, targets, max_labels, pos_row, plan,
                      sg, task, scratchpad, layout, ctx, cfg);
}

void Tokenizer::tokenize_into_row(int32_t* src,
                                  int32_t* targets,
                                  int      max_labels,
                                  int32_t* pos_row,
                                  const SectionPlan& plan,
                                  const SampledGraph& sg,
                                  const Task& task,
                                  const Scratchpad& scratchpad,
                                  const Layout& layout,
                                  const WorkerSharedContext& ctx,
                                  const GeneratorConfig& cfg) {
    GG_CHECK(src     != nullptr, "Tokenizer: src_tokens_row must not be null");
    GG_CHECK(targets != nullptr, "Tokenizer: targets_row must not be null");
    GG_CHECK(max_labels >= 1,
             "Tokenizer: max_labels must be >= 1 (got " << max_labels << ")");

    const TokenizationMode mode =
        tokenization_mode_from_string(cfg.tokenization_mode);
    const int struct_dim = struct_dim_for(mode);
    GG_ASSERT(struct_dim == layout.struct_dim,
              "Tokenizer: mode struct_dim=" << struct_dim
                  << " disagrees with layout.struct_dim=" << layout.struct_dim);

    const int num_special = num_special_from_ctx(ctx);

    const WriteView src_view{src, struct_dim};

    // Walk the section plan Layout used. Each section either writes
    // one marker token or dispatches to a content writer. `cursor` is
    // a SEQUENCE POSITION; WriteView maps (pos, col) -> flat offset.
    int cursor = 0;
    for (const Section& s : plan) {
        switch (s.kind) {
            case SectionKind::BOS:
            case SectionKind::EOS:
            case SectionKind::QueryStart:
            case SectionKind::QueryEnd:
            case SectionKind::ThinkStart:
            case SectionKind::ThinkEnd:
            case SectionKind::ScratchpadStart:
            case SectionKind::ScratchpadEnd:
            case SectionKind::TaskStart:
            case SectionKind::TaskEnd:
                cursor += write_marker(src_view, cursor, s.kind);
                break;
            case SectionKind::GraphEdges:
                cursor += write_graph_edges(src_view, cursor, sg, mode, cfg, num_special);
                break;
            case SectionKind::GraphNodes:
                cursor += write_graph_vertex_list(src_view, cursor, sg, num_special);
                break;
            case SectionKind::Query:
                cursor += write_query_content(src_view, cursor, sg, task, num_special, ctx);
                break;
            case SectionKind::Thinking:
                cursor += write_thinking_content(src_view, cursor, s.count);
                break;
            case SectionKind::Scratchpad:
                cursor += write_scratchpad_content(src_view, cursor, sg, task, scratchpad,
                                                   cfg, num_special);
                break;
            case SectionKind::Target:
                cursor += write_target_content(src_view, cursor, sg, task, num_special);
                break;
            case SectionKind::KhopsPrefix:
                cursor += write_khops_prefix_content(src_view, cursor, task);
                break;
        }
    }

    GG_ASSERT(cursor == layout.seq_len,
              "Tokenizer: wrote " << cursor << " tokens, layout says "
                  << layout.seq_len);

    fill_positions(pos_row, layout.seq_len, struct_dim);
    fill_targets(src_view, targets, max_labels, layout, sg, task,
                 struct_dim, num_special);
}

}  // namespace graphgen
