#include "graphgen/section_plan.h"

#include "graphgen/bfs_scratchpad_style.h"
#include "graphgen/check.h"
#include "graphgen/csr_graph.h"
#include "graphgen/generator_config.h"
#include "graphgen/sampled_graph.h"
#include "graphgen/scratchpad.h"
#include "graphgen/task.h"

namespace graphgen {

namespace {

// Each of the append_* helpers below owns one "slot" of the sequential
// axis. Adding a new task = one case per task-owned slot (query,
// target). Adding a new scratchpad = one case in the scratchpad slot.
// Feature flags (no_graph, num_thinking_tokens, ...) are checked in
// make_section_plan below and decide whether a slot appears at all.

void append_graph_sections(SectionPlan& plan,
                           const SampledGraph& sg,
                           const GeneratorConfig& cfg) {
    // Optional nodes prelude: list every vertex token before the
    // edges. Useful when the caller wants the model to see the
    // vertex vocabulary before parsing edge tuples. Flag lives on
    // cfg because it's a tokenization-side choice, not a graph
    // property.
    if (cfg.include_nodes_in_graph_tokenization) {
        plan.push_back({SectionKind::GraphNodes, sg.graph.num_vertices()});
    }
    // Edges. If include_duplicate_edges_in_graph_tokenization is set,
    // each undirected edge is emitted twice (u v EDGE then v u EDGE),
    // so the semantic count doubles.
    const int edge_count = cfg.include_duplicate_edges_in_graph_tokenization
        ? 2 * sg.graph.num_edges()
        : sg.graph.num_edges();
    plan.push_back({SectionKind::GraphEdges, edge_count});
}

void append_query_sections(SectionPlan& plan, const Task& task) {
    plan.push_back({SectionKind::QueryStart});
    switch (task.kind) {
        case TaskKind::ShortestPath:
            GG_CHECK(task.query.start_node.has_value(),
                     "make_section_plan: shortest_path Task missing query.start_node");
            GG_CHECK(task.query.end_node.has_value(),
                     "make_section_plan: shortest_path Task missing query.end_node");
            plan.push_back({SectionKind::Query, /*start,end*/ 2});
            break;
        case TaskKind::BFS:
            GG_CHECK(task.query.start_node.has_value(),
                     "make_section_plan: bfs Task missing query.start_node");
            plan.push_back({SectionKind::Query, /*start only*/ 1});
            break;
        case TaskKind::Center:
        case TaskKind::Centroid:
            GG_CHECK(task.query.vertex_set.has_value(),
                     "make_section_plan: " << to_string(task.kind)
                         << " Task missing query.vertex_set");
            GG_CHECK(!task.query.vertex_set->empty(),
                     "make_section_plan: " << to_string(task.kind)
                         << " Task has empty query.vertex_set");
            plan.push_back({SectionKind::Query,
                            static_cast<int>(task.query.vertex_set->size())});
            break;
        case TaskKind::KhopsGen:
            // Layout for khops_gen mirrors V1: the concatenated
            // segments become the pre-query prefix, and the LAST
            // segment token doubles as the query "cursor". We keep the
            // full concatenation in query.prefix and split the
            // rendering across two sections -- KhopsPrefix (all but
            // last) and Query (just the cursor). Preserving the split
            // lets downstream masking / positional treatment target
            // "the cursor" specifically without pattern-matching
            // token positions.
            GG_CHECK(task.query.prefix.has_value(),
                     "make_section_plan: khops_gen Task missing query.prefix");
            GG_CHECK(task.query.prefix->size() >= 2,
                     "make_section_plan: khops_gen prefix must have at least 2 tokens "
                     "(got " << task.query.prefix->size() << ")");
            plan.push_back({SectionKind::Query, /*cursor only*/ 1});
            break;
        case TaskKind::Khops:
            // Query for the per-position khops variant is a single
            // "D<k>" depth marker token. The k value is stored on
            // query.khops_k; the tokenizer resolves it to a token id
            // by looking up ctx.token_dict.at("D" + str(k)).
            GG_CHECK(task.query.khops_k.has_value(),
                     "make_section_plan: khops Task missing query.khops_k");
            plan.push_back({SectionKind::Query, /*D_k marker*/ 1});
            break;
        case TaskKind::None:
            GG_CHECK(false,
                     "make_section_plan: task_kind='" << to_string(task.kind)
                         << "' not implemented");
    }
    plan.push_back({SectionKind::QueryEnd});
}

// Thinking sits at the start of the generation region: right after
// QueryEnd and before ScratchpadStart (if present) or TaskStart
// (otherwise). Content is `num_thinking_tokens` repeated TOK_THINK
// tokens, so the count is copied straight from cfg.
void append_thinking_sections(SectionPlan& plan, const GeneratorConfig& cfg) {
    if (cfg.num_thinking_tokens <= 0) {
        return;
    }
    plan.push_back({SectionKind::ThinkStart});
    plan.push_back({SectionKind::Thinking, cfg.num_thinking_tokens});
    plan.push_back({SectionKind::ThinkEnd});
}

void append_scratchpad_sections(SectionPlan& plan,
                                const SampledGraph& sg,
                                const Task& task,
                                const Scratchpad& scratchpad,
                                const GeneratorConfig& cfg) {
    if (scratchpad.kind == ScratchpadKind::None) {
        return;
    }
    plan.push_back({SectionKind::ScratchpadStart});
    switch (scratchpad.kind) {
        case ScratchpadKind::BFS: {
            // Per-vertex block layout, by style:
            //   Plain              : v [ nbrs ]                           = 3 + deg(v)
            //   WithQueue          : { q } v [ nbrs ]                     = 5 + deg(v) + |q_v|
            //   ReverseAdjacency   : v [ nbrs_reversed ]                  = 3 + deg(v)
            //   DuplicateAdjacency : v [ nbrs ] { nbrs }                  = 5 + 2*deg(v)
            GG_CHECK(scratchpad.trace.has_value(),
                     "make_section_plan: bfs scratchpad missing trace");
            (void)task;  // future styles may depend on the task's query.
            const int n_visited = static_cast<int>(scratchpad.trace->size());
            int sum_deg = 0;
            for (int v : *scratchpad.trace) {
                sum_deg += sg.graph.degree(v);
            }
            const BFSScratchpadStyle style =
                bfs_scratchpad_style_from_string(cfg.bfs_scratchpad_style);
            int content_len = 0;
            switch (style) {
                case BFSScratchpadStyle::Plain:
                case BFSScratchpadStyle::ReverseAdjacency:
                    content_len = 3 * n_visited + sum_deg;
                    break;
                case BFSScratchpadStyle::DuplicateAdjacency:
                    content_len = 5 * n_visited + 2 * sum_deg;
                    break;
                case BFSScratchpadStyle::WithQueue: {
                    // Frontier data was cached on Scratchpad by
                    // sample_bfs (which ran FIFO BFS for exactly this
                    // reason). Reading it here avoids a second BFS.
                    GG_CHECK(scratchpad.frontier_at_pop.has_value(),
                             "make_section_plan: bfs+with_queue scratchpad "
                             "missing frontier_at_pop (sample_bfs should "
                             "have populated it)");
                    int sum_frontier = 0;
                    for (const auto& fr : *scratchpad.frontier_at_pop) {
                        sum_frontier += static_cast<int>(fr.size());
                    }
                    content_len = 5 * n_visited + sum_deg + sum_frontier;
                    break;
                }
            }
            plan.push_back({SectionKind::Scratchpad, content_len});
            break;
        }
        case ScratchpadKind::DFS:
            GG_CHECK(false,
                     "make_section_plan: scratchpad_kind='dfs' not implemented");
            break;
        case ScratchpadKind::None:
            // Unreachable -- checked above.
            break;
    }
    plan.push_back({SectionKind::ScratchpadEnd});
}

void append_target_sections(SectionPlan& plan, const Task& task) {
    plan.push_back({SectionKind::TaskStart});
    switch (task.kind) {
        case TaskKind::ShortestPath: {
            GG_CHECK(task.target.path.has_value(),
                     "make_section_plan: shortest_path Task missing target.path");
            const int k = static_cast<int>(task.target.path->size()) - 1;
            GG_CHECK(k >= 0,
                     "make_section_plan: shortest_path path length must be >= 0 (got "
                         << k << ")");
            plan.push_back({SectionKind::Target, k + 1});
            break;
        }
        case TaskKind::BFS: {
            GG_CHECK(task.target.path.has_value(),
                     "make_section_plan: bfs Task missing target.path (visit order)");
            const int visit_len = static_cast<int>(task.target.path->size());
            GG_CHECK(visit_len >= 1,
                     "make_section_plan: bfs visit order must have >= 1 vertex (got "
                         << visit_len << ")");
            plan.push_back({SectionKind::Target, visit_len});
            break;
        }
        case TaskKind::Center:
        case TaskKind::Centroid: {
            GG_CHECK(task.target.vertex_set.has_value(),
                     "make_section_plan: " << to_string(task.kind)
                         << " Task missing target.vertex_set");
            const int m = static_cast<int>(task.target.vertex_set->size());
            GG_CHECK(m >= 1,
                     "make_section_plan: " << to_string(task.kind)
                         << " target must have >= 1 center_node (got "
                         << m << ")");
            // Emit `|center_nodes|` sequential target positions. The
            // center_nodes have been shuffled at sample time so their
            // ORDER carries no signal; the tokenizer's targets tensor
            // applies cumulative label smoothing (every unemitted
            // center_node is valid at each step), turning this into a
            // set-generation objective rather than a fixed sequence.
            plan.push_back({SectionKind::Target, m});
            break;
        }
        case TaskKind::KhopsGen: {
            // Target = the k ground-truth back-hop tokens. Emitted as
            // `k` sequence positions of raw vocab tokens; the tokenizer
            // writes them directly without vertex_token indirection
            // (they're already node-vocab ids).
            GG_CHECK(task.target.path.has_value(),
                     "make_section_plan: khops_gen Task missing target.path");
            const int k = static_cast<int>(task.target.path->size());
            GG_CHECK(k >= 1,
                     "make_section_plan: khops_gen target must have >= 1 hop (got "
                         << k << ")");
            plan.push_back({SectionKind::Target, k});
            break;
        }
        case TaskKind::Khops: {
            // Target = the full length-P vocab sequence itself. The
            // per-position hop labels live on target.per_position_labels
            // and are patched into the targets tensor by fill_targets;
            // the src token at each seq position stays the seq token
            // (the model reads seq[i] as INPUT and predicts the hop
            // label as OUTPUT at that position).
            GG_CHECK(task.target.path.has_value(),
                     "make_section_plan: khops Task missing target.path (seq)");
            const int P = static_cast<int>(task.target.path->size());
            GG_CHECK(P >= 2,
                     "make_section_plan: khops seq must have >= 2 tokens (got "
                         << P << ")");
            plan.push_back({SectionKind::Target, P});
            break;
        }
        case TaskKind::None:
            // Rejection already happened in append_query_sections; keep
            // the switch exhaustive so new task kinds trip a compiler
            // warning until they're wired up.
            break;
    }
    plan.push_back({SectionKind::TaskEnd});
}

// K-hops prefix slot: sits immediately after BOS and before the (empty
// for khops_gen) graph slot. Emits `prefix.size() - 1` raw vocab
// tokens; the last prefix token becomes the query cursor and lives
// in the Query section instead.
void append_khops_prefix_sections(SectionPlan& plan, const Task& task) {
    if (task.kind != TaskKind::KhopsGen) return;
    GG_CHECK(task.query.prefix.has_value(),
             "make_section_plan: khops_gen Task missing query.prefix");
    const int prefix_len = static_cast<int>(task.query.prefix->size());
    GG_CHECK(prefix_len >= 2,
             "make_section_plan: khops_gen prefix must have at least 2 tokens (got "
                 << prefix_len << ")");
    plan.push_back({SectionKind::KhopsPrefix, prefix_len - 1});
}

}  // namespace

SectionPlan make_section_plan(const SampledGraph& sg,
                              const Task& task,
                              const Scratchpad& scratchpad,
                              const GeneratorConfig& cfg) {
    SectionPlan plan;
    // Rough upper bound: BOS + edges + (QS,Q,QE) + (SS,S,SE) + (TS,T,TE) + EOS.
    plan.reserve(13);

    plan.push_back({SectionKind::BOS});
    // KhopsPrefix (khops_gen only): sits between BOS and the graph
    // slot. No-op for every other task kind.
    append_khops_prefix_sections(plan, task);
    // Query placement: `query_at_end` (default true) puts the query
    // AFTER the graph, right before the generation region. When false
    // the query moves to before the graph, so the model sees the
    // question first. Both orderings still keep the query in the
    // prompt (never in the generation region).
    if (!cfg.query_at_end) {
        append_query_sections(plan, task);
    }
    if (cfg.include_graph_in_graph_tokenization) {
        append_graph_sections(plan, sg, cfg);
    }
    if (cfg.query_at_end) {
        append_query_sections(plan, task);
    }
    // Generation region: thinking -> scratchpad -> target. All three
    // are optional (each is a no-op if its relevant count is zero /
    // its kind is None).
    append_thinking_sections  (plan, cfg);
    append_scratchpad_sections(plan, sg, task, scratchpad, cfg);
    append_target_sections    (plan, task);
    plan.push_back({SectionKind::EOS});
    return plan;
}

int num_positions(const Section& s, TokenizationMode mode) {
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
            return 1;
        case SectionKind::GraphEdges:
            // SEAN spreads each edge across 3 sequence positions
            // (u, v, EDGE); STAN packs it into 1 position with 3
            // content columns. This is the sole mode-variable section.
            return mode == TokenizationMode::Sean ? 3 * s.count : s.count;
        case SectionKind::GraphNodes:
        case SectionKind::Query:
        case SectionKind::Thinking:
        case SectionKind::Scratchpad:
        case SectionKind::Target:
        case SectionKind::KhopsPrefix:
            return s.count;
    }
    return 0;  // unreachable; keeps -Wreturn-type quiet.
}

int struct_dim_for(TokenizationMode mode) {
    switch (mode) {
        case TokenizationMode::Sean: return 1;
        case TokenizationMode::Stan: return 3;
    }
    return 1;
}

}  // namespace graphgen
