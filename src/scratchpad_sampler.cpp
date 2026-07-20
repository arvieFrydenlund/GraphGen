#include "graphgen/scratchpad_sampler.h"

#include <utility>
#include <vector>

#include "graphgen/bfs_helpers.h"
#include "graphgen/bfs_scratchpad_style.h"
#include "graphgen/check.h"
#include "graphgen/csr_graph.h"
#include "graphgen/generator_config.h"
#include "graphgen/sampled_graph.h"
#include "graphgen/task.h"

namespace graphgen {

namespace {

// The BFS scratchpad's trace is the queue-pop visit order of a BFS run
// starting at the task's query start-node. Both Bfs and ShortestPath
// tasks populate query.start_node, so this helper works for either.
// Returning the visit order rather than reusing task.target.path lets
// the scratchpad be independent of what the task chose to emit -- e.g.
// for shortest_path the task's `path` is only the u->v answer, not a
// full BFS trace.
std::vector<int> bfs_visit_order(const CsrGraph& g, int start) {
    const int n = g.num_vertices();
    std::vector<int> visit_order;
    visit_order.reserve(static_cast<std::size_t>(n));
    std::vector<char> visited(static_cast<std::size_t>(n), 0);
    std::vector<int>  current;
    std::vector<int>  next;
    current.reserve(static_cast<std::size_t>(n));
    next.reserve(static_cast<std::size_t>(n));

    visited[static_cast<std::size_t>(start)] = 1;
    visit_order.push_back(start);
    current.push_back(start);
    while (!current.empty()) {
        for (int u : current) {
            for (int v : g.neighbours(u)) {
                if (!visited[static_cast<std::size_t>(v)]) {
                    visited[static_cast<std::size_t>(v)] = 1;
                    visit_order.push_back(v);
                    next.push_back(v);
                }
            }
        }
        current.swap(next);
        next.clear();
    }
    return visit_order;
}

}  // namespace

Scratchpad ScratchpadSampler::run(ScratchpadKind kind,
                                  const SampledGraph& sg,
                                  const Task& task,
                                  std::mt19937_64& rng,
                                  const GeneratorConfig& cfg) {
    switch (kind) {
        case ScratchpadKind::None: {
            Scratchpad s;
            s.kind = ScratchpadKind::None;
            return s;
        }
        case ScratchpadKind::BFS:
            return sample_bfs(sg, task, rng, cfg);
        case ScratchpadKind::DFS:
            return sample_dfs(sg, task, rng, cfg);
    }
    GG_CHECK(false, "ScratchpadSampler::run: unreachable ScratchpadKind");
}

Scratchpad ScratchpadSampler::sample_bfs(const SampledGraph& sg,
                                         const Task& task,
                                         std::mt19937_64& /*rng*/,
                                         const GeneratorConfig& cfg) {
    GG_CHECK(task.query.start_node.has_value(),
             "sample_bfs scratchpad: task must have query.start_node "
             "(both Bfs and ShortestPath tasks populate it)");

    Scratchpad s;
    s.kind = ScratchpadKind::BFS;

    // Only WithQueue needs the queue-at-pop snapshots. For every
    // other style the ping-pong bfs_visit_order suffices and we skip
    // both the frontier allocation and the FIFO-BFS run. Length and
    // token consumers below both read from scratchpad.frontier_at_pop
    // when it's populated, avoiding a second BFS pass at emit time.
    const BFSScratchpadStyle style =
        bfs_scratchpad_style_from_string(cfg.bfs_scratchpad_style);
    if (style == BFSScratchpadStyle::WithQueue) {
        auto pops = bfs_fifo_pops(sg.graph, *task.query.start_node);
        std::vector<int>              trace;
        std::vector<std::vector<int>> frontiers;
        trace.reserve(pops.size());
        frontiers.reserve(pops.size());
        for (auto& p : pops) {
            trace.push_back(p.vertex);
            frontiers.push_back(std::move(p.frontier));
        }
        s.trace           = std::move(trace);
        s.frontier_at_pop = std::move(frontiers);
    } else {
        s.trace = bfs_visit_order(sg.graph, *task.query.start_node);
    }
    return s;
}

Scratchpad ScratchpadSampler::sample_dfs(const SampledGraph&,
                                         const Task&,
                                         std::mt19937_64&,
                                         const GeneratorConfig&) {
    GG_CHECK(false, "sample_dfs scratchpad not implemented");
}

}  // namespace graphgen
