#include "doctest.h"

#include <algorithm>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>

#include "graphgen/csr_graph.h"
#include "graphgen/generator_config.h"
#include "graphgen/sampled_graph.h"
#include "graphgen/scratchpad.h"
#include "graphgen/task.h"
#include "graphgen/task_sampler.h"

using graphgen::CsrGraph;
using graphgen::Edge;
using graphgen::GeneratorConfig;
using graphgen::SampledGraph;
using graphgen::Scratchpad;
using graphgen::ScratchpadKind;
using graphgen::Task;
using graphgen::TaskKind;
using graphgen::TaskSampler;

namespace {

// Wrap a hand-built CsrGraph in a SampledGraph, populating a trivial
// identity vocab mapping so downstream code that reads it doesn't
// misbehave. sample_shortest_path itself doesn't touch vocab_ids.
SampledGraph make_sg(CsrGraph g) {
    SampledGraph sg;
    const int n = g.num_vertices();
    sg.graph = std::move(g);
    sg.internal_to_vocab.resize(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) sg.internal_to_vocab[i] = i;
    return sg;
}

// Minimal config bundle for sample_shortest_path. min/max_path_length
// are optionals on GeneratorConfig, hence the explicit assignments.
GeneratorConfig make_cfg(int min_len, int max_len, int max_attempts = 1000) {
    GeneratorConfig cfg;
    cfg.min_path_length = min_len;
    cfg.max_path_length = max_len;
    cfg.max_attempts    = max_attempts;
    return cfg;
}

TaskSampler make_sampler() { return TaskSampler{}; }

}  // namespace

TEST_CASE("sample_shortest_path on a line graph: unique path, single-choice DAG") {
    // 0 -- 1 -- 2 -- 3 -- 4
    std::vector<Edge> edges = {{0, 1}, {1, 2}, {2, 3}, {3, 4}};
    SampledGraph sg = make_sg(CsrGraph::from_undirected_edges(5, edges));
    GeneratorConfig cfg = make_cfg(1, 4);
    std::mt19937_64 rng(42);
    TaskSampler ts = make_sampler();

    auto [task, scratchpad] =
        ts.run(TaskKind::ShortestPath, ScratchpadKind::None, sg, rng, cfg);

    CHECK(task.kind == TaskKind::ShortestPath);
    REQUIRE(task.query.start_node.has_value());
    REQUIRE(task.query.end_node.has_value());
    REQUIRE(task.target.path.has_value());
    REQUIRE(task.target.valid_next_hops.has_value());
    CHECK(scratchpad.kind == ScratchpadKind::None);

    const auto& path = *task.target.path;
    const auto& hops = *task.target.valid_next_hops;

    // Path endpoints match query.
    CHECK(path.front() == *task.query.start_node);
    CHECK(path.back()  == *task.query.end_node);

    // Consecutive pairs are edges.
    for (std::size_t i = 0; i + 1 < path.size(); ++i) {
        CHECK(sg.graph.has_edge(path[i], path[i + 1]));
    }

    // valid_next_hops has one entry per step.
    CHECK(hops.size() + 1 == path.size());
    // On a line graph, at every step there is exactly one valid next hop.
    for (const auto& step_hops : hops) CHECK(step_hops.size() == 1);
    // And that hop is what got chosen.
    for (std::size_t i = 0; i < hops.size(); ++i) {
        CHECK(hops[i][0] == path[i + 1]);
    }
}

TEST_CASE("sample_shortest_path in a square: two equally-valid paths, DAG width 2 in the middle") {
    // Cycle 0-1-2-3-0. Distance 0->2 is 2 via either 1 or 3.
    std::vector<Edge> edges = {{0, 1}, {1, 2}, {2, 3}, {3, 0}};
    SampledGraph sg = make_sg(CsrGraph::from_undirected_edges(4, edges));
    // Pin path length to exactly 2 in a graph where the only length-2
    // unordered pairs are the two diagonals; the reject-loop will land
    // on one of them (deterministic under a seed).
    GeneratorConfig cfg = make_cfg(2, 2);
    std::mt19937_64 rng(0);
    TaskSampler ts = make_sampler();

    auto [task, scratchpad] =
        ts.run(TaskKind::ShortestPath, ScratchpadKind::None, sg, rng, cfg);

    REQUIRE(task.target.path.has_value());
    REQUIRE(task.target.valid_next_hops.has_value());
    const auto& path = *task.target.path;
    const auto& hops = *task.target.valid_next_hops;

    // Length should be exactly 2 (three vertices).
    CHECK(path.size() == 3);

    // At the middle step the DAG has two equally valid next hops.
    CHECK(hops[0].size() == 2);
    // The chosen middle vertex must be one of the valid options.
    const std::set<int> valid_set(hops[0].begin(), hops[0].end());
    CHECK(valid_set.count(path[1]) == 1);
    // The step from middle to end has exactly one option.
    CHECK(hops[1].size() == 1);
    CHECK(hops[1][0] == path[2]);
}

TEST_CASE("sample_shortest_path: chosen next hop is always inside valid_next_hops") {
    // Small graph with multiple shortest paths in places.
    std::vector<Edge> edges = {
        {0, 1}, {0, 2}, {1, 3}, {2, 3},
        {3, 4}, {4, 5}, {3, 5},
    };
    SampledGraph sg = make_sg(CsrGraph::from_undirected_edges(6, edges));
    GeneratorConfig cfg = make_cfg(1, 5);
    std::mt19937_64 rng(123);
    TaskSampler ts = make_sampler();

    for (int trial = 0; trial < 100; ++trial) {
        auto [task, _] =
            ts.run(TaskKind::ShortestPath, ScratchpadKind::None, sg, rng, cfg);
        const auto& path = *task.target.path;
        const auto& hops = *task.target.valid_next_hops;
        for (std::size_t i = 0; i < hops.size(); ++i) {
            const std::set<int> valid(hops[i].begin(), hops[i].end());
            CHECK(valid.count(path[i + 1]) == 1);
        }
    }
}

TEST_CASE("sample_shortest_path: path length matches BFS distance and is in range") {
    std::vector<Edge> edges = {
        {0, 1}, {0, 2}, {1, 3}, {2, 3}, {3, 4}, {4, 5}, {5, 6},
    };
    SampledGraph sg = make_sg(CsrGraph::from_undirected_edges(7, edges));
    GeneratorConfig cfg = make_cfg(2, 4);
    std::mt19937_64 rng(7);
    TaskSampler ts = make_sampler();

    for (int trial = 0; trial < 50; ++trial) {
        auto [task, _] =
            ts.run(TaskKind::ShortestPath, ScratchpadKind::None, sg, rng, cfg);
        const auto& path = *task.target.path;
        const int L = static_cast<int>(path.size()) - 1;
        CHECK(L >= 2);
        CHECK(L <= 4);
        // Path length equals the BFS distance from start to end.
        const auto dist = sg.graph.bfs(path.front());
        CHECK(dist[path.back()] == L);
    }
}

TEST_CASE("sample_shortest_path: deterministic under a fixed seed") {
    std::vector<Edge> edges = {
        {0, 1}, {1, 2}, {2, 3}, {3, 4}, {0, 4}, {1, 3},
    };
    SampledGraph sg = make_sg(CsrGraph::from_undirected_edges(5, edges));
    GeneratorConfig cfg = make_cfg(1, 4);
    TaskSampler a, b;

    std::mt19937_64 rng_a(9999), rng_b(9999);
    auto [task_a, _a] =
        a.run(TaskKind::ShortestPath, ScratchpadKind::None, sg, rng_a, cfg);
    auto [task_b, _b] =
        b.run(TaskKind::ShortestPath, ScratchpadKind::None, sg, rng_b, cfg);
    CHECK(task_a.query.start_node == task_b.query.start_node);
    CHECK(task_a.query.end_node   == task_b.query.end_node);
    CHECK(*task_a.target.path == *task_b.target.path);
}

TEST_CASE("sample_shortest_path: throws when no pair meets the length window") {
    // Complete graph K_5: every pair is distance 1. Ask for length 3.
    std::vector<Edge> edges;
    for (int i = 0; i < 5; ++i)
        for (int j = i + 1; j < 5; ++j)
            edges.push_back({i, j});
    SampledGraph sg = make_sg(CsrGraph::from_undirected_edges(5, edges));
    GeneratorConfig cfg = make_cfg(3, 3, /*max_attempts=*/50);
    std::mt19937_64 rng(1);
    TaskSampler ts = make_sampler();

    CHECK_THROWS_AS(
        ts.run(TaskKind::ShortestPath, ScratchpadKind::None, sg, rng, cfg),
        std::runtime_error);
}

TEST_CASE("TaskSampler: unimplemented task_kind throws") {
    // ShortestPath, Bfs, Center, and Centroid are wired up now;
    // Khops/KhopsGen still stub-throw. Use Khops as the representative
    // "not implemented" case.
    std::vector<Edge> edges = {{0, 1}, {1, 2}};
    SampledGraph sg = make_sg(CsrGraph::from_undirected_edges(3, edges));
    GeneratorConfig cfg;
    std::mt19937_64 rng(0);
    TaskSampler ts = make_sampler();

    CHECK_THROWS_AS(
        ts.run(TaskKind::Khops, ScratchpadKind::None, sg, rng, cfg),
        std::runtime_error);
}

// ---- sample_bfs -----------------------------------------------------------

TEST_CASE("sample_bfs on a line graph: visit order matches BFS from start") {
    // Line 0 - 1 - 2 - 3 - 4. BFS from 0 visits [0, 1, 2, 3, 4].
    std::vector<Edge> edges = {{0, 1}, {1, 2}, {2, 3}, {3, 4}};
    SampledGraph sg = make_sg(CsrGraph::from_undirected_edges(5, edges));
    GeneratorConfig cfg;
    TaskSampler ts = make_sampler();

    // Try several seeds; whatever start comes out, the visit order must
    // agree with an independent BFS from the same start.
    for (int seed = 0; seed < 30; ++seed) {
        std::mt19937_64 rng(seed);
        auto [task, _] =
            ts.run(TaskKind::BFS, ScratchpadKind::None, sg, rng, cfg);
        REQUIRE(task.kind == TaskKind::BFS);
        REQUIRE(task.query.start_node.has_value());
        REQUIRE(task.target.path.has_value());
        const int start = *task.query.start_node;
        const auto& path = *task.target.path;

        // First vertex is start.
        CHECK(path.front() == start);
        // All vertices distinct.
        std::set<int> seen(path.begin(), path.end());
        CHECK(seen.size() == path.size());
        // Every vertex reachable from start is in the path.
        const auto dist = sg.graph.bfs(start);
        std::size_t reachable = 0;
        for (int d : dist) if (d >= 0) ++reachable;
        CHECK(path.size() == reachable);
        // Visit order is non-decreasing in distance from start.
        for (std::size_t i = 1; i < path.size(); ++i) {
            CHECK(dist[static_cast<std::size_t>(path[i - 1])]
                  <= dist[static_cast<std::size_t>(path[i])]);
        }
    }
}

TEST_CASE("sample_bfs on a disconnected graph visits only reachable vertices") {
    // Two components: 0-1-2 and 3-4.
    std::vector<Edge> edges = {{0, 1}, {1, 2}, {3, 4}};
    SampledGraph sg = make_sg(CsrGraph::from_undirected_edges(5, edges));
    GeneratorConfig cfg;
    TaskSampler ts = make_sampler();

    // Force start=0 via a controlled RNG isn't easy; sample many times
    // and only assert on the ones that landed in either component.
    for (int seed = 0; seed < 50; ++seed) {
        std::mt19937_64 rng(seed);
        auto [task, _] =
            ts.run(TaskKind::BFS, ScratchpadKind::None, sg, rng, cfg);
        const int start = *task.query.start_node;
        const auto& path = *task.target.path;
        if (start <= 2) {
            CHECK(path.size() == 3);  // {0,1,2}
        } else {
            CHECK(path.size() == 2);  // {3,4}
        }
    }
}

TEST_CASE("TaskSampler: unimplemented scratchpad_kind throws") {
    std::vector<Edge> edges = {{0, 1}, {1, 2}};
    SampledGraph sg = make_sg(CsrGraph::from_undirected_edges(3, edges));
    GeneratorConfig cfg = make_cfg(1, 2);
    std::mt19937_64 rng(0);
    TaskSampler ts = make_sampler();

    // Dfs scratchpad still unimplemented; use it as the sentinel for
    // this rejection test now that Bfs scratchpad is wired up.
    CHECK_THROWS_AS(
        ts.run(TaskKind::ShortestPath, ScratchpadKind::DFS, sg, rng, cfg),
        std::runtime_error);
}

// ---- sample_center / sample_centroid --------------------------------------

TEST_CASE("sample_center on a line graph with given_query: middle wins") {
    // Line 0 - 1 - 2 - 3 - 4. With Q = {0, 4}, distances to Q are:
    //   v=0: max(0, 4) = 4
    //   v=1: max(1, 3) = 3
    //   v=2: max(2, 2) = 2   <-- min
    //   v=3: max(3, 1) = 3
    //   v=4: max(4, 0) = 4
    // Center should be {2} (unique min of the max-distance objective).
    std::vector<Edge> edges = {{0, 1}, {1, 2}, {2, 3}, {3, 4}};
    SampledGraph sg = make_sg(CsrGraph::from_undirected_edges(5, edges));
    GeneratorConfig cfg;
    cfg.given_query = std::vector<int>{0, 4};
    std::mt19937_64 rng(0);
    TaskSampler ts = make_sampler();

    auto [task, _] =
        ts.run(TaskKind::Center, ScratchpadKind::None, sg, rng, cfg);
    CHECK(task.kind == TaskKind::Center);
    REQUIRE(task.query.vertex_set.has_value());
    CHECK(*task.query.vertex_set == std::vector<int>{0, 4});
    REQUIRE(task.target.vertex_set.has_value());
    CHECK(*task.target.vertex_set == std::vector<int>{2});
}

TEST_CASE("sample_centroid on a line graph with given_query: middle wins") {
    // Same line 0..4, same Q = {0, 4}. Sums are:
    //   v=0: 0+4 = 4;  v=1: 1+3 = 4;  v=2: 2+2 = 4;
    //   v=3: 3+1 = 4;  v=4: 4+0 = 4.
    // Every vertex is tied for centroid on a line whose endpoints are
    // the query: all 5 vertices should be reported as center_nodes.
    std::vector<Edge> edges = {{0, 1}, {1, 2}, {2, 3}, {3, 4}};
    SampledGraph sg = make_sg(CsrGraph::from_undirected_edges(5, edges));
    GeneratorConfig cfg;
    cfg.given_query = std::vector<int>{0, 4};
    std::mt19937_64 rng(0);
    TaskSampler ts = make_sampler();

    auto [task, _] =
        ts.run(TaskKind::Centroid, ScratchpadKind::None, sg, rng, cfg);
    CHECK(task.kind == TaskKind::Centroid);
    REQUIRE(task.target.vertex_set.has_value());
    CHECK(task.target.vertex_set->size() == 5);
}

TEST_CASE("sample_centroid distinguishes centroid from center") {
    // Star graph: 0 is the hub, 1..4 are leaves.
    //   Q = {1, 2, 3, 4}.
    // Distances from v to each leaf:
    //   v=0: {1,1,1,1}      -> center max=1, centroid sum=4
    //   v=1: {0,2,2,2}      -> center max=2, centroid sum=6
    //   v=2: {2,0,2,2}      -> center max=2, centroid sum=6
    //   v=3: {2,2,0,2}      -> center max=2, centroid sum=6
    //   v=4: {2,2,2,0}      -> center max=2, centroid sum=6
    // Center: {0} (unique min max=1). Centroid: {0} (unique min sum=4).
    std::vector<Edge> edges = {{0, 1}, {0, 2}, {0, 3}, {0, 4}};
    SampledGraph sg = make_sg(CsrGraph::from_undirected_edges(5, edges));
    GeneratorConfig cfg;
    cfg.given_query = std::vector<int>{1, 2, 3, 4};
    std::mt19937_64 rng(0);
    TaskSampler ts = make_sampler();

    auto [t_center, _c] =
        ts.run(TaskKind::Center, ScratchpadKind::None, sg, rng, cfg);
    auto [t_centroid, _d] =
        ts.run(TaskKind::Centroid, ScratchpadKind::None, sg, rng, cfg);
    REQUIRE(t_center.target.vertex_set.has_value());
    REQUIRE(t_centroid.target.vertex_set.has_value());
    CHECK(*t_center.target.vertex_set == std::vector<int>{0});
    CHECK(*t_centroid.target.vertex_set == std::vector<int>{0});
}

TEST_CASE("sample_center samples a random query when given_query is unset") {
    std::vector<Edge> edges = {{0, 1}, {1, 2}, {2, 3}, {3, 4}};
    SampledGraph sg = make_sg(CsrGraph::from_undirected_edges(5, edges));
    GeneratorConfig cfg;
    cfg.min_query_size = 2;
    cfg.max_query_size = 3;
    std::mt19937_64 rng(42);
    TaskSampler ts = make_sampler();

    auto [task, _] =
        ts.run(TaskKind::Center, ScratchpadKind::None, sg, rng, cfg);
    REQUIRE(task.query.vertex_set.has_value());
    const int qs = static_cast<int>(task.query.vertex_set->size());
    CHECK(qs >= 2);
    CHECK(qs <= 3);
    // All query entries must be distinct and in-range.
    std::set<int> seen(task.query.vertex_set->begin(),
                       task.query.vertex_set->end());
    CHECK(static_cast<int>(seen.size()) == qs);
    for (int q : *task.query.vertex_set) {
        CHECK(q >= 0);
        CHECK(q < 5);
    }
    REQUIRE(task.target.vertex_set.has_value());
    CHECK(!task.target.vertex_set->empty());
}

TEST_CASE("sample_center throws on disconnected query with no common reach") {
    // Two disjoint edges: {0-1} and {2-3}. No vertex is reachable from
    // both 0 and 2, so center/centroid have no valid center_node.
    std::vector<Edge> edges = {{0, 1}, {2, 3}};
    SampledGraph sg = make_sg(CsrGraph::from_undirected_edges(4, edges));
    GeneratorConfig cfg;
    cfg.given_query = std::vector<int>{0, 2};
    std::mt19937_64 rng(0);
    TaskSampler ts = make_sampler();

    CHECK_THROWS_AS(
        ts.run(TaskKind::Center, ScratchpadKind::None, sg, rng, cfg),
        std::runtime_error);
}
