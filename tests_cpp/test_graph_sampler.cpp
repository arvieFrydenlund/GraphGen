// Tests for GraphSampler::sample_euclidean.
//
// The sampler generates random geometric graphs on the unit dim-cube.
// Positions are internal to the sampler (not surfaced on SampledGraph),
// so the test focuses on structural invariants and configuration
// dispatch rather than per-edge geometric checks. Property-based
// coverage of geometry is handled at the pytest end-to-end level once
// return_positions lands.

#include "doctest.h"

#include <algorithm>
#include <random>
#include <set>
#include <stdexcept>
#include <utility>

#include "graphgen/csr_graph.h"
#include "graphgen/generator_config.h"
#include "graphgen/graph_kind.h"
#include "graphgen/graph_sampler.h"
#include "graphgen/sampled_graph.h"

using graphgen::CsrGraph;
using graphgen::GeneratorConfig;
using graphgen::GraphKind;
using graphgen::GraphSampler;
using graphgen::SampledGraph;

namespace {

// Minimal Euclidean-sampler config. Fixed n via min==max removes one
// axis of variability from the invariant tests below.
GeneratorConfig make_euclidean_cfg(int n, int dim = 2,
                                   std::optional<double> emax = std::nullopt) {
    GeneratorConfig cfg;
    cfg.graph_kind      = "euclidean";
    cfg.min_num_nodes   = n;
    cfg.max_num_nodes   = n;
    cfg.dim             = dim;
    cfg.max_edge_length = emax;   // nullopt -> sampler defaults to 1/sqrt(n)
    return cfg;
}

}  // namespace

TEST_CASE("sample_euclidean: reports the configured number of vertices") {
    GraphSampler sampler;
    std::mt19937_64 rng(42);
    GeneratorConfig cfg = make_euclidean_cfg(/*n=*/25);

    SampledGraph sg = sampler.run(GraphKind::Euclidean, rng, cfg);
    CHECK(sg.graph.num_vertices() == 25);
    CHECK(sg.distance_semantic == graphgen::DistanceSemantic::HopCount);
    // Vocab mapping is now Worker's responsibility (assigned after
    // sampler.run using ctx.num_special / ctx.max_vocab). Direct
    // sampler.run leaves it empty -- see Worker end-to-end tests
    // for the vocab-space coverage.
    CHECK(sg.internal_to_vocab.empty());
}

TEST_CASE("sample_euclidean: builds an undirected graph, no self-loops") {
    GraphSampler sampler;
    std::mt19937_64 rng(1);
    GeneratorConfig cfg = make_euclidean_cfg(/*n=*/20);

    SampledGraph sg = sampler.run(GraphKind::Euclidean, rng, cfg);
    CHECK_FALSE(sg.graph.is_directed());
    for (auto e : sg.graph.edges()) {
        CHECK(e.u != e.v);          // no self-loops
        CHECK(e.u <  e.v);          // canonical u<v for undirected
    }
}

TEST_CASE("sample_euclidean: no duplicate edges") {
    // CsrGraph::from_undirected_edges deduplicates internally, but
    // sample_euclidean should never emit duplicates in the first
    // place (each pair (u, v) with u < v is visited once).
    GraphSampler sampler;
    std::mt19937_64 rng(7);
    GeneratorConfig cfg = make_euclidean_cfg(/*n=*/30);

    SampledGraph sg = sampler.run(GraphKind::Euclidean, rng, cfg);
    std::set<std::pair<int, int>> seen;
    for (auto e : sg.graph.edges()) {
        auto key = std::pair{e.u, e.v};
        CHECK(seen.insert(key).second);
    }
}

TEST_CASE("sample_euclidean: deterministic under a fixed seed") {
    GeneratorConfig cfg = make_euclidean_cfg(/*n=*/15);

    std::mt19937_64 rng_a(99);
    std::mt19937_64 rng_b(99);
    GraphSampler sampler_a;
    GraphSampler sampler_b;
    SampledGraph a = sampler_a.run(GraphKind::Euclidean, rng_a, cfg);
    SampledGraph b = sampler_b.run(GraphKind::Euclidean, rng_b, cfg);

    CHECK(a.graph.num_vertices() == b.graph.num_vertices());
    CHECK(a.graph.num_edges()    == b.graph.num_edges());
    // Same rng seed -> identical (u, v) pairs in the same emit
    // order. Compare pairwise; no sorting needed.
    std::vector<std::pair<int, int>> la;
    std::vector<std::pair<int, int>> lb;
    for (auto e : a.graph.edges()) la.emplace_back(e.u, e.v);
    for (auto e : b.graph.edges()) lb.emplace_back(e.u, e.v);
    REQUIRE(la.size() == lb.size());
    for (std::size_t i = 0; i < la.size(); ++i) {
        CHECK(la[i] == lb[i]);
    }
}

TEST_CASE("sample_euclidean: max_edge_length = 0 yields an empty graph") {
    // Edges accepted iff dist in [0, 0]. Vertex positions are drawn
    // from a continuous distribution so pairs at exact distance 0
    // have measure zero -> we expect no edges.
    GraphSampler sampler;
    std::mt19937_64 rng(0);
    GeneratorConfig cfg = make_euclidean_cfg(/*n=*/12, /*dim=*/2, /*emax=*/0.0);

    SampledGraph sg = sampler.run(GraphKind::Euclidean, rng, cfg);
    CHECK(sg.graph.num_edges() == 0);
}

TEST_CASE("sample_euclidean: max_edge_length large enough yields K_n") {
    // Unit dim-cube has diameter sqrt(dim); an emax >= sqrt(dim)
    // accepts every pair, giving the complete graph K_n with
    // n*(n-1)/2 undirected edges.
    const int n   = 8;
    const int dim = 2;
    const double diameter = std::sqrt(static_cast<double>(dim));
    GraphSampler sampler;
    std::mt19937_64 rng(0);
    GeneratorConfig cfg = make_euclidean_cfg(n, dim, /*emax=*/diameter + 1e-6);

    SampledGraph sg = sampler.run(GraphKind::Euclidean, rng, cfg);
    CHECK(sg.graph.num_edges() == n * (n - 1) / 2);
}

TEST_CASE("sample_euclidean: rejects directed=true") {
    GraphSampler sampler;
    std::mt19937_64 rng(0);
    GeneratorConfig cfg = make_euclidean_cfg(/*n=*/10);
    cfg.directed = true;
    CHECK_THROWS_AS(
        sampler.run(GraphKind::Euclidean, rng, cfg),
        std::runtime_error);
}

TEST_CASE("sample_euclidean: rejects cfg.dims") {
    GraphSampler sampler;
    std::mt19937_64 rng(0);
    GeneratorConfig cfg = make_euclidean_cfg(/*n=*/10);
    cfg.dims = std::vector<int>{5, 5};
    CHECK_THROWS_AS(
        sampler.run(GraphKind::Euclidean, rng, cfg),
        std::runtime_error);
}

TEST_CASE("sample_euclidean: rejects dim < 1") {
    GraphSampler sampler;
    std::mt19937_64 rng(0);
    GeneratorConfig cfg = make_euclidean_cfg(/*n=*/10, /*dim=*/0);
    CHECK_THROWS_AS(
        sampler.run(GraphKind::Euclidean, rng, cfg),
        std::runtime_error);
}

TEST_CASE("sample_euclidean: rejects min_edge_length > max_edge_length") {
    GraphSampler sampler;
    std::mt19937_64 rng(0);
    GeneratorConfig cfg = make_euclidean_cfg(/*n=*/10, /*dim=*/2,
                                             /*emax=*/0.1);
    cfg.min_edge_length = 0.5;   // greater than emax
    CHECK_THROWS_AS(
        sampler.run(GraphKind::Euclidean, rng, cfg),
        std::runtime_error);
}

// ---------------------------------------------------------------------------
// sample_path_star
// ---------------------------------------------------------------------------
//
// path_star is a rooted directed tree: vertex 0 is the root, and
// num_arms directed chains ("arms") emanate from it, each with an
// independently sampled length. The invariants tested here:
//   * vertex count == 1 + sum(arm_lengths)
//   * edge count   == n - 1  (a tree)
//   * built directed, no self-loops
//   * vertex 0 has in-degree 0; every non-root has in-degree 1
//   * connected as a rooted DAG: BFS from 0 reaches every vertex
// Range constraints (num_arms / arm_length within configured bounds)
// hold by construction, so we assert the derived n is in the
// expected interval rather than instrumenting the sampler.

namespace {

GeneratorConfig make_path_star_cfg(int min_arms, int max_arms,
                                   int min_len,  int max_len) {
    GeneratorConfig cfg;
    cfg.graph_kind     = "path_star";
    cfg.directed       = true;
    cfg.min_arms       = min_arms;
    cfg.max_arms       = max_arms;
    cfg.min_arm_length = min_len;
    cfg.max_arm_length = max_len;
    return cfg;
}

// Compute per-vertex in-degree from a directed CsrGraph.
std::vector<int> in_degrees(const CsrGraph& g) {
    std::vector<int> in_deg(static_cast<std::size_t>(g.num_vertices()), 0);
    for (auto e : g.edges()) {
        in_deg[static_cast<std::size_t>(e.v)]++;
    }
    return in_deg;
}

}  // namespace

TEST_CASE("sample_path_star: is a directed tree of the expected size") {
    // Fixed arm count and arm length -> deterministic n = 1 + A*L.
    GraphSampler sampler;
    std::mt19937_64 rng(42);
    GeneratorConfig cfg = make_path_star_cfg(/*arms=*/3, 3, /*len=*/4, 4);

    SampledGraph sg = sampler.run(GraphKind::PathStar, rng, cfg);
    const int expected_n = 1 + 3 * 4;
    CHECK(sg.graph.num_vertices() == expected_n);
    CHECK(sg.graph.num_edges()    == expected_n - 1);   // tree: |E| = |V| - 1
    CHECK(sg.graph.is_directed());
    CHECK(sg.distance_semantic == graphgen::DistanceSemantic::HopCount);
}

TEST_CASE("sample_path_star: vertex count within configured bounds") {
    // Sample many graphs and check n stays in
    // [1 + min_arms*min_len, 1 + max_arms*max_len].
    GraphSampler sampler;
    std::mt19937_64 rng(7);
    GeneratorConfig cfg = make_path_star_cfg(/*arms=*/2, 5, /*len=*/3, 7);
    const int n_min = 1 + 2 * 3;
    const int n_max = 1 + 5 * 7;
    for (int trial = 0; trial < 30; ++trial) {
        SampledGraph sg = sampler.run(GraphKind::PathStar, rng, cfg);
        const int n = sg.graph.num_vertices();
        CHECK(n >= n_min);
        CHECK(n <= n_max);
        CHECK(sg.graph.num_edges() == n - 1);   // still a tree
    }
}

TEST_CASE("sample_path_star: no self-loops") {
    GraphSampler sampler;
    std::mt19937_64 rng(1);
    GeneratorConfig cfg = make_path_star_cfg(3, 4, 2, 6);
    SampledGraph sg = sampler.run(GraphKind::PathStar, rng, cfg);
    for (auto e : sg.graph.edges()) CHECK(e.u != e.v);
}

TEST_CASE("sample_path_star: root has in-degree 0, every other vertex has 1") {
    // Root (vertex 0) has num_arms out-edges and 0 in-edges. Every
    // arm vertex has exactly one predecessor -- either the root
    // (arm-head) or the preceding arm vertex.
    GraphSampler sampler;
    std::mt19937_64 rng(3);
    GeneratorConfig cfg = make_path_star_cfg(2, 4, 2, 5);
    SampledGraph sg = sampler.run(GraphKind::PathStar, rng, cfg);
    auto in_deg = in_degrees(sg.graph);
    CHECK(in_deg[0] == 0);
    for (int v = 1; v < sg.graph.num_vertices(); ++v) {
        CHECK(in_deg[static_cast<std::size_t>(v)] == 1);
    }
}

TEST_CASE("sample_path_star: every vertex reachable from the root") {
    // BFS from vertex 0 must cover every other vertex on a rooted
    // directed tree -- no orphans.
    GraphSampler sampler;
    std::mt19937_64 rng(11);
    GeneratorConfig cfg = make_path_star_cfg(2, 4, 3, 6);
    SampledGraph sg = sampler.run(GraphKind::PathStar, rng, cfg);
    auto d = sg.graph.bfs(0);
    for (int v = 0; v < sg.graph.num_vertices(); ++v) {
        CHECK(d[static_cast<std::size_t>(v)] >= 0);   // reachable
    }
}

TEST_CASE("sample_path_star: deterministic under a fixed seed") {
    GeneratorConfig cfg = make_path_star_cfg(2, 4, 3, 5);
    std::mt19937_64 rng_a(99);
    std::mt19937_64 rng_b(99);
    GraphSampler sampler_a;
    GraphSampler sampler_b;
    SampledGraph a = sampler_a.run(GraphKind::PathStar, rng_a, cfg);
    SampledGraph b = sampler_b.run(GraphKind::PathStar, rng_b, cfg);

    REQUIRE(a.graph.num_vertices() == b.graph.num_vertices());
    REQUIRE(a.graph.num_edges()    == b.graph.num_edges());
    std::vector<std::pair<int, int>> la;
    std::vector<std::pair<int, int>> lb;
    for (auto e : a.graph.edges()) la.emplace_back(e.u, e.v);
    for (auto e : b.graph.edges()) lb.emplace_back(e.u, e.v);
    for (std::size_t i = 0; i < la.size(); ++i) CHECK(la[i] == lb[i]);
}

TEST_CASE("sample_path_star: rejects directed=false") {
    GraphSampler sampler;
    std::mt19937_64 rng(0);
    GeneratorConfig cfg = make_path_star_cfg(2, 3, 3, 5);
    cfg.directed = false;
    CHECK_THROWS_AS(
        sampler.run(GraphKind::PathStar, rng, cfg),
        std::runtime_error);
}

TEST_CASE("sample_path_star: rejects missing min_arms") {
    GraphSampler sampler;
    std::mt19937_64 rng(0);
    GeneratorConfig cfg;
    cfg.graph_kind     = "path_star";
    cfg.directed       = true;
    cfg.max_arms       = 3;
    cfg.min_arm_length = 3;
    cfg.max_arm_length = 5;
    // min_arms deliberately left unset
    CHECK_THROWS_AS(
        sampler.run(GraphKind::PathStar, rng, cfg),
        std::runtime_error);
}

TEST_CASE("sample_path_star: rejects min_arms < 1") {
    GraphSampler sampler;
    std::mt19937_64 rng(0);
    GeneratorConfig cfg = make_path_star_cfg(0, 3, 3, 5);
    CHECK_THROWS_AS(
        sampler.run(GraphKind::PathStar, rng, cfg),
        std::runtime_error);
}

TEST_CASE("sample_path_star: rejects min_arms > max_arms") {
    GraphSampler sampler;
    std::mt19937_64 rng(0);
    GeneratorConfig cfg = make_path_star_cfg(5, 2, 3, 5);
    CHECK_THROWS_AS(
        sampler.run(GraphKind::PathStar, rng, cfg),
        std::runtime_error);
}

TEST_CASE("sample_path_star: rejects min_arm_length > max_arm_length") {
    GraphSampler sampler;
    std::mt19937_64 rng(0);
    GeneratorConfig cfg = make_path_star_cfg(2, 3, /*min_len=*/6, /*max_len=*/4);
    CHECK_THROWS_AS(
        sampler.run(GraphKind::PathStar, rng, cfg),
        std::runtime_error);
}
