#include "doctest.h"

#include <algorithm>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

#include "graphgen/csr_graph.h"

using graphgen::CsrGraph;
using graphgen::Edge;
using graphgen::NeighbourEdge;
using graphgen::WeightedEdge;

TEST_CASE("directed unweighted round-trip") {
    // 0 -> 1, 0 -> 2, 2 -> 1, 3 isolated
    std::vector<Edge> edges = {{0, 1}, {0, 2}, {2, 1}};
    CsrGraph g = CsrGraph::from_directed_edges(4, edges);

    CHECK(g.num_vertices() == 4);
    CHECK(g.num_edges() == 3);
    CHECK(g.is_directed());
    CHECK_FALSE(g.has_weights());

    CHECK(g.degree(0) == 2);
    CHECK(g.degree(1) == 0);
    CHECK(g.degree(2) == 1);
    CHECK(g.degree(3) == 0);

    // Neighbour order matches insertion order.
    std::vector<int> nbrs0(g.neighbours(0).begin(), g.neighbours(0).end());
    CHECK(nbrs0 == std::vector<int>{1, 2});
    std::vector<int> nbrs2(g.neighbours(2).begin(), g.neighbours(2).end());
    CHECK(nbrs2 == std::vector<int>{1});
    CHECK(g.neighbours(1).size() == 0);
    CHECK(g.neighbours(3).size() == 0);
}

TEST_CASE("undirected mirroring: both endpoints see the edge, edges() dedups") {
    // Triangle 0-1-2 plus a pendant 1-3.
    std::vector<Edge> edges = {{0, 1}, {1, 2}, {2, 0}, {1, 3}};
    CsrGraph g = CsrGraph::from_undirected_edges(4, edges);

    CHECK_FALSE(g.is_directed());
    CHECK(g.num_edges() == 4);

    CHECK(g.degree(0) == 2);
    CHECK(g.degree(1) == 3);
    CHECK(g.degree(2) == 2);
    CHECK(g.degree(3) == 1);

    // Neighbour lists carry both orientations of each incident edge.
    for (auto [u, v] : g.edges()) {
        CHECK(g.has_edge(u, v));
        CHECK(g.has_edge(v, u));
    }
}

TEST_CASE("edges() on an undirected graph yields each edge once in canonical order") {
    std::vector<Edge> input = {{0, 1}, {1, 2}, {2, 0}, {1, 3}};
    CsrGraph g = CsrGraph::from_undirected_edges(4, input);

    std::set<std::pair<int, int>> seen;
    int count = 0;
    for (auto [u, v] : g.edges()) {
        CHECK(u < v);  // canonical: mirror slots are skipped
        seen.emplace(u, v);
        ++count;
    }
    CHECK(count == 4);
    CHECK(seen.size() == 4);
    CHECK(seen.count({0, 1}));
    CHECK(seen.count({1, 2}));
    CHECK(seen.count({0, 2}));
    CHECK(seen.count({1, 3}));
}

TEST_CASE("weighted directed graph") {
    std::vector<Edge>  edges   = {{0, 1}, {0, 2}, {1, 2}};
    std::vector<float> weights = {0.5f, 1.5f, 2.5f};
    CsrGraph g = CsrGraph::from_directed_edges(3, edges, weights);

    CHECK(g.has_weights());

    std::vector<NeighbourEdge> got;
    for (auto e : g.weighted_neighbours(0)) got.push_back(e);
    REQUIRE(got.size() == 2);
    CHECK(got[0].target == 1);
    CHECK(got[0].weight == doctest::Approx(0.5f));
    CHECK(got[1].target == 2);
    CHECK(got[1].weight == doctest::Approx(1.5f));

    // weighted_edges walks the whole graph.
    std::vector<WeightedEdge> all;
    for (auto e : g.weighted_edges()) all.push_back(e);
    CHECK(all.size() == 3);
}

TEST_CASE("weighted_edges on an undirected graph dedups the mirror") {
    std::vector<Edge>  input   = {{0, 1}, {1, 2}};
    std::vector<float> weights = {0.5f, 1.5f};
    CsrGraph g = CsrGraph::from_undirected_edges(3, input, weights);

    std::vector<WeightedEdge> got;
    for (auto e : g.weighted_edges()) got.push_back(e);
    REQUIRE(got.size() == 2);
    for (auto e : got) CHECK(e.u < e.v);
}

TEST_CASE("unweighted graph reports weight 1.0 from weighted_neighbours") {
    std::vector<Edge> edges = {{0, 1}, {1, 2}};
    CsrGraph g = CsrGraph::from_undirected_edges(3, edges);

    CHECK_FALSE(g.has_weights());
    for (auto [v, w] : g.weighted_neighbours(1)) {
        (void)v;
        CHECK(w == doctest::Approx(1.0f));
    }
}

TEST_CASE("has_edge walks the neighbour list") {
    std::vector<Edge> edges = {{0, 3}, {0, 1}, {0, 2}};
    CsrGraph g = CsrGraph::from_directed_edges(4, edges);

    CHECK(g.has_edge(0, 1));
    CHECK(g.has_edge(0, 2));
    CHECK(g.has_edge(0, 3));
    CHECK_FALSE(g.has_edge(0, 0));
    CHECK_FALSE(g.has_edge(1, 0));  // directed: reverse not stored
}

TEST_CASE("edges() covers every input edge exactly once (directed)") {
    std::vector<Edge> input = {{0, 1}, {1, 2}, {0, 3}, {3, 2}, {2, 0}};
    CsrGraph g = CsrGraph::from_directed_edges(4, input);

    std::set<std::pair<int, int>> seen;
    for (auto [u, v] : g.edges()) seen.emplace(u, v);
    CHECK(seen.size() == input.size());
    for (auto e : input) CHECK(seen.count({e.u, e.v}) == 1);
}

TEST_CASE("graph with only isolated vertices") {
    CsrGraph g = CsrGraph::from_directed_edges(5, {});
    CHECK(g.num_vertices() == 5);
    CHECK(g.num_edges() == 0);
    for (int u = 0; u < 5; ++u) {
        CHECK(g.degree(u) == 0);
        CHECK(g.neighbours(u).size() == 0);
    }
    // edges() on an empty graph yields an empty range without UB.
    int count = 0;
    for (auto e : g.edges()) { (void)e; ++count; }
    CHECK(count == 0);
}

TEST_CASE("construction rejects out-of-range endpoints") {
    std::vector<Edge> bad = {{0, 5}};
    CHECK_THROWS_AS(CsrGraph::from_directed_edges(3, bad), std::runtime_error);
}

TEST_CASE("construction rejects mismatched weight length") {
    std::vector<Edge>  edges   = {{0, 1}, {1, 2}};
    std::vector<float> weights = {1.0f};
    CHECK_THROWS_AS(
        CsrGraph::from_directed_edges(3, edges, weights),
        std::runtime_error);
}
