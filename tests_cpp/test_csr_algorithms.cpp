#include "doctest.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "graphgen/csr_graph.h"

using graphgen::Edge;
using graphgen::CsrGraph;

namespace {

// Line graph 0 -- 1 -- 2 -- 3 -- 4
CsrGraph make_line(int n) {
    std::vector<Edge> edges;
    for (int i = 0; i + 1 < n; ++i) edges.push_back({i, i + 1});
    return CsrGraph::from_undirected_edges(n, edges);
}

// Star: 0 at the centre connected to 1..n-1.
CsrGraph make_star(int n) {
    std::vector<Edge> edges;
    for (int i = 1; i < n; ++i) edges.push_back({0, i});
    return CsrGraph::from_undirected_edges(n, edges);
}

}  // namespace

TEST_CASE("bfs on a line graph returns distances 0..n-1") {
    CsrGraph g = make_line(5);
    auto d = g.bfs(0);
    REQUIRE(d.size() == 5);
    for (int i = 0; i < 5; ++i) CHECK(d[i] == i);
}

TEST_CASE("bfs from a non-zero source") {
    CsrGraph g = make_line(5);
    auto d = g.bfs(2);
    CHECK(d[0] == 2);
    CHECK(d[1] == 1);
    CHECK(d[2] == 0);
    CHECK(d[3] == 1);
    CHECK(d[4] == 2);
}

TEST_CASE("bfs on a star: centre reaches all leaves in 1 hop, leaves reach each other in 2") {
    CsrGraph g = make_star(5);
    auto d0 = g.bfs(0);
    for (int i = 1; i < 5; ++i) CHECK(d0[i] == 1);

    auto d1 = g.bfs(1);
    CHECK(d1[0] == 1);
    for (int i = 2; i < 5; ++i) CHECK(d1[i] == 2);
}

TEST_CASE("bfs reports -1 for unreachable vertices") {
    // Two disconnected triangles: 0-1-2 and 3-4-5.
    std::vector<Edge> edges = {{0, 1}, {1, 2}, {2, 0}, {3, 4}, {4, 5}, {5, 3}};
    CsrGraph g = CsrGraph::from_undirected_edges(6, edges);
    auto d = g.bfs(0);
    CHECK(d[0] == 0);
    CHECK(d[1] == 1);
    CHECK(d[2] == 1);
    CHECK(d[3] == -1);
    CHECK(d[4] == -1);
    CHECK(d[5] == -1);
}

TEST_CASE("bfs on an isolated-vertices graph") {
    CsrGraph g = CsrGraph::from_undirected_edges(4, {});
    auto d = g.bfs(2);
    CHECK(d[0] == -1);
    CHECK(d[1] == -1);
    CHECK(d[2] == 0);
    CHECK(d[3] == -1);
}

TEST_CASE("bfs source out of range throws") {
    CsrGraph g = make_line(3);
    CHECK_THROWS_AS(g.bfs(-1), std::runtime_error);
    CHECK_THROWS_AS(g.bfs(3), std::runtime_error);
}

TEST_CASE("distance_bounded_bfs caps expansion at max_dist") {
    CsrGraph g = make_line(6);  // 0-1-2-3-4-5
    auto d = g.distance_bounded_bfs(0, /*max_dist=*/2);
    CHECK(d[0] == 0);
    CHECK(d[1] == 1);
    CHECK(d[2] == 2);
    CHECK(d[3] == -1);  // beyond bound
    CHECK(d[4] == -1);
    CHECK(d[5] == -1);
}

TEST_CASE("distance_bounded_bfs with max_dist=0 marks only the source") {
    CsrGraph g = make_line(4);
    auto d = g.distance_bounded_bfs(1, /*max_dist=*/0);
    CHECK(d[0] == -1);
    CHECK(d[1] == 0);
    CHECK(d[2] == -1);
    CHECK(d[3] == -1);
}

TEST_CASE("distance_bounded_bfs matches full bfs when max_dist >= diameter") {
    CsrGraph g = make_line(5);
    CHECK(g.bfs(0) == g.distance_bounded_bfs(0, /*max_dist=*/100));
}

TEST_CASE("distance_bounded_bfs rejects negative max_dist") {
    CsrGraph g = make_line(3);
    CHECK_THROWS_AS(g.distance_bounded_bfs(0, -1), std::runtime_error);
}

// ---- connected_components -------------------------------------------------

TEST_CASE("connected_components: single connected graph has one component") {
    CsrGraph g = make_line(5);
    auto cc = g.connected_components();
    REQUIRE(cc.size() == 5);
    for (int c : cc) CHECK(c == 0);
    CHECK(*std::max_element(cc.begin(), cc.end()) + 1 == 1);
}

TEST_CASE("connected_components: three components with different sizes") {
    // {0,1,2} triangle, {3,4} edge, {5} isolated.
    std::vector<Edge> edges = {{0, 1}, {1, 2}, {2, 0}, {3, 4}};
    CsrGraph g = CsrGraph::from_undirected_edges(6, edges);
    auto cc = g.connected_components();

    // Same-component pairs share their id; different-component pairs don't.
    CHECK(cc[0] == cc[1]);
    CHECK(cc[1] == cc[2]);
    CHECK(cc[3] == cc[4]);
    CHECK(cc[0] != cc[3]);
    CHECK(cc[0] != cc[5]);
    CHECK(cc[3] != cc[5]);

    const int k = *std::max_element(cc.begin(), cc.end()) + 1;
    CHECK(k == 3);
}

TEST_CASE("connected_components: all-isolated graph gives one component per vertex") {
    CsrGraph g = CsrGraph::from_undirected_edges(4, {});
    auto cc = g.connected_components();
    REQUIRE(cc.size() == 4);
    const int k = *std::max_element(cc.begin(), cc.end()) + 1;
    CHECK(k == 4);
    // Every id must be distinct.
    std::vector<int> sorted(cc);
    std::sort(sorted.begin(), sorted.end());
    for (int i = 0; i < 4; ++i) CHECK(sorted[i] == i);
}

TEST_CASE("connected_components: empty graph returns an empty vector") {
    CsrGraph g = CsrGraph::from_undirected_edges(0, {});
    auto cc = g.connected_components();
    CHECK(cc.empty());
}
