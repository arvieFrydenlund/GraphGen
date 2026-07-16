#include "doctest.h"

#include <random>
#include <set>
#include <vector>

#include "graphgen/csr_graph.h"
#include "graphgen/sampled_graph.h"

using graphgen::Arc;
using graphgen::CsrGraph;
using graphgen::SampledGraph;

namespace {

SampledGraph make_line_graph(int n) {
    std::vector<Arc> edges;
    for (int i = 0; i + 1 < n; ++i) edges.push_back({i, i + 1});
    SampledGraph sg;
    sg.graph = CsrGraph::from_undirected_edges(n, edges);
    return sg;
}

}  // namespace

TEST_CASE("assign_random_vocab_ids maps every vertex into [min_vocab, max_vocab)") {
    std::mt19937_64 rng(0);
    SampledGraph sg = make_line_graph(6);
    sg.assign_random_vocab_ids(/*min_vocab=*/10, /*max_vocab=*/50, rng);

    REQUIRE(sg.internal_to_vocab.size() == 6);
    std::set<int> unique(sg.internal_to_vocab.begin(), sg.internal_to_vocab.end());
    CHECK(unique.size() == sg.internal_to_vocab.size());
    for (int id : sg.internal_to_vocab) {
        CHECK(id >= 10);
        CHECK(id < 50);
    }
}

TEST_CASE("assign_random_vocab_ids falls back to [0, n) when max_vocab is negative") {
    std::mt19937_64 rng(0);
    SampledGraph sg = make_line_graph(5);
    sg.assign_random_vocab_ids(/*min_vocab=*/-1, /*max_vocab=*/-1, rng);

    REQUIRE(sg.internal_to_vocab.size() == 5);
    std::vector<int> sorted = sg.internal_to_vocab;
    std::sort(sorted.begin(), sorted.end());
    for (int i = 0; i < 5; ++i) CHECK(sorted[i] == i);
}

TEST_CASE("assign_random_vocab_ids is deterministic under a fixed seed") {
    SampledGraph a = make_line_graph(8);
    SampledGraph b = make_line_graph(8);
    std::mt19937_64 rng_a(42), rng_b(42);
    a.assign_random_vocab_ids(0, 100, rng_a);
    b.assign_random_vocab_ids(0, 100, rng_b);
    CHECK(a.internal_to_vocab == b.internal_to_vocab);
}
