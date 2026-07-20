#include "doctest.h"

#include <cstddef>
#include <random>
#include <stdexcept>
#include <vector>

#include "graphgen/csr_graph.h"
#include "graphgen/generator_config.h"
#include "graphgen/layout.h"
#include "graphgen/sampled_graph.h"
#include "graphgen/scratchpad.h"
#include "graphgen/task.h"

using graphgen::BatchLayout;
using graphgen::CsrGraph;
using graphgen::Edge;
using graphgen::GeneratorConfig;
using graphgen::Layout;
using graphgen::SampledGraph;
using graphgen::Scratchpad;
using graphgen::ScratchpadKind;
using graphgen::Task;
using graphgen::TaskKind;

namespace {

SampledGraph make_line_sg(int n) {
    std::vector<Edge> edges;
    for (int i = 0; i + 1 < n; ++i) edges.push_back({i, i + 1});
    SampledGraph sg;
    sg.graph = CsrGraph::from_undirected_edges(n, edges);
    sg.internal_to_vocab.resize(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) sg.internal_to_vocab[i] = i;
    return sg;
}

// A minimal, hand-constructed shortest_path Task with start/end and a
// path already populated -- used to exercise Layout::from directly
// without running the sampler.
Task make_shortest_path_task(int start, int end, std::vector<int> path) {
    Task t;
    t.kind             = TaskKind::ShortestPath;
    t.query.start_node = start;
    t.query.end_node   = end;
    t.target.path      = std::move(path);
    // valid_next_hops not needed for Layout::from's formula.
    return t;
}

GeneratorConfig make_sean_cfg() {
    GeneratorConfig cfg;
    cfg.tokenization_mode = "sean";
    return cfg;
}

}  // namespace

TEST_CASE("Layout::from (Sean, ShortestPath, None) on a 3-edge, k=2 path") {
    // 4-vertex line 0-1-2-3 has m=3 undirected edges. Pick start=0, end=2
    // giving path length k=2, so path = [0, 1, 2] (3 vertices).
    SampledGraph sg = make_line_sg(4);
    Task t = make_shortest_path_task(/*start=*/0, /*end=*/2, {0, 1, 2});
    Scratchpad sp;  // ScratchpadKind::None
    GeneratorConfig cfg = make_sean_cfg();

    const Layout L = Layout::from(sg, t, sp, cfg);

    // Expected geometry: m=3, k=2.
    //   pos 0             BOS
    //   pos 1..9          edges (3 * m = 9)
    //   pos 10            QUERY_START
    //   pos 11..12        start, end (query_len = 2)
    //   pos 13            QUERY_END
    //   pos 14            TASK_START
    //   pos 15..17        path vertices (k+1 = 3)
    //   pos 18            TASK_END
    //   pos 19            EOS
    //   seq_len = 20 = 3m + k + 9

    CHECK(L.struct_dim        == 1);
    CHECK(L.graph_edge_start  == 1);
    CHECK(L.graph_edge_len    == 9);
    CHECK(L.query_start       == 11);
    CHECK(L.query_len         == 2);
    CHECK(L.task_target_start == 15);
    CHECK(L.task_target_len   == 3);
    CHECK(L.seq_len           == 20);
    CHECK(L.num_nodes         == 4);
    CHECK(L.num_edges         == 3);
}

TEST_CASE("Layout::from: seq_len formula holds for varying m and k") {
    Scratchpad sp;
    GeneratorConfig cfg = make_sean_cfg();

    for (int n = 2; n <= 8; ++n) {
        SampledGraph sg = make_line_sg(n);       // m = n - 1
        for (int k = 1; k <= n - 1; ++k) {       // path length up to diameter
            std::vector<int> path;
            for (int i = 0; i <= k; ++i) path.push_back(i);
            Task t = make_shortest_path_task(0, k, path);

            const Layout L = Layout::from(sg, t, sp, cfg);
            CHECK(L.seq_len == 3 * (n - 1) + k + 9);
        }
    }
}

TEST_CASE("Layout::from under STAN packs edges into 1 position of 3 columns") {
    // Line 0-1-2 (m=2), SP 0->2 (path=[0,1,2], k=2). Under STAN each
    // edge takes 1 sequence position (not 3), so seq_len is smaller
    // than the SEAN equivalent by 2*m = 4 positions.
    //   Section walk:
    //     BOS(1), edges(2), QS(1), query(2), QE(1),
    //     TS(1), target(3), TE(1), EOS(1) = 13
    //   struct_dim = 3.
    SampledGraph sg = make_line_sg(3);
    Task t = make_shortest_path_task(0, 2, {0, 1, 2});
    Scratchpad sp;
    GeneratorConfig cfg;
    cfg.tokenization_mode = "stan";

    const Layout L = Layout::from(sg, t, sp, cfg);
    CHECK(L.struct_dim         == 3);
    CHECK(L.graph_edge_start   == 1);
    CHECK(L.graph_edge_len     == 2);           // 1 position per edge
    CHECK(L.query_start        == 4);
    CHECK(L.query_len          == 2);
    CHECK(L.task_target_start  == 8);
    CHECK(L.task_target_len    == 3);
    CHECK(L.seq_len            == 13);
    CHECK(L.num_edges          == 2);
}

TEST_CASE("Layout::from rejects still-unimplemented task_kinds") {
    // Bfs and ShortestPath both wire up now; Center/Khops/KhopsGen
    // remain stubs. Verify one of the remaining unimplemented kinds
    // still throws so the regression fence stays live.
    SampledGraph sg = make_line_sg(3);
    Task t;
    t.kind = TaskKind::Center;
    Scratchpad sp;
    GeneratorConfig cfg = make_sean_cfg();

    CHECK_THROWS_AS(Layout::from(sg, t, sp, cfg), std::runtime_error);
}

TEST_CASE("Layout::from (Sean, Bfs, None) on a line graph: seq_len = 3m + n_visited + 8") {
    // BFS from vertex 0 on the 5-vertex line 0-1-2-3-4 visits all 5
    // vertices, so visit_len = 5, m = 4.
    SampledGraph sg = make_line_sg(5);
    Task t;
    t.kind             = TaskKind::BFS;
    t.query.start_node = 0;
    t.target.path      = std::vector<int>{0, 1, 2, 3, 4};
    Scratchpad sp;  // ScratchpadKind::None
    GeneratorConfig cfg = make_sean_cfg();

    const Layout L = Layout::from(sg, t, sp, cfg);

    // Expected geometry (BFS section list):
    //   BOS(1), edges(3*4=12), QS(1), start(1), QE(1), TS(1),
    //   visit_order(5), TE(1), EOS(1) = 24
    CHECK(L.struct_dim        == 1);
    CHECK(L.graph_edge_start  == 1);
    CHECK(L.graph_edge_len    == 12);
    CHECK(L.query_start       == 14);  // right after QS at pos 13
    CHECK(L.query_len         == 1);   // start only, no end
    CHECK(L.task_target_start == 17);  // right after QE(15) + TS(16)
    CHECK(L.task_target_len   == 5);
    CHECK(L.seq_len           == 24);
    CHECK(L.num_nodes         == 5);
    CHECK(L.num_edges         == 4);
}

TEST_CASE("Layout::from (Sean, ShortestPath, BFS/Plain) on a line graph") {
    // Line 0-1-2-3-4, m=4. SP from 0 to 4: k=4, path=[0..4].
    // BFS scratchpad from start (=0) visits all 5, n_visited=5.
    // deg = [1,2,2,2,1], Σdeg = 8.
    // scratchpad content = 3*5 + 8 = 23.
    SampledGraph sg = make_line_sg(5);
    Task t;
    t.kind             = TaskKind::ShortestPath;
    t.query.start_node = 0;
    t.query.end_node   = 4;
    t.target.path      = std::vector<int>{0, 1, 2, 3, 4};
    Scratchpad sp;
    sp.kind  = ScratchpadKind::BFS;
    sp.trace = std::vector<int>{0, 1, 2, 3, 4};
    GeneratorConfig cfg = make_sean_cfg();

    const Layout L = Layout::from(sg, t, sp, cfg);

    // Section walk (positions 0-indexed):
    //   0        BOS
    //   1..12    edges (12)
    //   13       QS
    //   14..15   query (2)
    //   16       QE
    //   17       SS
    //   18..40   scratchpad (23)
    //   41       SE
    //   42       TS
    //   43..47   target (5)
    //   48       TE
    //   49       EOS
    //   seq_len = 50
    CHECK(L.struct_dim         == 1);
    CHECK(L.graph_edge_start   == 1);
    CHECK(L.graph_edge_len     == 12);
    CHECK(L.query_start        == 14);
    CHECK(L.query_len          == 2);
    CHECK(L.scratchpad_start   == 18);
    CHECK(L.scratchpad_len     == 23);
    CHECK(L.task_target_start  == 43);
    CHECK(L.task_target_len    == 5);
    CHECK(L.seq_len            == 50);
    CHECK(L.num_nodes          == 5);
    CHECK(L.num_edges          == 4);
}

TEST_CASE("Layout::from rejects (bfs, bfs) as redundant via config validation") {
    // Redundant combination is caught at GeneratorConfig construction
    // time (see test_generator_config.py), not at Layout::from -- but
    // keep a smoke test that at least one still-unimplemented
    // (task_kind, scratchpad_kind) pair still throws. Use dfs which
    // remains a stub.
    SampledGraph sg = make_line_sg(3);
    Task t = make_shortest_path_task(0, 1, {0, 1});
    Scratchpad sp;
    sp.kind = ScratchpadKind::DFS;
    GeneratorConfig cfg = make_sean_cfg();

    CHECK_THROWS_AS(Layout::from(sg, t, sp, cfg), std::runtime_error);
}

TEST_CASE("BatchLayout::from computes max seq_len and preserves items") {
    Scratchpad sp;
    GeneratorConfig cfg = make_sean_cfg();

    std::vector<Layout> items;
    for (int n : {3, 5, 4}) {
        SampledGraph sg = make_line_sg(n);
        std::vector<int> path;
        const int k = n - 2;
        for (int i = 0; i <= k; ++i) path.push_back(i);
        items.push_back(Layout::from(sg, make_shortest_path_task(0, k, path), sp, cfg));
    }
    // seq_len values: for n=3,k=1 -> 3*2+1+9=16
    //                 for n=5,k=3 -> 3*4+3+9=24
    //                 for n=4,k=2 -> 3*3+2+9=20
    BatchLayout bl = BatchLayout::from(std::move(items));
    CHECK(bl.batch_size  == 3);
    CHECK(bl.max_seq_len == 24);
    CHECK(bl.struct_dim  == 1);
    REQUIRE(bl.items.size() == 3);
    CHECK(bl.items[0].seq_len == 16);
    CHECK(bl.items[1].seq_len == 24);
    CHECK(bl.items[2].seq_len == 20);
}

TEST_CASE("BatchLayout::from handles an empty item list") {
    BatchLayout bl = BatchLayout::from({});
    CHECK(bl.batch_size  == 0);
    CHECK(bl.max_seq_len == 0);
    CHECK(bl.items.empty());
}
