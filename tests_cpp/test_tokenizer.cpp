#include "doctest.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "graphgen/csr_graph.h"
#include "graphgen/generator_config.h"
#include "graphgen/layout.h"
#include "graphgen/sampled_graph.h"
#include "graphgen/scratchpad.h"
#include "graphgen/task.h"
#include "graphgen/tokenizer.h"
#include "graphgen/worker_shared_context.h"

using graphgen::CsrGraph;
using graphgen::Edge;
using graphgen::GeneratorConfig;
using graphgen::Layout;
using graphgen::SampledGraph;
using graphgen::Scratchpad;
using graphgen::ScratchpadKind;
using graphgen::Task;
using graphgen::TaskKind;
using graphgen::Tokenizer;
using graphgen::WorkerSharedContext;

namespace {

// Small fixture: hand-built line graph 0-1-2-3, path 0->1->2, vocab
// mapping is the identity (internal_to_vocab[u] = u).
struct Fixture {
    SampledGraph sg;
    Task         task;
    Scratchpad   scratchpad;   // ScratchpadKind::None
    Layout       layout;
    WorkerSharedContext ctx;   // empty dict -- tokenizer falls back to NUM_SPECIAL_DEFAULT
    GeneratorConfig cfg;
};

Fixture make_fixture() {
    Fixture f;

    // Graph: 0 - 1 - 2 - 3 (4 vertices, 3 undirected edges).
    std::vector<Edge> edges = {{0, 1}, {1, 2}, {2, 3}};
    f.sg.graph = CsrGraph::from_undirected_edges(4, edges);
    // Vocab is now token-id space: vertex u -> token id NS + u
    // (matches what the Worker would produce via sample_distinct_ints
    // over [ctx.num_special, ctx.max_vocab)). Tokenizer no longer
    // applies any shift.
    const int NS = WorkerSharedContext::NUM_SPECIAL_DEFAULT;
    f.sg.internal_to_vocab = {NS + 0, NS + 1, NS + 2, NS + 3};

    // Task: shortest path 0 -> 2 through 1.
    f.task.kind             = TaskKind::ShortestPath;
    f.task.query.start_node = 0;
    f.task.query.end_node   = 2;
    f.task.target.path      = std::vector<int>{0, 1, 2};

    // Scratchpad::None by default construction.

    f.cfg.tokenization_mode = "sean";

    f.layout = Layout::from(f.sg, f.task, f.scratchpad, f.cfg);
    return f;
}

}  // namespace

TEST_CASE("Tokenizer writes the expected SEAN sequence for a small path task") {
    Fixture f = make_fixture();
    const int seq_len = f.layout.seq_len;  // 3m+k+9 = 9+2+9 = 20
    REQUIRE(seq_len == 20);

    // Pre-fill row buffers with pad, exactly as BatchOutputArrays would.
    // Targets is (gen_len, max_labels) row-major; SP through a strict
    // line graph has exactly one valid next hop per step, so
    // max_labels=1 suffices for this fixture.
    const int max_labels = 1;
    std::vector<int32_t> src (seq_len, WorkerSharedContext::TOK_PAD);
    std::vector<int32_t> targets(f.layout.gen_len * max_labels,
                                 WorkerSharedContext::TOK_PAD);
    std::vector<int32_t> pos (seq_len, 0);

    Tokenizer tk;
    tk.tokenize_into_row(src.data(), targets.data(), max_labels, pos.data(),
                         f.sg, f.task, f.scratchpad, f.layout, f.ctx, f.cfg);

    const int NS = WorkerSharedContext::NUM_SPECIAL_DEFAULT;

    // Expected sequence written out explicitly (see Layout doc block):
    //   BOS, [u,v,EDGE] x 3, QUERY_START, start, end, QUERY_END,
    //   TASK_START, p0, p1, p2, TASK_END, EOS
    const std::vector<int32_t> expected_src = {
        WorkerSharedContext::TOK_BOS,
        NS + 0, NS + 1, WorkerSharedContext::TOK_EDGE,  // edge 0-1
        NS + 1, NS + 2, WorkerSharedContext::TOK_EDGE,  // edge 1-2
        NS + 2, NS + 3, WorkerSharedContext::TOK_EDGE,  // edge 2-3
        WorkerSharedContext::TOK_QUERY_START,
        NS + 0, NS + 2,                                  // start=0, end=2
        WorkerSharedContext::TOK_QUERY_END,
        WorkerSharedContext::TOK_TASK_START,
        NS + 0, NS + 1, NS + 2,                          // path 0,1,2
        WorkerSharedContext::TOK_TASK_END,
        WorkerSharedContext::TOK_EOS,
    };
    REQUIRE(static_cast<int>(expected_src.size()) == seq_len);
    for (int i = 0; i < seq_len; ++i) {
        CHECK(src[i] == expected_src[i]);
    }

    // Positional ids: 0..seq_len-1.
    for (int i = 0; i < seq_len; ++i) {
        CHECK(pos[i] == i);
    }

    // targets baseline: chosen label at k=0 matches src at each
    // generation-region column (prefix_len + gp). max_labels==1 here
    // so the slot is targets[gp] directly.
    for (int gp = 0; gp < f.layout.gen_len; ++gp) {
        CHECK(targets[gp] == src[f.layout.prefix_len + gp]);
    }
}

TEST_CASE("Tokenizer padding tail stays untouched past seq_len") {
    Fixture f = make_fixture();
    // Allocate rows wider than seq_len (like BatchOutputArrays would).
    const int pad = 10;
    const int stride = f.layout.seq_len + pad;
    const int max_labels = 1;
    std::vector<int32_t> src (stride, WorkerSharedContext::TOK_PAD);
    std::vector<int32_t> targets(f.layout.gen_len * max_labels + pad,
                                 WorkerSharedContext::TOK_PAD);
    std::vector<int32_t> pos (stride, 0);

    Tokenizer tk;
    tk.tokenize_into_row(src.data(), targets.data(), max_labels, pos.data(),
                         f.sg, f.task, f.scratchpad, f.layout, f.ctx, f.cfg);

    for (int i = f.layout.seq_len; i < stride; ++i) {
        CHECK(src[i]  == WorkerSharedContext::TOK_PAD);
        CHECK(pos[i]  == 0);
    }
    // targets tail past gen_len stays TOK_PAD.
    for (int i = f.layout.gen_len * max_labels;
             i < static_cast<int>(targets.size()); ++i) {
        CHECK(targets[i] == WorkerSharedContext::TOK_PAD);
    }
}

TEST_CASE("Tokenizer respects a null positions_row (return_pos_ids=false)") {
    Fixture f = make_fixture();
    std::vector<int32_t> src (f.layout.seq_len, WorkerSharedContext::TOK_PAD);
    const int max_labels = 1;
    std::vector<int32_t> targets(f.layout.seq_len, WorkerSharedContext::TOK_PAD);

    Tokenizer tk;
    // No crash when pos == nullptr.
    tk.tokenize_into_row(src.data(), targets.data(), max_labels, /*positions_row=*/nullptr,
                         f.sg, f.task, f.scratchpad, f.layout, f.ctx, f.cfg);

    // src still fully populated.
    CHECK(src.front() == WorkerSharedContext::TOK_BOS);
    CHECK(src.back()  == WorkerSharedContext::TOK_EOS);
}

TEST_CASE("Tokenizer rejects unimplemented mode / task / scratchpad combinations") {
    Fixture f = make_fixture();
    std::vector<int32_t> src (f.layout.seq_len, WorkerSharedContext::TOK_PAD);
    const int max_labels = 1;
    std::vector<int32_t> targets(f.layout.seq_len, WorkerSharedContext::TOK_PAD);
    std::vector<int32_t> pos (f.layout.seq_len, 0);

    Tokenizer tk;

    SUBCASE("stan mode") {
        GeneratorConfig cfg = f.cfg;
        cfg.tokenization_mode = "stan";
        CHECK_THROWS_AS(
            tk.tokenize_into_row(src.data(), targets.data(), max_labels, pos.data(),
                                 f.sg, f.task, f.scratchpad, f.layout, f.ctx, cfg),
            std::runtime_error);
    }

    SUBCASE("center task (still unimplemented)") {
        Task other = f.task;
        other.kind = TaskKind::Center;
        CHECK_THROWS_AS(
            tk.tokenize_into_row(src.data(), targets.data(), max_labels, pos.data(),
                                 f.sg, other, f.scratchpad, f.layout, f.ctx, f.cfg),
            std::runtime_error);
    }

    SUBCASE("dfs scratchpad (still unimplemented)") {
        // Bfs scratchpad on shortest_path is now supported; Dfs
        // remains a stub and stays the rejection sentinel.
        Scratchpad other;
        other.kind = ScratchpadKind::DFS;
        other.trace = std::vector<int>{0, 1};  // any non-empty trace
        CHECK_THROWS_AS(
            tk.tokenize_into_row(src.data(), targets.data(), max_labels, pos.data(),
                                 f.sg, f.task, other, f.layout, f.ctx, f.cfg),
            std::runtime_error);
    }
}

TEST_CASE("Tokenizer uses custom num_special from a loaded dictionary") {
    Fixture f = make_fixture();
    // Load the default dictionary with a specific max_num_nodes so
    // ctx.num_special becomes the real (dict-driven) value; verify the
    // vertex-token offset picks it up.
    f.ctx.set_default_dictionary(/*max_num_nodes=*/4, /*extra_after=*/0);
    REQUIRE(f.ctx.num_special == WorkerSharedContext::NUM_SPECIAL_DEFAULT);

    std::vector<int32_t> src (f.layout.seq_len, WorkerSharedContext::TOK_PAD);
    const int max_labels = 1;
    std::vector<int32_t> targets(f.layout.seq_len, WorkerSharedContext::TOK_PAD);
    std::vector<int32_t> pos (f.layout.seq_len, 0);

    Tokenizer tk;
    tk.tokenize_into_row(src.data(), targets.data(), max_labels, pos.data(),
                         f.sg, f.task, f.scratchpad, f.layout, f.ctx, f.cfg);

    // Vertex-vocab tokens must land in [num_special, num_special + n).
    const int NS = f.ctx.num_special;
    const int n  = f.sg.graph.num_vertices();
    CHECK(src[1] == NS + 0);  // first edge u
    CHECK(src[2] == NS + 1);  // first edge v
    for (int i = 0; i < f.layout.seq_len; ++i) {
        if (src[i] >= NS) {
            CHECK(src[i] < NS + n);
        }
    }
}

TEST_CASE("Tokenizer writes the expected SEAN sequence for a small BFS task") {
    // Reuse the shortest-path fixture's graph (line 0-1-2-3) but build
    // a BFS task instead: start=0, visit order = [0, 1, 2, 3].
    Fixture f = make_fixture();
    f.task.kind             = TaskKind::BFS;
    f.task.query.start_node = 0;
    f.task.query.end_node   = std::nullopt;
    f.task.target.path      = std::vector<int>{0, 1, 2, 3};
    f.task.target.valid_next_hops = std::nullopt;
    f.layout = Layout::from(f.sg, f.task, f.scratchpad, f.cfg);

    // Expected geometry: m=3, visit_len=4  →  seq_len = 3*3 + 4 + 7 = 20
    // (6 markers + edges + query_len(1) + target_len = 6 + 9 + 1 + 4)
    REQUIRE(f.layout.seq_len == 20);

    std::vector<int32_t> src (f.layout.seq_len, WorkerSharedContext::TOK_PAD);
    const int max_labels = 1;
    std::vector<int32_t> targets(f.layout.seq_len, WorkerSharedContext::TOK_PAD);
    std::vector<int32_t> pos (f.layout.seq_len, 0);

    Tokenizer tk;
    tk.tokenize_into_row(src.data(), targets.data(), max_labels, pos.data(),
                         f.sg, f.task, f.scratchpad, f.layout, f.ctx, f.cfg);

    const int NS = WorkerSharedContext::NUM_SPECIAL_DEFAULT;
    const std::vector<int32_t> expected_src = {
        WorkerSharedContext::TOK_BOS,
        NS + 0, NS + 1, WorkerSharedContext::TOK_EDGE,  // edge 0-1
        NS + 1, NS + 2, WorkerSharedContext::TOK_EDGE,  // edge 1-2
        NS + 2, NS + 3, WorkerSharedContext::TOK_EDGE,  // edge 2-3
        WorkerSharedContext::TOK_QUERY_START,
        NS + 0,                                          // start only (no end)
        WorkerSharedContext::TOK_QUERY_END,
        WorkerSharedContext::TOK_TASK_START,
        NS + 0, NS + 1, NS + 2, NS + 3,                  // BFS visit order
        WorkerSharedContext::TOK_TASK_END,
        WorkerSharedContext::TOK_EOS,
    };
    REQUIRE(static_cast<int>(expected_src.size()) == f.layout.seq_len);
    for (int i = 0; i < f.layout.seq_len; ++i) {
        CHECK(src[i] == expected_src[i]);
    }
}

TEST_CASE("Tokenizer writes SEAN + ShortestPath + BFS/Plain scratchpad") {
    // Line 0-1-2-3 fixture: m=3, deg=[1,2,2,1], SP 0->3 has path
    // [0,1,2,3] (k=3). BFS from start=0 visits all 4 vertices, so
    // n_visited=4, sum_deg=6, scratchpad content = 3*4 + 6 = 18.
    // Section walk positions:
    //   0        BOS
    //   1..9     edges (9)
    //   10       QS
    //   11..12   query (2: start,end)
    //   13       QE
    //   14       SS
    //   15..32   scratchpad (18)
    //   33       SE
    //   34       TS
    //   35..38   target (4: path)
    //   39       TE
    //   40       EOS
    //   seq_len = 41
    Fixture f = make_fixture();
    // Fixture task is (SP: 0->2). Widen to 0->3 so BFS covers the graph.
    f.task.query.end_node = 3;
    f.task.target.path    = std::vector<int>{0, 1, 2, 3};

    f.scratchpad.kind  = ScratchpadKind::BFS;
    f.scratchpad.trace = std::vector<int>{0, 1, 2, 3};

    f.layout = Layout::from(f.sg, f.task, f.scratchpad, f.cfg);
    REQUIRE(f.layout.seq_len         == 41);
    REQUIRE(f.layout.scratchpad_len  == 18);
    REQUIRE(f.layout.task_target_len == 4);

    std::vector<int32_t> src (f.layout.seq_len, WorkerSharedContext::TOK_PAD);
    const int max_labels = 1;
    std::vector<int32_t> targets(f.layout.seq_len, WorkerSharedContext::TOK_PAD);
    std::vector<int32_t> pos (f.layout.seq_len, 0);

    Tokenizer tk;
    tk.tokenize_into_row(src.data(), targets.data(), max_labels, pos.data(),
                         f.sg, f.task, f.scratchpad, f.layout, f.ctx, f.cfg);

    const int NS = WorkerSharedContext::NUM_SPECIAL_DEFAULT;
    const std::vector<int32_t> expected_src = {
        WorkerSharedContext::TOK_BOS,
        NS + 0, NS + 1, WorkerSharedContext::TOK_EDGE,
        NS + 1, NS + 2, WorkerSharedContext::TOK_EDGE,
        NS + 2, NS + 3, WorkerSharedContext::TOK_EDGE,
        WorkerSharedContext::TOK_QUERY_START,
        NS + 0, NS + 3,                                      // start, end
        WorkerSharedContext::TOK_QUERY_END,
        WorkerSharedContext::TOK_SCRATCH_START,
        // v=0: neighbours = {1}
        NS + 0, WorkerSharedContext::TOK_BFS_ADJ_START,
        NS + 1,
        WorkerSharedContext::TOK_BFS_ADJ_END,
        // v=1: neighbours = {0, 2}
        NS + 1, WorkerSharedContext::TOK_BFS_ADJ_START,
        NS + 0, NS + 2,
        WorkerSharedContext::TOK_BFS_ADJ_END,
        // v=2: neighbours = {1, 3}
        NS + 2, WorkerSharedContext::TOK_BFS_ADJ_START,
        NS + 1, NS + 3,
        WorkerSharedContext::TOK_BFS_ADJ_END,
        // v=3: neighbours = {2}
        NS + 3, WorkerSharedContext::TOK_BFS_ADJ_START,
        NS + 2,
        WorkerSharedContext::TOK_BFS_ADJ_END,
        WorkerSharedContext::TOK_SCRATCH_END,
        WorkerSharedContext::TOK_TASK_START,
        NS + 0, NS + 1, NS + 2, NS + 3,                       // shortest path
        WorkerSharedContext::TOK_TASK_END,
        WorkerSharedContext::TOK_EOS,
    };
    REQUIRE(static_cast<int>(expected_src.size()) == f.layout.seq_len);
    for (int i = 0; i < f.layout.seq_len; ++i) {
        CHECK(src[i] == expected_src[i]);
    }
}

TEST_CASE("Tokenizer writes SEAN + SP + BFS/ReverseAdjacency scratchpad") {
    // Same line-0-1-2-3 fixture; ReverseAdjacency has same length as
    // Plain (3n + Σdeg = 18) but neighbour order is reversed inside
    // each per-vertex block.
    Fixture f = make_fixture();
    f.task.query.end_node = 3;
    f.task.target.path    = std::vector<int>{0, 1, 2, 3};
    f.scratchpad.kind     = ScratchpadKind::BFS;
    f.scratchpad.trace    = std::vector<int>{0, 1, 2, 3};
    f.cfg.bfs_scratchpad_style = "reverse_adjacency";

    f.layout = Layout::from(f.sg, f.task, f.scratchpad, f.cfg);
    REQUIRE(f.layout.scratchpad_len == 18);
    REQUIRE(f.layout.seq_len         == 41);

    std::vector<int32_t> src (f.layout.seq_len, WorkerSharedContext::TOK_PAD);
    const int max_labels = 1;
    std::vector<int32_t> targets(f.layout.seq_len, WorkerSharedContext::TOK_PAD);
    std::vector<int32_t> pos (f.layout.seq_len, 0);
    Tokenizer tk;
    tk.tokenize_into_row(src.data(), targets.data(), max_labels, pos.data(),
                         f.sg, f.task, f.scratchpad, f.layout, f.ctx, f.cfg);

    const int NS = WorkerSharedContext::NUM_SPECIAL_DEFAULT;
    // Only inspect the scratchpad content (start..start+len).
    const int s = f.layout.scratchpad_start;
    const std::vector<int32_t> expected_scratch = {
        // v=0, neighbours = {1} (reversed = {1})
        NS + 0, WorkerSharedContext::TOK_BFS_ADJ_START,
        NS + 1,
        WorkerSharedContext::TOK_BFS_ADJ_END,
        // v=1, neighbours = {0, 2} reversed = {2, 0}
        NS + 1, WorkerSharedContext::TOK_BFS_ADJ_START,
        NS + 2, NS + 0,
        WorkerSharedContext::TOK_BFS_ADJ_END,
        // v=2, neighbours = {1, 3} reversed = {3, 1}
        NS + 2, WorkerSharedContext::TOK_BFS_ADJ_START,
        NS + 3, NS + 1,
        WorkerSharedContext::TOK_BFS_ADJ_END,
        // v=3, neighbours = {2} reversed = {2}
        NS + 3, WorkerSharedContext::TOK_BFS_ADJ_START,
        NS + 2,
        WorkerSharedContext::TOK_BFS_ADJ_END,
    };
    REQUIRE(static_cast<int>(expected_scratch.size()) == f.layout.scratchpad_len);
    for (int i = 0; i < f.layout.scratchpad_len; ++i) {
        CHECK(src[s + i] == expected_scratch[i]);
    }
}

TEST_CASE("Tokenizer writes SEAN + SP + BFS/DuplicateAdjacency scratchpad") {
    // Per-vertex block: v [ nbrs ] { nbrs }, so content_len =
    // 5n + 2*Σdeg = 20 + 12 = 32.
    Fixture f = make_fixture();
    f.task.query.end_node = 3;
    f.task.target.path    = std::vector<int>{0, 1, 2, 3};
    f.scratchpad.kind     = ScratchpadKind::BFS;
    f.scratchpad.trace    = std::vector<int>{0, 1, 2, 3};
    f.cfg.bfs_scratchpad_style = "duplicate_adjacency";

    f.layout = Layout::from(f.sg, f.task, f.scratchpad, f.cfg);
    REQUIRE(f.layout.scratchpad_len == 32);
    REQUIRE(f.layout.seq_len         == 55);  // 41 (Plain) + (32 - 18)

    std::vector<int32_t> src (f.layout.seq_len, WorkerSharedContext::TOK_PAD);
    const int max_labels = 1;
    std::vector<int32_t> targets(f.layout.seq_len, WorkerSharedContext::TOK_PAD);
    std::vector<int32_t> pos (f.layout.seq_len, 0);
    Tokenizer tk;
    tk.tokenize_into_row(src.data(), targets.data(), max_labels, pos.data(),
                         f.sg, f.task, f.scratchpad, f.layout, f.ctx, f.cfg);

    const int NS = WorkerSharedContext::NUM_SPECIAL_DEFAULT;
    const int s = f.layout.scratchpad_start;
    const std::vector<int32_t> expected_scratch = {
        // v=0
        NS + 0, WorkerSharedContext::TOK_BFS_ADJ_START,
        NS + 1,
        WorkerSharedContext::TOK_BFS_ADJ_END,
        WorkerSharedContext::TOK_CURLY_START, NS + 1, WorkerSharedContext::TOK_CURLY_END,
        // v=1
        NS + 1, WorkerSharedContext::TOK_BFS_ADJ_START,
        NS + 0, NS + 2,
        WorkerSharedContext::TOK_BFS_ADJ_END,
        WorkerSharedContext::TOK_CURLY_START, NS + 0, NS + 2, WorkerSharedContext::TOK_CURLY_END,
        // v=2
        NS + 2, WorkerSharedContext::TOK_BFS_ADJ_START,
        NS + 1, NS + 3,
        WorkerSharedContext::TOK_BFS_ADJ_END,
        WorkerSharedContext::TOK_CURLY_START, NS + 1, NS + 3, WorkerSharedContext::TOK_CURLY_END,
        // v=3
        NS + 3, WorkerSharedContext::TOK_BFS_ADJ_START,
        NS + 2,
        WorkerSharedContext::TOK_BFS_ADJ_END,
        WorkerSharedContext::TOK_CURLY_START, NS + 2, WorkerSharedContext::TOK_CURLY_END,
    };
    REQUIRE(static_cast<int>(expected_scratch.size()) == f.layout.scratchpad_len);
    for (int i = 0; i < f.layout.scratchpad_len; ++i) {
        CHECK(src[s + i] == expected_scratch[i]);
    }
}

TEST_CASE("Tokenizer writes SEAN + SP + BFS/WithQueue scratchpad") {
    // Per-vertex block: { q } v [ nbrs ].
    // Line 0-1-2-3, FIFO BFS from 0:
    //   frontier before popping 0: [0]   -> queue token = {0}
    //   frontier before popping 1: [1]   -> queue token = {1}
    //   frontier before popping 2: [2]   -> queue token = {2}
    //   frontier before popping 3: [3]   -> queue token = {3}
    // Σ|q| = 4. content_len = 5n + Σdeg + Σ|q| = 20 + 6 + 4 = 30.
    Fixture f = make_fixture();
    f.task.query.end_node = 3;
    f.task.target.path    = std::vector<int>{0, 1, 2, 3};
    f.scratchpad.kind     = ScratchpadKind::BFS;
    f.scratchpad.trace    = std::vector<int>{0, 1, 2, 3};
    // Cached FIFO frontiers -- normally populated by sample_bfs when
    // style == with_queue, but we hand-build the scratchpad here so
    // supply them explicitly.
    f.scratchpad.frontier_at_pop = std::vector<std::vector<int>>{
        {0}, {1}, {2}, {3},
    };
    f.cfg.bfs_scratchpad_style = "with_queue";

    f.layout = Layout::from(f.sg, f.task, f.scratchpad, f.cfg);
    REQUIRE(f.layout.scratchpad_len == 30);
    REQUIRE(f.layout.seq_len         == 53);  // 41 (Plain) + (30 - 18)

    std::vector<int32_t> src (f.layout.seq_len, WorkerSharedContext::TOK_PAD);
    const int max_labels = 1;
    std::vector<int32_t> targets(f.layout.seq_len, WorkerSharedContext::TOK_PAD);
    std::vector<int32_t> pos (f.layout.seq_len, 0);
    Tokenizer tk;
    tk.tokenize_into_row(src.data(), targets.data(), max_labels, pos.data(),
                         f.sg, f.task, f.scratchpad, f.layout, f.ctx, f.cfg);

    const int NS = WorkerSharedContext::NUM_SPECIAL_DEFAULT;
    const int s = f.layout.scratchpad_start;
    const std::vector<int32_t> expected_scratch = {
        // v=0
        WorkerSharedContext::TOK_CURLY_START, NS + 0, WorkerSharedContext::TOK_CURLY_END,
        NS + 0, WorkerSharedContext::TOK_BFS_ADJ_START,
        NS + 1,
        WorkerSharedContext::TOK_BFS_ADJ_END,
        // v=1
        WorkerSharedContext::TOK_CURLY_START, NS + 1, WorkerSharedContext::TOK_CURLY_END,
        NS + 1, WorkerSharedContext::TOK_BFS_ADJ_START,
        NS + 0, NS + 2,
        WorkerSharedContext::TOK_BFS_ADJ_END,
        // v=2
        WorkerSharedContext::TOK_CURLY_START, NS + 2, WorkerSharedContext::TOK_CURLY_END,
        NS + 2, WorkerSharedContext::TOK_BFS_ADJ_START,
        NS + 1, NS + 3,
        WorkerSharedContext::TOK_BFS_ADJ_END,
        // v=3
        WorkerSharedContext::TOK_CURLY_START, NS + 3, WorkerSharedContext::TOK_CURLY_END,
        NS + 3, WorkerSharedContext::TOK_BFS_ADJ_START,
        NS + 2,
        WorkerSharedContext::TOK_BFS_ADJ_END,
    };
    REQUIRE(static_cast<int>(expected_scratch.size()) == f.layout.scratchpad_len);
    for (int i = 0; i < f.layout.scratchpad_len; ++i) {
        CHECK(src[s + i] == expected_scratch[i]);
    }
}

TEST_CASE("Tokenizer under STAN packs edges into 1 position of 3 columns") {
    // Fixture line 0-1-2-3, SP 0->2 (default fixture task).
    // Under STAN, m=3 edges take 3 sequence positions (not 9). Full
    // seq_len = 1(BOS) + 3(edges) + 1(QS) + 2(query) + 1(QE) + 1(TS)
    //         + 3(target=[0,1,2]) + 1(TE) + 1(EOS) = 14.
    Fixture f = make_fixture();
    f.cfg.tokenization_mode = "stan";
    f.layout = Layout::from(f.sg, f.task, f.scratchpad, f.cfg);
    REQUIRE(f.layout.struct_dim == 3);
    REQUIRE(f.layout.seq_len    == 14);

    const int S = f.layout.seq_len;
    const int D = f.layout.struct_dim;
    std::vector<int32_t> src (S * D, WorkerSharedContext::TOK_PAD);
    const int max_labels = 1;
    std::vector<int32_t> targets(S * D, WorkerSharedContext::TOK_PAD);
    std::vector<int32_t> pos (S * D, 0);
    Tokenizer tk;
    tk.tokenize_into_row(src.data(), targets.data(), max_labels, pos.data(),
                         f.sg, f.task, f.scratchpad, f.layout, f.ctx, f.cfg);

    const int NS = WorkerSharedContext::NUM_SPECIAL_DEFAULT;
    auto tok = [&](int p, int c) { return src[p * D + c]; };

    // Column 0 across positions holds the section token stream.
    // Under STAN, edge positions carry the u vertex in col 0.
    CHECK(tok(0, 0) == WorkerSharedContext::TOK_BOS);

    // Edges at positions 1..3, col 0 = u, col 1 = v, col 2 = EDGE.
    // Line 0-1-2-3 canonical edges: (0,1), (1,2), (2,3).
    for (int i = 0; i < 3; ++i) {
        CHECK(tok(1 + i, 0) == NS + i);
        CHECK(tok(1 + i, 1) == NS + i + 1);
        CHECK(tok(1 + i, 2) == WorkerSharedContext::TOK_EDGE);
    }

    // Non-edge positions: col 0 is the token, cols 1 and 2 stay PAD.
    CHECK(tok(4, 0) == WorkerSharedContext::TOK_QUERY_START);
    CHECK(tok(4, 1) == WorkerSharedContext::TOK_PAD);
    CHECK(tok(4, 2) == WorkerSharedContext::TOK_PAD);
    CHECK(tok(5, 0) == NS + 0);                                // start
    CHECK(tok(5, 1) == WorkerSharedContext::TOK_PAD);
    CHECK(tok(6, 0) == NS + 2);                                // end
    CHECK(tok(7, 0) == WorkerSharedContext::TOK_QUERY_END);
    CHECK(tok(8, 0) == WorkerSharedContext::TOK_TASK_START);
    CHECK(tok(9, 0) == NS + 0);
    CHECK(tok(10, 0) == NS + 1);
    CHECK(tok(11, 0) == NS + 2);
    CHECK(tok(12, 0) == WorkerSharedContext::TOK_TASK_END);
    CHECK(tok(13, 0) == WorkerSharedContext::TOK_EOS);
}

TEST_CASE("Tokenizer sorts BFS-scratchpad adjacency lists by first-mention edge order") {
    // 4-cycle graph with edges pushed in a specific non-canonical order
    // so CSR row-order and first-mention-edge-order disagree. Verifies
    // the emitted adjacency list uses first-mention order, not CSR.
    //
    // Input edge order:  (2,3), (0,3), (1,2), (0,1)
    // -> edges() yield:  (0,3), (0,1), (1,2), (2,3)   [row-scan of stored slots]
    //    edge indices:     0       1       2       3
    // -> first_mention:  0->0, 3->0, 1->1, 2->2
    //
    // For vertex 1: CSR neighbour order is [2, 0] (input order preserved
    // in row 1's slots), but first-mention order is [0, 2] (0 is fm=0, 2 is fm=2).
    Fixture f = make_fixture();

    std::vector<Edge> edges = {{2, 3}, {0, 3}, {1, 2}, {0, 1}};
    f.sg.graph = CsrGraph::from_undirected_edges(4, edges);
    // Token-id space; matches make_fixture()'s convention.
    const int NS_local = WorkerSharedContext::NUM_SPECIAL_DEFAULT;
    f.sg.internal_to_vocab.assign(4, 0);
    for (int i = 0; i < 4; ++i) f.sg.internal_to_vocab[i] = NS_local + i;

    // BFS task from vertex 1. Target path is the BFS visit order --
    // we just need SOMETHING that walks the whole graph so the
    // adjacency-list sort has meaningful work to do.
    f.task.kind             = TaskKind::BFS;
    f.task.query.start_node = 1;
    f.task.query.end_node   = std::nullopt;
    // BFS from 1 with CSR order: pop 1 -> enqueue [2, 0] -> pop 2 -> enqueue [3]
    // -> pop 0 -> (3 already visited) -> pop 3.  Trace = [1, 2, 0, 3].
    f.task.target.path      = std::vector<int>{1, 2, 0, 3};
    f.task.target.valid_next_hops = std::nullopt;

    f.scratchpad.kind  = ScratchpadKind::BFS;
    f.scratchpad.trace = std::vector<int>{1, 2, 0, 3};

    f.layout = Layout::from(f.sg, f.task, f.scratchpad, f.cfg);

    std::vector<int32_t> src (f.layout.seq_len, WorkerSharedContext::TOK_PAD);
    const int max_labels = 1;
    std::vector<int32_t> targets(f.layout.seq_len, WorkerSharedContext::TOK_PAD);
    std::vector<int32_t> pos (f.layout.seq_len, 0);
    Tokenizer tk;
    tk.tokenize_into_row(src.data(), targets.data(), max_labels, pos.data(),
                         f.sg, f.task, f.scratchpad, f.layout, f.ctx, f.cfg);

    const int NS = WorkerSharedContext::NUM_SPECIAL_DEFAULT;
    // Scratchpad content: per-vertex `v [ nbrs_sorted_by_first_mention ]`.
    // Expected sorted orders:
    //   v=1: nbrs {0, 2} in fm order (0 has fm=0, 2 has fm=2)  -> [0, 2]
    //   v=2: nbrs {3, 1} in fm order (3 has fm=0, 1 has fm=1)  -> [3, 1]
    //   v=0: nbrs {3, 1} in fm order                            -> [3, 1]
    //   v=3: nbrs {0, 2} in fm order                            -> [0, 2]
    const int s = f.layout.scratchpad_start;
    const std::vector<int32_t> expected_scratch = {
        // v=1
        NS + 1, WorkerSharedContext::TOK_BFS_ADJ_START,
        NS + 0, NS + 2,
        WorkerSharedContext::TOK_BFS_ADJ_END,
        // v=2
        NS + 2, WorkerSharedContext::TOK_BFS_ADJ_START,
        NS + 3, NS + 1,
        WorkerSharedContext::TOK_BFS_ADJ_END,
        // v=0
        NS + 0, WorkerSharedContext::TOK_BFS_ADJ_START,
        NS + 3, NS + 1,
        WorkerSharedContext::TOK_BFS_ADJ_END,
        // v=3
        NS + 3, WorkerSharedContext::TOK_BFS_ADJ_START,
        NS + 0, NS + 2,
        WorkerSharedContext::TOK_BFS_ADJ_END,
    };
    REQUIRE(static_cast<int>(expected_scratch.size()) == f.layout.scratchpad_len);
    for (int i = 0; i < f.layout.scratchpad_len; ++i) {
        CHECK(src[s + i] == expected_scratch[i]);
    }
}
