// Tests for the hop-distance-matrix accessor on SampledGraph.
//
// Covers:
//   * lazy compute into an owned buffer (contiguous n*n storage)
//   * lazy compute into an attached view (strided batch-tensor row)
//   * (n, n) clipping past the attached view's max_n
//   * unreachable pairs marked with HOP_UNREACHABLE (-1)
//   * padded slots outside (n, n) region are HOP_PAD when attached to
//     a pre-filled batch tensor
//   * release_hop_distances drops owned storage but leaves attached
//     views alone
//   * attach after compute rejected (single-attach discipline)

#include "doctest.h"

#include <array>
#include <climits>
#include <vector>

#include "graphgen/csr_graph.h"
#include "graphgen/hop_distance_matrix.h"
#include "graphgen/mdspan_shim.h"
#include "graphgen/sampled_graph.h"

using graphgen::CsrGraph;
using graphgen::Edge;
using graphgen::HopDistView;
using graphgen::HOP_PAD;
using graphgen::HOP_UNREACHABLE;
using graphgen::SampledGraph;

namespace {

// Build an owned SampledGraph wrapping an undirected line 0-1-2-3-4.
SampledGraph make_line_sg(int n) {
    std::vector<Edge> edges;
    for (int i = 0; i + 1 < n; ++i) edges.push_back({i, i + 1});
    SampledGraph sg;
    sg.graph = CsrGraph::from_undirected_edges(n, edges);
    sg.internal_to_vocab.resize(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) sg.internal_to_vocab[i] = i;
    return sg;
}

// Build a HopDistView over a caller-owned buffer sized (rows, cols).
// Row stride == cols, column stride == 1.
HopDistView make_view(std::vector<std::int32_t>& buf, int rows, int cols) {
    std::array<int, 2> strides{cols, 1};
    return HopDistView{
        buf.data(),
        graphgen::md::layout_stride::mapping{
            graphgen::md::dextents<int, 2>{rows, cols},
            strides}};
}

}  // namespace

TEST_CASE("hop_distances: owned storage, line graph") {
    // Line 0-1-2-3-4. Distance(u, v) == |u - v|.
    SampledGraph sg = make_line_sg(5);
    CHECK_FALSE(sg.hop_distances_computed());

    HopDistView D = sg.hop_distances();
    CHECK(sg.hop_distances_computed());
    CHECK(D.extent(0) == 5);
    CHECK(D.extent(1) == 5);
    for (int u = 0; u < 5; ++u) {
        for (int v = 0; v < 5; ++v) {
            CHECK(D[u, v] == std::abs(u - v));
        }
    }
}

TEST_CASE("hop_distances: unreachable pairs report HOP_UNREACHABLE") {
    // Two disjoint edges: {0-1}, {2-3}. Pairs across components are
    // unreachable in both directions.
    std::vector<Edge> edges = {{0, 1}, {2, 3}};
    SampledGraph sg;
    sg.graph = CsrGraph::from_undirected_edges(4, edges);
    sg.internal_to_vocab = {0, 1, 2, 3};

    HopDistView D = sg.hop_distances();
    // Within-component pairs are finite.
    CHECK(D[0, 1] == 1);
    CHECK(D[2, 3] == 1);
    // Cross-component pairs are HOP_UNREACHABLE.
    CHECK(D[0, 2] == HOP_UNREACHABLE);
    CHECK(D[1, 3] == HOP_UNREACHABLE);
    CHECK(D[2, 0] == HOP_UNREACHABLE);
}

TEST_CASE("hop_distances: repeated calls return the same values (memoised)") {
    SampledGraph sg = make_line_sg(4);
    HopDistView D1 = sg.hop_distances();
    HopDistView D2 = sg.hop_distances();
    for (int u = 0; u < 4; ++u) {
        for (int v = 0; v < 4; ++v) {
            CHECK(D1[u, v] == D2[u, v]);
        }
    }
}

TEST_CASE("hop_distances: attached view fills batch tensor, clipped to n") {
    // Batch tensor with max_n = 6, but our graph has n = 4. Fill with
    // HOP_PAD so we can verify padding survives around the (4, 4)
    // filled region.
    const int max_n = 6;
    std::vector<std::int32_t> buf(
        static_cast<std::size_t>(max_n) * static_cast<std::size_t>(max_n),
        HOP_PAD);
    HopDistView batch_view = make_view(buf, max_n, max_n);

    SampledGraph sg = make_line_sg(4);
    sg.attach_hop_distances_view(batch_view);
    CHECK_FALSE(sg.hop_distances_computed());

    HopDistView D = sg.hop_distances();
    CHECK(sg.hop_distances_computed());
    // Clipped extents: consumer sees only (n, n).
    CHECK(D.extent(0) == 4);
    CHECK(D.extent(1) == 4);
    // Inside the (n, n) region: real BFS values (line graph).
    for (int u = 0; u < 4; ++u) {
        for (int v = 0; v < 4; ++v) {
            CHECK(D[u, v] == std::abs(u - v));
        }
    }
    // Outside the (n, n) region: unchanged HOP_PAD, both on the
    // (max_n - n) rows and on the (max_n - n) columns of filled rows.
    for (int v = 0; v < max_n; ++v) {
        for (int u = 4; u < max_n; ++u) {
            CHECK(buf[u * max_n + v] == HOP_PAD);
        }
    }
    for (int u = 0; u < 4; ++u) {
        for (int v = 4; v < max_n; ++v) {
            CHECK(buf[u * max_n + v] == HOP_PAD);
        }
    }
}

TEST_CASE("hop_distances: attach must happen before first access") {
    SampledGraph sg = make_line_sg(3);
    (void)sg.hop_distances();  // triggers owned-storage compute
    CHECK(sg.hop_distances_computed());

    std::vector<std::int32_t> buf(9, HOP_PAD);
    HopDistView view = make_view(buf, 3, 3);
    CHECK_THROWS_AS(sg.attach_hop_distances_view(view), std::runtime_error);
}

TEST_CASE("hop_distances: attach rejects double-attach") {
    SampledGraph sg = make_line_sg(3);
    std::vector<std::int32_t> buf(9, HOP_PAD);
    HopDistView view = make_view(buf, 3, 3);
    sg.attach_hop_distances_view(view);

    std::vector<std::int32_t> buf2(9, HOP_PAD);
    HopDistView view2 = make_view(buf2, 3, 3);
    CHECK_THROWS_AS(sg.attach_hop_distances_view(view2), std::runtime_error);
}

TEST_CASE("hop_distances: attached view smaller than n throws at attach") {
    SampledGraph sg = make_line_sg(5);
    std::vector<std::int32_t> buf(16, HOP_PAD);  // 4x4, too small for n=5
    HopDistView view = make_view(buf, 4, 4);
    // Reject at attach time -- earliest point we can detect the
    // mismatch, so consumers don't discover it only when the fill
    // actually runs.
    CHECK_THROWS_AS(sg.attach_hop_distances_view(view), std::runtime_error);
}

TEST_CASE("release_hop_distances: drops owned buffer, resets computed flag") {
    SampledGraph sg = make_line_sg(4);
    (void)sg.hop_distances();
    CHECK(sg.hop_distances_computed());
    sg.release_hop_distances();
    CHECK_FALSE(sg.hop_distances_computed());
    // Re-compute works cleanly after release.
    HopDistView D = sg.hop_distances();
    CHECK(D[0, 3] == 3);
}

TEST_CASE("release_hop_distances: leaves attached view alone") {
    // When storage is an attached view, the caller owns the buffer's
    // lifetime; release_hop_distances must NOT clear the computed
    // flag or reset the view.
    const int n = 4;
    std::vector<std::int32_t> buf(
        static_cast<std::size_t>(n) * static_cast<std::size_t>(n),
        HOP_PAD);
    HopDistView view = make_view(buf, n, n);

    SampledGraph sg = make_line_sg(n);
    sg.attach_hop_distances_view(view);
    (void)sg.hop_distances();
    CHECK(sg.hop_distances_computed());

    sg.release_hop_distances();
    // Attached-view storage keeps the compute state.
    CHECK(sg.hop_distances_computed());
    // Buffer still holds the BFS values.
    CHECK(buf[0 * n + 3] == 3);  // D[0, 3] == 3 on the line graph
}

TEST_CASE("hop_distances: directed graph, asymmetric matrix") {
    // Directed 0 -> 1 -> 2. Forward BFS: D[0, 2]=2 but D[2, 0]=unreachable.
    std::vector<Edge> edges = {{0, 1}, {1, 2}};
    SampledGraph sg;
    sg.graph = CsrGraph::from_directed_edges(3, edges);
    sg.internal_to_vocab = {0, 1, 2};

    HopDistView D = sg.hop_distances();
    CHECK(D[0, 0] == 0);
    CHECK(D[0, 1] == 1);
    CHECK(D[0, 2] == 2);
    CHECK(D[1, 2] == 1);
    CHECK(D[1, 0] == HOP_UNREACHABLE);
    CHECK(D[2, 0] == HOP_UNREACHABLE);
    CHECK(D[2, 1] == HOP_UNREACHABLE);
}
