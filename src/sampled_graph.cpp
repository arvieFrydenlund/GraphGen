// SampledGraph out-of-line implementations.
//
// Everything else on SampledGraph is trivial data. Only the lazy
// hop-distance accessor + lifecycle hooks live here because they need
// BFS and stride-aware fill logic.

#include "graphgen/sampled_graph.h"

#include <array>
#include <cstddef>
#include <cstdint>

#include "graphgen/check.h"

namespace graphgen {

namespace {

// Build a HopDistView over `data` with the given (rows, cols) extents
// and (row_stride, 1) strides. Small utility shared by both storage
// paths and the "clip to (n, n)" operation.
HopDistView make_view(std::int32_t* data, int rows, int cols, int row_stride) {
    std::array<int, 2> strides{row_stride, 1};
    return HopDistView{
        data,
        md::layout_stride::mapping{md::dextents<int, 2>{rows, cols}, strides}};
}

// Clip `v` to a top-left (rows, cols) sub-region without calling
// submdspan (which is C++26, not present in libc++ 18 yet). Since
// HopDistView already uses layout_stride, clipping is just a new
// mapping with smaller extents but the same strides and data
// pointer -- consumer sees a smaller view over the same memory.
HopDistView clip_top_left(HopDistView v, int rows, int cols) {
    return make_view(v.data_handle(), rows, cols,
                     static_cast<int>(v.stride(0)));
}

// Fill an (n, n) region of hop distances by running one BFS per source
// vertex. `write(u, v, d)` is a small stride-agnostic setter -- callers
// pass a lambda that knows whether the underlying storage is contiguous
// (owned) or strided (batch tensor view).
template <typename Write>
void fill_hop_distances_bfs(const CsrGraph& g, Write write) {
    const int n = g.num_vertices();
    if (n == 0) return;

    // Reusable scratch buffers across the n BFS runs. Swapping avoids
    // reallocation per source; contents are reset per source.
    std::vector<int>          current;
    std::vector<int>          next;
    std::vector<std::int32_t> dist;
    current.reserve(static_cast<std::size_t>(n));
    next.reserve(static_cast<std::size_t>(n));
    dist.assign(static_cast<std::size_t>(n), HOP_UNREACHABLE);

    for (int src = 0; src < n; ++src) {
        // Reset dist to -1 for the new source's BFS. Only touched
        // entries need clearing; simplest to reset all n each source.
        std::fill(dist.begin(), dist.end(), HOP_UNREACHABLE);
        current.clear();
        next.clear();

        dist[static_cast<std::size_t>(src)] = 0;
        current.push_back(src);

        int layer = 0;
        while (!current.empty()) {
            for (int u : current) {
                for (int v : g.neighbours(u)) {
                    if (dist[static_cast<std::size_t>(v)] == HOP_UNREACHABLE) {
                        dist[static_cast<std::size_t>(v)] =
                            static_cast<std::int32_t>(layer + 1);
                        next.push_back(v);
                    }
                }
            }
            ++layer;
            current.swap(next);
            next.clear();
        }

        // Publish the row for this source. write() places (src, v, d)
        // into whichever backing storage the caller supplied.
        for (int v = 0; v < n; ++v) {
            write(src, v, dist[static_cast<std::size_t>(v)]);
        }
    }
}

}  // namespace

HopDistView SampledGraph::hop_distances() const {
    const int n = graph.num_vertices();

    // Fast path: already computed. Return the same clipped view we
    // returned last time.
    if (hop_computed_) {
        if (hop_view_) {
            // Clip the attached (max_n, max_n) view to (n, n) so
            // consumers don't see the batch-tensor padding.
            return clip_top_left(*hop_view_, n, n);
        }
        // Owned buffer: contiguous n*n; row stride == n.
        return make_view(hop_owned_.data(), n, n, /*row_stride=*/n);
    }

    // First-call compute path.
    if (hop_view_) {
        // Storage is the attached batch-tensor slice. Row stride is
        // whatever the caller's view had (typically max_n). Write into
        // the (n, n) subregion of it.
        HopDistView v = *hop_view_;
        GG_CHECK(v.extent(0) >= n && v.extent(1) >= n,
                 "SampledGraph::hop_distances: attached view too small ("
                     << v.extent(0) << "x" << v.extent(1)
                     << ") for n=" << n);
        fill_hop_distances_bfs(graph, [&](int u, int vtx, std::int32_t d) {
            v[u, vtx] = d;
        });
        hop_computed_ = true;
        return clip_top_left(v, n, n);
    }

    // Owned path: allocate contiguous n*n and fill via row-major
    // indexing.
    hop_owned_.assign(static_cast<std::size_t>(n) * static_cast<std::size_t>(n),
                      HOP_UNREACHABLE);
    fill_hop_distances_bfs(graph, [&](int u, int vtx, std::int32_t d) {
        hop_owned_[static_cast<std::size_t>(u) *
                       static_cast<std::size_t>(n) +
                   static_cast<std::size_t>(vtx)] = d;
    });
    hop_computed_ = true;

    return make_view(hop_owned_.data(), n, n, /*row_stride=*/n);
}

void SampledGraph::attach_hop_distances_view(HopDistView v) {
    GG_CHECK(!hop_computed_,
             "SampledGraph::attach_hop_distances_view: already computed; "
             "attach must happen before first hop_distances() call");
    GG_CHECK(!hop_view_.has_value(),
             "SampledGraph::attach_hop_distances_view: view already attached");
    const int n = graph.num_vertices();
    GG_CHECK(v.extent(0) >= n && v.extent(1) >= n,
             "SampledGraph::attach_hop_distances_view: view extents ("
                 << v.extent(0) << "x" << v.extent(1)
                 << ") smaller than num_vertices n=" << n);
    hop_view_ = v;
}

void SampledGraph::release_hop_distances() {
    // If we're viewing into a batch tensor the caller owns it -- do
    // nothing there. Only drop the owned buffer.
    hop_owned_.clear();
    hop_owned_.shrink_to_fit();
    // Note: hop_view_ and hop_computed_ are left alone. Once attached,
    // the caller controls that buffer's lifetime; and if compute
    // already happened into the view, we mustn't pretend it hasn't.
    if (!hop_view_) {
        hop_computed_ = false;
    }
}

}  // namespace graphgen
