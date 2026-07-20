// Hop-distance matrix types and sentinels.
//
// "Hop distance" is the number of edges on a shortest path between two
// vertices in an unweighted graph -- what BFS gives you. This header
// exists on its own because we plan to introduce a separate
// WeightedDistanceMatrix (Dijkstra / edge-weight-sum) alongside it when
// weighted graphs land; keeping the naming semantics-specific from day
// one avoids a rename later.
//
// The view type is a stride-aware 2D mdspan of int. Two backing storages
// share the same view type:
//   1. Owned per-graph buffer (std::vector<int>, contiguous n*n).
//   2. Slice of a batch tensor (B, max_n, max_n) attached from the
//      worker before tasks run. Strided (row stride = max_n), clipped
//      to (n_i, n_i) via submdspan so consumers never see padding.
//
// Sentinels:
//   HOP_UNREACHABLE (-1): pair is disconnected in the graph. Set by BFS
//   HOP_PAD (INT_MIN):    slot is outside the valid [0, n_i) x [0, n_i)
//                         region of a batch tensor row. Set by
//                         BatchOutputArrays at allocation time and never
//                         touched afterwards.

#ifndef GRAPHGEN_HOP_DISTANCE_MATRIX_H
#define GRAPHGEN_HOP_DISTANCE_MATRIX_H

#include <climits>
#include <cstdint>

#include "graphgen/mdspan_shim.h"

namespace graphgen {

// Stride-aware 2D mdspan of int32_t. Used for both owned (contiguous)
// and batch-tensor-attached (strided) storage; consumers never need to
// know which they hold. Extents are dynamic in both dimensions.
//
// int32_t (rather than plain int) so the element type matches the
// exposed numpy dtype exactly regardless of platform int width.
//
// md:: routes to std::mdspan when the toolchain provides it
// (C++23, __cpp_lib_mdspan) and to the vendored Kokkos reference
// implementation otherwise -- see mdspan_shim.h.
using HopDistView = md::mdspan<
    std::int32_t,
    md::dextents<int, 2>,
    md::layout_stride>;

// Sentinels. See file-level comment for semantics.
inline constexpr std::int32_t HOP_UNREACHABLE = -1;
inline constexpr std::int32_t HOP_PAD         = INT32_MIN;

}  // namespace graphgen

#endif  // GRAPHGEN_HOP_DISTANCE_MATRIX_H
