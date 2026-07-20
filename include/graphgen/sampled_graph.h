// SampledGraph -- the output of a graph sampler, before the task and
// tokenizer stages run.
//
// The graph itself uses contiguous internal vertex ids 0..n-1, which
// are just construction indices. This struct carries the two other
// pieces of per-graph state that travel with a sample all the way to
// the tokenizer:
//   * how the sampler wants distances measured on this graph, so
//     downstream tasks pick BFS or Dijkstra without having to guess;
//   * the internal -> vocabulary token id mapping that the tokenizer
//     will emit for each vertex.
//
// Vocab ids stay out of the internal graph representaion on purpose. 
// Every algorithm allocates scratch space proportional to vertex count 
// (BFS distances, component ids, all-pairs matrices ~ n^2).
// Indexing those by vocab id instead would size them by the vocab range.
// The mapping is applied once at emit time by the tokenizer and never
// touches any algorithm.

#ifndef GRAPHGEN_SAMPLED_GRAPH_H
#define GRAPHGEN_SAMPLED_GRAPH_H

#include <cstdint>
#include <optional>
#include <vector>

#include "graphgen/csr_graph.h"
#include "graphgen/hop_distance_matrix.h"

namespace graphgen {

// Distance interpretation the sampler intends for this graph. Downstream
// tasks pick BFS vs Dijkstra based on this rather than sniffing weights.

// these do by various names depending on the domain:
// EdgeWeight = Weighted Distance / Path Cost / Length
// HopCount = Hop Distance / Unweighted Distance /
//            Topological Distance / Geodesic Distance

enum class DistanceSemantic {
    HopCount,    // shortest path = number of edges (BFS)
    EdgeWeight,  // shortest path = sum of edge weights (Dijkstra)
};

// Bundle populated by GraphSampler::run and read by downstream stages.
//
// Two derived quantities are exposed by lazy accessor rather than raw
// fields so the pipeline can control lifetime and storage:
//
//   * hop_distances() -- integer, all-pairs, BFS-derived shortest-path
//     lengths in edges. Computed on first call (via one BFS per source
//     vertex, O(n * (n+m))). Cached until release_hop_distances() or
//     until this SampledGraph is destroyed. The backing storage is
//     either an owned std::vector<int> (contiguous n*n) or a slice of a
//     batch tensor previously attached via attach_hop_distances_view;
//     consumers see the same clipped (n, n) HopDistView either way.
//
//   * A future weighted_distances() (Dijkstra / sum-of-weights) will
//     live next to hop_distances but is a separate matrix -- the two
//     semantics are independent, not either/or, since a weighted graph
//     may still be asked for hop counts (topology only).
class SampledGraph {
public:
    CsrGraph         graph;
    DistanceSemantic distance_semantic = DistanceSemantic::HopCount;

    // Mapping internal vertex u -> vocabulary token id. Length matches
    // graph.num_vertices() once GraphSampler::run has populated it;
    // empty before that point.
    //
    // Vocab ids are kept out of the internal graph representation on
    // purpose. Every algorithm allocates scratch space proportional to
    // vertex count (BFS distances, component ids, all-pairs matrices
    // ~ n^2); indexing those by vocab id would size them by the vocab
    // range instead. The mapping is applied once at emit time by the
    // tokenizer.
    std::vector<int> internal_to_vocab;

    // Geometric positions (only populated by geometric-graph samplers
    // like sample_euclidean). Flat row-major layout of shape
    // (num_vertices, dim), so position of internal vertex u is
    // positions[u*dim + k] for k in [0, dim). `dim` records the
    // stride so downstream code can iterate without an out-of-band
    // dim parameter. Left empty (dim=0) for graph kinds without a
    // natural coordinate system; the pipeline treats "empty
    // positions" as "no geometry to surface" and skips the
    // node_positions batch tensor even if cfg.return_positions is on.
    std::vector<double> positions;
    int                 dim = 0;

    // ---- Hop-distance matrix -------------------------------------------
    // Get an (n, n) clipped view of the hop-distance matrix. Computes
    // on the first call, cached thereafter. Semantics: entry (u, v) is
    // the number of edges on a shortest path from u to v, or
    // HOP_UNREACHABLE (-1) if no path exists. For undirected graphs the
    // matrix is symmetric; for directed graphs it is not (BFS from u
    // follows out-edges).
    //
    // mutable so const-correct consumers (task samplers take sg by
    // const&) can still trigger the memoised compute.
    HopDistView hop_distances() const;

    // Attach an external (max_n, max_n) view -- typically a slice of
    // a (B, max_n, max_n) batch tensor allocated by BatchOutputArrays.
    // Must be called before hop_distances() and only once; on next
    // access the fill goes into the attached view rather than an owned
    // buffer. n = graph.num_vertices() must be <= min(v.extent(0),
    // v.extent(1)).
    void attach_hop_distances_view(HopDistView v);

    // Drop the owned buffer if this SampledGraph owned one; no-op if
    // the storage was an attached view (caller owns that lifetime).
    // Called by the pipeline after the task stage when the caller did
    // not request the matrix as a Python return value.
    void release_hop_distances();

    // True iff hop_distances() has been computed (owned or into view).
    // Test / debug hook; production code should not need this.
    bool hop_distances_computed() const { return hop_computed_; }

private:
    // Owned per-graph buffer. Contiguous n*n int32_t when in use, empty
    // otherwise. Mutable for the lazy accessor.
    mutable std::vector<std::int32_t>      hop_owned_;
    // Externally attached storage. When set, hop_distances() writes
    // into this view and hop_owned_ stays empty.
    mutable std::optional<HopDistView>     hop_view_;
    // Set true after the first successful hop_distances() call.
    mutable bool                           hop_computed_ = false;
};

}  // namespace graphgen

#endif  // GRAPHGEN_SAMPLED_GRAPH_H
