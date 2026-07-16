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

#include <vector>

#include "graphgen/csr_graph.h"

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

// Passive bundle. Populated by GraphSampler::run and read (never
// mutated in place) by downstream stages.
struct SampledGraph {
    CsrGraph         graph;
    DistanceSemantic distance_semantic = DistanceSemantic::HopCount;
    // Mapping internal vertex u -> vocabulary token id. Length matches
    // graph.num_vertices() once GraphSampler::run has populated it;
    // empty before that point.
    std::vector<int> internal_to_vocab;
};

}  // namespace graphgen

#endif  // GRAPHGEN_SAMPLED_GRAPH_H
