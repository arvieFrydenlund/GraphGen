// GraphSampler -- dispatches on GraphKind and returns a SampledGraph.
//
// The Worker holds one GraphSampler and calls `run` per batch item; the
// sampler itself is stateless (RNG is threaded through the argument, so
// samplers do not carry any per-worker mutable state).
//
// This header only declares the surface; the per-kind implementations
// are stubs in the .cpp that throw via GG_CHECK. Real algorithms land
// in later steps.

#ifndef GRAPHGEN_GRAPH_SAMPLER_H
#define GRAPHGEN_GRAPH_SAMPLER_H

#include <random>
#include <vector>

#include "graphgen/csr_graph.h"
#include "graphgen/graph_kind.h"
#include "graphgen/sampled_graph.h"

namespace graphgen {

struct GeneratorConfig;  // forward decl -- only referenced by pointer/ref

class GraphSampler {
public:
    // Dispatches to sample_<kind> for the given GraphKind. `rng` is the
    // per-worker mt19937_64 owned by Worker; samplers must draw only from
    // it (never from thread-local or global state) for reproducibility.
    SampledGraph run(GraphKind kind,
                     std::mt19937_64& rng,
                     const GeneratorConfig& cfg);

private:
    SampledGraph sample_erdos_renyi(std::mt19937_64& rng, const GeneratorConfig& cfg);
    SampledGraph sample_euclidean  (std::mt19937_64& rng, const GeneratorConfig& cfg);
    SampledGraph sample_random_tree(std::mt19937_64& rng, const GeneratorConfig& cfg);
    SampledGraph sample_path_star  (std::mt19937_64& rng, const GeneratorConfig& cfg);
    SampledGraph sample_balanced   (std::mt19937_64& rng, const GeneratorConfig& cfg);
    SampledGraph sample_khops      (std::mt19937_64& rng, const GeneratorConfig& cfg);
    SampledGraph sample_khops_gen  (std::mt19937_64& rng, const GeneratorConfig& cfg);

    // Randomise the edge list before CsrGraph::build consumes it.
    // CsrGraph stores per-row slots in input order, so shuffling here
    // shows up as randomised neighbour order within each vertex's
    // adjacency list. The tokenizer emits edges in that CSR-derived
    // order (via g.edges()), so the shuffled order is the emission
    // order -- and derived quantities like the "first-mention edge
    // index" the BFS scratchpad's adjacency-list sort uses are stable
    // with respect to the same rng seed.
    //
    // Every sample_* helper should call this before handing its
    // edges vector to CsrGraph so the model never sees a canonical
    // construction order leak through the tokens. Static because it
    // touches no per-instance state; lives on GraphSampler (not
    // CsrGraph) because CsrGraph is a pure data structure that
    // shouldn't take an rng.
    static void shuffle_edges(std::vector<Edge>& edges,
                              std::mt19937_64& rng);
};

}  // namespace graphgen

#endif  // GRAPHGEN_GRAPH_SAMPLER_H
