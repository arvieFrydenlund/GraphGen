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

#include "graphgen/graph_kind.h"
#include "graphgen/sampled_graph.h"

namespace graphgen {

class GeneratorConfig;  // forward decl -- only referenced by pointer/ref

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
};

}  // namespace graphgen

#endif  // GRAPHGEN_GRAPH_SAMPLER_H
