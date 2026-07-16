#include "graphgen/graph_sampler.h"

#include "graphgen/check.h"
#include "graphgen/generator_config.h"

namespace graphgen {

SampledGraph GraphSampler::run(GraphKind kind,
                               std::mt19937_64& rng,
                               const GeneratorConfig& cfg) {
    SampledGraph sg = [&]() -> SampledGraph {
        switch (kind) {
            case GraphKind::ErdosRenyi: return sample_erdos_renyi(rng, cfg);
            case GraphKind::Euclidean:  return sample_euclidean  (rng, cfg);
            case GraphKind::RandomTree: return sample_random_tree(rng, cfg);
            case GraphKind::PathStar:   return sample_path_star  (rng, cfg);
            case GraphKind::Balanced:   return sample_balanced   (rng, cfg);
            case GraphKind::Khops:      return sample_khops      (rng, cfg);
            case GraphKind::KhopsGen:   return sample_khops_gen  (rng, cfg);
        }
        GG_CHECK(false, "GraphSampler::run: unreachable -- unknown GraphKind");
    }();

    // Assign vocab IDs eagerly, right when we know n. This lives on the
    // graph itself so downstream stages never see a "raw" SampledGraph
    // without a vocab mapping.
    sg.assign_random_vocab_ids(cfg.min_vocab, cfg.max_vocab, rng);

    return sg;
}

SampledGraph GraphSampler::sample_erdos_renyi(std::mt19937_64&, const GeneratorConfig&) {
    GG_CHECK(false, "sample_erdos_renyi not implemented");
}
SampledGraph GraphSampler::sample_euclidean(std::mt19937_64&, const GeneratorConfig&) {
    GG_CHECK(false, "sample_euclidean not implemented");
}
SampledGraph GraphSampler::sample_random_tree(std::mt19937_64&, const GeneratorConfig&) {
    GG_CHECK(false, "sample_random_tree not implemented");
}
SampledGraph GraphSampler::sample_path_star(std::mt19937_64&, const GeneratorConfig&) {
    GG_CHECK(false, "sample_path_star not implemented");
}
SampledGraph GraphSampler::sample_balanced(std::mt19937_64&, const GeneratorConfig&) {
    GG_CHECK(false, "sample_balanced not implemented");
}
SampledGraph GraphSampler::sample_khops(std::mt19937_64&, const GeneratorConfig&) {
    GG_CHECK(false, "sample_khops not implemented");
}
SampledGraph GraphSampler::sample_khops_gen(std::mt19937_64&, const GeneratorConfig&) {
    GG_CHECK(false, "sample_khops_gen not implemented");
}

}  // namespace graphgen
