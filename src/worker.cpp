// Worker implementation.
//
// generate_batch parses the config's graph_kind string into a GraphKind
// enum, then hands off to GraphSampler. Downstream stages (task, tokenize,
// pack into arrays) come in later steps -- for now the sampler stubs
// throw and that error propagates to Python.

#include "graphgen/worker.h"

#include <utility>

#include "graphgen/check.h"
#include "graphgen/graph_kind.h"
#include "graphgen/sampled_graph.h"

namespace graphgen {

Worker::Worker(std::shared_ptr<WorkerSharedContext> ctx, std::uint64_t seed)
    : ctx_(std::move(ctx)), gen_(seed) {}

py::dict Worker::generate_batch(const GeneratorConfig& cfg) {
    // Unknown names throw std::invalid_argument with the offending string;
    // pybind11 surfaces that as ValueError on the Python side.
    const GraphKind kind = graph_kind_from_string(cfg.graph_kind);

    // Rest of the pipeline (task computer, tokenizer, batch packing)
    // arrives in later steps; for now the sampler stubs throw.
    SampledGraph sampled = sampler_.run(kind, gen_, cfg);
    (void)sampled;

    return py::dict{};
}

}  // namespace graphgen
