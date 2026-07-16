// Per-worker mutable state (currently just the RNG) plus a handle to the
// shared, immutable WorkerSharedContext.
//
// One Worker is instantiated per PyTorch DataLoader worker process.  Each
// call to `generate_batch(cfg)` samples, tokenises, and packages one full
// batch of graph tasks according to `cfg`. Dispatch on `cfg.graph_kind`
// selects the topology sampler.

#ifndef GRAPHGEN_WORKER_H
#define GRAPHGEN_WORKER_H

#include <cstdint>
#include <memory>
#include <random>

#include <pybind11/pybind11.h>

#include "graphgen/generator_config.h"
#include "graphgen/graph_sampler.h"
#include "graphgen/worker_shared_context.h"

namespace graphgen {

namespace py = pybind11;

class Worker {
public:
    Worker(std::shared_ptr<WorkerSharedContext> ctx, std::uint64_t seed);

    py::dict generate_batch(const GeneratorConfig& cfg);

private:
    std::shared_ptr<WorkerSharedContext> ctx_;
    std::mt19937_64 gen_;
    GraphSampler sampler_;
};

}  // namespace graphgen

#endif  // GRAPHGEN_WORKER_H
