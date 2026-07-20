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
#include "graphgen/task_sampler.h"
#include "graphgen/tokenizer.h"
#include "graphgen/worker_shared_context.h"

namespace graphgen {

namespace py = pybind11;

class Worker {
public:
    Worker(std::shared_ptr<WorkerSharedContext> ctx, std::uint64_t seed);

    py::dict generate_batch(const GeneratorConfig& cfg);

    // Test-only introspection: runs one full graph_sampler_.run(...) call
    // and returns a small dict of invariants (num_vertices, num_edges,
    // num_components, vocab_ids) for pytest to check. Removed when
    // generate_batch() gains a real return shape.
    py::dict sample_graph_stats(const GeneratorConfig& cfg);

    // Test-only introspection: runs graph_sampler_ then task_sampler_
    // and returns the query + target fields (start, end, path,
    // valid_next_hops) plus a couple of graph stats. Removed when
    // generate_batch() gains a real return shape.
    py::dict sample_shortest_path_stats(const GeneratorConfig& cfg);

private:
    std::shared_ptr<WorkerSharedContext> ctx_;
    std::mt19937_64 gen_;
    GraphSampler graph_sampler_;
    TaskSampler  task_sampler_;
    Tokenizer    tokenizer_;
};

}  // namespace graphgen

#endif  // GRAPHGEN_WORKER_H
