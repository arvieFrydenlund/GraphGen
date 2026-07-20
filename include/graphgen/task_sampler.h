// TaskSampler -- dispatches on (TaskKind, ScratchpadKind) and returns
// the Task + Scratchpad pair for one sampled item.
//
// The Worker holds one TaskSampler and calls `run` once per accepted
// item in a batch. Like GraphSampler, TaskSampler is stateless (the
// per-worker RNG is threaded through the argument), so its instance is
// really just a namespace for the per-kind private methods and the
// dispatcher. Every per-kind method involves stochastic picks (which
// (start, end) pair to ask about, which shortest path to sample from
// the DAG, which neighbour a traversal visits first), hence "sampler".

#ifndef GRAPHGEN_TASK_SAMPLER_H
#define GRAPHGEN_TASK_SAMPLER_H

#include <random>
#include <utility>

#include "graphgen/random_utils.h"
#include "graphgen/scratchpad.h"
#include "graphgen/scratchpad_sampler.h"
#include "graphgen/task.h"

namespace graphgen {

struct GeneratorConfig;  // forward decl
class SampledGraph;     // forward decl

class TaskSampler {
public:
    // Dispatches on both enums. Currently only the (ShortestPath, None)
    // combination is wired up; every other combination throws.
    //
    // `node_vocab_lo` / `node_vocab_hi` bracket the raw vocab range
    // [lo, hi) that node ids draw from (== [ctx.num_special,
    // ctx.max_vocab)). Only synthetic-sequence tasks (khops_gen /
    // khops) touch these -- graph-based tasks ignore them entirely,
    // so tests using graph tasks can leave the defaults.
    std::pair<Task, Scratchpad> run(TaskKind task_kind,
                                    ScratchpadKind scratchpad_kind,
                                    const SampledGraph& sg,
                                    std::mt19937_64& rng,
                                    const GeneratorConfig& cfg,
                                    int node_vocab_lo = 0,
                                    int node_vocab_hi = 0);

private:
    Task sample_shortest_path(const SampledGraph& sg,
                              std::mt19937_64& rng,
                              const GeneratorConfig& cfg);

    Task sample_bfs(const SampledGraph& sg,
                    std::mt19937_64& rng,
                    const GeneratorConfig& cfg);

    // Center / Centroid share the aggregation-over-query-distances
    // shape and differ only in the aggregator: Center uses MAX (find
    // vertex minimising the worst-case query distance -- the graph
    // center with respect to Q); Centroid uses SUM (minimise total
    // distance -- the graph centroid). Both use |Q| single-source
    // BFS calls (or reverse-BFS calls under directed graphs), never
    // materialising a full n x n distance matrix.
    Task sample_center_centroid(const SampledGraph& sg,
                                std::mt19937_64& rng,
                                const GeneratorConfig& cfg,
                                bool is_center);

    // Synthetic-sequence task: sample a k-hop backtrace over a random
    // prefix drawn from [node_vocab_lo, node_vocab_hi). No graph
    // needed; the shape of the answer is baked into the prefix layout.
    Task sample_khops_gen(std::mt19937_64& rng,
                          const GeneratorConfig& cfg,
                          int node_vocab_lo,
                          int node_vocab_hi);

    // Per-position khops. Samples a sequence over the vocab range
    // (either via k+1 shuffled vocab permutations or by uniform
    // sampling with adjacent-repeat rejection), then computes the
    // per-position k-hop back-pointers and materialises the labels
    // that the tokenizer's fill_targets step will patch in.
    Task sample_khops(std::mt19937_64& rng,
                      const GeneratorConfig& cfg,
                      int node_vocab_lo,
                      int node_vocab_hi);

    // Scratchpad sampling is a private step of task sampling: it reads
    // Task fields (start, end) but never mutates them, keeping Task the
    // single source of truth for path/valid_next_hops.
    ScratchpadSampler scratchpad_sampler_;

    // Cached partition sampler for khops_gen. Keeping it as a member
    // amortises the DP cache across every item a worker processes.
    SampleIntPartition partition_sampler_;
};

}  // namespace graphgen

#endif  // GRAPHGEN_TASK_SAMPLER_H
