// ScratchpadSampler -- dispatches on ScratchpadKind and returns the
// Scratchpad for one sampled item.
//
// Owned by TaskSampler, not by Worker: scratchpad construction reads
// Task fields (start, end) that the task sampler has just produced, so
// keeping the two stages inside the same public "sample a training
// example" call preserves the invariant that scratchpad never writes to
// task. The file split is purely for code organisation -- each pipeline
// stage lives in its own translation unit -- and does not change the
// stage-count Worker orchestrates.

#ifndef GRAPHGEN_SCRATCHPAD_SAMPLER_H
#define GRAPHGEN_SCRATCHPAD_SAMPLER_H

#include <random>

#include "graphgen/scratchpad.h"

namespace graphgen {

struct GeneratorConfig;  // forward decl
class SampledGraph;     // forward decl
struct Task;             // forward decl

class ScratchpadSampler {
public:
    // Dispatches on scratchpad_kind. Only ScratchpadKind::None is wired
    // up at this stage; per-kind bodies for Bfs/Dfs throw until ported.
    Scratchpad run(ScratchpadKind kind,
                   const SampledGraph& sg,
                   const Task& task,
                   std::mt19937_64& rng,
                   const GeneratorConfig& cfg);

private:
    Scratchpad sample_bfs(const SampledGraph& sg,
                          const Task& task,
                          std::mt19937_64& rng,
                          const GeneratorConfig& cfg);

    Scratchpad sample_dfs(const SampledGraph& sg,
                          const Task& task,
                          std::mt19937_64& rng,
                          const GeneratorConfig& cfg);
};

}  // namespace graphgen

#endif  // GRAPHGEN_SCRATCHPAD_SAMPLER_H
