// Scratchpad -- the model's "shown work" trace, produced alongside the
// Task by TaskSampler (via ScratchpadSampler).
//
// A scratchpad is what the model emits *before* the answer to explain
// how it got there (e.g. a BFS traversal trace, a DFS visit order). The
// None kind carries no trace; every other kind fills `trace` with a
// sequence of vertex ids that will be tokenised as thinking-token
// output ahead of the target sequence.
//
// Invariant vs Task: Scratchpad never writes to Task fields. Task owns
// (start, end, path, valid_next_hops); Scratchpad owns its own trace
// and any per-scratchpad-kind metadata added later.

#ifndef GRAPHGEN_SCRATCHPAD_H
#define GRAPHGEN_SCRATCHPAD_H

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace graphgen {

enum class ScratchpadKind {
    None,
    BFS,
    DFS,
};

std::string_view to_string(ScratchpadKind s);

// Throws std::invalid_argument on unknown names.
ScratchpadKind scratchpad_kind_from_string(std::string_view s);

struct Scratchpad {
    ScratchpadKind kind = ScratchpadKind::None;

    // Sequence of vertex ids in visit order for BFS/DFS traces. Empty
    // when kind == None.
    std::optional<std::vector<int>> trace;

    // FIFO-BFS queue snapshot right BEFORE each pop, aligned with
    // `trace`: frontier_at_pop[i][0] == trace[i]. Populated by
    // sample_bfs only when the tokenization needs it (currently:
    // bfs_scratchpad_style == "with_queue"). Left empty otherwise so
    // the memory cost is paid only when actually consumed.
    std::optional<std::vector<std::vector<int>>> frontier_at_pop;
};

}  // namespace graphgen

#endif  // GRAPHGEN_SCRATCHPAD_H
