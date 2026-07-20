// Which BFS scratchpad variant the tokenizer emits.
//
// The scratchpad walks the BFS visit order and, at each popped vertex,
// records some combination of vertex identity, adjacency list, and
// frontier queue. Legacy code exposed three orthogonal bool flags
// (`include_queue`, `reverse_adjacency_lists`, `duplicate_adjacency_lists`),
// but in practice the useful settings were mutually exclusive: pick
// one alternative to the plain form, not several at once. This enum
// names each valid recipe so invalid combinations are unrepresentable.
//
//   Plain              -- per-vertex block is `v [ n_1 n_2 ... n_deg(v) ]`.
//                         Neighbours in natural CSR order. Default.
//
//   WithQueue          -- per-vertex block is `{ q_1 ... q_k } v [ n_1 ... n_deg ]`
//                         where the `{...}` chunk lists the BFS frontier
//                         at the moment v was popped. Teaches the model
//                         the frontier -> pop step alongside expansion.
//
//   ReverseAdjacency   -- per-vertex block is `v [ n_deg ... n_1 ]`.
//                         Same content as Plain, neighbours reversed.
//                         Probes CSR-order dependence.
//
//   DuplicateAdjacency -- per-vertex block is `v [ n_1 ... n_deg ] { n_1 ... n_deg }`.
//                         Adjacency list emitted twice with different
//                         bracket kinds; supports the legacy validator's
//                         "alt list" fallback path.
//
// Consumers dispatch on the enum and never see the underlying bools.

#ifndef GRAPHGEN_BFS_SCRATCHPAD_STYLE_H
#define GRAPHGEN_BFS_SCRATCHPAD_STYLE_H

#include <string>
#include <string_view>

namespace graphgen {

enum class BFSScratchpadStyle {
    Plain,
    WithQueue,
    ReverseAdjacency,
    DuplicateAdjacency,
};

// Canonical lowercase spelling used on GeneratorConfig.bfs_scratchpad_style
// and in error messages.
std::string_view to_string(BFSScratchpadStyle s);

// Throws std::invalid_argument on unknown names.
BFSScratchpadStyle bfs_scratchpad_style_from_string(std::string_view s);

}  // namespace graphgen

#endif  // GRAPHGEN_BFS_SCRATCHPAD_STYLE_H
