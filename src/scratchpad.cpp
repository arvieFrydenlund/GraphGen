#include "graphgen/scratchpad.h"

#include <stdexcept>
#include <string>

namespace graphgen {

std::string_view to_string(ScratchpadKind s) {
    switch (s) {
        case ScratchpadKind::None: return "none";
        case ScratchpadKind::BFS:  return "bfs";
        case ScratchpadKind::DFS:  return "dfs";
    }
    return "<invalid>";
}

ScratchpadKind scratchpad_kind_from_string(std::string_view s) {
    if (s == "none") return ScratchpadKind::None;
    if (s == "bfs")  return ScratchpadKind::BFS;
    if (s == "dfs")  return ScratchpadKind::DFS;
    throw std::invalid_argument("unknown scratchpad_kind='" + std::string(s) + "'");
}

}  // namespace graphgen
