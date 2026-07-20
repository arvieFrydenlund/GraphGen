#include "graphgen/bfs_scratchpad_style.h"

#include <stdexcept>
#include <string>

namespace graphgen {

std::string_view to_string(BFSScratchpadStyle s) {
    switch (s) {
        case BFSScratchpadStyle::Plain:              return "plain";
        case BFSScratchpadStyle::WithQueue:          return "with_queue";
        case BFSScratchpadStyle::ReverseAdjacency:   return "reverse_adjacency";
        case BFSScratchpadStyle::DuplicateAdjacency: return "duplicate_adjacency";
    }
    return "<invalid>";
}

BFSScratchpadStyle bfs_scratchpad_style_from_string(std::string_view s) {
    if (s == "plain")               return BFSScratchpadStyle::Plain;
    if (s == "with_queue")          return BFSScratchpadStyle::WithQueue;
    if (s == "reverse_adjacency")   return BFSScratchpadStyle::ReverseAdjacency;
    if (s == "duplicate_adjacency") return BFSScratchpadStyle::DuplicateAdjacency;
    throw std::invalid_argument(
        "unknown bfs_scratchpad_style='" + std::string(s) +
        "' (expected one of: plain, with_queue, reverse_adjacency, duplicate_adjacency)");
}

}  // namespace graphgen
