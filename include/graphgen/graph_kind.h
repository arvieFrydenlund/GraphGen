// Graph topology kind (which sampler runs for a batch).
//
// Kept as a small standalone header so both the sampler and any downstream
// module (task computer, tokenizer, tests) can name kinds without pulling
// the sampler's transitive includes.

#ifndef GRAPHGEN_GRAPH_KIND_H
#define GRAPHGEN_GRAPH_KIND_H

#include <string>
#include <string_view>

namespace graphgen {

enum class GraphKind {
    ErdosRenyi,
    Euclidean,
    RandomTree,
    PathStar,
    Balanced,
    Khops,
    KhopsGen,
};

// String form used on GeneratorConfig.graph_kind and in error messages.
std::string_view to_string(GraphKind k);

// Throws std::invalid_argument on unknown names; the message includes the
// offending string so the caller can surface it verbatim.
GraphKind graph_kind_from_string(std::string_view s);

}  // namespace graphgen

#endif  // GRAPHGEN_GRAPH_KIND_H
