#include "graphgen/graph_kind.h"

#include <stdexcept>
#include <string>

namespace graphgen {

std::string_view to_string(GraphKind k) {
    switch (k) {
        case GraphKind::ErdosRenyi: return "erdos_renyi";
        case GraphKind::Euclidean:  return "euclidean";
        case GraphKind::RandomTree: return "random_tree";
        case GraphKind::PathStar:   return "path_star";
        case GraphKind::Balanced:   return "balanced";
        case GraphKind::Khops:      return "khops";
        case GraphKind::KhopsGen:   return "khops_gen";
    }
    return "<invalid>";
}

GraphKind graph_kind_from_string(std::string_view s) {
    if (s == "erdos_renyi") return GraphKind::ErdosRenyi;
    if (s == "euclidean")   return GraphKind::Euclidean;
    if (s == "random_tree") return GraphKind::RandomTree;
    if (s == "path_star")   return GraphKind::PathStar;
    if (s == "balanced")    return GraphKind::Balanced;
    if (s == "khops")       return GraphKind::Khops;
    if (s == "khops_gen")   return GraphKind::KhopsGen;
    throw std::invalid_argument("unknown graph_kind='" + std::string(s) + "'");
}

}  // namespace graphgen
