#include "graphgen/task.h"

#include <stdexcept>
#include <string>

namespace graphgen {

std::string_view to_string(TaskKind t) {
    switch (t) {
        case TaskKind::None:         return "none";
        case TaskKind::ShortestPath: return "shortest_path";
        case TaskKind::BFS:          return "bfs";
        case TaskKind::Center:       return "center";
        case TaskKind::Centroid:     return "centroid";
        case TaskKind::Khops:        return "khops";
        case TaskKind::KhopsGen:     return "khops_gen";
    }
    return "<invalid>";
}

TaskKind task_kind_from_string(std::string_view s) {
    if (s == "none")          return TaskKind::None;
    if (s == "shortest_path") return TaskKind::ShortestPath;
    if (s == "bfs")           return TaskKind::BFS;
    if (s == "center")        return TaskKind::Center;
    if (s == "centroid")      return TaskKind::Centroid;
    if (s == "khops")         return TaskKind::Khops;
    if (s == "khops_gen")     return TaskKind::KhopsGen;
    throw std::invalid_argument("unknown task_kind='" + std::string(s) + "'");
}

}  // namespace graphgen
