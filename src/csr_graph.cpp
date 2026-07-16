#include "graphgen/csr_graph.h"

#include <algorithm>
#include <cstddef>
#include <utility>

#include "graphgen/check.h"

namespace graphgen {

CsrGraph CsrGraph::build(int n,
                         std::span<const Arc>   edges,
                         std::span<const float> weights,
                         bool mirror,
                         bool is_directed) {
    GG_CHECK(n >= 0, "CsrGraph: n must be non-negative");
    const bool has_weights = !weights.empty();
    if (has_weights) {
        GG_CHECK(weights.size() == edges.size(),
                 "CsrGraph: weights.size() must equal edges.size()");
    }
    for (const auto& e : edges) {
        GG_CHECK(e.source >= 0 && e.source < n,
                 "CsrGraph: edge source out of range [0, n)");
        GG_CHECK(e.target >= 0 && e.target < n,
                 "CsrGraph: edge target out of range [0, n)");
    }

    const std::size_t num_arcs = mirror ? edges.size() * 2 : edges.size();

    // Pass 1: per-source out-degrees, stored in row_offsets[u+1].
    std::vector<int> row_offsets(static_cast<std::size_t>(n) + 1, 0);
    for (const auto& e : edges) {
        ++row_offsets[static_cast<std::size_t>(e.source) + 1];
        if (mirror) ++row_offsets[static_cast<std::size_t>(e.target) + 1];
    }
    // Pass 2: prefix-sum -> start offsets.
    for (int u = 0; u < n; ++u) {
        row_offsets[static_cast<std::size_t>(u) + 1] +=
            row_offsets[static_cast<std::size_t>(u)];
    }

    // Pass 3: scatter arcs (and weights) using a cursor copy of row_offsets.
    std::vector<int>   col_indices(num_arcs);
    std::vector<float> arc_weights;
    if (has_weights) arc_weights.assign(num_arcs, 0.0f);

    std::vector<int> cursor(row_offsets.begin(), row_offsets.end() - 1);
    for (std::size_t i = 0; i < edges.size(); ++i) {
        const Arc e = edges[i];
        const int slot_uv = cursor[static_cast<std::size_t>(e.source)]++;
        col_indices[static_cast<std::size_t>(slot_uv)] = e.target;
        if (has_weights)
            arc_weights[static_cast<std::size_t>(slot_uv)] = weights[i];

        if (mirror) {
            const int slot_vu = cursor[static_cast<std::size_t>(e.target)]++;
            col_indices[static_cast<std::size_t>(slot_vu)] = e.source;
            if (has_weights)
                arc_weights[static_cast<std::size_t>(slot_vu)] = weights[i];
        }
    }

    CsrGraph g;
    g.n_           = n;
    g.is_directed_ = is_directed;
    g.row_offsets_ = std::move(row_offsets);
    g.col_indices_ = std::move(col_indices);
    g.weights_     = std::move(arc_weights);
    return g;
}

CsrGraph CsrGraph::from_directed_edges(int n,
                                       std::span<const Arc>   edges,
                                       std::span<const float> weights) {
    return build(n, edges, weights, /*mirror=*/false, /*is_directed=*/true);
}

CsrGraph CsrGraph::from_undirected_edges(int n,
                                         std::span<const Arc>   edges,
                                         std::span<const float> weights) {
    return build(n, edges, weights, /*mirror=*/true, /*is_directed=*/false);
}

bool CsrGraph::has_arc(int u, int v) const {
    for (int nbr : neighbours(u)) {
        if (nbr == v) return true;
    }
    return false;
}

// --- Range accessors -------------------------------------------------------

CsrGraph::NeighbourEdgeRange CsrGraph::weighted_neighbours(int u) const {
    return NeighbourEdgeRange{neighbours(u), neighbour_weights(u), has_weights()};
}

CsrGraph::ArcRange CsrGraph::arcs() const {
    return ArcRange{n_, row_offsets_.data(), col_indices_.data()};
}

CsrGraph::WeightedArcRange CsrGraph::weighted_arcs() const {
    return WeightedArcRange{n_, row_offsets_.data(), col_indices_.data(),
                            weights_.data(), has_weights()};
}

}  // namespace graphgen
