#include "graphgen/csr_graph.h"

#include <algorithm>
#include <cstddef>
#include <utility>

#include "graphgen/check.h"

namespace graphgen {

CsrGraph CsrGraph::build(int n,
                         std::span<const Edge>  edges,
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
        GG_CHECK(e.u >= 0 && e.u < n,
                 "CsrGraph: edge endpoint u out of range [0, n)");
        GG_CHECK(e.v >= 0 && e.v < n,
                 "CsrGraph: edge endpoint v out of range [0, n)");
    }

    const std::size_t num_slots = mirror ? edges.size() * 2 : edges.size();

    // Pass 1: per-source out-degrees, stored in row_offsets[u+1].
    std::vector<int> row_offsets(static_cast<std::size_t>(n) + 1, 0);
    for (const auto& e : edges) {
        ++row_offsets[static_cast<std::size_t>(e.u) + 1];
        if (mirror) ++row_offsets[static_cast<std::size_t>(e.v) + 1];
    }
    // Pass 2: prefix-sum -> start offsets.
    for (int u = 0; u < n; ++u) {
        row_offsets[static_cast<std::size_t>(u) + 1] +=
            row_offsets[static_cast<std::size_t>(u)];
    }

    // Pass 3: scatter edges (and weights) using a cursor copy of
    // row_offsets. Preserves input order within each row -- if the
    // caller shuffled `edges` before calling build(), the shuffle
    // shows up as randomised within-row neighbour order.
    std::vector<int>   col_indices(num_slots);
    std::vector<float> edge_weights;
    if (has_weights) edge_weights.assign(num_slots, 0.0f);

    std::vector<int> cursor(row_offsets.begin(), row_offsets.end() - 1);
    for (std::size_t i = 0; i < edges.size(); ++i) {
        const Edge e = edges[i];
        const int slot_uv = cursor[static_cast<std::size_t>(e.u)]++;
        col_indices[static_cast<std::size_t>(slot_uv)] = e.v;
        if (has_weights)
            edge_weights[static_cast<std::size_t>(slot_uv)] = weights[i];

        if (mirror) {
            const int slot_vu = cursor[static_cast<std::size_t>(e.v)]++;
            col_indices[static_cast<std::size_t>(slot_vu)] = e.u;
            if (has_weights)
                edge_weights[static_cast<std::size_t>(slot_vu)] = weights[i];
        }
    }

    CsrGraph g;
    g.n_           = n;
    g.is_directed_ = is_directed;
    g.row_offsets_ = std::move(row_offsets);
    g.col_indices_ = std::move(col_indices);
    g.weights_     = std::move(edge_weights);
    return g;
}

CsrGraph CsrGraph::from_directed_edges(int n,
                                       std::span<const Edge>  edges,
                                       std::span<const float> weights) {
    return build(n, edges, weights, /*mirror=*/false, /*is_directed=*/true);
}

CsrGraph CsrGraph::from_undirected_edges(int n,
                                         std::span<const Edge>  edges,
                                         std::span<const float> weights) {
    return build(n, edges, weights, /*mirror=*/true, /*is_directed=*/false);
}

bool CsrGraph::has_edge(int u, int v) const {
    for (int nbr : neighbours(u)) {
        if (nbr == v) return true;
    }
    return false;
}

// --- Graph algorithms ------------------------------------------------------

std::vector<int> CsrGraph::bfs(int source) const {
    GG_CHECK(source >= 0 && source < n_,
             "CsrGraph::bfs: source out of range [0, num_vertices())");

    std::vector<int> dist(static_cast<std::size_t>(n_), -1);
    if (n_ == 0) return dist;

    // Two ping-pong buffers instead of std::queue: contiguous memory and
    // no per-push allocation. current/next hold vertices to expand at
    // the current and next BFS layer respectively.
    std::vector<int> current;
    std::vector<int> next;
    current.reserve(static_cast<std::size_t>(n_));
    next.reserve(static_cast<std::size_t>(n_));

    dist[static_cast<std::size_t>(source)] = 0;
    current.push_back(source);

    int layer = 0;
    while (!current.empty()) {
        for (int u : current) {
            for (int v : neighbours(u)) {
                if (dist[static_cast<std::size_t>(v)] == -1) {
                    dist[static_cast<std::size_t>(v)] = layer + 1;
                    next.push_back(v);
                }
            }
        }
        ++layer;
        current.swap(next);
        next.clear();
    }
    return dist;
}

std::vector<int> CsrGraph::distance_bounded_bfs(int source, int max_dist) const {
    GG_CHECK(source >= 0 && source < n_,
             "CsrGraph::distance_bounded_bfs: source out of range [0, num_vertices())");
    GG_CHECK(max_dist >= 0,
             "CsrGraph::distance_bounded_bfs: max_dist must be >= 0");

    std::vector<int> dist(static_cast<std::size_t>(n_), -1);
    if (n_ == 0) return dist;

    std::vector<int> current;
    std::vector<int> next;
    current.reserve(static_cast<std::size_t>(n_));
    next.reserve(static_cast<std::size_t>(n_));

    dist[static_cast<std::size_t>(source)] = 0;
    if (max_dist == 0) return dist;
    current.push_back(source);

    int layer = 0;
    while (!current.empty() && layer < max_dist) {
        for (int u : current) {
            for (int v : neighbours(u)) {
                if (dist[static_cast<std::size_t>(v)] == -1) {
                    dist[static_cast<std::size_t>(v)] = layer + 1;
                    next.push_back(v);
                }
            }
        }
        ++layer;
        current.swap(next);
        next.clear();
    }
    return dist;
}

std::vector<int> CsrGraph::connected_components() const {
    std::vector<int> comp(static_cast<std::size_t>(n_), -1);
    if (n_ == 0) return comp;

    std::vector<int> stack;
    stack.reserve(static_cast<std::size_t>(n_));
    int next_id = 0;

    for (int start = 0; start < n_; ++start) {
        if (comp[static_cast<std::size_t>(start)] != -1) continue;

        // Flood-fill from `start`. Iterative DFS with an explicit stack
        // avoids blowing the C stack on pathologically deep graphs and
        // reuses the scratch buffer across components.
        comp[static_cast<std::size_t>(start)] = next_id;
        stack.push_back(start);
        while (!stack.empty()) {
            const int u = stack.back();
            stack.pop_back();
            for (int v : neighbours(u)) {
                if (comp[static_cast<std::size_t>(v)] == -1) {
                    comp[static_cast<std::size_t>(v)] = next_id;
                    stack.push_back(v);
                }
            }
        }
        ++next_id;
    }
    return comp;
}

// --- Range accessors -------------------------------------------------------

CsrGraph::NeighbourEdgeRange CsrGraph::weighted_neighbours(int u) const {
    return NeighbourEdgeRange{neighbours(u), neighbour_weights(u), has_weights()};
}

CsrGraph::EdgeRange CsrGraph::edges() const {
    return EdgeRange{n_, row_offsets_.data(), col_indices_.data(), is_directed_};
}

CsrGraph::WeightedEdgeRange CsrGraph::weighted_edges() const {
    return WeightedEdgeRange{n_, row_offsets_.data(), col_indices_.data(),
                             weights_.data(), has_weights(), is_directed_};
}

}  // namespace graphgen
