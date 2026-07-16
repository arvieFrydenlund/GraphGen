// Compressed Sparse Row (CSR) graph.
//
// Layout (all vertex IDs are internal, 0..n-1):
//
//   col_indices  flat array of arc targets, grouped by source vertex.
//                Size = num_arcs (== 2*m for undirected, m for directed).
//   row_offsets  size n+1. Arcs out of vertex u live at the half-open
//                slice col_indices[row_offsets[u], row_offsets[u+1]).
//   weights      parallel to col_indices when the graph is weighted;
//                empty vector when unweighted.
//
// Why CSR (vs vector<vector<int>>):
//   - Two heap allocations for the whole graph vs n+1 -- matters when we
//     build millions of graphs per epoch.
//   - Contiguous storage -> the CPU prefetcher wins traversal; BFS on
//     small graphs is measurably faster.
//   - Trivially fork-safe for PyTorch DataLoader workers.
//   - Zero-copy interop with SciPy sparse / numpy on the Python side.
//
// Why std::span in the accessors:
//   - std::span<T> is a non-owning (pointer, length) pair -- a lightweight
//     view over the caller's memory, not a copy or a container.
//   - neighbours(u) returns a span pointing straight into col_indices at
//     row_offsets[u], with length degree(u). No copy, no allocation.
//   - Callers iterate it like any container:
//         for (int v : g.neighbours(u)) { ... }
//     with .size(), .begin()/.end(), and [] working as expected.
//
// Callers should NEVER touch row_offsets_ / col_indices_ directly. All
// offset math lives in this file. Use the named accessors below; if you
// need an iteration shape that isn't provided, add a wrapper here rather
// than reach into the raw arrays at the call site.

#ifndef GRAPHGEN_CSR_GRAPH_H
#define GRAPHGEN_CSR_GRAPH_H

#include <cstddef>
#include <span>
#include <vector>

namespace graphgen {

// (target, weight) pair yielded by weighted-neighbour iteration.
// weight is 1.0f whenever the underlying graph is unweighted, so
// weighted-algorithm loops work uniformly on both.
struct NeighbourEdge {
    int   target;
    float weight;
};

// (source, target) pair yielded by unweighted arc iteration.
struct Arc {
    int source;
    int target;
};

// (source, target, weight) yielded by weighted arc iteration.
struct WeightedArc {
    int   source;
    int   target;
    float weight;
};

class CsrGraph {
public:
    // ---------- Construction ---------------------------------------------
    // Directed: each Arc becomes exactly one arc. If `weights` is provided
    // it must satisfy weights.size() == edges.size(); it is stored parallel
    // to the arcs.
    static CsrGraph from_directed_edges(
        int n,
        std::span<const Arc>   edges,
        std::span<const float> weights = {});

    // Undirected: each input Arc becomes two arcs (u->v and v->u). Weights
    // are mirrored across both orientations.
    static CsrGraph from_undirected_edges(
        int n,
        std::span<const Arc>   edges,
        std::span<const float> weights = {});

    // ---------- Sizes ----------------------------------------------------
    int  num_vertices() const { return n_; }
    int  num_arcs()     const { return static_cast<int>(col_indices_.size()); }
    int  num_edges()    const { return is_directed_ ? num_arcs() : num_arcs() / 2; }
    bool is_directed()  const { return is_directed_; }
    bool has_weights()  const { return !weights_.empty(); }

    // ---------- Per-vertex access ----------------------------------------
    int degree(int u) const {
        return row_offsets_[u + 1] - row_offsets_[u];
    }

    // for (int v : g.neighbours(u)) { ... }
    std::span<const int> neighbours(int u) const {
        return std::span<const int>(
            col_indices_.data() + row_offsets_[u],
            static_cast<std::size_t>(degree(u)));
    }

    // Parallel to neighbours(u); empty span when unweighted.
    // Prefer weighted_neighbours(u) for zipped iteration.
    std::span<const float> neighbour_weights(int u) const {
        if (weights_.empty()) return {};
        return std::span<const float>(
            weights_.data() + row_offsets_[u],
            static_cast<std::size_t>(degree(u)));
    }

    // ---------- Zipped iteration (no offset math at the call site) -------
    class NeighbourEdgeRange;
    class ArcRange;
    class WeightedArcRange;

    // for (auto [v, w] : g.weighted_neighbours(u)) { ... }
    // Weight is 1.0f when the graph is unweighted so weighted algorithm
    // bodies compile and behave correctly on both.
    NeighbourEdgeRange weighted_neighbours(int u) const;

    // for (auto [u, v] : g.arcs()) { ... }
    // In an undirected graph each edge appears twice (once per orientation).
    ArcRange arcs() const;

    // for (auto [u, v, w] : g.weighted_arcs()) { ... }
    WeightedArcRange weighted_arcs() const;

    // ---------- Queries --------------------------------------------------
    // O(degree(u)). Fine for sanity checks; not for hot loops.
    bool has_arc(int u, int v) const;

private:
    int  n_            = 0;
    bool is_directed_  = false;
    std::vector<int>   row_offsets_;
    std::vector<int>   col_indices_;
    std::vector<float> weights_;

    // Shared backend for from_directed_edges / from_undirected_edges.
    // When `mirror` is true each input arc yields both orientations.
    static CsrGraph build(int n,
                          std::span<const Arc>   edges,
                          std::span<const float> weights,
                          bool mirror,
                          bool is_directed);
};

// ---------- Iteration ranges (thin wrappers) ---------------------------

class CsrGraph::NeighbourEdgeRange {
public:
    class Iterator {
    public:
        NeighbourEdge operator*() const {
            return NeighbourEdge{
                *target_,
                has_weights_ ? *weight_ : 1.0f,
            };
        }
        Iterator& operator++() {
            ++target_;
            if (has_weights_) ++weight_;
            return *this;
        }
        bool operator!=(const Iterator& other) const { return target_ != other.target_; }

        const int* target_;
        const float* weight_;
        bool has_weights_;
    };

    Iterator begin() const { return {targets_.data(), weights_.data(), has_weights_}; }
    Iterator end()   const {
        return {targets_.data() + targets_.size(),
                weights_.data() + (has_weights_ ? weights_.size() : 0),
                has_weights_};
    }

    std::span<const int>   targets_;
    std::span<const float> weights_;
    bool                   has_weights_;
};

class CsrGraph::ArcRange {
public:
    class Iterator {
    public:
        Arc operator*() const { return Arc{u_, col_indices_[i_]}; }
        Iterator& operator++() {
            ++i_;
            // Advance source vertex when we've walked past its arcs.
            while (u_ + 1 < n_ && i_ >= row_offsets_[u_ + 1]) ++u_;
            return *this;
        }
        bool operator!=(const Iterator& other) const { return i_ != other.i_; }

        int i_;
        int u_;
        int n_;
        const int* row_offsets_;
        const int* col_indices_;
    };

    Iterator begin() const {
        int u = 0;
        while (u + 1 < n_ && row_offsets_[u] == row_offsets_[u + 1]) ++u;  // skip isolated vertices
        return {row_offsets_[0], u, n_, row_offsets_, col_indices_};
    }
    Iterator end() const {
        return {row_offsets_[n_], n_ - 1, n_, row_offsets_, col_indices_};
    }

    int n_;
    const int* row_offsets_;
    const int* col_indices_;
};

class CsrGraph::WeightedArcRange {
public:
    class Iterator {
    public:
        WeightedArc operator*() const {
            return WeightedArc{
                u_,
                col_indices_[i_],
                has_weights_ ? weights_[i_] : 1.0f,
            };
        }
        Iterator& operator++() {
            ++i_;
            while (u_ + 1 < n_ && i_ >= row_offsets_[u_ + 1]) ++u_;
            return *this;
        }
        bool operator!=(const Iterator& other) const { return i_ != other.i_; }

        int i_;
        int u_;
        int n_;
        const int* row_offsets_;
        const int* col_indices_;
        const float* weights_;
        bool has_weights_;
    };

    Iterator begin() const {
        int u = 0;
        while (u + 1 < n_ && row_offsets_[u] == row_offsets_[u + 1]) ++u;
        return {row_offsets_[0], u, n_, row_offsets_, col_indices_, weights_, has_weights_};
    }
    Iterator end() const {
        return {row_offsets_[n_], n_ - 1, n_, row_offsets_, col_indices_, weights_, has_weights_};
    }

    int n_;
    const int* row_offsets_;
    const int* col_indices_;
    const float* weights_;
    bool has_weights_;
};

}  // namespace graphgen

#endif  // GRAPHGEN_CSR_GRAPH_H
