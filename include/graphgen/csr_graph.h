// Compressed Sparse Row (CSR) graph.
//
// Layout (all vertex IDs are internal, 0..n-1):
//
//   col_indices  flat array of neighbour vertex IDs, grouped by source
//                vertex. For an undirected graph each edge {u, v} is
//                stored twice (once in u's slice, once in v's) -- the
//                CSR row is inherently a per-vertex adjacency list.
//   row_offsets  size n+1. Neighbours of vertex u live at the half-open
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

// Endpoint pair used both as input to the CsrGraph constructors and as
// the yield type of edges(). The graph itself carries the directedness
// flag, so this struct does not commit to a direction: on a directed
// graph a (u, v) Edge means "directed edge from u to v"; on an
// undirected graph it means "connection between u and v", and edges()
// yields each such pair exactly once in canonical order (u < v).
struct Edge {
    int u;
    int v;
};

// Same shape as Edge plus a weight.
struct WeightedEdge {
    int   u;
    int   v;
    float weight;
};

class CsrGraph {
public:
    // ---------- Construction ---------------------------------------------
    // Each input Edge {u, v} becomes one stored directed edge u -> v. If
    // `weights` is provided it must satisfy weights.size() == edges.size();
    // it is stored parallel to the edges.
    static CsrGraph from_directed_edges(
        int n,
        std::span<const Edge>  edges,
        std::span<const float> weights = {});

    // Each input Edge {u, v} becomes two stored directed edges
    // (u -> v and v -> u) so per-vertex iteration works from either
    // endpoint. Weights are mirrored across both orientations.
    static CsrGraph from_undirected_edges(
        int n,
        std::span<const Edge>  edges,
        std::span<const float> weights = {});

    // ---------- Sizes ----------------------------------------------------
    int  num_vertices() const { return n_; }
    // Number of logical edges: for directed, one per input Edge; for
    // undirected, one per input Edge (i.e. half the internal storage
    // slot count, which is not exposed).
    int  num_edges() const {
        const int slots = static_cast<int>(col_indices_.size());
        return is_directed_ ? slots : slots / 2;
    }
    bool is_directed() const { return is_directed_; }
    bool has_weights() const { return !weights_.empty(); }

    // ---------- Per-vertex access ----------------------------------------
    int degree(int u) const {
        return row_offsets_[u + 1] - row_offsets_[u];
    }

    // for (int v : g.neighbours(u)) { ... }
    // On undirected graphs both orientations of each incident edge are
    // present here -- that is the whole point of iterating out of `u`.
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
    class EdgeRange;
    class WeightedEdgeRange;

    // for (auto [v, w] : g.weighted_neighbours(u)) { ... }
    // Weight is 1.0f when the graph is unweighted so weighted algorithm
    // bodies compile and behave correctly on both.
    NeighbourEdgeRange weighted_neighbours(int u) const;

    // for (auto [u, v] : g.edges()) { ... }
    // Yields each logical edge exactly once. Directed: all stored
    // edges. Undirected: only the (u, v) slot with u < v (dedup of
    // the mirror). Self-loops on undirected graphs are not currently
    // supported and would be skipped by this iterator.
    EdgeRange edges() const;

    // for (auto [u, v, w] : g.weighted_edges()) { ... }
    // Same dedup rule as edges().
    WeightedEdgeRange weighted_edges() const;

    // ---------- Queries --------------------------------------------------
    // Is v in u's neighbour list? On undirected graphs this is symmetric
    // (both has_edge(u, v) and has_edge(v, u) hold when the edge exists).
    // O(degree(u)); fine for sanity checks, not for hot loops.
    bool has_edge(int u, int v) const;

    // ---------- Graph algorithms -----------------------------------------
    // Breadth-first shortest hop distances from `source`. Returns a
    // vector of size num_vertices() where entry v is the number of edges
    // on the shortest source->v path, or -1 if v is unreachable. Ignores
    // edge weights (see the plan for Dijkstra when we add weights).
    std::vector<int> bfs(int source) const;

    // Like bfs(source) but stops expanding beyond `max_dist`: vertices
    // strictly farther than max_dist stay -1 in the result. Useful for
    // k-hop task setups that don't need the full distance field.
    std::vector<int> distance_bounded_bfs(int source, int max_dist) const;

    // Connected-component id per vertex, dense in [0, K) where K is the
    // number of components. For a directed graph edges are treated as
    // undirected (weakly connected components). K = 0 for a graph with
    // zero vertices; otherwise K = max(result) + 1.
    std::vector<int> connected_components() const;

private:
    int  n_            = 0;
    bool is_directed_  = false;
    std::vector<int>   row_offsets_;
    std::vector<int>   col_indices_;
    std::vector<float> weights_;

    // Shared backend for from_directed_edges / from_undirected_edges.
    // When `mirror` is true each input edge yields both orientations
    // in storage.
    static CsrGraph build(int n,
                          std::span<const Edge>  edges,
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

class CsrGraph::EdgeRange {
public:
    class Iterator {
    public:
        Edge operator*() const { return Edge{u_, col_indices_[i_]}; }
        Iterator& operator++() {
            ++i_;
            advance_to_valid();
            return *this;
        }
        bool operator!=(const Iterator& other) const { return i_ != other.i_; }

        // Move (u_, i_) to the next slot that this range should emit.
        // Bumps u_ past empty rows; on undirected graphs, skips mirror
        // slots (those where u_ > col_indices_[i_]) so each edge is
        // yielded only once in canonical (u < v) form.
        void advance_to_valid() {
            const int end = row_offsets_[n_];
            while (i_ < end) {
                while (u_ + 1 < n_ && i_ >= row_offsets_[u_ + 1]) ++u_;
                if (is_directed_ || u_ < col_indices_[i_]) return;
                ++i_;
            }
        }

        int i_;
        int u_;
        int n_;
        const int* row_offsets_;
        const int* col_indices_;
        bool is_directed_;
    };

    Iterator begin() const {
        Iterator it{0, 0, n_, row_offsets_, col_indices_, is_directed_};
        it.advance_to_valid();
        return it;
    }
    Iterator end() const {
        return {row_offsets_[n_], n_ - 1, n_, row_offsets_, col_indices_, is_directed_};
    }

    int n_;
    const int* row_offsets_;
    const int* col_indices_;
    bool is_directed_;
};

class CsrGraph::WeightedEdgeRange {
public:
    class Iterator {
    public:
        WeightedEdge operator*() const {
            return WeightedEdge{
                u_,
                col_indices_[i_],
                has_weights_ ? weights_[i_] : 1.0f,
            };
        }
        Iterator& operator++() {
            ++i_;
            advance_to_valid();
            return *this;
        }
        bool operator!=(const Iterator& other) const { return i_ != other.i_; }

        void advance_to_valid() {
            const int end = row_offsets_[n_];
            while (i_ < end) {
                while (u_ + 1 < n_ && i_ >= row_offsets_[u_ + 1]) ++u_;
                if (is_directed_ || u_ < col_indices_[i_]) return;
                ++i_;
            }
        }

        int i_;
        int u_;
        int n_;
        const int* row_offsets_;
        const int* col_indices_;
        const float* weights_;
        bool has_weights_;
        bool is_directed_;
    };

    Iterator begin() const {
        Iterator it{0, 0, n_, row_offsets_, col_indices_, weights_, has_weights_, is_directed_};
        it.advance_to_valid();
        return it;
    }
    Iterator end() const {
        return {row_offsets_[n_], n_ - 1, n_, row_offsets_, col_indices_,
                weights_, has_weights_, is_directed_};
    }

    int n_;
    const int* row_offsets_;
    const int* col_indices_;
    const float* weights_;
    bool has_weights_;
    bool is_directed_;
};

}  // namespace graphgen

#endif  // GRAPHGEN_CSR_GRAPH_H
