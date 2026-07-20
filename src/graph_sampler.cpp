#include "graphgen/graph_sampler.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "graphgen/check.h"
#include "graphgen/csr_graph.h"
#include "graphgen/generator_config.h"
#include "graphgen/random_utils.h"

namespace graphgen {

SampledGraph GraphSampler::run(GraphKind kind,
                               std::mt19937_64& rng,
                               const GeneratorConfig& cfg) {
    // Pure topology. The Worker assigns `sg.internal_to_vocab` after
    // this call using the dictionary-derived node-vocab range on the
    // ctx -- the sampler doesn't need to know about vocab layout to
    // build a graph. Callers that don't go through Worker (raw
    // sampler unit tests) can leave `internal_to_vocab` empty; only
    // the tokenizer requires it populated.
    switch (kind) {
        case GraphKind::ErdosRenyi: return sample_erdos_renyi(rng, cfg);
        case GraphKind::Euclidean:  return sample_euclidean  (rng, cfg);
        case GraphKind::RandomTree: return sample_random_tree(rng, cfg);
        case GraphKind::PathStar:   return sample_path_star  (rng, cfg);
        case GraphKind::Balanced:   return sample_balanced   (rng, cfg);
        case GraphKind::Khops:      return sample_khops      (rng, cfg);
        case GraphKind::KhopsGen:   return sample_khops_gen  (rng, cfg);
    }
    GG_CHECK(false, "GraphSampler::run: unreachable -- unknown GraphKind");
}

SampledGraph GraphSampler::sample_erdos_renyi(std::mt19937_64& rng,
                                              const GeneratorConfig& cfg) {
    GG_CHECK(cfg.min_num_nodes > 0,
             "sample_erdos_renyi: min_num_nodes must be > 0 (got "
                 << cfg.min_num_nodes << ")");
    GG_CHECK(cfg.edge_prob.has_value(),
             "sample_erdos_renyi: edge_prob must be set");
    const float p = *cfg.edge_prob;
    GG_CHECK(p >= 0.0f && p <= 1.0f,
             "sample_erdos_renyi: edge_prob must be in [0, 1] (got " << p << ")");

    // Sample n uniformly from [min_num_nodes, max_num_nodes]. A negative
    // max means "use min only" -- pin the range to a single value.
    const int lo = cfg.min_num_nodes;
    const int hi = (cfg.max_num_nodes > 0 ? cfg.max_num_nodes : lo) + 1;
    IntSampler n_sampler(lo, hi);
    const int n = n_sampler(rng);

    // G(n, p): iterate every candidate edge and keep it with
    // probability p. Undirected considers the n*(n-1)/2 unordered
    // pairs; directed considers all n*(n-1) ordered pairs (each
    // direction is an independent Bernoulli trial, so both u->v
    // AND v->u can be present). O(n^2) in both modes; swap in
    // Batagelj-Brandes geometric-skip sampling if this ever shows
    // up in a profile at large n.
    std::vector<Edge> edges;
    std::bernoulli_distribution keep(p);
    if (cfg.directed) {
        for (int u = 0; u < n; ++u) {
            for (int v = 0; v < n; ++v) {
                if (u == v) continue;
                if (keep(rng)) edges.push_back({u, v});
            }
        }
    } else {
        for (int u = 0; u < n; ++u) {
            for (int v = u + 1; v < n; ++v) {
                if (keep(rng)) edges.push_back({u, v});
            }
        }
    }

    // Randomise input order before build. See GraphSampler::shuffle_edges.
    shuffle_edges(edges, rng);

    SampledGraph sg;
    sg.graph             = cfg.directed
        ? CsrGraph::from_directed_edges(n, edges)
        : CsrGraph::from_undirected_edges(n, edges);
    sg.distance_semantic = DistanceSemantic::HopCount;
    return sg;
}
SampledGraph GraphSampler::sample_euclidean(std::mt19937_64& rng,
                                            const GeneratorConfig& cfg) {
    // Random geometric graph on the unit dim-cube. Each of n vertices
    // draws an independent uniform position in [0, 1]^dim; an edge is
    // added iff the Euclidean distance falls in [min_edge_length,
    // max_edge_length]. Undirected only (distances are symmetric).
    //
    // Defaults match the classic RGG literature:
    //   * dim              = 2
    //   * min_edge_length  = 0.0   (any short edge is accepted)
    //   * max_edge_length  = 1/sqrt(n)  -- the connectivity threshold
    //                        for RGG in the unit square; ~O(1) expected
    //                        degree independent of n.
    //
    // Positions are consumed here and not surfaced to Python. If the
    // caller ever needs them for plotting or downstream evaluation,
    // add a return_positions cfg flag mirroring return_hop_distances
    // (allocate a (B, max_n, dim) float64 tensor, attach per-row
    // views, adopt into BatchOutputArrays).
    GG_CHECK(cfg.min_num_nodes > 0,
             "sample_euclidean: min_num_nodes must be > 0 (got "
                 << cfg.min_num_nodes << ")");
    GG_CHECK(!cfg.directed,
             "sample_euclidean: directed=true is not supported "
             "(Euclidean distance is symmetric)");
    GG_CHECK(!cfg.dims.has_value(),
             "sample_euclidean: cfg.dims is reserved for a future "
             "per-dimension extent API; use cfg.dim for the dimension "
             "count on the unit cube");

    const int dim = cfg.dim.value_or(2);
    GG_CHECK(dim >= 1,
             "sample_euclidean: dim must be >= 1 (got " << dim << ")");

    // Sample n uniformly from [min_num_nodes, max_num_nodes]. Matches
    // sample_erdos_renyi's convention: a negative or zero max pins
    // the range to `min_num_nodes` exactly.
    const int lo = cfg.min_num_nodes;
    const int hi = (cfg.max_num_nodes > 0 ? cfg.max_num_nodes : lo) + 1;
    IntSampler n_sampler(lo, hi);
    const int n = n_sampler(rng);
    GG_CHECK(n >= 1,
             "sample_euclidean: sampled n must be >= 1 (got " << n << ")");

    const double e_min = cfg.min_edge_length.value_or(0.0);
    const double e_max = cfg.max_edge_length.value_or(1.0 / std::sqrt(
                             static_cast<double>(n)));
    GG_CHECK(e_min >= 0.0,
             "sample_euclidean: min_edge_length must be >= 0 (got "
                 << e_min << ")");
    GG_CHECK(e_max >= e_min,
             "sample_euclidean: max_edge_length ("
                 << e_max << ") must be >= min_edge_length (" << e_min << ")");

    // Positions: flat vector<double> of length n*dim, indexed as
    // positions[u*dim + k]. Contiguous layout keeps the pairwise
    // distance loop cache-friendly and is the pattern the future
    // return_positions tensor would use verbatim.
    std::vector<double> positions(static_cast<std::size_t>(n) *
                                  static_cast<std::size_t>(dim));
    std::uniform_real_distribution<double> coord(0.0, 1.0);
    for (std::size_t i = 0; i < positions.size(); ++i) {
        positions[i] = coord(rng);
    }

    // Compare squared distances against the squared band bounds; only
    // pay for sqrt on the final radius-band bounds, not on every
    // pair. Same acceptance region (dist^2 is monotone in dist for
    // dist >= 0), so no numerical drift versus the naive sqrt-per-pair
    // form apart from the constant e_min^2 / e_max^2.
    const double e_min_sq = e_min * e_min;
    const double e_max_sq = e_max * e_max;

    std::vector<Edge> edges;
    // Rough capacity hint: expected O(n * pi * e_max^2) edges in
    // dim=2. Reserve conservatively to avoid pathological growth.
    edges.reserve(static_cast<std::size_t>(std::max(n, 8)));

    for (int u = 0; u < n; ++u) {
        const double* pu = &positions[static_cast<std::size_t>(u) *
                                      static_cast<std::size_t>(dim)];
        for (int v = u + 1; v < n; ++v) {
            const double* pv = &positions[static_cast<std::size_t>(v) *
                                          static_cast<std::size_t>(dim)];
            double d2 = 0.0;
            for (int k = 0; k < dim; ++k) {
                const double delta = pu[k] - pv[k];
                d2 += delta * delta;
            }
            if (d2 >= e_min_sq && d2 <= e_max_sq) {
                edges.push_back({u, v});
            }
        }
    }

    // Randomise input order before build. See GraphSampler::shuffle_edges.
    shuffle_edges(edges, rng);

    SampledGraph sg;
    sg.graph             = CsrGraph::from_undirected_edges(n, edges);
    sg.distance_semantic = DistanceSemantic::HopCount;
    // Retain the flat (n, dim) position vector on the graph. The
    // worker copies this into the batch tensor when
    // cfg.return_positions is set; otherwise the pipeline drops it
    // on the floor when the SampledGraph goes out of scope.
    sg.positions         = std::move(positions);
    sg.dim               = dim;
    return sg;
}
SampledGraph GraphSampler::sample_random_tree(std::mt19937_64&, const GeneratorConfig&) {
    GG_CHECK(false, "sample_random_tree not implemented");
}
SampledGraph GraphSampler::sample_path_star(std::mt19937_64& rng,
                                            const GeneratorConfig& cfg) {
    // Rooted directed tree ("star of paths"). Vertex 0 is the root;
    // `num_arms` directed chains ("arms") emanate from it. Each arm
    // has an independently sampled length in [min_arm_length,
    // max_arm_length], so a single graph typically contains arms of
    // several different lengths -- an intentional signal for the
    // shortest_path task, which learns to walk down each arm the
    // right distance.
    //
    // Vertex numbering (construction order):
    //   0                             -- root
    //   1 .. L_0                      -- arm 0 (root -> 1 -> 2 -> ... -> L_0)
    //   L_0 + 1 .. L_0 + L_1          -- arm 1
    //   ...
    // Total n = 1 + sum_i L_i.
    //
    // Directed only: an undirected star would collapse into a hub
    // with equidistant leaves and lose the sequential-reasoning
    // signal that makes path_star useful. Mirror sample_euclidean's
    // opinionated stance: reject cfg.directed=false at runtime rather
    // than silently building the wrong graph.
    GG_CHECK(cfg.directed,
             "sample_path_star: requires directed=true (path_star is a "
             "rooted directed tree; undirected would collapse into a "
             "hub with equidistant leaves)");
    GG_CHECK(cfg.min_arms.has_value() && cfg.max_arms.has_value(),
             "sample_path_star: cfg.min_arms and cfg.max_arms must both be set");
    GG_CHECK(cfg.min_arm_length.has_value() && cfg.max_arm_length.has_value(),
             "sample_path_star: cfg.min_arm_length and cfg.max_arm_length "
             "must both be set");

    const int min_arms       = *cfg.min_arms;
    const int max_arms       = *cfg.max_arms;
    const int min_arm_length = *cfg.min_arm_length;
    const int max_arm_length = *cfg.max_arm_length;

    GG_CHECK(min_arms >= 1,
             "sample_path_star: min_arms must be >= 1 (got "
                 << min_arms << ")");
    GG_CHECK(max_arms >= min_arms,
             "sample_path_star: max_arms (" << max_arms
                 << ") must be >= min_arms (" << min_arms << ")");
    GG_CHECK(min_arm_length >= 1,
             "sample_path_star: min_arm_length must be >= 1 (got "
                 << min_arm_length << ")");
    GG_CHECK(max_arm_length >= min_arm_length,
             "sample_path_star: max_arm_length (" << max_arm_length
                 << ") must be >= min_arm_length (" << min_arm_length << ")");

    // IntSampler is upper-exclusive; +1 to include the upper bound.
    IntSampler arms_sampler(min_arms, max_arms + 1);
    IntSampler arm_len_sampler(min_arm_length, max_arm_length + 1);

    const int num_arms = arms_sampler(rng);

    // Build the edge list in one pass. Vertex numbering is implicit:
    // 0 = root, then one contiguous chunk per arm. `next_vertex`
    // tracks the next free id; each arm chains `prev -> cur` edges
    // starting from the root.
    std::vector<Edge> edges;
    edges.reserve(static_cast<std::size_t>(num_arms) *
                  static_cast<std::size_t>(max_arm_length));

    int next_vertex = 1;   // 0 is the root
    for (int a = 0; a < num_arms; ++a) {
        const int arm_length = arm_len_sampler(rng);
        int prev = 0;
        for (int j = 0; j < arm_length; ++j) {
            const int cur = next_vertex++;
            edges.push_back({prev, cur});
            prev = cur;
        }
    }
    const int n = next_vertex;   // 1 root + all arm vertices

    // Randomise input order before build. See GraphSampler::shuffle_edges.
    // Also randomises which vertex-id chunk is "arm 0" from the
    // tokeniser's perspective, since CsrGraph builds neighbour lists
    // in input order.
    shuffle_edges(edges, rng);

    SampledGraph sg;
    sg.graph             = CsrGraph::from_directed_edges(n, edges);
    sg.distance_semantic = DistanceSemantic::HopCount;
    return sg;
}
SampledGraph GraphSampler::sample_balanced(std::mt19937_64&, const GeneratorConfig&) {
    GG_CHECK(false, "sample_balanced not implemented");
}
SampledGraph GraphSampler::sample_khops(std::mt19937_64&, const GeneratorConfig&) {
    // Per-position khops is a synthetic-sequence task -- like
    // khops_gen there is no graph structure. Return an empty
    // SampledGraph so the rest of the pipeline stays uniform.
    SampledGraph sg;
    sg.graph             = CsrGraph::from_directed_edges(/*n=*/0, /*edges=*/{});
    sg.distance_semantic = DistanceSemantic::HopCount;
    return sg;
}
SampledGraph GraphSampler::sample_khops_gen(std::mt19937_64&, const GeneratorConfig&) {
    // K-hops-gen is a synthetic-sequence task: there is no graph
    // structure to sample. The task sampler will fill in the prefix
    // and ground truths from the raw vocab range read out of the
    // shared context. Returning an empty SampledGraph (n=0, no
    // edges, no positions) keeps the rest of the pipeline uniform:
    // the graph sections in the plan collapse to zero-count content
    // sections and produce no positions, while the tokenizer walks
    // the same section list it does for every other task.
    SampledGraph sg;
    sg.graph             = CsrGraph::from_directed_edges(/*n=*/0, /*edges=*/{});
    sg.distance_semantic = DistanceSemantic::HopCount;
    return sg;
}

void GraphSampler::shuffle_edges(std::vector<Edge>& edges,
                                 std::mt19937_64& rng) {
    std::shuffle(edges.begin(), edges.end(), rng);
}

}  // namespace graphgen
