// TaskSampler implementation.
//
// sample_shortest_path picks one (start, end) pair with path length in
// the configured range, then walks the shortest-path DAG from start to
// end sampling uniformly at each step -- yielding both the chosen path
// and, at each step, the set of vertices that would have been equally
// correct.
//
// Two full-graph BFS calls per accepted item: forward BFS from start
// (distances FROM start), and either forward BFS from end (undirected,
// symmetric) or reverse BFS from end (directed, walk predecessors).
// The DAG walk needs distance TO end, not FROM end, so `distance_to`
// below dispatches on g.is_directed().

#include "graphgen/task_sampler.h"

#include <algorithm>
#include <cstddef>
#include <random>
#include <utility>
#include <vector>

#include "graphgen/check.h"
#include "graphgen/csr_graph.h"
#include "graphgen/generator_config.h"
#include "graphgen/hop_distance_matrix.h"
#include "graphgen/random_utils.h"
#include "graphgen/sampled_graph.h"

namespace graphgen {

namespace {

// Path-length range with sensible defaults if the caller left the config
// fields unset. min defaults to 1 (paths of length 0 are trivial); max
// defaults to n-1 (the diameter upper bound).
struct PathLengthRange {
    int min;
    int max;
};

PathLengthRange resolve_path_length_range(const GeneratorConfig& cfg, int n) {
    const int lo = cfg.min_path_length.value_or(1);
    const int hi = cfg.max_path_length.value_or(n > 1 ? n - 1 : 1);
    GG_CHECK(lo >= 1, "sample_shortest_path: min_path_length must be >= 1");
    GG_CHECK(hi >= lo,
             "sample_shortest_path: max_path_length must be >= min_path_length");
    return {lo, hi};
}

// Distance FROM every vertex TO `sink`. For undirected graphs this
// equals g.bfs(sink) (distances are symmetric). For directed graphs
// we build an in-neighbours map from the edges list and BFS from sink
// over that reverse-adjacency -- CsrGraph doesn't materialise the
// reverse-adjacency at construction time, but we only need it for
// this one call so the on-the-fly build (O(n + m)) is fine.
std::vector<int> distance_to(const CsrGraph& g, int sink) {
    if (!g.is_directed()) return g.bfs(sink);

    const int n = g.num_vertices();
    std::vector<std::vector<int>> in_nbrs(static_cast<std::size_t>(n));
    for (auto e : g.edges()) {
        in_nbrs[static_cast<std::size_t>(e.v)].push_back(e.u);
    }

    std::vector<int> dist(static_cast<std::size_t>(n), -1);
    dist[static_cast<std::size_t>(sink)] = 0;
    std::vector<int> current, next;
    current.reserve(static_cast<std::size_t>(n));
    next.reserve(static_cast<std::size_t>(n));
    current.push_back(sink);
    int layer = 0;
    while (!current.empty()) {
        for (int v : current) {
            for (int pred : in_nbrs[static_cast<std::size_t>(v)]) {
                if (dist[static_cast<std::size_t>(pred)] == -1) {
                    dist[static_cast<std::size_t>(pred)] = layer + 1;
                    next.push_back(pred);
                }
            }
        }
        current.swap(next);
        next.clear();
        ++layer;
    }
    return dist;
}

}  // namespace

std::pair<Task, Scratchpad> TaskSampler::run(TaskKind task_kind,
                                             ScratchpadKind scratchpad_kind,
                                             const SampledGraph& sg,
                                             std::mt19937_64& rng,
                                             const GeneratorConfig& cfg,
                                             int node_vocab_lo,
                                             int node_vocab_hi) {
    Task task;
    switch (task_kind) {
        case TaskKind::None:
            GG_CHECK(false, "TaskSampler::run: task_kind='none' has no work");
        case TaskKind::ShortestPath:
            task = sample_shortest_path(sg, rng, cfg);
            break;
        case TaskKind::BFS:
            task = sample_bfs(sg, rng, cfg);
            break;
        case TaskKind::Center:
            task = sample_center_centroid(sg, rng, cfg, /*is_center=*/true);
            break;
        case TaskKind::Centroid:
            task = sample_center_centroid(sg, rng, cfg, /*is_center=*/false);
            break;
        case TaskKind::Khops:
            task = sample_khops(rng, cfg, node_vocab_lo, node_vocab_hi);
            break;
        case TaskKind::KhopsGen:
            task = sample_khops_gen(rng, cfg, node_vocab_lo, node_vocab_hi);
            break;
    }
    Scratchpad scratchpad =
        scratchpad_sampler_.run(scratchpad_kind, sg, task, rng, cfg);
    return {std::move(task), std::move(scratchpad)};
}

Task TaskSampler::sample_shortest_path(const SampledGraph& sg,
                                       std::mt19937_64& rng,
                                       const GeneratorConfig& cfg) {
    const CsrGraph& g = sg.graph;
    const int n = g.num_vertices();
    GG_CHECK(n >= 2,
             "sample_shortest_path: graph must have >= 2 vertices (got "
                 << n << ")");

    const PathLengthRange range = resolve_path_length_range(cfg, n);
    const int max_attempts = cfg.max_attempts;
    GG_CHECK(max_attempts > 0,
             "sample_shortest_path: cfg.max_attempts must be > 0");

    IntSampler vertex_sampler(0, n);

    for (int attempt = 0; attempt < max_attempts; ++attempt) {
        const int start = vertex_sampler(rng);
        const int end   = vertex_sampler(rng);
        if (start == end) continue;

        const std::vector<int> d_from_start = g.bfs(start);
        const int L = d_from_start[static_cast<std::size_t>(end)];
        if (L < 0) continue;                             // disconnected pair
        if (L < range.min || L > range.max) continue;    // wrong length bucket

        // Distance from every vertex TO `end`. For undirected graphs
        // this is just another forward BFS from end (symmetric); for
        // directed graphs it's a reverse BFS. See distance_to() above.
        const std::vector<int> d_to_end = distance_to(g, end);

        // Walk start -> end on the DAG, sampling uniformly at each step
        // and recording the full valid set for label smoothing.
        std::vector<int>               path;
        std::vector<std::vector<int>>  valid_next_hops;
        path.reserve(static_cast<std::size_t>(L + 1));
        valid_next_hops.reserve(static_cast<std::size_t>(L));

        path.push_back(start);
        int current = start;
        for (int step = 0; step < L; ++step) {
            std::vector<int> valid;
            const int remaining = L - step - 1;  // distance from next vertex to end
            for (int y : g.neighbours(current)) {
                if (d_from_start[static_cast<std::size_t>(y)] == step + 1 &&
                    d_to_end    [static_cast<std::size_t>(y)] == remaining) {
                    valid.push_back(y);
                }
            }
            // BFS distances guarantee at least one valid next hop exists
            // (the reconstruct trail from `end`'s predecessor chain).
            GG_ASSERT(!valid.empty(),
                      "sample_shortest_path: DAG step had no valid next hop");

            IntSampler pick(0, static_cast<int>(valid.size()));
            const int next = valid[static_cast<std::size_t>(pick(rng))];

            valid_next_hops.push_back(std::move(valid));
            path.push_back(next);
            current = next;
        }
        GG_ASSERT(current == end,
                  "sample_shortest_path: DAG walk didn't land on end vertex");

        Task task;
        task.kind             = TaskKind::ShortestPath;
        task.query.start_node = start;
        task.query.end_node   = end;
        task.target.path            = std::move(path);
        task.target.valid_next_hops = std::move(valid_next_hops);
        return task;
    }

    GG_CHECK(false,
             "sample_shortest_path: no (start, end) pair with path length in ["
                 << range.min << ", " << range.max << "] found after "
                 << max_attempts << " attempts");
}

// BFS traversal from a random start vertex. Target is the visit order
// (queue-pop order) of a BFS run on the full graph; only vertices
// reachable from `start` appear in the output. Length = number of
// reachable vertices, which equals the graph's size when the graph is
// connected (the common case for ER at reasonable p) and is smaller
// otherwise.
//
// No length filter / reject-loop: unlike shortest_path, BFS doesn't
// have a natural "path length range" to bucket into. If a task needs
// one it can be added as a config-conditional filter here.
Task TaskSampler::sample_bfs(const SampledGraph& sg,
                             std::mt19937_64& rng,
                             const GeneratorConfig& cfg) {
    (void)cfg;  // no BFS-specific knobs yet
    const CsrGraph& g = sg.graph;
    const int n = g.num_vertices();
    GG_CHECK(n >= 1, "sample_bfs: graph must have at least 1 vertex");

    IntSampler vertex_sampler(0, n);
    const int start = vertex_sampler(rng);

    // Standard BFS from start; record vertices in queue-pop order.
    // Two ping-pong buffers rather than std::queue to keep the visit
    // sequence contiguous and free of per-push allocation, matching the
    // convention we already use for CsrGraph::bfs.
    std::vector<int> visit_order;
    visit_order.reserve(static_cast<std::size_t>(n));
    std::vector<char> visited(static_cast<std::size_t>(n), 0);
    std::vector<int>  current;
    std::vector<int>  next;
    current.reserve(static_cast<std::size_t>(n));
    next.reserve(static_cast<std::size_t>(n));

    visited[static_cast<std::size_t>(start)] = 1;
    visit_order.push_back(start);
    current.push_back(start);
    while (!current.empty()) {
        for (int u : current) {
            for (int v : g.neighbours(u)) {
                if (!visited[static_cast<std::size_t>(v)]) {
                    visited[static_cast<std::size_t>(v)] = 1;
                    visit_order.push_back(v);
                    next.push_back(v);
                }
            }
        }
        current.swap(next);
        next.clear();
    }

    Task task;
    task.kind             = TaskKind::BFS;
    task.query.start_node = start;
    task.target.path      = std::move(visit_order);
    return task;
}

// Sample a query set Q from cfg.given_query if supplied, else uniformly
// at random from all vertices with size drawn from
// [min_query_size, max_query_size].
namespace {

std::vector<int> resolve_query(const CsrGraph& g,
                               std::mt19937_64& rng,
                               const GeneratorConfig& cfg) {
    const int n = g.num_vertices();
    if (cfg.given_query.has_value() && !cfg.given_query->empty()) {
        // Trust caller-provided query. Bounds-check the entries.
        for (int q : *cfg.given_query) {
            GG_CHECK(q >= 0 && q < n,
                     "sample_center_centroid: given_query vertex "
                         << q << " out of range [0, " << n << ")");
        }
        return *cfg.given_query;
    }
    const int lo_default = 2;
    const int lo = cfg.min_query_size.value_or(lo_default);
    const int hi_default = n;
    const int hi_cap = cfg.max_query_size.value_or(hi_default);
    // Clamp hi to n (can't sample more distinct vertices than exist)
    // and >= lo. Legacy allowed hi == -1 as "unset"; we treat any
    // hi <= 0 as "use n".
    const int hi = std::min(n, hi_cap > 0 ? hi_cap : hi_default);
    GG_CHECK(lo >= 1,
             "sample_center_centroid: min_query_size must be >= 1 (got "
                 << lo << ")");
    GG_CHECK(hi >= lo,
             "sample_center_centroid: max_query_size (" << hi
                 << ") must be >= min_query_size (" << lo << ")");
    GG_CHECK(hi <= n,
             "sample_center_centroid: max_query_size (" << hi
                 << ") exceeds num_vertices (" << n << ")");

    // Pick query length uniformly in [lo, hi], then draw that many
    // distinct vertex ids without replacement.
    IntSampler len_sampler(lo, hi + 1);  // IntSampler is half-open
    const int q_len = len_sampler(rng);
    return sample_distinct_ints(q_len, 0, n, rng);
}

}  // namespace

Task TaskSampler::sample_center_centroid(const SampledGraph& sg,
                                         std::mt19937_64& rng,
                                         const GeneratorConfig& cfg,
                                         bool is_center) {
    const CsrGraph& g = sg.graph;
    const int n = g.num_vertices();
    GG_CHECK(n >= 1, "sample_center_centroid: graph must have >= 1 vertex");

    std::vector<int> query = resolve_query(g, rng, cfg);

    // Center/Centroid needs distances from every vertex TO each q in
    // Q. That's |Q| BFS-worth of information, and |Q| grows with n
    // in the general case -- so ask sg for the full hop-distance
    // matrix, which is O(n * (n+m)) BFS runs once per graph. When the
    // caller has already attached a batch-tensor view (i.e.
    // return_hop_distances is set), this fill lands directly in the
    // returned tensor with zero copy. Otherwise it lives in a
    // per-graph owned buffer until release_hop_distances() drops it.
    // Semantics: D[u, v] is the hop distance from u to v, or
    // HOP_UNREACHABLE (-1) if no path exists. For directed graphs BFS
    // follows out-edges, so D[v, q] is exactly "steps from v to q".
    const HopDistView D = sg.hop_distances();

    // Per-vertex aggregated distance to Q. Center: max; Centroid: sum.
    // If any q is unreachable from some v, mark v disqualified so it
    // can't win the min-selection later -- conflating unreachable with
    // distance 0 would place spurious center_nodes on isolated components.
    std::vector<long long> agg(static_cast<std::size_t>(n), 0);
    std::vector<char>      reachable_from_all(
        static_cast<std::size_t>(n), 1);

    for (int q : query) {
        for (int v = 0; v < n; ++v) {
            const int dv = D[v, q];
            if (dv == HOP_UNREACHABLE) {
                reachable_from_all[static_cast<std::size_t>(v)] = 0;
                continue;
            }
            if (is_center) {
                // MAX aggregator: take the running max.
                if (dv > agg[static_cast<std::size_t>(v)]) {
                    agg[static_cast<std::size_t>(v)] = dv;
                }
            } else {
                // SUM aggregator (centroid).
                agg[static_cast<std::size_t>(v)] +=
                    static_cast<long long>(dv);
            }
        }
    }

    // Find the minimum aggregated distance among vertices reachable
    // from every q in Q, then collect all vertices tied at that
    // minimum -- multiple center_nodes are the norm for center/centroid.
    long long best = std::numeric_limits<long long>::max();
    for (int v = 0; v < n; ++v) {
        if (!reachable_from_all[static_cast<std::size_t>(v)]) continue;
        if (agg[static_cast<std::size_t>(v)] < best) {
            best = agg[static_cast<std::size_t>(v)];
        }
    }
    GG_CHECK(best != std::numeric_limits<long long>::max(),
             "sample_center_centroid: no vertex reachable from every "
             "query vertex; the graph is too disconnected for this Q");

    // Collect every vertex tied at the minimum, then shuffle so the
    // emitted order isn't a leak of internal vertex ids -- the model
    // shouldn't learn to output center_nodes in ascending-id order.
    // Downstream label smoothing accepts any unemitted node as a
    // valid target at every position (see fill_targets in tokenizer).
    std::vector<int> center_nodes;
    for (int v = 0; v < n; ++v) {
        if (reachable_from_all[static_cast<std::size_t>(v)] &&
            agg[static_cast<std::size_t>(v)] == best) {
            center_nodes.push_back(v);
        }
    }
    std::shuffle(center_nodes.begin(), center_nodes.end(), rng);

    Task task;
    task.kind              = is_center ? TaskKind::Center : TaskKind::Centroid;
    task.query.vertex_set  = std::move(query);
    task.target.vertex_set = std::move(center_nodes);
    return task;
}

// -----------------------------------------------------------------------------
// sample_khops_gen
// -----------------------------------------------------------------------------
//
// Constructs a synthetic prefix + backtrace task. Given the raw vocab
// range [lo, hi), a sampled k in [min_khops, max_khops], and a sampled
// prefix_length P in [min_prefix_length, max_prefix_length]:
//
//   1. Split (P - k) into k positive segment lengths via
//      SampleIntPartition (uniform or non_uniform per cfg.partition_method).
//      Each segment then occupies (segment_len + 1) prefix positions.
//   2. Walk k segments. At entry to segment i, `cur_value` is the
//      current backtrace token. Emit segment[i] as (segment_len + 1)
//      random vocab tokens (each drawn from [lo, hi) minus cur_value),
//      then OVERWRITE either position [-1] (right_side_connect or the
//      last segment) or position [-2] (otherwise) with cur_value so
//      the model can find it. `cur_value` for the NEXT segment is
//      whichever of positions [-1] / [-2] holds the token that would
//      appear "just past" the overwritten slot -- i.e. exactly the
//      slot the model would need to look at after finding cur_value.
//   3. The k values of cur_value at segment entry are the k-hop ground
//      truths. The concatenated segment tokens are the prefix; the
//      LAST token is the query cursor.
//   4. khops_no_repeats forces the k+1 cur_value stream to be a
//      shuffled sequence of unique vocab tokens (path[0..k]); each
//      segment then also gets its "next cur_value" slot patched to
//      path[i+1] so the invariant holds through the overwrite.
Task TaskSampler::sample_khops_gen(std::mt19937_64& rng,
                                   const GeneratorConfig& cfg,
                                   int node_vocab_lo,
                                   int node_vocab_hi) {
    // Config resolution. All khops_gen-specific fields must be set;
    // config validation upstream already checks the ranges and the
    // Q >= N feasibility relation, but we defensively re-check the
    // presence bits here so an internal bug doesn't produce garbage.
    GG_CHECK(cfg.min_khops.has_value() && cfg.max_khops.has_value(),
             "sample_khops_gen: min_khops / max_khops must be set");
    GG_CHECK(cfg.min_prefix_length.has_value() && cfg.max_prefix_length.has_value(),
             "sample_khops_gen: min_prefix_length / max_prefix_length must be set");

    const int min_k = *cfg.min_khops;
    const int max_k = *cfg.max_khops;
    const int min_P = *cfg.min_prefix_length;
    const int max_P = *cfg.max_prefix_length;
    GG_CHECK(min_k >= 1 && max_k >= min_k,
             "sample_khops_gen: need 1 <= min_khops <= max_khops (got "
                 << min_k << ", " << max_k << ")");
    GG_CHECK(min_P >= 2 * max_k,
             "sample_khops_gen: min_prefix_length (" << min_P
                 << ") must be >= 2 * max_khops (" << (2 * max_k)
                 << ") so every (k, P) pair has a feasible partition");
    GG_CHECK(max_P >= min_P,
             "sample_khops_gen: max_prefix_length must be >= min_prefix_length");

    GG_CHECK(node_vocab_hi > node_vocab_lo,
             "sample_khops_gen: empty vocab range");
    const int vocab_size = node_vocab_hi - node_vocab_lo;
    // uniform_int_distribution over [lo, hi-1]; matches V1 semantics.
    std::uniform_int_distribution<int> vocab_dist(node_vocab_lo, node_vocab_hi - 1);

    const bool right_side_connect = cfg.right_side_connect.value_or(true);
    const bool no_repeats         = cfg.khops_no_repeats.value_or(false);
    const std::string partition_method = cfg.partition_method.value_or("uniform");

    // Sample k and P.
    std::uniform_int_distribution<int> k_dist(min_k, max_k);
    const int k = k_dist(rng);
    std::uniform_int_distribution<int> P_dist(min_P, max_P);
    const int P = P_dist(rng);
    // Partition of Q = P - k into N = k positive parts.
    const int Q = P - k;
    const int N = k;
    GG_CHECK(Q >= N,
             "sample_khops_gen: infeasible partition (Q=" << Q << " < N=" << N
                 << "); should have been caught by min_prefix_length check");

    std::vector<int> segment_lengths;
    if (partition_method == "uniform") {
        segment_lengths = partition_sampler_.uniform_random_partition(
            Q, N, rng, /*shuffle=*/true);
    } else if (partition_method == "non_uniform") {
        segment_lengths = partition_sampler_.non_uniform_random_partition(
            Q, N, rng, /*shuffle=*/true);
    } else {
        GG_CHECK(false,
                 "sample_khops_gen: unknown partition_method='" << partition_method
                     << "' (expected 'uniform' or 'non_uniform')");
    }

    // Pre-sample the unique-value chain when khops_no_repeats is on.
    // path has k+1 entries: cur_value at entry of segment i is path[i],
    // and the "next slot" of segment i gets patched to path[i+1].
    std::vector<int> path;
    if (no_repeats) {
        GG_CHECK(vocab_size >= k + 1,
                 "sample_khops_gen: khops_no_repeats requires vocab_size >= max_khops + 1 "
                 "(got vocab_size=" << vocab_size << ", k=" << k << ")");
        path = sample_distinct_ints(k + 1, node_vocab_lo, node_vocab_hi, rng);
    }

    int cur_value = no_repeats ? path[0] : vocab_dist(rng);

    std::vector<int> prefix;
    prefix.reserve(static_cast<std::size_t>(P));
    std::vector<int> ground_truths;
    ground_truths.reserve(static_cast<std::size_t>(k));

    for (int i = 0; i < k; ++i) {
        ground_truths.push_back(cur_value);
        const bool is_last     = (i == k - 1);
        const int  segment_len = segment_lengths[static_cast<std::size_t>(i)];
        const int  seg_size    = segment_len + 1;

        // Fill segment with random vocab tokens != cur_value.
        // Sampling from [lo, hi - 1) and shifting past cur_value keeps
        // the distribution uniform without rejection loops.
        std::vector<int> segment(static_cast<std::size_t>(seg_size));
        std::uniform_int_distribution<int> seg_dist(node_vocab_lo, node_vocab_hi - 2);
        for (int j = 0; j < seg_size; ++j) {
            int v = seg_dist(rng);
            if (v >= cur_value) ++v;   // hop over cur_value
            segment[static_cast<std::size_t>(j)] = v;
        }
        // Insert cur_value at the "connect" slot. On the last segment
        // it always goes at [-1] so the query cursor equals the final
        // ground truth's back-hop target.
        if (is_last || right_side_connect) {
            segment.back() = cur_value;
        } else {
            segment[static_cast<std::size_t>(seg_size - 2)] = cur_value;
        }

        // Compute next cur_value. Under right_side_connect the segment
        // pattern is (..., next, cur_value): pos [-2] is "next" (what
        // sits just before cur in the token stream, i.e. what the
        // model would read after finding cur when scanning backward
        // and stepping one to the right). Under !right_side_connect
        // the pattern is (..., cur_value, next): next is at pos [-1].
        //
        // no_repeats mode: overwrite the "next" slot with path[i+1]
        // BEFORE reading, so the emitted ground truth stays unique.
        int next_value;
        if (right_side_connect) {
            if (no_repeats) {
                segment[static_cast<std::size_t>(seg_size - 2)] =
                    path[static_cast<std::size_t>(i + 1)];
            }
            next_value = segment[static_cast<std::size_t>(seg_size - 2)];
        } else {
            if (no_repeats) {
                segment.back() = path[static_cast<std::size_t>(i + 1)];
            }
            next_value = segment.back();
        }

        prefix.insert(prefix.end(), segment.begin(), segment.end());
        cur_value = next_value;
    }
    GG_CHECK(static_cast<int>(prefix.size()) == P,
             "sample_khops_gen: internal error, emitted prefix length "
                 << prefix.size() << " != P=" << P);

    Task task;
    task.kind         = TaskKind::KhopsGen;
    task.query.prefix = std::move(prefix);
    task.target.path  = std::move(ground_truths);
    return task;
}

// -----------------------------------------------------------------------------
// sample_khops (per-position variant)
// -----------------------------------------------------------------------------
//
// Ports V1's KHopsTask constructor. Produces a length-P sequence over
// [lo, hi), computes per-position back-pointers, iterates k times to
// materialise hops[0..k-1][i], and packages the labels the tokenizer's
// fill_targets step will patch in.
//
// Two sampling paths:
//   * permutation_version=true : seq = concatenation of (k + 1)
//     shuffled full-vocab permutations. Guarantees no adjacent
//     repeats (a full reshuffle handles the seam case) and, by
//     construction, every seq position at index >= vocab_size has at
//     least one earlier match -- so hops[0][i] is defined for all
//     later positions. Length is (k + 1) * vocab_size.
//   * otherwise                : uniform sample from [lo, hi) with
//     adjacent-repeat rejection. Length is drawn from
//     [min_prefix_length, max_prefix_length].
//
// Masking is applied at label-materialisation time:
//   * mask_to_size > 0        : positions with i <= mask_to_size are masked.
//   * mask_to_vocab_size      : only positions with i >= P - vocab_size
//                               get labels ("only the last vocab_size").
//   * neither                 : all positions with defined hops[k-1][i]
//                               get labels.
//
// intermediate_labels controls how many labels each un-masked position
// gets: 1 (just hops[k-1][i]) if false, up to k (deduplicated backward
// chain hops[k-1..0][i]) if true.
Task TaskSampler::sample_khops(std::mt19937_64& rng,
                               const GeneratorConfig& cfg,
                               int node_vocab_lo,
                               int node_vocab_hi) {
    GG_CHECK(cfg.min_khops.has_value() && cfg.max_khops.has_value(),
             "sample_khops: min_khops / max_khops must be set");
    const int min_k = *cfg.min_khops;
    const int max_k = *cfg.max_khops;
    GG_CHECK(min_k >= 1 && max_k >= min_k,
             "sample_khops: need 1 <= min_khops <= max_khops (got "
                 << min_k << ", " << max_k << ")");

    GG_CHECK(node_vocab_hi > node_vocab_lo,
             "sample_khops: empty vocab range");
    const int vocab_size = node_vocab_hi - node_vocab_lo;
    // sample_khops assumes vocab_size >= 3 so adjacent-repeat rejection
    // in the standard sampler is non-degenerate (and the V1 code
    // asserts the same). >=3 lets us always find a next_token != prev.
    GG_CHECK(vocab_size >= 3,
             "sample_khops: node vocab size must be >= 3 (got "
                 << vocab_size << ")");

    const bool permutation = cfg.permutation_version.value_or(false);
    const bool right_side  = cfg.right_side_connect.value_or(true);
    const bool intermediate = cfg.intermediate_labels.value_or(false);
    const int  mask_to_size = cfg.mask_to_size.value_or(-1);
    const bool mask_to_vocab_size = cfg.mask_to_vocab_size.value_or(false);

    // Sample k.
    std::uniform_int_distribution<int> k_dist(min_k, max_k);
    const int k = k_dist(rng);

    // Build the seq.
    std::vector<int> seq;
    if (permutation) {
        // (k + 1) shuffled copies of the full vocab, glued together.
        // A single reshuffle at the seam removes the rare case where
        // the last token of the prior copy equals the first token of
        // the next copy -- avoids the infinite loop V1 warns about
        // downstream in the hop chain.
        seq.reserve(static_cast<std::size_t>((k + 1) * vocab_size));
        std::vector<int> vocab(static_cast<std::size_t>(vocab_size));
        std::iota(vocab.begin(), vocab.end(), node_vocab_lo);
        int prior_last = -1;
        for (int copy = 0; copy < k + 1; ++copy) {
            std::shuffle(vocab.begin(), vocab.end(), rng);
            if (prior_last >= 0 && vocab.front() == prior_last) {
                std::shuffle(vocab.begin(), vocab.end(), rng);
            }
            for (int v : vocab) seq.push_back(v);
            prior_last = vocab.back();
        }
    } else {
        GG_CHECK(cfg.min_prefix_length.has_value() && cfg.max_prefix_length.has_value(),
                 "sample_khops: min_prefix_length / max_prefix_length must be set "
                 "when permutation_version=false");
        const int min_P = *cfg.min_prefix_length;
        const int max_P = *cfg.max_prefix_length;
        GG_CHECK(min_P >= 2 && max_P >= min_P,
                 "sample_khops: need 2 <= min_prefix_length <= max_prefix_length "
                 "(got " << min_P << ", " << max_P << ")");
        std::uniform_int_distribution<int> P_dist(min_P, max_P);
        const int P = P_dist(rng);

        seq.reserve(static_cast<std::size_t>(P));
        std::uniform_int_distribution<int> dist(node_vocab_lo, node_vocab_hi - 1);
        seq.push_back(dist(rng));
        for (int i = 1; i < P; ++i) {
            int next = dist(rng);
            while (next == seq.back()) next = dist(rng);
            seq.push_back(next);
        }
    }

    const int P = static_cast<int>(seq.size());

    // Compute back_pointer[i] = j + offset where seq[j] == seq[i] and j
    // is the nearest earlier index; -1 if no such j (or if j + offset
    // falls outside [0, P)).
    const int offset = right_side ? 1 : -1;
    std::vector<int> back_pointer(static_cast<std::size_t>(P), -1);

    // hops[h] is a length-P vector; hops[h][i] is the value at
    // back-pointer step h+1 starting from position i, or -1 if the
    // chain terminates before h+1 steps.
    std::vector<std::vector<int>> hops(
        static_cast<std::size_t>(k),
        std::vector<int>(static_cast<std::size_t>(P), -1));

    for (int i = P - 1; i >= 0; --i) {
        const int target = seq[static_cast<std::size_t>(i)];
        for (int j = i - 1; j >= 0; --j) {
            if (seq[static_cast<std::size_t>(j)] == target) {
                const int ptr_idx = j + offset;
                if (ptr_idx >= 0 && ptr_idx < P) {
                    back_pointer[static_cast<std::size_t>(i)] = ptr_idx;
                    hops[0][static_cast<std::size_t>(i)] =
                        seq[static_cast<std::size_t>(ptr_idx)];
                }
                break;
            }
        }
    }

    // Iterate k - 1 more times to chain the back-pointer.
    // cur_back_pointer[i] tracks where we are after h back-hops from i.
    std::vector<int> cur_back_pointer(back_pointer);
    for (int h = 1; h < k; ++h) {
        for (int i = 0; i < P; ++i) {
            const int new_ptr = cur_back_pointer[static_cast<std::size_t>(i)];
            if (new_ptr == -1) continue;
            cur_back_pointer[static_cast<std::size_t>(i)] =
                back_pointer[static_cast<std::size_t>(new_ptr)];
            const int nxt = cur_back_pointer[static_cast<std::size_t>(i)];
            if (nxt == -1) continue;
            hops[static_cast<std::size_t>(h)][static_cast<std::size_t>(i)] =
                seq[static_cast<std::size_t>(nxt)];
        }
    }

    // Materialise per-position labels. Empty inner vector = masked
    // (fill_targets writes PAD across all label slots).
    std::vector<std::vector<int>> per_position_labels(
        static_cast<std::size_t>(P));
    for (int i = 0; i < P; ++i) {
        const int final_hop = hops[static_cast<std::size_t>(k - 1)]
                                  [static_cast<std::size_t>(i)];
        if (final_hop < 0) continue;   // chain didn't reach k hops

        // Position-level mask conditions. V1 semantics: mask_to_size
        // and mask_to_vocab_size are UNMASK flags -- only positions
        // satisfying one of them get labels. If NEITHER is set, all
        // positions with a full chain get labels.
        bool unmasked = false;
        if (mask_to_size > 0) {
            if (i > mask_to_size) unmasked = true;
        }
        if (mask_to_vocab_size) {
            if (i >= P - vocab_size) unmasked = true;
        }
        if (mask_to_size < 0 && !mask_to_vocab_size) {
            unmasked = true;
        }
        if (!unmasked) continue;

        auto& labels = per_position_labels[static_cast<std::size_t>(i)];
        if (!intermediate) {
            labels.push_back(final_hop);
            continue;
        }
        // Intermediate: hops[k-1][i], hops[k-2][i], ..., hops[0][i],
        // deduplicated (stop at first repeat, matching V1's
        // set-of-hops-based break to prevent the loss from
        // over-weighting a repeated token).
        std::vector<char> seen_bit(static_cast<std::size_t>(vocab_size), 0);
        for (int h = 0; h < k; ++h) {
            const int v = hops[static_cast<std::size_t>(k - 1 - h)]
                              [static_cast<std::size_t>(i)];
            if (v < 0) break;
            const int rel = v - node_vocab_lo;
            if (rel >= 0 && rel < vocab_size) {
                if (seen_bit[static_cast<std::size_t>(rel)]) break;
                seen_bit[static_cast<std::size_t>(rel)] = 1;
            }
            labels.push_back(v);
        }
    }

    Task task;
    task.kind                       = TaskKind::Khops;
    task.query.khops_k              = k;
    task.target.path                = std::move(seq);
    task.target.per_position_labels = std::move(per_position_labels);
    return task;
}

}  // namespace graphgen
