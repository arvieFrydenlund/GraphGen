// Small random utilities that show up everywhere in the sampler pipeline.
// Kept header-only in dependency terms (just <random> + <vector> + <span>)
// so any core file -- CsrGraph builders, SampledGraph, topology samplers,
// task samplers -- can pull it in without dragging pybind11 or config
// along.

#ifndef GRAPHGEN_RANDOM_UTILS_H
#define GRAPHGEN_RANDOM_UTILS_H

#include <cstdint>
#include <random>
#include <span>
#include <unordered_map>
#include <vector>

namespace graphgen {

// Draw `k` distinct integers uniformly from the half-open range [lo, hi).
// Result is returned in random order (not sorted); it is a uniformly
// random k-subset of the range, with each ordering equally likely.
//
// Preconditions (violations throw std::invalid_argument):
//   k >= 0
//   hi >= lo
//   hi - lo >= k                (span wide enough to host k distinct ids)
//
// Complexity: O(hi - lo) time and space via shuffle-then-truncate. For
// the small ranges we sample from (hundreds of ids) this beats a
// reservoir sampler in constant factor and is straightforward to reason
// about; swap for reservoir sampling if we ever start sampling from
// truly large ranges.
std::vector<int> sample_distinct_ints(int k,
                                      int lo,
                                      int hi,
                                      std::mt19937_64& rng);

// Reusable single-integer sampler over a half-open range [lo, hi).
//
// Built once, drawn many times: callers (path-length sampling in tasks,
// khops-k sampling, branching-factor in trees, ...) construct one per
// batch and invoke operator() per sample. Rebuilding a
// std::discrete_distribution on every draw was one of the perf pitfalls
// of the V1 code; this class amortises that.
//
// Two constructors, one type:
//   IntSampler(lo, hi)           uniform over [lo, hi)
//   IntSampler(lo, hi, weights)  discrete over [lo, hi) with the given
//                                per-bin weights (weights.size() must
//                                equal hi - lo)
//
// The Python side sends weights as a plain list of floats; the caller
// passes a span into them. Empty weights are not allowed on the
// weighted ctor -- use the uniform ctor for that case, so intent stays
// explicit at the call site.
//
// Preconditions (violations throw std::invalid_argument):
//   hi > lo                                            (both ctors)
//   weights.size() == hi - lo                          (weighted)
//   every weight >= 0 and at least one > 0             (weighted)
class IntSampler {
public:
    IntSampler(int lo, int hi);
    IntSampler(int lo, int hi, std::span<const float> weights);

    int operator()(std::mt19937_64& rng);

    int lo() const { return lo_; }
    int hi() const { return hi_; }
    bool is_weighted() const { return weighted_; }

private:
    int  lo_       = 0;
    int  hi_       = 0;
    bool weighted_ = false;
    // Both distributions are cheap to default-construct; only the one
    // matching `weighted_` is ever drawn from. A std::variant would save
    // a few bytes at the cost of an extra branch per draw -- not worth it.
    std::uniform_int_distribution<int> uniform_;
    std::discrete_distribution<int>    discrete_;
};

// Sample integer partitions -- split Q into N positive parts.
//
// Two flavours, both invoked per-item during khops task generation:
//   * `uniform_random_partition(Q, N, rng, shuffle)` samples uniformly
//     from the set of all partitions of Q into N parts, via a DP that
//     counts partition_QN / partition_QNK (memoised across calls).
//     The DP is O(Q * N) per fresh cell; the cache pays for itself as
//     soon as the same (Q, N) pair recurs across a batch. See Locey
//     2013 for the underlying algorithm.
//   * `non_uniform_random_partition(Q, N, rng, shuffle)` is the cheap
//     greedy alternative: each part starts at 1, then randomly
//     absorbs a slice of the remaining length. Biased toward the
//     first parts under the "shuffle=false" path but that bias is
//     removed by the trailing shuffle.
//
// Both return a `vector<int>` of length exactly N summing to Q, sorted
// ascending when `shuffle=false` and randomly permuted otherwise.
//
// The class holds two caches for the uniform sampler:
//   * QN_cache_ : (Q, N) -> partition count
//   * QNK_cache_: (Q, N, K) -> partition count with max part K
// Keys are packed into 32-/48-bit int64_t for constant-time hashing.
// Caches shrink to zero when they exceed `max_cache_size` slots so
// long-running workers can't leak unbounded memory; the same DP
// results just get recomputed the next time.
class SampleIntPartition {
public:
    // `suggested_cache_size` is a soft target; the cache is cleared
    // when it grows past `max_cache_multiplier * suggested_cache_size`.
    explicit SampleIntPartition(int suggested_cache_size    = 10'000'000,
                                int max_cache_multiplier    = 10);

    std::vector<int> uniform_random_partition(int Q, int N,
                                              std::mt19937_64& rng,
                                              bool shuffle = true);
    std::vector<int> non_uniform_random_partition(int Q, int N,
                                                  std::mt19937_64& rng,
                                                  bool shuffle = true);

    // Test / observability hooks.
    std::size_t qn_cache_size()  const { return qn_cache_.size(); }
    std::size_t qnk_cache_size() const { return qnk_cache_.size(); }

private:
    std::int64_t partition_QN (int Q, int N);
    std::int64_t partition_QNK(int Q, int N, int K);
    static int   min_max_part_size(int Q, int N);
    void         clear_if_needed();

    // Pack (Q, N) or (Q, N, K) into an int64 key. Q, N, K assumed to
    // fit in 20 bits each (comfortably above any realistic khops
    // prefix length).
    static std::uint64_t key_qn (int Q, int N) {
        return (static_cast<std::uint64_t>(Q) << 20)
             |  static_cast<std::uint64_t>(N);
    }
    static std::uint64_t key_qnk(int Q, int N, int K) {
        return (static_cast<std::uint64_t>(Q) << 40)
             | (static_cast<std::uint64_t>(N) << 20)
             |  static_cast<std::uint64_t>(K);
    }

    std::unordered_map<std::uint64_t, std::int64_t> qn_cache_;
    std::unordered_map<std::uint64_t, std::int64_t> qnk_cache_;
    int suggested_cache_size_;
    int max_cache_size_;
};

}  // namespace graphgen

#endif  // GRAPHGEN_RANDOM_UTILS_H
