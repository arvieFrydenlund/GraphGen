// Small random utilities that show up everywhere in the sampler pipeline.
// Kept header-only in dependency terms (just <random> + <vector> + <span>)
// so any core file -- CsrGraph builders, SampledGraph, topology samplers,
// task samplers -- can pull it in without dragging pybind11 or config
// along.

#ifndef GRAPHGEN_RANDOM_UTILS_H
#define GRAPHGEN_RANDOM_UTILS_H

#include <random>
#include <span>
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

}  // namespace graphgen

#endif  // GRAPHGEN_RANDOM_UTILS_H
