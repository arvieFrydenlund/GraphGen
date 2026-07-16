#include "graphgen/random_utils.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>

namespace graphgen {

std::vector<int> sample_distinct_ints(int k,
                                      int lo,
                                      int hi,
                                      std::mt19937_64& rng) {
    if (k < 0) {
        throw std::invalid_argument(
            "sample_distinct_ints: k=" + std::to_string(k) + " must be >= 0");
    }
    if (hi < lo) {
        throw std::invalid_argument(
            "sample_distinct_ints: hi (" + std::to_string(hi) +
            ") < lo (" + std::to_string(lo) + ")");
    }
    if (hi - lo < k) {
        throw std::invalid_argument(
            "sample_distinct_ints: range [" + std::to_string(lo) + ", " +
            std::to_string(hi) + ") too narrow for k=" + std::to_string(k));
    }

    std::vector<int> pool(static_cast<std::size_t>(hi - lo));
    std::iota(pool.begin(), pool.end(), lo);
    std::shuffle(pool.begin(), pool.end(), rng);
    pool.resize(static_cast<std::size_t>(k));
    return pool;
}

IntSampler::IntSampler(int lo, int hi)
    : lo_(lo),
      hi_(hi),
      weighted_(false),
      // uniform_int_distribution is inclusive on both ends; we want the
      // usual half-open [lo, hi) convention, so pass hi - 1 as the max.
      uniform_(lo, hi - 1) {
    if (hi <= lo) {
        throw std::invalid_argument(
            "IntSampler: hi (" + std::to_string(hi) +
            ") must be > lo (" + std::to_string(lo) + ")");
    }
}

IntSampler::IntSampler(int lo, int hi, std::span<const float> weights)
    : lo_(lo),
      hi_(hi),
      weighted_(true) {
    if (hi <= lo) {
        throw std::invalid_argument(
            "IntSampler: hi (" + std::to_string(hi) +
            ") must be > lo (" + std::to_string(lo) + ")");
    }
    const std::size_t expected = static_cast<std::size_t>(hi - lo);
    if (weights.size() != expected) {
        throw std::invalid_argument(
            "IntSampler: weights.size() (" + std::to_string(weights.size()) +
            ") must equal hi - lo (" + std::to_string(expected) + ")");
    }
    bool any_positive = false;
    for (float w : weights) {
        if (w < 0.0f) {
            throw std::invalid_argument(
                "IntSampler: weights must be non-negative (got " +
                std::to_string(w) + ")");
        }
        if (w > 0.0f) any_positive = true;
    }
    if (!any_positive) {
        // std::discrete_distribution would silently fall back to uniform
        // in this case; reject it here so a stray all-zero vector fails
        // loudly instead of masquerading as uniform sampling.
        throw std::invalid_argument("IntSampler: at least one weight must be > 0");
    }

    discrete_ = std::discrete_distribution<int>(weights.begin(), weights.end());
}

int IntSampler::operator()(std::mt19937_64& rng) {
    // Weighted path returns a bin index in [0, hi - lo); shift into [lo, hi).
    if (weighted_) return discrete_(rng) + lo_;
    return uniform_(rng);
}

}  // namespace graphgen
