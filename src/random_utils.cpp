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

// -----------------------------------------------------------------------------
// SampleIntPartition
// -----------------------------------------------------------------------------
//
// Uniform-random integer partitioning of Q into N parts follows Locey
// 2013: enumerate partition counts partition_QN(Q, N) and
// partition_QNK(Q, N, K) (partitions with max part K), draw a uniform
// index in [1, partition_QN(Q, N)], and walk the K-dimension one
// segment at a time, subtracting the count of partitions that come
// BEFORE the chosen index. Both count tables are memoised so the
// per-item cost after warm-up is O(N * max_K) rather than O(Q * N).

SampleIntPartition::SampleIntPartition(int suggested_cache_size,
                                       int max_cache_multiplier)
    : suggested_cache_size_(suggested_cache_size),
      max_cache_size_(suggested_cache_size * max_cache_multiplier) {}

void SampleIntPartition::clear_if_needed() {
    // Drop the whole cache when either half exceeds the soft budget.
    // Cheaper than eviction and correct: we just recompute lazily.
    if (static_cast<int>(qn_cache_.size())  > suggested_cache_size_) qn_cache_.clear();
    if (static_cast<int>(qnk_cache_.size()) > suggested_cache_size_) qnk_cache_.clear();
}

int SampleIntPartition::min_max_part_size(int Q, int N) {
    // Smallest K such that N parts of size <= K can sum to Q,
    // i.e. ceil(Q / N).
    int m = Q / N;
    if (Q % N > 0) ++m;
    return m;
}

std::int64_t SampleIntPartition::partition_QN(int Q, int N) {
    // Number of partitions of Q into exactly N positive parts.
    // Recurrence:
    //   partition_QN(Q, 0) = [Q == 0]
    //   partition_QN(Q, N) = 0                if N > Q or Q <= 0
    //   partition_QN(Q, N) = 1                if N == Q
    //   partition_QN(Q, N) = partition_QN(Q-1, N-1)   // smallest part == 1
    //                      + partition_QN(Q-N, N)     // else shrink all parts by 1
    const auto key = key_qn(Q, N);
    if (auto it = qn_cache_.find(key); it != qn_cache_.end()) return it->second;
    if (N == 0)         return Q == 0 ? 1 : 0;
    if (N > Q || Q <= 0) return 0;
    if (N == Q)          return 1;

    const std::int64_t result = partition_QN(Q - 1, N - 1)
                              + partition_QN(Q - N, N);
    // Hard-cap the cache: reset entirely if we've blown past the
    // absolute ceiling. Under normal batch sizes this never fires.
    if (static_cast<int>(qn_cache_.size()) > max_cache_size_) qn_cache_.clear();
    qn_cache_.emplace(key, result);
    return result;
}

std::int64_t SampleIntPartition::partition_QNK(int Q, int N, int K) {
    // Number of partitions of Q into exactly N parts, each <= K.
    // Recurrence:
    //   partition_QNK(Q, N, K) = 0                    if Q > N*K or K <= 0
    //   partition_QNK(Q, N, K) = 1                    if Q == N*K
    //   partition_QNK(Q, N, K) = sum_{i=0..N-1} partition_QNK(Q - i*K, N - i, K - 1)
    //                              (i parts of size exactly K, rest <= K-1)
    const auto key = key_qnk(Q, N, K);
    if (auto it = qnk_cache_.find(key); it != qnk_cache_.end()) return it->second;
    if (Q > N * K || K <= 0) return 0;
    if (Q == N * K)          return 1;

    std::int64_t result = 0;
    for (int i = 0; i < N; ++i) {
        result += partition_QNK(Q - i * K, N - i, K - 1);
    }
    if (static_cast<int>(qnk_cache_.size()) > max_cache_size_) qnk_cache_.clear();
    qnk_cache_.emplace(key, result);
    return result;
}

std::vector<int> SampleIntPartition::uniform_random_partition(
    int Q, int N,
    std::mt19937_64& rng,
    bool shuffle) {
    if (N <= 0 || Q < N) {
        throw std::invalid_argument(
            "uniform_random_partition: need N > 0 and Q >= N (got Q=" +
            std::to_string(Q) + ", N=" + std::to_string(N) + ")");
    }
    clear_if_needed();
    std::vector<int> segment_lengths;
    segment_lengths.reserve(static_cast<std::size_t>(N));

    int lo_K   = min_max_part_size(Q, N);
    int hi_K   = Q - N + 1;
    std::int64_t total = partition_QN(Q, N);
    std::uniform_int_distribution<std::int64_t> dist(1, total);
    std::int64_t which = dist(rng);

    while (Q > 0) {
        // Walk K = lo_K..hi_K; the first K whose partition_QNK(Q, N, K)
        // is >= `which` becomes the next chosen part. The count of
        // partitions with a strictly smaller max part gets subtracted
        // from `which` to zoom into the chosen K's sub-slice.
        int          chosen_K = lo_K;
        std::int64_t prev_cnt = 0;
        for (int K = lo_K; K <= hi_K; ++K) {
            std::int64_t cnt = partition_QNK(Q, N, K);
            if (cnt >= which) {
                prev_cnt = partition_QNK(Q, N, K - 1);
                chosen_K = K;
                break;
            }
        }
        segment_lengths.push_back(chosen_K);
        Q -= chosen_K;
        if (Q <= 0) break;   // == would suffice; <= is defensive
        --N;
        which -= prev_cnt;
        lo_K = min_max_part_size(Q, N);
        hi_K = chosen_K;   // subsequent parts can't exceed the last chosen
    }

    if (shuffle) std::shuffle(segment_lengths.begin(), segment_lengths.end(), rng);
    return segment_lengths;
}

std::vector<int> SampleIntPartition::non_uniform_random_partition(
    int Q, int N,
    std::mt19937_64& rng,
    bool shuffle) {
    if (N <= 0 || Q < N) {
        throw std::invalid_argument(
            "non_uniform_random_partition: need N > 0 and Q >= N (got Q=" +
            std::to_string(Q) + ", N=" + std::to_string(N) + ")");
    }
    // Every part starts at 1 (partitions must be strictly positive).
    // Randomly absorb chunks of the remaining budget into the first
    // N-1 parts; the last part gets whatever is left. Non-uniform in
    // the "trailing parts are smaller" sense, but a final shuffle
    // usually cancels that visual bias -- callers that need real
    // uniformity use uniform_random_partition above.
    std::vector<int> segment_lengths(static_cast<std::size_t>(N), 1);
    int remaining = Q - N;
    for (int i = 0; i < N - 1; ++i) {
        if (remaining <= 0) break;
        std::uniform_int_distribution<int> dist(0, remaining);
        int r = dist(rng);
        segment_lengths[static_cast<std::size_t>(i)] += r;
        remaining -= r;
    }
    segment_lengths.back() += remaining;

    if (shuffle) {
        std::shuffle(segment_lengths.begin(), segment_lengths.end(), rng);
    } else {
        std::sort(segment_lengths.begin(), segment_lengths.end());
    }
    return segment_lengths;
}

}  // namespace graphgen
