#include "doctest.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>

#include "graphgen/random_utils.h"

using graphgen::IntSampler;
using graphgen::sample_distinct_ints;

TEST_CASE("sample_distinct_ints returns k distinct values in [lo, hi)") {
    std::mt19937_64 rng(0xC0FFEEULL);
    auto v = sample_distinct_ints(/*k=*/10, /*lo=*/5, /*hi=*/50, rng);

    CHECK(v.size() == 10);
    std::set<int> unique(v.begin(), v.end());
    CHECK(unique.size() == v.size());
    for (int x : v) {
        CHECK(x >= 5);
        CHECK(x < 50);
    }
}

TEST_CASE("sample_distinct_ints is deterministic under a fixed seed") {
    std::mt19937_64 a(12345);
    std::mt19937_64 b(12345);
    CHECK(sample_distinct_ints(20, 0, 100, a) ==
          sample_distinct_ints(20, 0, 100, b));
}

TEST_CASE("sample_distinct_ints k == hi-lo yields a permutation of the range") {
    std::mt19937_64 rng(1);
    auto v = sample_distinct_ints(/*k=*/8, /*lo=*/0, /*hi=*/8, rng);
    std::sort(v.begin(), v.end());
    for (int i = 0; i < 8; ++i) CHECK(v[i] == i);
}

TEST_CASE("sample_distinct_ints k=0 returns empty") {
    std::mt19937_64 rng(0);
    CHECK(sample_distinct_ints(0, 0, 100, rng).empty());
}

TEST_CASE("sample_distinct_ints rejects a too-narrow range") {
    std::mt19937_64 rng(0);
    CHECK_THROWS_AS(sample_distinct_ints(10, 0, 5, rng), std::invalid_argument);
}

TEST_CASE("sample_distinct_ints rejects hi < lo") {
    std::mt19937_64 rng(0);
    CHECK_THROWS_AS(sample_distinct_ints(1, 10, 5, rng), std::invalid_argument);
}

TEST_CASE("sample_distinct_ints rejects negative k") {
    std::mt19937_64 rng(0);
    CHECK_THROWS_AS(sample_distinct_ints(-1, 0, 10, rng), std::invalid_argument);
}

// ---- IntSampler --------------------------------------------------------

TEST_CASE("IntSampler uniform stays inside [lo, hi) and hits every bin") {
    std::mt19937_64 rng(0xABC);
    IntSampler s(10, 15);  // 5 bins
    CHECK_FALSE(s.is_weighted());
    CHECK(s.lo() == 10);
    CHECK(s.hi() == 15);

    std::array<int, 5> counts{};
    for (int i = 0; i < 10000; ++i) {
        int x = s(rng);
        REQUIRE(x >= 10);
        REQUIRE(x < 15);
        ++counts[x - 10];
    }
    for (int c : counts) CHECK(c > 0);
}

TEST_CASE("IntSampler weighted concentrates draws in high-weight bins") {
    // Weights heavily favour bin 3 (offset lo). ~90% of draws should land
    // on lo + 3 = 8.
    std::mt19937_64 rng(0xDEAD);
    std::vector<float> weights = {0.1f, 0.1f, 0.1f, 9.0f, 0.1f};  // sums to 9.4
    IntSampler s(5, 10, weights);
    CHECK(s.is_weighted());

    int hits_on_favourite = 0;
    const int trials = 20000;
    for (int i = 0; i < trials; ++i) {
        int x = s(rng);
        REQUIRE(x >= 5);
        REQUIRE(x < 10);
        if (x == 8) ++hits_on_favourite;
    }
    const double frac = static_cast<double>(hits_on_favourite) / trials;
    CHECK(frac > 0.85);
    CHECK(frac < 0.99);
}

TEST_CASE("IntSampler is reusable: one construction, many draws") {
    // Regression against rebuilding the underlying distribution per draw.
    std::mt19937_64 rng(1);
    std::vector<float> weights = {1.0f, 1.0f, 1.0f, 1.0f};
    IntSampler s(0, 4, weights);
    for (int i = 0; i < 1000; ++i) {
        int x = s(rng);
        CHECK(x >= 0);
        CHECK(x < 4);
    }
}

TEST_CASE("IntSampler determinism: same seed, same draws") {
    std::mt19937_64 a(999);
    std::mt19937_64 b(999);
    IntSampler sa(0, 10);
    IntSampler sb(0, 10);
    for (int i = 0; i < 50; ++i) CHECK(sa(a) == sb(b));

    std::mt19937_64 ca(999);
    std::mt19937_64 cb(999);
    std::vector<float> w = {1.0f, 2.0f, 3.0f};
    IntSampler wa(0, 3, w);
    IntSampler wb(0, 3, w);
    for (int i = 0; i < 50; ++i) CHECK(wa(ca) == wb(cb));
}

TEST_CASE("IntSampler rejects empty range") {
    std::vector<float> w;
    CHECK_THROWS_AS(IntSampler(5, 5), std::invalid_argument);
    CHECK_THROWS_AS(IntSampler(5, 3), std::invalid_argument);
}

TEST_CASE("IntSampler weighted ctor rejects mismatched weight length") {
    std::vector<float> w = {1.0f, 2.0f};  // 2 weights for a 3-bin range
    CHECK_THROWS_AS(IntSampler(0, 3, w), std::invalid_argument);
}

TEST_CASE("IntSampler weighted ctor rejects negative weight") {
    std::vector<float> w = {1.0f, -0.5f, 2.0f};
    CHECK_THROWS_AS(IntSampler(0, 3, w), std::invalid_argument);
}

TEST_CASE("IntSampler weighted ctor rejects all-zero weights") {
    // std::discrete_distribution would silently fall back to uniform; we
    // reject up front so a stray all-zero input fails loudly.
    std::vector<float> w = {0.0f, 0.0f, 0.0f};
    CHECK_THROWS_AS(IntSampler(0, 3, w), std::invalid_argument);
}
