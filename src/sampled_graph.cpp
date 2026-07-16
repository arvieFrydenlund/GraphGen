#include "graphgen/sampled_graph.h"

#include "graphgen/random_utils.h"

namespace graphgen {

void SampledGraph::assign_random_vocab_ids(int min_vocab,
                                           int max_vocab,
                                           std::mt19937_64& rng) {
    const int n  = graph.num_vertices();
    int       lo = min_vocab;
    int       hi = max_vocab;
    // "vocab unset" fallback: treat the span as [0, n).
    if (hi < 0) {
        lo = 0;
        hi = n;
    }
    internal_to_vocab = sample_distinct_ints(n, lo, hi, rng);
}

}  // namespace graphgen
