// WorkerSharedContext implementation.
//
// Ports V1's dictionaries.h layout and counting logic onto the new struct.
// Semantics preserved verbatim so tokenised outputs stay byte-for-byte
// comparable against the pre-cutover baseline.

#include "graphgen/worker_shared_context.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "graphgen/generator_config.h"

namespace graphgen {

WorkerSharedContext::WorkerSharedContext() {
    set_default_dictionary();
    set_default_pos_dictionary();
}

void WorkerSharedContext::set_dictionary(std::map<std::string, int> d,
                                         int max_num_nodes,
                                         int extra_after,
                                         std::string extra_after_symbol_) {
    token_dict = std::move(d);
    extra_after_symbol = std::move(extra_after_symbol_);
    num_special = 0;
    num_extra = 0;

    // V1 heuristic: entries whose key parses as an integer count as vocab
    // tokens; entries whose key starts with `extra_after_symbol` followed
    // by more characters count as extras; everything else is special.
    int num_node_vocab = 0;
    for (const auto& [key, _value] : token_dict) {
        try {
            (void)std::stoi(key);
            ++num_node_vocab;
        } catch (const std::invalid_argument&) {
            if (key.rfind(extra_after_symbol, 0) == 0 &&
                key.size() > extra_after_symbol.size()) {
                ++num_extra;
            } else {
                ++num_special;
            }
        }
    }
    max_vocab = num_special + num_node_vocab;

    if (max_num_nodes > 0 && num_node_vocab != max_num_nodes) {
        throw std::invalid_argument(
            "Dictionary vocab size (" + std::to_string(num_node_vocab) +
            ") does not match max_num_nodes (" + std::to_string(max_num_nodes) + ")");
    }
    if (extra_after > 0 && num_extra != extra_after) {
        throw std::invalid_argument(
            "Dictionary extra tokens (" + std::to_string(num_extra) +
            ") do not match extra_after (" + std::to_string(extra_after) + ")");
    }
}

void WorkerSharedContext::set_default_dictionary(int max_num_nodes,
                                                 int extra_after,
                                                 std::string extra_after_symbol_) {
    extra_after_symbol = std::move(extra_after_symbol_);
    token_dict = {
        {"<s>",   TOK_BOS},
        {"<pad>", TOK_PAD},
        {"</s>",  TOK_EOS},
        {"<unk>", TOK_UNK},
        {"|",     TOK_EDGE},
        {"!",     TOK_THINK},
        {"=",     TOK_TASK_START},
        {".",     TOK_TASK_END},
        {"t1",    TOK_T1},
        {"t2",    TOK_T2},
        {"t3",    TOK_T3},
        {"t4",    TOK_T4},
        {"t5",    TOK_T5},
        {"/",     TOK_QUERY_START},
        {"?",     TOK_QUERY_END},
        {"@",     TOK_AT},
        {"#",     TOK_SCRATCH_START},
        {"[",     TOK_BFS_ADJ_START},
        {"]",     TOK_BFS_ADJ_END},
        {"{",     TOK_CURLY_START},
        {"}",     TOK_CURLY_END},
        {"$",     TOK_DOLLAR},
        {extra_after_symbol, TOK_D_PREFIX},
    };

    // Sanity: indices must be contiguous starting at zero.
    int max_idx = 0;
    for (const auto& [_key, value] : token_dict) {
        if (value > max_idx) max_idx = value;
    }
    if (max_idx + 1 != static_cast<int>(token_dict.size())) {
        throw std::invalid_argument("Default dictionary indices are not contiguous");
    }
    num_special = static_cast<int>(token_dict.size());
    num_extra = 0;
    max_vocab = num_special;

    if (max_num_nodes > 0) {
        max_vocab = num_special + max_num_nodes;
        for (int i = num_special; i < num_special + max_num_nodes; ++i) {
            token_dict[std::to_string(i - num_special)] = i;
        }
    }
    if (extra_after > 0) {
        num_extra = extra_after;
        const int current_size = static_cast<int>(token_dict.size());
        for (int i = 0; i < extra_after; ++i) {
            token_dict[extra_after_symbol + std::to_string(i)] = current_size + i;
        }
    }
}

void WorkerSharedContext::set_pos_dictionary(std::map<std::string, PosRange> d) {
    // Structural validation of each range in isolation.
    for (const auto& [name, range] : d) {
        const auto [start, end] = range;
        if (start < 0) {
            throw std::invalid_argument(
                "pos_dictionary entry '" + name + "' has negative start (" +
                std::to_string(start) + ")");
        }
        if (start >= end) {
            throw std::invalid_argument(
                "pos_dictionary entry '" + name + "' has start >= end (" +
                std::to_string(start) + ", " + std::to_string(end) +
                "); ranges are half-open [start, end) so start < end is required");
        }
    }

    // Cross-entry validation: sort by start, verify no overlap. Gaps allowed.
    std::vector<std::pair<PosRange, std::string>> sorted;
    sorted.reserve(d.size());
    for (const auto& [name, range] : d) sorted.emplace_back(range, name);
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b) { return a.first.first < b.first.first; });
    for (std::size_t i = 1; i < sorted.size(); ++i) {
        const auto& [prev_range, prev_name] = sorted[i - 1];
        const auto& [cur_range,  cur_name]  = sorted[i];
        if (cur_range.first < prev_range.second) {
            throw std::invalid_argument(
                "pos_dictionary entries '" + prev_name + "' [" +
                std::to_string(prev_range.first) + ", " +
                std::to_string(prev_range.second) + ") and '" + cur_name + "' [" +
                std::to_string(cur_range.first) + ", " +
                std::to_string(cur_range.second) + ") overlap");
        }
    }

    pos_dict = std::move(d);
}

void WorkerSharedContext::set_default_pos_dictionary() {
    // Half-open [start, end) ranges. Numeric coverage matches V1's inclusive
    // layout (V1 misc_start=11, misc_end=49 -> [11, 50), etc.).
    pos_dict = {
        // Scalars: single-position markers, encoded as 1-wide ranges.
        {"pad",                   {   0,    1}},
        {"query_invariance",      {   1,    2}},
        {"edge_invariance",       {   2,    3}},
        {"node_invariance",       {   3,    4}},
        {"graph_invariance",      {   4,    5}},
        {"scratchpad_invariance", {   5,    6}},
        {"task_invariance",       {   6,    7}},

        // Ranges. Note gap [7, 11) between the invariance scalars and 'misc'
        // is intentional and matches V1.
        {"misc",                  {  11,   50}},
        {"query",                 {  50,  200}},
        {"graph",                 { 200,  940}},
        {"graph_sub",             { 940,  950}},
        {"thinking",              { 950, 1000}},
        {"task",                  {1000, 5001}},
    };
}

std::string WorkerSharedContext::validate_for_config(const GeneratorConfig& cfg) const {
    const auto has_range = [&](const std::string& name) {
        return pos_dict.find(name) != pos_dict.end();
    };

    // Universally required: every task uses 'pad' as the embedding pad.
    if (!has_range("pad")) {
        return "pos_dictionary missing required range 'pad'";
    }

    // Topology-based graph_kinds tokenize a real graph and need graph
    // positional embeddings. khops / khops_gen work on hop-prefix data
    // and do not.
    static const std::vector<std::string> kinds_needing_graph = {
        "erdos_renyi", "euclidean", "random_tree", "path_star", "balanced",
    };
    const bool needs_graph =
        std::find(kinds_needing_graph.begin(), kinds_needing_graph.end(),
                  cfg.graph_kind) != kinds_needing_graph.end();
    if (needs_graph && !has_range("graph")) {
        return "pos_dictionary missing required range 'graph' for graph_kind='" +
               cfg.graph_kind + "'";
    }

    return "";
}

void WorkerSharedContext::extend_validation_hashes(const std::vector<std::uint64_t>& hashes) {
    validation_hashes.insert(hashes.begin(), hashes.end());
}

void WorkerSharedContext::extend_test_hashes(const std::vector<std::uint64_t>& hashes) {
    test_hashes.insert(hashes.begin(), hashes.end());
}

}  // namespace graphgen
