// Per-run state shared across every Worker instance.
//
// Owned by a std::shared_ptr; populated once at start-up (default ctor
// materialises V1's default token + positional dictionaries and leaves
// hash-based filters empty) and then treated as read-only while workers
// consume it. Setters exist so the Python side can override defaults
// *before* handing the context to any Worker; mutating after the fact
// races the workers.
//
// The TOK_* static constants document the fixed positions of the special
// tokens in the default dictionary; custom dictionaries passed via
// set_dictionary() must honour them so downstream sampler / tokenizer
// code can name tokens by symbol without a runtime lookup.

#ifndef GRAPHGEN_WORKER_SHARED_CONTEXT_H
#define GRAPHGEN_WORKER_SHARED_CONTEXT_H

#include <cstdint>
#include <map>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace graphgen {

struct GeneratorConfig;  // fwd -- validate_for_config takes one by const ref

// Positional-embedding range. Half-open interval [start, end) into the
// positional-embedding table. `start == end - 1` represents a scalar
// position (e.g. "pad", per-kind invariance markers).
using PosRange = std::pair<int, int>;

struct WorkerSharedContext {
    // Fixed special-token positions (indices into token_dict).
    // Matches the layout materialised by set_default_dictionary().
    static constexpr int TOK_BOS           = 0;   // "<s>"
    static constexpr int TOK_PAD           = 1;   // "<pad>"
    static constexpr int TOK_EOS           = 2;   // "</s>"
    static constexpr int TOK_UNK           = 3;   // "<unk>"
    static constexpr int TOK_EDGE          = 4;   // "|"  edge marker
    static constexpr int TOK_THINK         = 5;   // "!"  thinking token
    static constexpr int TOK_TASK_START    = 6;   // "="
    static constexpr int TOK_TASK_END      = 7;   // "."
    static constexpr int TOK_T1            = 8;   // "t1" task-type markers for
    static constexpr int TOK_T2            = 9;   // "t2" multi-task training
    static constexpr int TOK_T3            = 10;  // "t3"
    static constexpr int TOK_T4            = 11;  // "t4"
    static constexpr int TOK_T5            = 12;  // "t5"
    static constexpr int TOK_QUERY_START   = 13;  // "/"
    static constexpr int TOK_QUERY_END     = 14;  // "?"
    static constexpr int TOK_AT            = 15;  // "@"
    static constexpr int TOK_SCRATCH_START = 16;  // "#"
    static constexpr int TOK_SCRATCH_END   = 17;  // "%"
    static constexpr int TOK_BFS_ADJ_START = 18;  // "["
    static constexpr int TOK_BFS_ADJ_END   = 19;  // "]"
    static constexpr int TOK_CURLY_START   = 20;  // "{"
    static constexpr int TOK_CURLY_END     = 21;  // "}"
    static constexpr int TOK_DOLLAR        = 22;  // "$"
    static constexpr int TOK_D_PREFIX      = 23;  // "D"  extra-token prefix
    static constexpr int TOK_THINK_START   = 24;  // "<"
    static constexpr int TOK_THINK_END     = 25;  // ">"

    // Number of special tokens in the default dictionary (indices [0, 26)).
    static constexpr int NUM_SPECIAL_DEFAULT = 26;

    // Dictionaries. token_dict is a flat string -> id lookup;
    // pos_dict maps a name to a half-open [start, end) range in the
    // positional-embedding table. Filled by the ctor or by the setters.
    std::map<std::string, int> token_dict;
    std::map<std::string, PosRange> pos_dict;

    // Layout metadata for token_dict. Computed by set_dictionary /
    // set_default_dictionary; consumed by samplers to know where the
    // vocab range starts and by tokenizers for validity checks.
    int num_special = 0;
    int num_extra = 0;
    int max_vocab = 0;
    std::string extra_after_symbol = "D";

    // Hash-based filters. Empty by default; populated once at start-up if
    // the caller wants validation / test splits enforced.
    std::unordered_set<std::uint64_t> validation_hashes;
    std::unordered_set<std::uint64_t> test_hashes;

    // Constructs with V1's default token dictionary (max_num_nodes=50,
    // extra_after=0, extra_after_symbol="D") and V1's default positional
    // dictionary. Hash sets start empty.
    WorkerSharedContext();

    // Replaces the token dictionary with the caller-supplied mapping.
    // Counts special / extra / vocab entries; when the corresponding
    // argument is non-negative, verifies the counts match.
    // Throws std::invalid_argument on mismatch.
    void set_dictionary(std::map<std::string, int> d,
                        int max_num_nodes = -1,
                        int extra_after = -1,
                        std::string extra_after_symbol_ = "D");

    // Overwrites the token dictionary with the canonical default layout
    // (special tokens 0..22, then max_num_nodes integer-named vocab
    // tokens, then extra_after "<symbol><n>" tokens).
    void set_default_dictionary(int max_num_nodes = 50,
                                int extra_after = 0,
                                std::string extra_after_symbol_ = "D");

    // Replaces the positional dictionary. Validates each entry
    // structurally (start >= 0, start < end) and rejects overlapping
    // ranges. Gaps between ranges are allowed. Content-level rules
    // ("a graph task needs a 'graph' range") are enforced separately
    // by validate_for_config().
    void set_pos_dictionary(std::map<std::string, PosRange> d);

    // Overwrites the positional dictionary with the canonical default:
    // scalars for pad + per-kind invariance markers, then ranges for
    // misc / query / graph / graph_sub / thinking / task. Numeric layout
    // mirrors V1's coverage under half-open semantics.
    void set_default_pos_dictionary();

    // Returns "" if the current pos_dict has every range the given
    // config needs, else a human-readable error message describing what's
    // missing. Content requirements are minimal today ("pad" always;
    // "graph" for topology-based graph_kinds); more rules are added as
    // samplers and tokenizers land.
    [[nodiscard]] std::string validate_for_config(const GeneratorConfig& cfg) const;

    // Appends to the hash-set filters. May be called multiple times.
    void extend_validation_hashes(const std::vector<std::uint64_t>& hashes);
    void extend_test_hashes(const std::vector<std::uint64_t>& hashes);

    // Fast single-element membership checks. Used by batch pybind helpers
    // and by any future C++ code that needs to skip generated instances.
    bool in_validation(std::uint64_t h) const {
        return validation_hashes.find(h) != validation_hashes.end();
    }
    bool in_test(std::uint64_t h) const {
        return test_hashes.find(h) != test_hashes.end();
    }
};

}  // namespace graphgen

#endif  // GRAPHGEN_WORKER_SHARED_CONTEXT_H
