// Python bindings for the graphgen extension module.
//
// Everything of substance lives in graphgen/ headers and src/*.cpp; this
// translation unit only exposes the public API to Python via pybind11.
// Three types are bound: WorkerSharedContext (immutable, worker-shared
// state), Worker (per-DataLoader-worker sampler/tokeniser), and
// GeneratorConfig (declarative request describing one batch).

#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "graphgen/check.h"
#include "graphgen/generator_config.h"
#include "graphgen/worker_shared_context.h"
#include "graphgen/worker.h"
#include "graphgen/random_utils.h"

namespace py = pybind11;

namespace {

// Batched membership check helper. Returns a numpy bool array `out` where
// out[i] = pred(hashes[i]). Kept local because it's only wanted by the
// pybind glue; core C++ code uses the per-element predicate directly.
template <class Pred>
py::array_t<bool> batch_contains(const py::array_t<std::uint64_t, py::array::c_style>& hashes,
                                 Pred pred) {
    const auto n = static_cast<py::ssize_t>(hashes.size());
    py::array_t<bool> out(n);
    auto out_ = out.mutable_unchecked<1>();
    auto in_  = hashes.unchecked<1>();
    for (py::ssize_t i = 0; i < n; ++i) out_(i) = pred(in_(i));
    return out;
}

}  // namespace

PYBIND11_MODULE(generator, m) {
    m.doc() = "graphgen -- graph generation for language-model training";

    py::class_<graphgen::WorkerSharedContext,
               std::shared_ptr<graphgen::WorkerSharedContext>>(
        m, "WorkerSharedContext")
        .def(py::init<>(),
             "Constructs a context with V1's default token + positional "
             "dictionaries and empty hash filters.")
        // Layout metadata read-only from Python (mutated by setters below).
        .def_readonly("token_dict",         &graphgen::WorkerSharedContext::token_dict)
        .def_readonly("pos_dict",           &graphgen::WorkerSharedContext::pos_dict)
        .def_readonly("num_special",        &graphgen::WorkerSharedContext::num_special)
        .def_readonly("num_extra",          &graphgen::WorkerSharedContext::num_extra)
        .def_readonly("max_vocab",          &graphgen::WorkerSharedContext::max_vocab)
        .def_readonly("extra_after_symbol", &graphgen::WorkerSharedContext::extra_after_symbol)
        // Special-token IDs surfaced for Python consumers that need them.
        .def_readonly_static("NUM_SPECIAL_DEFAULT", &graphgen::WorkerSharedContext::NUM_SPECIAL_DEFAULT)
        .def_readonly_static("TOK_BOS",        &graphgen::WorkerSharedContext::TOK_BOS)
        .def_readonly_static("TOK_PAD",        &graphgen::WorkerSharedContext::TOK_PAD)
        .def_readonly_static("TOK_EOS",        &graphgen::WorkerSharedContext::TOK_EOS)
        .def_readonly_static("TOK_UNK",        &graphgen::WorkerSharedContext::TOK_UNK)
        .def_readonly_static("TOK_EDGE",       &graphgen::WorkerSharedContext::TOK_EDGE)
        // Dictionary setters (call before handing the ctx to any Worker).
        .def("set_dictionary", &graphgen::WorkerSharedContext::set_dictionary,
             py::arg("d"), py::arg("max_num_nodes") = -1, py::arg("extra_after") = -1,
             py::arg("extra_after_symbol") = "D",
             "Replace the token dictionary with the caller's mapping. "
             "Validates counts if max_num_nodes/extra_after are >= 0.")
        .def("set_default_dictionary", &graphgen::WorkerSharedContext::set_default_dictionary,
             py::arg("max_num_nodes") = 50, py::arg("extra_after") = 0,
             py::arg("extra_after_symbol") = "D",
             "Overwrite the token dictionary with the canonical default layout.")
        .def("set_pos_dictionary", &graphgen::WorkerSharedContext::set_pos_dictionary,
             py::arg("d"),
             "Replace the positional dictionary with a name -> (start, end) "
             "mapping. Half-open [start, end) semantics; rejects overlapping "
             "ranges and structurally invalid entries.")
        .def("set_default_pos_dictionary", &graphgen::WorkerSharedContext::set_default_pos_dictionary,
             "Overwrite the positional dictionary with the canonical default.")
        .def("validate_for_config", &graphgen::WorkerSharedContext::validate_for_config,
             py::arg("cfg"),
             "Return '' if pos_dict has every range the config needs, else "
             "a human-readable error message.")
        // Hash-filter population + queries.
        .def("extend_validation_hashes", &graphgen::WorkerSharedContext::extend_validation_hashes,
             py::arg("hashes"))
        .def("extend_test_hashes", &graphgen::WorkerSharedContext::extend_test_hashes,
             py::arg("hashes"))
        .def_property_readonly("validation_size",
             [](const graphgen::WorkerSharedContext& c) { return c.validation_hashes.size(); })
        .def_property_readonly("test_size",
             [](const graphgen::WorkerSharedContext& c) { return c.test_hashes.size(); })
        .def("in_validation", &graphgen::WorkerSharedContext::in_validation, py::arg("h"))
        .def("in_test",       &graphgen::WorkerSharedContext::in_test,       py::arg("h"))
        .def("is_in_validation",
             [](const graphgen::WorkerSharedContext& c,
                const py::array_t<std::uint64_t, py::array::c_style>& hashes) {
                 return batch_contains(hashes, [&](std::uint64_t h) { return c.in_validation(h); });
             }, py::arg("hashes"),
             "Batch membership check against validation_hashes. Returns "
             "a numpy bool array parallel to the input.")
        .def("is_in_test",
             [](const graphgen::WorkerSharedContext& c,
                const py::array_t<std::uint64_t, py::array::c_style>& hashes) {
                 return batch_contains(hashes, [&](std::uint64_t h) { return c.in_test(h); });
             }, py::arg("hashes"))
        .def("is_invalid_example",
             [](const graphgen::WorkerSharedContext& c,
                const py::array_t<std::uint64_t, py::array::c_style>& hashes) {
                 return batch_contains(hashes, [&](std::uint64_t h) {
                     return c.in_validation(h) || c.in_test(h);
                 });
             }, py::arg("hashes"),
             "True where the hash is in *either* the validation or test filter.");

    py::class_<graphgen::Worker>(m, "Worker")
        .def(py::init<std::shared_ptr<graphgen::WorkerSharedContext>, std::uint64_t>(),
             py::arg("ctx"), py::arg("seed") = 0)
        .def("generate_batch", &graphgen::Worker::generate_batch, py::arg("cfg"),
             "Sample, tokenize, and package one batch of graph tasks.")
        .def("sample_graph_stats", &graphgen::Worker::sample_graph_stats, py::arg("cfg"),
             "Test-only: sample one graph and return "
             "{num_vertices, num_edges, num_components, vocab_ids}. Goes away "
             "once generate_batch grows a real return shape.")
        .def("sample_shortest_path_stats", &graphgen::Worker::sample_shortest_path_stats,
             py::arg("cfg"),
             "Test-only: run graph_sampler + task_computer for one item and "
             "return {num_vertices, num_edges, start, end, path, path_length, "
             "valid_next_hops}. Goes away once generate_batch grows a real "
             "return shape.");

    py::class_<graphgen::GeneratorConfig>(m, "GeneratorConfig")
        .def(py::init<>())
        .def(py::init<const py::kwargs &>(),
             "Construct from Python kwargs; validates cross-field constraints "
             "and raises ValueError on failure.")
        .def("validate", &graphgen::GeneratorConfig::validate,
             "Returns '' if the config is valid, else a human-readable error "
             "message.")
        .def("to_dict", &graphgen::GeneratorConfig::to_dict,
             "Round-trip the config to a Python dict. Unset optional fields "
             "appear as None.")
        .def("__repr__", &graphgen::GeneratorConfig::to_string)
        // Shared
        .def_readwrite("min_num_nodes", &graphgen::GeneratorConfig::min_num_nodes)
        .def_readwrite("max_num_nodes", &graphgen::GeneratorConfig::max_num_nodes)
        .def_readwrite("batch_size",    &graphgen::GeneratorConfig::batch_size)
        .def_readwrite("max_edges",     &graphgen::GeneratorConfig::max_edges)
        .def_readwrite("max_attempts",  &graphgen::GeneratorConfig::max_attempts)
        // Dispatch
        .def_readwrite("graph_kind",      &graphgen::GeneratorConfig::graph_kind)
        .def_readwrite("task_kind",       &graphgen::GeneratorConfig::task_kind)
        .def_readwrite("scratchpad_kind", &graphgen::GeneratorConfig::scratchpad_kind)
        // Graph structure
        .def_readwrite("directed", &graphgen::GeneratorConfig::directed)
        .def_readwrite("weighted", &graphgen::GeneratorConfig::weighted)
        // Tokenization
        .def_readwrite("tokenization_mode",                   &graphgen::GeneratorConfig::tokenization_mode)
        .def_readwrite("query_at_end",                                 &graphgen::GeneratorConfig::query_at_end)
        .def_readwrite("include_graph_in_graph_tokenization",          &graphgen::GeneratorConfig::include_graph_in_graph_tokenization)
        .def_readwrite("include_duplicate_edges_in_graph_tokenization", &graphgen::GeneratorConfig::include_duplicate_edges_in_graph_tokenization)
        .def_readwrite("include_nodes_in_graph_tokenization",          &graphgen::GeneratorConfig::include_nodes_in_graph_tokenization)
        .def_readwrite("num_thinking_tokens",                          &graphgen::GeneratorConfig::num_thinking_tokens)
        .def_readwrite("align_prefix_front_pad",                       &graphgen::GeneratorConfig::align_prefix_front_pad)
        // Pos ids
        .def_readwrite("return_pos_ids",       &graphgen::GeneratorConfig::return_pos_ids)
        // Distance / geometry returns
        .def_readwrite("return_hop_distances", &graphgen::GeneratorConfig::return_hop_distances)
        .def_readwrite("return_positions",     &graphgen::GeneratorConfig::return_positions)
        .def_readwrite("return_distance_rank_targets",
                       &graphgen::GeneratorConfig::return_distance_rank_targets)
        // Task-specific
        .def_readwrite("min_path_length",     &graphgen::GeneratorConfig::min_path_length)
        .def_readwrite("max_path_length",     &graphgen::GeneratorConfig::max_path_length)
        .def_readwrite("task_sample_dist",    &graphgen::GeneratorConfig::task_sample_dist)
        .def_readwrite("start_at_root",       &graphgen::GeneratorConfig::start_at_root)
        .def_readwrite("end_at_leaf",         &graphgen::GeneratorConfig::end_at_leaf)
        .def_readwrite("probs",               &graphgen::GeneratorConfig::probs)
        .def_readwrite("min_query_size",      &graphgen::GeneratorConfig::min_query_size)
        .def_readwrite("max_query_size",      &graphgen::GeneratorConfig::max_query_size)
        .def_readwrite("given_query",         &graphgen::GeneratorConfig::given_query)
        .def_readwrite("min_khops",           &graphgen::GeneratorConfig::min_khops)
        .def_readwrite("max_khops",           &graphgen::GeneratorConfig::max_khops)
        .def_readwrite("min_prefix_length",   &graphgen::GeneratorConfig::min_prefix_length)
        .def_readwrite("max_prefix_length",   &graphgen::GeneratorConfig::max_prefix_length)
        .def_readwrite("right_side_connect",  &graphgen::GeneratorConfig::right_side_connect)
        .def_readwrite("khops_no_repeats",    &graphgen::GeneratorConfig::khops_no_repeats)
        .def_readwrite("permutation_version", &graphgen::GeneratorConfig::permutation_version)
        .def_readwrite("mask_to_vocab_size",  &graphgen::GeneratorConfig::mask_to_vocab_size)
        .def_readwrite("mask_to_size",        &graphgen::GeneratorConfig::mask_to_size)
        .def_readwrite("intermediate_labels", &graphgen::GeneratorConfig::intermediate_labels)
        .def_readwrite("partition_method",    &graphgen::GeneratorConfig::partition_method)
        // Scratchpad
        .def_readwrite("bfs_scratchpad_style",      &graphgen::GeneratorConfig::bfs_scratchpad_style)
        .def_readwrite("stop_once_found",           &graphgen::GeneratorConfig::stop_once_found)
        // Graph-kind-specific
        .def_readwrite("edge_prob",       &graphgen::GeneratorConfig::edge_prob)
        .def_readwrite("dim",             &graphgen::GeneratorConfig::dim)
        .def_readwrite("min_edge_length", &graphgen::GeneratorConfig::min_edge_length)
        .def_readwrite("max_edge_length", &graphgen::GeneratorConfig::max_edge_length)
        .def_readwrite("dims",            &graphgen::GeneratorConfig::dims)
        .def_readwrite("min_arms",        &graphgen::GeneratorConfig::min_arms)
        .def_readwrite("max_arms",        &graphgen::GeneratorConfig::max_arms)
        .def_readwrite("min_arm_length",  &graphgen::GeneratorConfig::min_arm_length)
        .def_readwrite("max_arm_length",  &graphgen::GeneratorConfig::max_arm_length)
        .def_readwrite("min_lookahead",     &graphgen::GeneratorConfig::min_lookahead)
        .def_readwrite("max_lookahead",     &graphgen::GeneratorConfig::max_lookahead)
        .def_readwrite("min_noise_reserve", &graphgen::GeneratorConfig::min_noise_reserve)
        .def_readwrite("max_num_parents",   &graphgen::GeneratorConfig::max_num_parents)
        .def_readwrite("max_noise",         &graphgen::GeneratorConfig::max_noise)
        // Debug
        .def_readwrite("print_cpp_args", &graphgen::GeneratorConfig::print_cpp_args);

    // -------------------------------------------------------------------
    // Free-function helpers for khops / khops_gen consumers on the
    // Python side. Kept as module-level defs (not methods on any class)
    // because they're pure math with no need for a Worker / ctx.
    // -------------------------------------------------------------------

    // Sample an integer partition of Q into N positive parts.
    //   method='uniform'      -- uniform over all partitions (Locey 2013 DP).
    //   method='non_uniform'  -- cheap greedy; each part >= 1, biased.
    // Returns a Python list of length N summing to Q. `seed=None`
    // constructs a fresh random_device-seeded RNG; passing an int
    // seed gives determinism.
    m.def("uniform_random_int_partition",
          [](int Q, int N, std::optional<std::uint64_t> seed,
             const std::string& method, bool shuffle) {
              std::mt19937_64 rng(seed.value_or(std::random_device{}()));
              graphgen::SampleIntPartition sampler;
              if (method == "uniform") {
                  return sampler.uniform_random_partition(Q, N, rng, shuffle);
              }
              if (method == "non_uniform") {
                  return sampler.non_uniform_random_partition(Q, N, rng, shuffle);
              }
              throw std::invalid_argument(
                  "uniform_random_int_partition: method must be 'uniform' or "
                  "'non_uniform' (got '" + method + "')");
          },
          py::arg("Q"), py::arg("N"),
          py::arg("seed") = py::none(),
          py::arg("method") = "uniform",
          py::arg("shuffle") = true,
          "Sample an integer partition of Q into N positive parts.");

    // Verify a batch of khops_gen (prefix, ground_truths) pairs. Walks
    // the prefix from the cursor position backward `k` times and
    // checks that each hop lands on the corresponding ground truth.
    // Returns a (B,) int32 array of 0/1 flags.
    m.def("verify_khop_gens",
          [](py::array_t<int, py::array::c_style> prefixes,
             py::array_t<int, py::array::c_style> prefix_lengths,
             py::array_t<int, py::array::c_style> ground_truths,
             bool right_side_connect) {
              GG_CHECK(prefixes.ndim() == 2,
                       "verify_khop_gens: prefixes must be 2-D (B, max_prefix_len)");
              GG_CHECK(prefix_lengths.ndim() == 1,
                       "verify_khop_gens: prefix_lengths must be 1-D (B,)");
              GG_CHECK(ground_truths.ndim() == 2,
                       "verify_khop_gens: ground_truths must be 2-D (B, k)");
              const auto B     = prefixes.shape(0);
              const auto max_P = prefixes.shape(1);
              GG_CHECK(prefix_lengths.shape(0) == B,
                       "verify_khop_gens: prefix_lengths.shape[0] != B");
              GG_CHECK(ground_truths.shape(0) == B,
                       "verify_khop_gens: ground_truths.shape[0] != B");
              const auto k = ground_truths.shape(1);

              py::array_t<int> out(B);
              auto out_       = out.mutable_unchecked<1>();
              auto pfx_       = prefixes.unchecked<2>();
              auto len_       = prefix_lengths.unchecked<1>();
              auto gt_        = ground_truths.unchecked<2>();

              for (py::ssize_t b = 0; b < B; ++b) {
                  const int P = len_(b);
                  GG_CHECK(P > 0 && P <= max_P,
                           "verify_khop_gens: prefix_length out of range");
                  // Reconstruct semantics ported verbatim from V1's
                  // verify_khop_gen: build the backtrace list
                  // [cursor, ..., gt_0] by walking the prefix
                  // backward, then reverse and compare to the
                  // provided ground truths (which are in FORWARD
                  // order: ground_truths[0] = gt at entry of segment
                  // 0, ..., ground_truths[k-1] = gt at entry of last
                  // segment == cursor).
                  std::vector<int> reconstruct;
                  reconstruct.reserve(static_cast<std::size_t>(k));
                  int cur_value = pfx_(b, P - 2);   // cursor
                  reconstruct.push_back(cur_value);
                  int cur_idx = P - 4;
                  // V1 used `cur_idx > 0` here, which silently missed
                  // matches at index 0 when the very first segment was
                  // short. Relax to `>= 0` under right_side_connect
                  // (safe: we index cur_idx + 1) and `>= 1` under
                  // left-side (need cur_idx - 1 to be in-range).
                  const int min_idx = right_side_connect ? 0 : 1;
                  while (cur_idx >= min_idx) {
                      if (pfx_(b, cur_idx) == cur_value) {
                          if (right_side_connect) {
                              cur_value = pfx_(b, cur_idx + 1);
                          } else {
                              cur_value = pfx_(b, cur_idx - 1);
                              // Extra decrement matches V1: skip the
                              // slot we just consumed as the "next"
                              // pointer so it can't be double-matched
                              // on the following iteration.
                              --cur_idx;
                          }
                          reconstruct.push_back(cur_value);
                      }
                      --cur_idx;
                  }
                  std::reverse(reconstruct.begin(), reconstruct.end());

                  bool ok = (static_cast<py::ssize_t>(reconstruct.size()) == k);
                  if (ok) {
                      for (py::ssize_t j = 0; j < k; ++j) {
                          if (reconstruct[static_cast<std::size_t>(j)] != gt_(b, j)) {
                              ok = false; break;
                          }
                      }
                  }
                  out_(b) = ok ? 1 : 0;
              }
              return out;
          },
          py::arg("prefixes"),
          py::arg("prefix_lengths"),
          py::arg("ground_truths"),
          py::arg("right_side_connect") = true,
          "Verify khops_gen backtraces. Returns a (B,) int32 array of "
          "0/1 flags: 1 iff walking backward from prefix[-2] hits each "
          "ground truth in order.");
}
