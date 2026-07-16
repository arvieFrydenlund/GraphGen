// Python bindings for the graphgen extension module.
//
// Everything of substance lives in graphgen/ headers and src/*.cpp; this
// translation unit only exposes the public API to Python via pybind11.
// Three types are bound: WorkerSharedContext (immutable, worker-shared
// state), Worker (per-DataLoader-worker sampler/tokeniser), and
// GeneratorConfig (declarative request describing one batch).

#include <cstdint>
#include <memory>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "graphgen/generator_config.h"
#include "graphgen/worker_shared_context.h"
#include "graphgen/worker.h"

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
             "Sample, tokenize, and package one batch of graph tasks.");

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
        .def_readwrite("min_vocab",     &graphgen::GeneratorConfig::min_vocab)
        .def_readwrite("max_vocab",     &graphgen::GeneratorConfig::max_vocab)
        .def_readwrite("batch_size",    &graphgen::GeneratorConfig::batch_size)
        .def_readwrite("max_edges",     &graphgen::GeneratorConfig::max_edges)
        .def_readwrite("max_attempts",  &graphgen::GeneratorConfig::max_attempts)
        // Dispatch
        .def_readwrite("graph_kind",      &graphgen::GeneratorConfig::graph_kind)
        .def_readwrite("task_kind",       &graphgen::GeneratorConfig::task_kind)
        .def_readwrite("scratchpad_kind", &graphgen::GeneratorConfig::scratchpad_kind)
        // Tokenization
        .def_readwrite("is_causal",                           &graphgen::GeneratorConfig::is_causal)
        .def_readwrite("is_direct_ranking",                   &graphgen::GeneratorConfig::is_direct_ranking)
        .def_readwrite("query_at_end",                        &graphgen::GeneratorConfig::query_at_end)
        .def_readwrite("no_graph",                            &graphgen::GeneratorConfig::no_graph)
        .def_readwrite("concat_edges",                        &graphgen::GeneratorConfig::concat_edges)
        .def_readwrite("duplicate_edges",                     &graphgen::GeneratorConfig::duplicate_edges)
        .def_readwrite("include_nodes_in_graph_tokenization", &graphgen::GeneratorConfig::include_nodes_in_graph_tokenization)
        .def_readwrite("num_thinking_tokens",                 &graphgen::GeneratorConfig::num_thinking_tokens)
        .def_readwrite("scratchpad_as_prefix",                &graphgen::GeneratorConfig::scratchpad_as_prefix)
        .def_readwrite("is_flat_model",                       &graphgen::GeneratorConfig::is_flat_model)
        .def_readwrite("align_prefix_front_pad",              &graphgen::GeneratorConfig::align_prefix_front_pad)
        // Pos ids
        .def_readwrite("return_pos_ids",       &graphgen::GeneratorConfig::return_pos_ids)
        .def_readwrite("use_edges_invariance", &graphgen::GeneratorConfig::use_edges_invariance)
        .def_readwrite("use_node_invariance",  &graphgen::GeneratorConfig::use_node_invariance)
        .def_readwrite("use_graph_invariance", &graphgen::GeneratorConfig::use_graph_invariance)
        .def_readwrite("use_query_invariance", &graphgen::GeneratorConfig::use_query_invariance)
        .def_readwrite("use_graph_structure",  &graphgen::GeneratorConfig::use_graph_structure)
        .def_readwrite("use_full_structure",   &graphgen::GeneratorConfig::use_full_structure)
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
        .def_readwrite("sort_adjacency_lists",      &graphgen::GeneratorConfig::sort_adjacency_lists)
        .def_readwrite("use_unique_depth_markers",  &graphgen::GeneratorConfig::use_unique_depth_markers)
        .def_readwrite("stop_once_found",           &graphgen::GeneratorConfig::stop_once_found)
        .def_readwrite("include_queue",             &graphgen::GeneratorConfig::include_queue)
        .def_readwrite("reverse_adjacency_lists",   &graphgen::GeneratorConfig::reverse_adjacency_lists)
        .def_readwrite("duplicate_adjacency_lists", &graphgen::GeneratorConfig::duplicate_adjacency_lists)
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
}
