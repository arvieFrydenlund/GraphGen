//
// Created by arvie on 4/6/25.
//

#include <iostream>
#include <random>
#include <Python.h>
#include <chrono>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <thread>
#include <mutex>

#include "undirected_graphs.h"
#include "utils.h"
#include "instance.h"
#include "dictionaries.h"

using namespace std;
namespace py = pybind11;
using namespace py::literals;

/* ************************************************
 *  Seeding, needed for datastream in python
 *  ***********************************************/

inline unsigned int seed_ = std::random_device{}();
static thread_local auto gen = std::mt19937(seed_); // so each thread in the dataloader is different

inline unsigned int get_seed() {
    return seed_;
}

inline void set_seed(const unsigned int seed = 0) {
    gen.seed(seed);
    seed_ = seed;
}

inline int rand_int(int min, int max) {
    std::uniform_int_distribution<int> d(min, max);
    return d(gen);
}


/* ************************************************
 *  Test and valid datasets and hashing
 *  ***********************************************/

static unordered_set<std::uint64_t> validation_hashes;
static unordered_set<std::uint64_t> test_hashes;

int get_validation_size() {
    return validation_hashes.size();
}

int get_test_size() {
    return test_hashes.size();
}

inline void set_validation_hashes(const py::array_t<std::uint64_t, py::array::c_style> &hashes) {
    auto ra = hashes.unchecked();
    auto shape = hashes.shape();
    for (int i = 0; i < shape[0]; i++) {
        validation_hashes.insert(ra(i));
    }
}

inline void set_test_hashes(const py::array_t<std::uint64_t, py::array::c_style> &hashes) {
    auto ra = hashes.unchecked();
    auto shape = hashes.shape();
    for (int i = 0; i < shape[0]; i++) {
        test_hashes.insert(ra(i));
    }
}


inline py::array_t<bool, py::array::c_style> is_in_validation(
    const py::array_t<std::uint64_t, py::array::c_style> &hashes) {
    py::array_t<bool, py::array::c_style> arr(static_cast<int>(hashes.size()));
    auto ra = arr.mutable_unchecked();
    auto rh = hashes.unchecked();
    auto shape = hashes.shape();
    for (int i = 0; i < shape[0]; i++) {
        if (validation_hashes.find(rh(i)) != validation_hashes.end()) {
            ra(i) = true;
        } else {
            ra(i) = false;
        }
    }
    return arr;
}

inline py::array_t<bool, py::array::c_style> is_in_test(const py::array_t<std::uint64_t, py::array::c_style> &hashes) {
    py::array_t<bool, py::array::c_style> arr(static_cast<int>(hashes.size()));
    auto ra = arr.mutable_unchecked();
    auto rh = hashes.unchecked();
    auto shape = hashes.shape();
    for (int i = 0; i < shape[0]; i++) {
        if (test_hashes.find(rh(i)) != test_hashes.end()) {
            ra(i) = true;
        } else {
            ra(i) = false;
        }
    }
    return arr;
}

inline py::array_t<bool, py::array::c_style> is_invalid_example(
    const py::array_t<std::uint64_t, py::array::c_style> &hashes) {
    py::array_t<bool, py::array::c_style> arr(static_cast<int>(hashes.size()));
    auto ra = arr.mutable_unchecked();
    arr[py::make_tuple(py::ellipsis())] = false; // initialize array
    auto rh = hashes.unchecked();
    auto shape = hashes.shape();
    for (int i = 0; i < shape[0]; i++) {
        if (validation_hashes.find(rh(i)) != validation_hashes.end()) {
            ra(i) = true;
        } else if (test_hashes.find(rh(i)) != test_hashes.end()) {
            ra(i) = true;
        }
    }
    return arr;
}

inline bool task_type_check(const std::string &task_type) {
    // Check if the task type is valid
    static const std::set<std::string> valid_task_types = {
        "shortest_path", "path", "center", "centroid", "none", "None",
    };
    return valid_task_types.find(task_type) != valid_task_types.end();
}


/* ************************************************
 *  Batched graph generation
 *  ***********************************************/

void check_args(const int c_min, const int c_max,
                const int batch_size,
                const int min_path_length, const int max_path_length,
                const int num_thinking_tokens) {
    if (c_min > c_max) { throw std::invalid_argument("Invalid arguments: c_min > c_max"); }
    if (batch_size <= 0) { throw std::invalid_argument("Invalid arguments: batch_size <= 0"); }
    if (min_path_length > max_path_length) { throw std::invalid_argument("Invalid arguments: min_path_length > max_path_length"); }
    if (num_thinking_tokens < 0) {
        throw std::invalid_argument("Invalid arguments: num_thinking_tokens < 0");
    }
    if (dictionary.empty()) {
        throw std::invalid_argument("Invalid arguments: dictionary is empty.  Please set it first.");
    }
}

vector<int> check_and_set_vocab_limits(int min_num_nodes, int &max_num_nodes,
                                         int min_vocab, int &max_vocab) {
    if (min_num_nodes <= 0) { throw std::invalid_argument("Invalid arguments: min_num_nodes <= 0"); }
    if (max_num_nodes == -1) {
        max_num_nodes = min_num_nodes;
    }
    if (min_vocab == -1 and max_vocab == -1){
        min_vocab = dictionary_num_special;
        max_vocab = dictionary_max_vocab;
    }else if (max_vocab == -1) {
        max_vocab = max_num_nodes;
    }
    if (max_vocab - min_vocab < max_num_nodes) {
        auto s = "Invalid arguments: max_vocab - min_vocab < max_num_nodes:  " +
                 to_string(max_vocab) + " - " + to_string(min_vocab) + " < " + to_string(max_num_nodes);
        throw std::invalid_argument(s);
    }
    return {min_num_nodes, max_num_nodes, min_vocab, max_vocab};
}

inline int sample_num_nodes(const int min_num_nodes, const int max_num_nodes) {
    if (max_num_nodes == min_num_nodes) {
        return max_num_nodes;
    }
    uniform_int_distribution<int> d(min_num_nodes, max_num_nodes);
    return d(gen);
};


inline int attempt_check(const int E, const int max_edges, const int attempts, const int max_attempts) {
    if (E > max_edges) {
        if (attempts > max_attempts) {
            cout << "Failed to generate graph after " << attempts << " attempts" << endl;
        }
        return 1;
    }
    return 0;
}


inline py::dict erdos_renyi_n(
    int min_num_nodes, int max_num_nodes, float p = -1.0, const int c_min = 75, const int c_max = 125,
    const string &task_type = "shortest_path",
    const int max_path_length = 10, const int min_path_length = 1,
    const bool sort_adjacency_lists = true, const bool use_unique_depth_markers = true,
    int max_query_size = -1, const int min_query_size = 2,
    const bool is_causal = false, const bool is_direct_ranking = false,
    const bool shuffle_edges = false, const bool shuffle_nodes = false,
    int min_vocab = -1, int max_vocab = -1,
    const int batch_size = 256, const int max_edges = 512, int max_attempts = 1000,
    const bool concat_edges = true,
    const bool duplicate_edges = false,
    const bool include_nodes_in_graph_tokenization = false,
    const bool query_at_end = true,
    const int num_thinking_tokens = 0,
    const string scratchpad_type = "none",
    const bool is_flat_model = true,
    const bool align_prefix_front_pad = false,
    const bool use_edges_invariance = false,  // for concated edges this allows true permutation invariance
    const bool use_node_invariance = false,
    const bool use_graph_invariance = false,
    const bool use_query_invariance = false,
    const bool use_task_structure = false,  // divide positions by task structure
    const bool use_graph_structure = false,  // 2d positions by graph structure
    const py::kwargs& kwargs = py::kwargs()) {

    check_args(c_min, c_max, batch_size, min_path_length, max_path_length, num_thinking_tokens);
    if (p > 1.0) { throw std::invalid_argument("Invalid arguments: p > 1.0"); }

    auto m = check_and_set_vocab_limits(min_num_nodes, max_num_nodes, min_vocab, max_vocab);
    min_num_nodes = m[0], max_num_nodes = m[1], min_vocab = m[2], max_vocab = m[3];

    optional<vector<float>> task_sample_dist = nullopt;
    if (kwargs.contains("task_sample_dist")) {
        if (!kwargs["task_sample_dist"].is_none() and !kwargs["task_sample_dist"].cast<py::list>().empty()) {
            task_sample_dist = kwargs["task_sample_dist"].cast<vector<float> >();
        }
    }
    optional<vector<int>> given_query = nullopt;

    auto batched_instances = BatchedInstances<boost::undirectedS>(
        "erdos_renyi", task_type,
                    min_vocab, max_vocab,
                    query_at_end, num_thinking_tokens, is_flat_model, align_prefix_front_pad);

    int attempts = 0;
    while (batched_instances.size() < batch_size && attempts < max_attempts) {
        int num_nodes = sample_num_nodes(min_num_nodes, max_num_nodes);
        unique_ptr<Graph<boost::undirectedS> > g_ptr;
        // auto graph_t = time_before();
        erdos_renyi_generator(g_ptr, num_nodes, gen, p, c_min, c_max, false);
        // time_after(graph_t, "graph gen");
        // const auto N = num_vertices(*g_ptr);
        const auto E = num_edges(*g_ptr);
        if (const auto a = attempt_check(E, max_edges, attempts, max_attempts)) {
            attempts += a;
            continue;
        }
        // auto pack_t = time_before();
        auto instance = Instance<boost::undirectedS>(g_ptr, gen, dictionary,
            min_vocab, max_vocab, shuffle_nodes, shuffle_edges,
            task_type, scratchpad_type,
            max_path_length, min_path_length, -1, -1, task_sample_dist,
            sort_adjacency_lists, use_unique_depth_markers,
            given_query, max_query_size, min_query_size,
            is_causal, is_direct_ranking, concat_edges, duplicate_edges, include_nodes_in_graph_tokenization,
            pos_dictionary, use_edges_invariance, use_node_invariance, use_graph_invariance, use_query_invariance,
            use_task_structure, use_graph_structure
            );
        // time_after(pack_t, "pack");
        batched_instances.add(instance);
    }
    return batched_instances.package_for_model(dictionary, pos_dictionary);
}



PYBIND11_MODULE(generator, m) {
    m.doc() = "Graph generation module"; // optional module docstring

    // seeding
    m.def("set_seed", &set_seed, "Sets random seed (unique to thread)", py::arg("seed") = 0);
    m.def("get_seed", &get_seed, "Gets random seed (unique to thread)");
    m.def("rand_int", &rand_int, "Gets random int (unique to thread)", py::arg("min") = 0, py::arg("max") = 100);

    // hashing test and validation sets
    m.def("get_validation_size", &get_validation_size,
          "Gets the size of the validation set.\n"
          "Returns:\n\t"
          "int: size of the validation set\n");

    m.def("get_test_size", &get_test_size,
          "Gets the size of the test set.\n"
          "Returns:\n\t"
          "int: size of the test set\n");

    m.def("set_validation_hashes", &set_validation_hashes,
          "Sets the validation hashes for the graph generation.  This is used to check if the graph generation is correct.\n"
          "Parameters:\n\t"
          "hashes: numpy [N] of uint64_t hashes\n\t",
          py::arg("hashes"));

    m.def("set_test_hashes", &set_test_hashes,
          "Sets the test hashes for the graph generation.  This is used to check if the graph generation is correct.\n"
          "Parameters:\n\t"
          "hashes: numpy [N] of uint64_t hashes\n\t",
          py::arg("hashes"));

    m.def("is_in_validation", &is_in_validation,
          "Checks if the graph generation is correct.\n"
          "Parameters:\n\t"
          "hashes: numpy [N] of uint64_t hashes\n\t"
          "Returns:\n\t"
          "numpy [N] True if the graph generation is correct, False otherwise\n",
          py::arg("hashes"));

    m.def("is_in_test", &is_in_test,
          "Checks if the graph generation is correct.\n"
          "Parameters:\n\t"
          "hashes: numpy [N] of uint64_t hashes\n\t"
          "Returns:\n\t"
          "numpy [N] True if the graph generation is correct, False otherwise\n",
          py::arg("hashes"));

    m.def("is_invalid_example", &is_invalid_example,
          "Validation and test sets check\n"
          "Parameters:\n\t"
          "hashes: numpy [N] of uint64_t hashes\n\t"
          "Returns:\n\t"
          "numpy [N] True if the graph is in the validation or test sets, False otherwise\n",
          py::arg("hashes"));

    m.def("task_type_check", &task_type_check,
          "Checks if the task type is valid.\n"
          "Parameters:\n\t"
          "task_type: str",
          py::arg("task_type"));

    // dictionary/vocabulary
    m.def("set_dictionary", &set_dictionary,
          "Sets the dictionary/vocabulary of token to token_idx.\n"
          "Parameters:\n\t"
          "dictionary: of str -> int\n\t",
          py::arg("dictionary"), py::arg("verbose") = false,
          py::arg("max_num_nodes") = 50,
          py::arg("extra_after") = 0,
          py::arg("extra_after_symbol") = "D");

    m.def("set_default_dictionary", &set_default_dictionary,
          "Sets the dictionary/vocabulary of token to token_idx.\n"
          "Parameters:\n\t"
          "None\n",
          py::arg("max_num_nodes") = 50,
          py::arg("extra_after") = 0,
          py::arg("extra_after_symbol") = "D");

    m.def("get_dictionary", &get_dictionary,
          "Gets the dictionary/vocabulary of token to token_idx.\n"
          "Parameters:\n\t"
          "None\n"
          "Returns:\n\t"
          "dictionary: of str -> int\n");

    m.def("get_dictionary_vocab_limits", &get_dictionary_vocab_limits,
          "1) num special symbols, 2) max vocab size");

    m.def("set_pos_dictionary", &set_pos_dictionary,
      "Sets the pos dictionary.\n"
      "Parameters:\n\t"
      "dictionary: of str -> int\n\t",
      py::arg("dictionary"), py::arg("verbose") = false);

    m.def("set_default_pos_dictionary", &set_default_pos_dictionary,
          "Sets the pos dictionary.\n"
          "Parameters:\n\t"
          "None\n");

    m.def("get_pos_dictionary", &get_pos_dictionary,
          "Gets the pos dictionary.\n"
          "Parameters:\n\t"
          "None\n"
          "Returns:\n\t"
          "dictionary: of str -> int\n");

    /*
    m.def("verify_paths", &verify_paths<int>,
          "Batch varies the that any predicted paths are valid given the distance matrices.\n"
          "Parameters:\n\t"
          "distances: [batch_size, vocab_size, vocab_size]\n\t"
          "queries: [batch_size, 2] of start, end\n\t"
          "paths: [batch_size, max_path_length]\n\t"
          "path_lengths: [batch_size]\n\t"
          "Returns:\n\t"
          "is_valid [batch_size], int, -1 if not valid, 0 if valid but not shortest, 1 if valid and shortest.\n",
          py::arg("distances"), py::arg("queries"), py::arg("paths"), py::arg("path_lengths"));
    */

    /*
    m.def("balanced_graph_size_check", &balanced_graph_size_check,
          "Check that the balanced graph size is valid.  Will fail assert otherwise.",
          py::arg("num_nodes"), py::arg("lookahead"), py::arg("min_noise_reserve") = 0);
    */

    // batched graph generation
    m.def("erdos_renyi_n", &erdos_renyi_n,
          "Generate a batch of Erdos Renyi graphs\nGraph specific parameters:\n\t"
          "p: probability of edge creation.  If -1 then 1/num_nodes.\n\t",

          py::arg("min_num_nodes"),
          py::arg("max_num_nodes"),
          py::arg("p") = -1.0,
          py::arg("c_min") = 75,
          py::arg("c_max") = 125,
          py::arg("task_type") = "shortest_path",
          py::arg("max_path_length") = 10,
          py::arg("min_path_length") = 1,
          py::arg("sort_adjacency_lists") = true,
          py::arg("use_unique_depth_markers") = true,
          // optional 'task_sample_dist' can be passed by kwarg
          py::arg("max_query_size") = -1,
          py::arg("min_query_size") = 2,
          py::arg("is_causal") = false,
          py::arg("is_direct_ranking") = false,
          py::arg("shuffle_edges") = false,
          py::arg("shuffle_nodes") = false,
          py::arg("min_vocab") = 0,
          py::arg("max_vocab") = -1,
          py::arg("batch_size") = 256,
          py::arg("max_edges") = 512,
          py::arg("max_attempts") = 1000,
          py::arg("concat_edges") = true,
          py::arg("duplicate_edges") = false,
          py::arg("include_nodes_in_graph_tokenization") = false,
          py::arg("query_at_end") = true,
          py::arg("num_thinking_tokens") = 0,
          py::arg("scratchpad_type") = "none",
          py::arg("is_flat_model") = true,
          py::arg("align_prefix_front_pad") = false,
          py::arg("use_edges_invariance") = false,
          py::arg("use_node_invariance") = false,
          py::arg("use_graph_invariance") = false,
          py::arg("use_query_invariance") = false,
          py::arg("use_task_structure") = false,
          py::arg("use_graph_structure") = false);
}
