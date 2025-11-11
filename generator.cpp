//
// Created by arvie on 4/6/25.
// Uitls for generation.cpp
//

#include <iostream>
#include <random>
#include <Python.h>
#include <chrono>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <limits>
#include <cmath>

#include <thread>
#include <mutex>

#include "undirected_graphs.h"
#include "utils.h"
#include "instance.h"

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
 *  Dictionary for mapping vocabulary to ids
 *  ***********************************************/

static map<std::string, int> dictionary; // token to idx map

inline void set_dictionary(py::dict &py_dictionary, const bool verbose = false) {
    if (verbose) {
        cout << "Setting dictionary" << endl;
    }
    for (std::pair<py::handle, py::handle> item: py_dictionary) {
        auto key = item.first.cast<std::string>();
        auto value = item.second.cast<int>();
        if (verbose) {
            cout << "\tkey: " << key << ", value=" << value << endl;
        }
        dictionary[key] = value;
    }
}

inline void set_default_dictionary(const int max_vocab = 100, const int extra_after=0) {
    /* Sets a default dictionary
    * {'<s>': 0, '<pad>': 1, '</s>': 2, '<unk>': 3,
    * '|': 4, '!': 5, '=': 6, '.': 7,
    * 't1': 8, 't2': 9, 't3': 10, 't4': 11, 't5': 12,
    * '/': 13, '?': 14, '@': 15, '#': 16,
    * 's1': 17, 's2': 18, 's3': 19, 's4': 20, 's5': 21,
    */
    dictionary = {
        {"<s>", 0},
        {"<pad>", 1},
        {"</s>", 2},
        {"<unk>", 3},
        {"|", 4},  // edge marker
        {"!", 5},  // thinking token
        {"=", 6},  // task start
        {".", 7},  // task end
        {"t1", 8}, // potentially mark task type for muliti-task learning
        {"t2", 9},
        {"t1", 10},
        {"t2", 11},
        {"t3", 12},
        {"/", 13}, // query start
        {"?", 14}, // query end
        {"@", 15},
        {"#", 16}, // scratchpad start
        {"[", 17}, // bfs adjacency start
        {"]", 18}, // bfs adjacency end
        {"{", 19},
        {"}", 20},
        {"$", 21},
        {"D", 22},
    };

    if (max_vocab > 0) {
        auto num_special = static_cast<int>(dictionary.size());
        assert(max_vocab >=  num_special);
        for (int i = num_special; i < max_vocab; i++) {
            dictionary[std::to_string(i - num_special)] = i;
        }
    }
    if (extra_after > 0) {
        auto current_size = static_cast<int>(dictionary.size());
        for (int i = 0; i < extra_after; i++) {
            dictionary["D" + std::to_string(i)] = current_size + i;
        }
    }
}

map<std::string, int> get_dictionary() {
    return dictionary;
}

/* ************************************************
 *  Dictionary for mapping positions to ids
 *  ***********************************************/

static map<std::string, int> pos_dictionary; // token to idx map

inline void set_pos_dictionary(py::dict &py_dictionary, const bool verbose = false) {
    if (verbose) {
        cout << "Setting pos dictionary" << endl;
    }
    for (std::pair<py::handle, py::handle> item: py_dictionary) {
        auto key = item.first.cast<std::string>();
        auto value = item.second.cast<int>();
        if (verbose) {
            cout << "\tkey: " << key << ", value=" << value << endl;
        }
        pos_dictionary[key] = value;
    }
    // verify that all keys are present
    vector<string> required_keys = {
        "pad",
        "misc_start",
        "misc_end",
        "query_start",
        "query_end",
        "graph_start",
        "graph_end",
        "graph_sub_start",
        "graph_sub_end",
        "task_start",
        "task_end",
    };
    for (const auto &key : required_keys) {
        if (pos_dictionary.find(key) == pos_dictionary.end()) {
            throw std::invalid_argument("Key " + key + " not found in pos_dictionary");
        }
    }
}

inline void set_default_pos_dictionary() {
    /* Sets a default pos dictionary
    */
    pos_dictionary = {
        {"pad", 0},  // needed for embedding look-up
        {"query_invariance", 1},
        {"edge_invariance", 2},
        {"node_invariance", 3},
        {"graph_invariance", 4},
        {"misc_start", 11},
        {"misc_end", 99},
        {"query_start", 100},
        {"query_end", 199},
        {"graph_start", 200},
        {"graph_end", 499},
        {"graph_sub_start", 500},
        {"graph_sub_end", 599},  // only really need 3
        {"thinking_start", 600},
        {"thinking_end", 699},
        {"task_start", 700},
        {"task_end", 999},
    };
}

map<std::string, int> get_pos_dictionary() {
    return pos_dictionary;
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

pair<int, int> check_and_set_vocab_limits(int min_num_nodes, int &max_num_nodes,
                                         const int min_vocab, int &max_vocab) {
    if (min_num_nodes <= 0) { throw std::invalid_argument("Invalid arguments: min_num_nodes <= 0"); }
    if (max_num_nodes == -1) {
        max_num_nodes = min_num_nodes;
    }
    if (max_vocab == -1) {
        max_vocab = max_num_nodes;
    }
    if (max_vocab - min_vocab < max_num_nodes) {
        auto s = "Invalid arguments: max_vocab - min_vocab < max_num_nodes:  " +
                 to_string(max_vocab) + " - " + to_string(min_vocab) + " < " + to_string(max_num_nodes);
        throw std::invalid_argument(s);
    }
    return {min_num_nodes, max_num_nodes};
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
    const bool is_causal = false, const bool is_direct_ranking = false, const bool shuffle_edges = false,
    const bool shuffle_nodes = false, const int min_vocab = 0, int max_vocab = -1,
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
    min_num_nodes = m.first;
    max_num_nodes = m.second;

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
          py::arg("dictionary"), py::arg("verbose") = false);

    m.def("set_default_dictionary", &set_default_dictionary,
          "Sets the dictionary/vocabulary of token to token_idx.\n"
          "Parameters:\n\t"
          "None\n",
          py::arg("max_vocab") = 100,
          py::arg("extra_after") = 0);

    m.def("get_dictionary", &get_dictionary,
          "Gets the dictionary/vocabulary of token to token_idx.\n"
          "Parameters:\n\t"
          "None\n"
          "Returns:\n\t"
          "dictionary: of str -> int\n");

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
