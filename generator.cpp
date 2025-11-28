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
#include "directed_graphs.h"
#include "utils.h"
#include "instance.h"
#include "tasks.h"
#include "dictionaries.h"

using namespace std;
namespace py = pybind11;
using namespace py::literals;

/* ************************************************
 *  Seeding, needed for datastream in python
 *  ***********************************************/

inline unsigned int seed_ = std::random_device{}();
static thread_local auto gen = std::mt19937(seed_); // thread_local so each thread in the dataloader is different

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


/* ************************************************************************
 *  Dictionary for mapping vocabulary to ids and mapping positions to ids
 *  ***********************************************************************/

map<std::string, int> dictionary = {};  // token to idx map
int dictionary_num_special = 0;  // special tokens are the first N tokens in the dictionary
int dictionary_num_extra = 0;  //extra tokens
int dictionary_max_vocab = 0;  //then the rest are vocab tokens up to max_vocab
string dictionary_extra_after_symbol = "D";  // then rest are special extras of indeterminate number of D1, D2, ...
map<std::string, int> pos_dictionary = {}; // token to idx map


/* *********************************
 *  Integer Partitioning (for khops)
 *  ********************************/
SampleIntPartition sample_int_partition{};

void set_int_partition_cache_size(const int suggested_cache_size=1000000000, const int max_cache_size=10){
    sample_int_partition.suggested_cache_size = suggested_cache_size;
    sample_int_partition.max_cache_size = max_cache_size;
}

py::array_t<int, py::array::c_style> uniform_random_int_partition(int Q, int N, const bool shuffle=true){
    vector<int> segment_lengths;
    sample_int_partition.uniform_random_partition(Q, N, segment_lengths, gen, shuffle);
    py::array_t<int, py::array::c_style> arr(static_cast<int>(segment_lengths.size()));
    auto ra = arr.mutable_unchecked();
    for (int i = 0; i < static_cast<int>(segment_lengths.size()); i++){
        ra(i) = segment_lengths[i];
    }
    return arr;
}



/* ************************************************
 *  Batched graph generation
 *  ***********************************************/

void check_args(const int c_min, const int c_max,
                const int batch_size,
                const int min_path_length, const int max_path_length,
                const int num_thinking_tokens) {
    if (c_min > 0 and c_min > c_max) { throw std::invalid_argument("Invalid arguments: c_min > c_max"); }
    if (batch_size <= 0) { throw std::invalid_argument("Invalid arguments: batch_size <= 0"); }
    if (min_path_length > 0 and min_path_length > max_path_length) {
        throw std::invalid_argument("Invalid arguments: min_path_length > max_path_length");
    }
    if (num_thinking_tokens < 0) {
        throw std::invalid_argument("Invalid arguments: num_thinking_tokens < 0");
    }
    if (dictionary.empty()) {
        throw std::invalid_argument("Invalid arguments: dictionary is empty.  Please set it first.");
    }
}

vector<int> check_and_set_vocab_limits(int min_num_nodes, int max_num_nodes,
                                       int min_vocab, int max_vocab) {
    if (min_num_nodes <= 0) { throw std::invalid_argument("Invalid arguments: min_num_nodes <= 0"); }
    if (max_num_nodes == -1) {
        max_num_nodes = min_num_nodes;
    }
    if (min_vocab == -1 and max_vocab == -1) {
        min_vocab = dictionary_num_special;
        max_vocab = dictionary_max_vocab;
    } else if (max_vocab == -1) {
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
        const bool scratchpad_as_prefix = false,
        const bool no_graph = false,
        const bool is_flat_model = true,
        const bool align_prefix_front_pad = false,
        const bool use_edges_invariance = false,  // for concated edges this allows true permutation invariance
        const bool use_node_invariance = false,
        const bool use_graph_invariance = false,
        const bool use_query_invariance = false,
        const bool use_task_structure = false,  // divide positions by task structure
        const bool use_graph_structure = false,  // 2d positions by graph structure
        const py::kwargs &kwargs = py::kwargs()) {

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
                                                     is_causal, is_direct_ranking, concat_edges, duplicate_edges,
                                                     include_nodes_in_graph_tokenization,
                                                     scratchpad_as_prefix, no_graph,
                                                     pos_dictionary, use_edges_invariance, use_node_invariance,
                                                     use_graph_invariance, use_query_invariance,
                                                     use_task_structure, use_graph_structure
        );
        // time_after(pack_t, "pack");
        batched_instances.add(instance);
    }
    return batched_instances.package_for_model(dictionary, pos_dictionary);
}


inline py::dict euclidean_n(
        int min_num_nodes, int max_num_nodes, const int dim = 2, float radius = -1.0,
        const int c_min = 75, const int c_max = 125,
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
        const bool scratchpad_as_prefix = false,
        const bool no_graph = false,
        const bool is_flat_model = true,
        const bool align_prefix_front_pad = false,
        const bool use_edges_invariance = false,
        const bool use_node_invariance = false,
        const bool use_graph_invariance = false,
        const bool use_query_invariance = false,
        const bool use_task_structure = false,
        const bool use_graph_structure = false,
        const py::kwargs &kwargs = py::kwargs()) {

    check_args(c_min, c_max, batch_size, min_path_length, max_path_length, num_thinking_tokens);
    if (dim < 0) { throw std::invalid_argument("Invalid arguments: dim < 0"); }
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
            "euclidean", task_type,
            min_vocab, max_vocab,
            query_at_end, num_thinking_tokens, is_flat_model, align_prefix_front_pad);

    int attempts = 0;
    while (batched_instances.size() < batch_size && attempts < max_attempts) {
        int num_nodes = sample_num_nodes(min_num_nodes, max_num_nodes);
        unique_ptr<Graph<boost::undirectedS> > g_ptr;
        unique_ptr<vector<vector<float> > > positions_ptr;
        euclidean_generator(g_ptr, positions_ptr, num_nodes, gen, dim, radius, c_min, c_max, false);
        const auto E = num_edges(*g_ptr);
        if (const auto a = attempt_check(E, max_edges, attempts, max_attempts)) {
            attempts += a;
            continue;
        }
        auto instance = Instance<boost::undirectedS>(g_ptr, gen, dictionary,
                                                     min_vocab, max_vocab, shuffle_nodes, shuffle_edges,
                                                     task_type, scratchpad_type,
                                                     max_path_length, min_path_length, -1, -1, task_sample_dist,
                                                     sort_adjacency_lists, use_unique_depth_markers,
                                                     given_query, max_query_size, min_query_size,
                                                     is_causal, is_direct_ranking, concat_edges, duplicate_edges,
                                                     include_nodes_in_graph_tokenization,
                                                     scratchpad_as_prefix, no_graph,
                                                     pos_dictionary, use_edges_invariance, use_node_invariance,
                                                     use_graph_invariance, use_query_invariance,
                                                     use_task_structure, use_graph_structure,
                                                     optional<unique_ptr<vector<vector<float> > > >(
                                                             std::move(positions_ptr))
        );
        batched_instances.add(instance);
    }
    return batched_instances.package_for_model(dictionary, pos_dictionary);
}


inline py::dict random_tree_n(
        int min_num_nodes, int max_num_nodes, vector<float> &probs, const int max_depth,
        const int c_min = 75, const int c_max = 125,
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
        const bool scratchpad_as_prefix = false,
        const bool no_graph = false,
        const bool is_flat_model = true,
        const bool align_prefix_front_pad = false,
        const bool use_edges_invariance = false,
        const bool use_node_invariance = false,
        const bool use_graph_invariance = false,
        const bool use_query_invariance = false,
        const bool use_task_structure = false,
        const bool use_graph_structure = false,
        const py::kwargs &kwargs = py::kwargs()) {

    check_args(c_min, c_max, batch_size, min_path_length, max_path_length, num_thinking_tokens);
    auto m = check_and_set_vocab_limits(min_num_nodes, max_num_nodes, min_vocab, max_vocab);
    min_num_nodes = m[0], max_num_nodes = m[1], min_vocab = m[2], max_vocab = m[3];

    bool start_at_root = true;
    if (kwargs.contains("start_at_root")) {
        start_at_root = kwargs["start_at_root"].cast<bool>();
    }

    optional<vector<float>> task_sample_dist = nullopt;
    if (kwargs.contains("task_sample_dist")) {
        if (!kwargs["task_sample_dist"].is_none() and !kwargs["task_sample_dist"].cast<py::list>().empty()) {
            task_sample_dist = kwargs["task_sample_dist"].cast<vector<float> >();
        }
    }
    optional<vector<int>> given_query = nullopt;

    auto batched_instances = BatchedInstances<boost::undirectedS>(
            "euclidean", task_type,
            min_vocab, max_vocab,
            query_at_end, num_thinking_tokens, is_flat_model, align_prefix_front_pad);

    int attempts = 0;
    while (batched_instances.size() < batch_size && attempts < max_attempts) {
        int num_nodes = sample_num_nodes(min_num_nodes, max_num_nodes);
        unique_ptr<Graph<boost::undirectedS> > g_ptr;
        int start = random_tree_generator(g_ptr, num_nodes, gen, probs, max_depth, false);
        if (!start_at_root) {
            start = -1;
        }
        const auto E = num_edges(*g_ptr);
        if (const auto a = attempt_check(E, max_edges, attempts, max_attempts)) {
            attempts += a;
            continue;
        }
        auto instance = Instance<boost::undirectedS>(g_ptr, gen, dictionary,
                                                     min_vocab, max_vocab, shuffle_nodes, shuffle_edges,
                                                     task_type, scratchpad_type,
                                                     max_path_length, min_path_length, start, -1, task_sample_dist,
                                                     sort_adjacency_lists, use_unique_depth_markers,
                                                     given_query, max_query_size, min_query_size,
                                                     is_causal, is_direct_ranking, concat_edges, duplicate_edges,
                                                     include_nodes_in_graph_tokenization,
                                                     scratchpad_as_prefix, no_graph,
                                                     pos_dictionary, use_edges_invariance, use_node_invariance,
                                                     use_graph_invariance, use_query_invariance,
                                                     use_task_structure, use_graph_structure
        );
        batched_instances.add(instance);
    }
    return batched_instances.package_for_model(dictionary, pos_dictionary);
}

inline py::dict path_star_n(
        const int min_num_arms, const int max_num_arms, const int min_arm_length, const int max_arm_length,
        const string &task_type = "shortest_path",
        const bool sort_adjacency_lists = true, const bool use_unique_depth_markers = true,
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
        const bool scratchpad_as_prefix = false,
        const bool no_graph = false,
        const bool is_flat_model = true,
        const bool align_prefix_front_pad = false,
        const bool use_edges_invariance = false,
        const bool use_node_invariance = false,
        const bool use_graph_invariance = false,
        const bool use_query_invariance = false,
        const bool use_task_structure = false,
        const bool use_graph_structure = false,
        const py::kwargs &kwargs = py::kwargs()) {

    check_args(-1, -1, batch_size, -1, -1, num_thinking_tokens);
    auto m = check_and_set_vocab_limits(1, 1, min_vocab, max_vocab);
    min_vocab = m[2], max_vocab = m[3];

    optional<vector<float>> task_sample_dist = nullopt;
    if (kwargs.contains("task_sample_dist")) {
        if (!kwargs["task_sample_dist"].is_none() and !kwargs["task_sample_dist"].cast<py::list>().empty()) {
            task_sample_dist = kwargs["task_sample_dist"].cast<vector<float> >();
        }
    }

    auto batched_instances = BatchedInstances<boost::directedS>(
            "euclidean", task_type,
            min_vocab, max_vocab,
            query_at_end, num_thinking_tokens, is_flat_model, align_prefix_front_pad);

    int attempts = 0;
    while (batched_instances.size() < batch_size && attempts < max_attempts) {
        unique_ptr<Graph<boost::directedS> > g_ptr;
        auto start_end = path_star_generator(g_ptr, min_num_arms, max_num_arms, min_arm_length, max_arm_length, gen,
                                             false);
        const auto E = num_edges(*g_ptr);
        if (const auto a = attempt_check(E, max_edges, attempts, max_attempts)) {
            attempts += a;
            continue;
        }
        auto instance = Instance<boost::directedS>(g_ptr, gen, dictionary,
                                                   min_vocab, max_vocab, shuffle_nodes, shuffle_edges,
                                                   task_type, scratchpad_type,
                                                   -1, -1, start_end.first, start_end.second, task_sample_dist,
                                                   sort_adjacency_lists, use_unique_depth_markers,
                                                   (optional<vector<int>> &) nullopt, -1, -1,
                                                   is_causal, is_direct_ranking, concat_edges, duplicate_edges,
                                                   include_nodes_in_graph_tokenization,
                                                   scratchpad_as_prefix, no_graph,
                                                   pos_dictionary, use_edges_invariance, use_node_invariance,
                                                   use_graph_invariance, use_query_invariance,
                                                   use_task_structure, use_graph_structure
        );
        batched_instances.add(instance);
    }
    return batched_instances.package_for_model(dictionary, pos_dictionary);
}


inline py::dict balanced_n(
        const int min_num_nodes, int max_num_nodes, const int min_lookahead, const int max_lookahead,
        const int min_noise_reserve = 0, const int max_num_parents = 4, int max_noise = -1,
        const string &task_type = "shortest_path",
        int max_query_size = -1, const int min_query_size = 2,
        const bool sort_adjacency_lists = true, const bool use_unique_depth_markers = true,
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
        const bool scratchpad_as_prefix = false,
        const bool no_graph = false,
        const bool is_flat_model = true,
        const bool align_prefix_front_pad = false,
        const bool use_edges_invariance = false,
        const bool use_node_invariance = false,
        const bool use_graph_invariance = false,
        const bool use_query_invariance = false,
        const bool use_task_structure = false,
        const bool use_graph_structure = false,
        const py::kwargs &kwargs = py::kwargs()) {

    check_args(-1, -1, batch_size, -1, -1, num_thinking_tokens);
    auto m = check_and_set_vocab_limits(min_num_nodes, max_num_nodes, min_vocab, max_vocab);
    min_vocab = m[2], max_vocab = m[3];

    if (min_lookahead <= 0) { throw std::invalid_argument("Invalid arguments: min_lookahead <= 0"); }
    if (max_lookahead <= 0) { throw std::invalid_argument("Invalid arguments: max_lookahead <= 0"); }
    if (max_num_parents <= 0) { throw std::invalid_argument("Invalid arguments: max_num_parents <= 0"); }
    if (!balanced_graph_size_check(min_num_nodes, max_lookahead, min_noise_reserve)) {
        throw std::invalid_argument("Invalid arguments: balanced_graph_size_check failed");
    }

    optional<vector<float>> task_sample_dist = nullopt;
    if (kwargs.contains("task_sample_dist")) {
        if (!kwargs["task_sample_dist"].is_none() and !kwargs["task_sample_dist"].cast<py::list>().empty()) {
            task_sample_dist = kwargs["task_sample_dist"].cast<vector<float> >();
        }
    }

    auto batched_instances = BatchedInstances<boost::directedS>(
            "euclidean", task_type,
            min_vocab, max_vocab,
            query_at_end, num_thinking_tokens, is_flat_model, align_prefix_front_pad);

    int attempts = 0;
    while (batched_instances.size() < batch_size && attempts < max_attempts) {
        int num_nodes;
        if (max_num_nodes == min_num_nodes) {
            num_nodes = max_num_nodes;
        } else {
            uniform_int_distribution<int> d(min_num_nodes, max_num_nodes);
            num_nodes = d(gen);
        }
        auto lookahead = uniform_int_distribution<int>(min_lookahead, max_lookahead)(gen);
        int max_noise_sample;
        if (max_noise > 0) {
            max_noise_sample = uniform_int_distribution<int>(0, max_noise)(gen);
        } else {
            max_noise_sample = uniform_int_distribution<int>(-1, lookahead)(gen); // -1 means all remaining
        }
        unique_ptr<Graph<boost::directedS> > g_ptr;
        // auto graph_t = time_before();
        auto start_end = balanced_generator(g_ptr, num_nodes, gen, lookahead, min_noise_reserve, max_num_parents,
                                            max_noise_sample);
        const auto E = num_edges(*g_ptr);
        if (const auto a = attempt_check(E, max_edges, attempts, max_attempts)) {
            attempts += a;
            continue;
        }
        auto instance = Instance<boost::directedS>(g_ptr, gen, dictionary,
                                                   min_vocab, max_vocab, shuffle_nodes, shuffle_edges,
                                                   task_type, scratchpad_type,
                                                   -1, -1, start_end.first, start_end.second, task_sample_dist,
                                                   sort_adjacency_lists, use_unique_depth_markers,
                                                   (optional<vector<int>> &) nullopt, max_query_size, min_query_size,
                                                   is_causal, is_direct_ranking, concat_edges, duplicate_edges,
                                                   include_nodes_in_graph_tokenization,
                                                   scratchpad_as_prefix, no_graph,
                                                   pos_dictionary, use_edges_invariance, use_node_invariance,
                                                   use_graph_invariance, use_query_invariance,
                                                   use_task_structure, use_graph_structure
        );
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

    // dictionary/vocabulary
    m.def("set_dictionary", &set_dictionary,
          "Sets the dictionary/vocabulary of token to token_idx.\n"
          "Parameters:\n\t"
          "dictionary: of str -> int\n\t",
          py::arg("dictionary"), py::arg("verbose") = false,
          py::arg("max_num_nodes") = -1,
          py::arg("extra_after") = -1,
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

    m.def("verify_path", &ShortestPathTask::verify_path<int>,
          "Verifies that any predicted paths are valid given the distance matrices.\n"
          "Parameters:\n\t"
          "distances: [vocab_size, vocab_size]\n\t"
          "path: [max_path_length]\n\t"
          "Returns:\n\t"
          "is_valid bool, int, -1 if not valid, 0 if valid but not shortest, 1 if valid and shortest.\n",
          py::arg("distance"), py::arg("path"));

    m.def("verify_paths", &ShortestPathTask::verify_paths<int>,
          "Batch version. "
          "Verifies that any predicted paths are valid given the distance matrices.\n"
          "Parameters:\n\t"
          "distances: [batch_size, vocab_size, vocab_size]\n\t"
          "queries: [batch_size, 2] of start, end\n\t"
          "paths: [batch_size, max_path_length]\n\t"
          "path_lengths: [batch_size]\n\t"
          "Returns:\n\t"
          "is_valid [batch_size], int, -1 if not valid, 0 if valid but not shortest, 1 if valid and shortest.\n",
          py::arg("distances"), py::arg("queries"), py::arg("paths"), py::arg("path_lengths"));

    m.def("verify_bfs_gen", &BFSScratchPad::verify_bfs_gen<int>,
          "Verifies that any predicted paths are valid given the distance matrices.\n"
          "Parameters:\n\t"
          "distances: [vocab_size, vocab_size]\n\t"
          "gen: [max_path_length]\n\t"
          "Returns:\n\t"
          "is_valid bool, int,"
          " -1 if not valid due to special tokens, 0 if not valid due bfs, 1 if valid and bfs.\n",
          py::arg("distance"),  py::arg("start"),  py::arg("end"), py::arg("gen"),
          py::arg("check_special_tokens") = true);

    m.def("verify_bfs_gens", &BFSScratchPad::verify_bfs_gens<int>,
          "Batch version. "
          "Verifies that any predicted paths are valid given the distance matrices.\n"
          "Parameters:\n\t"
          "distances: [batch_size, vocab_size, vocab_size]\n\t"
          "queries: [batch_size, 2] of start, end\n\t"
          "gens: [batch_size, max_path_length]\n\t"
          "path_lengths: [batch_size]\n\t"
          "Returns:\n\t"
          "is_valid [batch_size], int, "
          " -1 if not valid due to special tokens, 0 if not valid due bfs, 1 if valid and bfs.\n",
          py::arg("distances"), py::arg("queries"), py::arg("gens"), py::arg("path_lengths"),
          py::arg("check_special_tokens") = true);

    m.def("set_int_partition_cache_size", &set_int_partition_cache_size,
          "Sets the integer partition cache size used in balanced graph generation.\n"
          "Parameters:\n\t"
          "size: suggested_cache_size\n\t",
          py::arg("suggested_cache_size"),
          py::arg("max_cache_size"));

    m.def("uniform_random_int_partition", &uniform_random_int_partition,
          "Generates a uniform random integer partition of total into num_parts parts.\n"
          "Parameters:\n\t"
          "total: total integer to partition\n\t"
          "num_parts: number of parts to partition into\n\t"
          "Returns:\n\t"
          "vector<int>: partitioned integers\n",
          py::arg("Q"), py::arg("N"), py::arg("shuffle") = true);

    m.def("balanced_graph_size_check", &balanced_graph_size_check,
          "Check that the balanced graph size is valid.  Will fail assert otherwise.",
          py::arg("num_nodes"), py::arg("lookahead"), py::arg("min_noise_reserve") = 0);

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
          py::arg("scratchpad_as_prefix") = false,
          py::arg("no_graph") = false,
          py::arg("is_flat_model") = true,
          py::arg("align_prefix_front_pad") = false,
          py::arg("use_edges_invariance") = false,
          py::arg("use_node_invariance") = false,
          py::arg("use_graph_invariance") = false,
          py::arg("use_query_invariance") = false,
          py::arg("use_task_structure") = false,
          py::arg("use_graph_structure") = false);

    m.def("euclidean_n", &euclidean_n,
          "Generate a batch of Euclidean graphs\nGraph specific parameters:\n\t"
          "dims: number of dimensions\n\t"
          "radius: radius of graph.  If -1 then 1/sqrt(num_nodes).\n\t",

          py::arg("min_num_nodes"),
          py::arg("max_num_nodes"),
          py::arg("dim") = 2,
          py::arg("radius") = -1.0,
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
          py::arg("scratchpad_as_prefix") = false,
          py::arg("no_graph") = false,
          py::arg("is_flat_model") = true,
          py::arg("align_prefix_front_pad") = false,
          py::arg("use_edges_invariance") = false,
          py::arg("use_node_invariance") = false,
          py::arg("use_graph_invariance") = false,
          py::arg("use_query_invariance") = false,
          py::arg("use_task_structure") = false,
          py::arg("use_graph_structure") = false);

    m.def("random_tree_n", &random_tree_n,
          "Generate a batch of random tree graphs\nGraph specific parameters:\n\t"
          "probs: list of probs of branching.\n\t"
          "max_depth: max branching factor.\n\t",

          py::arg("min_num_nodes"),
          py::arg("max_num_nodes"),
          py::arg("probs") = vector<float>{0.5, 0.5},
          py::arg("max_depth") = -1.0,
          py::arg("c_min") = 75,
          py::arg("c_max") = 125,
          py::arg("task_type") = "shortest_path",
          py::arg("max_path_length") = 10,
          py::arg("min_path_length") = 1,
            // optional start_at_root can be passed by kwarg, default is true
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
          py::arg("scratchpad_as_prefix") = false,
          py::arg("no_graph") = false,
          py::arg("is_flat_model") = true,
          py::arg("align_prefix_front_pad") = false,
          py::arg("use_edges_invariance") = false,
          py::arg("use_node_invariance") = false,
          py::arg("use_graph_invariance") = false,
          py::arg("use_query_invariance") = false,
          py::arg("use_task_structure") = false,
          py::arg("use_graph_structure") = false);

    m.def("path_star_n", &path_star_n,
          "Generate a batch of path star graphs\nGraph specific parameters:\n\t"
          "min_num_arms: min number of arms.\n\t"
          "max_num_arms: max number of arms.\n\t"
          "min_arm_length: min arm length.\n\t"
          "max_arm_length: max arm length.\n\t",

          py::arg("min_num_arms"),
          py::arg("max_num_arms"),
          py::arg("min_arm_length"),
          py::arg("max_arm_length"),
          py::arg("task_type") = "shortest_path",
          py::arg("sort_adjacency_lists") = true,
          py::arg("use_unique_depth_markers") = true,
            // optional 'task_sample_dist' can be passed by kwarg
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
          py::arg("scratchpad_as_prefix") = false,
          py::arg("no_graph") = false,
          py::arg("is_flat_model") = true,
          py::arg("align_prefix_front_pad") = false,
          py::arg("use_edges_invariance") = false,
          py::arg("use_node_invariance") = false,
          py::arg("use_graph_invariance") = false,
          py::arg("use_query_invariance") = false,
          py::arg("use_task_structure") = false,
          py::arg("use_graph_structure") = false);

    m.def("balanced_n", &balanced_n,
          "Generate a batch of balanced graphs\nParameters:\n\t"
          "min_num_nodes: min number of nodes. We strongly recommend using shuffle_nodes and a vocab range map.\n\t"
          "max_num_nodes: min number of nodes.  If -1 use min only.\n\t"
          "min_lookahead: min number of lookahead nodes.\n\t"
          "max_lookahead: max number of lookahead nodes.\n\t"
          "min_noise_reserve: min noise reserve.\n\t"
          "max_num_parents: max number of parents.\n\t"
          "max_noise: max noise.\n\t",

          py::arg("min_num_nodes"),
          py::arg("max_num_nodes"),
          py::arg("min_lookahead"),
          py::arg("max_lookahead"),
          py::arg("min_noise_reserve") = 0,
          py::arg("max_num_parents") = 4,
          py::arg("max_noise") = -1,
          py::arg("task_type") = "shortest_path",
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
          py::arg("scratchpad_as_prefix") = false,
          py::arg("no_graph") = false,
          py::arg("is_flat_model") = true,
          py::arg("align_prefix_front_pad") = false,
          py::arg("use_edges_invariance") = false,
          py::arg("use_node_invariance") = false,
          py::arg("use_graph_invariance") = false,
          py::arg("use_query_invariance") = false,
          py::arg("use_task_structure") = false,
          py::arg("use_graph_structure") = false);
}
