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
#include "args.h"

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

void set_int_partition_cache_size(const int suggested_cache_size = 1000000000, const int max_cache_size = 10) {
    sample_int_partition.suggested_cache_size = suggested_cache_size;
    sample_int_partition.max_cache_size = max_cache_size;
}

py::array_t<int, py::array::c_style> uniform_random_int_partition(int Q, int N, const bool shuffle = true) {
    auto segment_lengths = sample_int_partition.uniform_random_partition(Q, N, gen, shuffle);
    py::array_t<int, py::array::c_style> arr(static_cast<int>(segment_lengths.size()));
    auto ra = arr.mutable_unchecked();
    for (int i = 0; i < static_cast<int>(segment_lengths.size()); i++) {
        ra(i) = segment_lengths[i];
    }
    return arr;
}


/* ************************************************
 *  Batched graph generation
 *  ***********************************************/
void checks_and_sets(int &min_num_nodes, int &max_num_nodes,
                     int &min_vocab, int &max_vocab,
                     const int batch_size) {
    if (batch_size <= 0) { throw std::invalid_argument("Invalid arguments: batch_size <= 0"); }

    if (dictionary.empty()) {
        throw std::invalid_argument("Invalid arguments: dictionary is empty.  Please set it first.");
    }

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
}

inline py::dict erdos_renyi_n(
        int min_num_nodes, int max_num_nodes, float p = -1.0, const int c_min = 75, const int c_max = 125,
        const string &task_type = "shortest_path",
        const string &scratchpad_type = "none",
        const bool shuffle_edges = false, const bool shuffle_nodes = false,
        int min_vocab = -1, int max_vocab = -1,
        const int batch_size = 256, const int max_edges = 512, int max_attempts = 1000,
        const bool is_causal = false,
        const bool is_direct_ranking = false,
        const bool concat_edges = true,
        const bool duplicate_edges = false,
        const bool include_nodes_in_graph_tokenization = false,
        const bool query_at_end = true,
        const int num_thinking_tokens = 0,
        const bool scratchpad_as_prefix = false,
        const bool no_graph = false,
        const bool is_flat_model = true,
        const bool align_prefix_front_pad = false,
        const py::kwargs &kwargs = py::kwargs()) {

    if (p > 1.0) { throw std::invalid_argument("Invalid arguments: p > 1.0"); }
    checks_and_sets(min_num_nodes, max_num_nodes, min_vocab, max_vocab, batch_size);

    Args args(task_type, scratchpad_type, "erdos_renyi", min_vocab, max_vocab,
              is_causal, is_direct_ranking, query_at_end, no_graph,
              concat_edges, duplicate_edges, include_nodes_in_graph_tokenization,
              num_thinking_tokens, scratchpad_as_prefix, is_flat_model, align_prefix_front_pad,
              kwargs);

    int attempts = 0;
    auto batched_instances = BatchedInstances<boost::undirectedS>(args);
    while (batched_instances.size() < batch_size && attempts < max_attempts) {
        auto graph = make_unique<GraphWrapper<boost::undirectedS> >(min_num_nodes, max_num_nodes, min_vocab, max_vocab,
                                                                    shuffle_edges, shuffle_nodes, c_min, c_max);
        graph->make_erdos_renyi(gen, p);
        if (const auto a = graph->attempt_check(max_edges, attempts, max_attempts)) {
            attempts += a;
            continue;
        }
        // auto pack_t = time_before();
        Instance<boost::undirectedS> instance(gen, graph, args, dictionary, pos_dictionary);
        // time_after(pack_t, "pack");
        batched_instances.add(instance);
    }
    return batched_instances.package_for_model(dictionary, pos_dictionary);
}

inline py::dict euclidean_n(
        int min_num_nodes, int max_num_nodes, const int dim = 2, float radius = -1.0, const int c_min = 75, const int c_max = 125,
        const string &task_type = "shortest_path",
        const string &scratchpad_type = "none",
        const bool shuffle_edges = false, const bool shuffle_nodes = false,
        int min_vocab = -1, int max_vocab = -1,
        const int batch_size = 256, const int max_edges = 512, int max_attempts = 1000,
        const bool is_causal = false,
        const bool is_direct_ranking = false,
        const bool concat_edges = true,
        const bool duplicate_edges = false,
        const bool include_nodes_in_graph_tokenization = false,
        const bool query_at_end = true,
        const int num_thinking_tokens = 0,
        const bool scratchpad_as_prefix = false,
        const bool no_graph = false,
        const bool is_flat_model = true,
        const bool align_prefix_front_pad = false,
        const py::kwargs &kwargs = py::kwargs()) {

    if (dim < 0) { throw std::invalid_argument("Invalid arguments: dim < 0"); }
    checks_and_sets(min_num_nodes, max_num_nodes, min_vocab, max_vocab, batch_size);

    Args args(task_type, scratchpad_type, "euclidean", min_vocab, max_vocab,
              is_causal, is_direct_ranking, query_at_end, no_graph,
              concat_edges, duplicate_edges, include_nodes_in_graph_tokenization,
              num_thinking_tokens, scratchpad_as_prefix, is_flat_model, align_prefix_front_pad,
              kwargs);

    int attempts = 0;
    auto batched_instances = BatchedInstances<boost::undirectedS>(args);
    while (batched_instances.size() < batch_size && attempts < max_attempts) {
        GraphWrapper<boost::undirectedS> graph(min_num_nodes, max_num_nodes, min_vocab, max_vocab,
                                               shuffle_edges, shuffle_nodes, c_min, c_max);
        graph.make_euclidean(gen, dim, radius);
        if (const auto a = graph.attempt_check(max_edges, attempts, max_attempts)) {
            attempts += a;
            continue;
        }
        auto instance = Instance<boost::undirectedS>(gen, graph, args, dictionary, pos_dictionary);
        );
        batched_instances.add(instance);
    }
    return batched_instances.package_for_model(dictionary, pos_dictionary);
}


inline py::dict random_tree_n(
        int min_num_nodes, int max_num_nodes, const int max_degree, const int max_depth, const float bernoulli_p,
        const string &task_type = "shortest_path",
        const string &scratchpad_type = "none",
        const bool shuffle_edges = false, const bool shuffle_nodes = false,
        int min_vocab = -1, int max_vocab = -1,
        const int batch_size = 256, const int max_edges = 512, int max_attempts = 1000,
        const bool is_causal = false,
        const bool is_direct_ranking = false,
        const bool concat_edges = true,
        const bool duplicate_edges = false,
        const bool include_nodes_in_graph_tokenization = false,
        const bool query_at_end = true,
        const int num_thinking_tokens = 0,
        const bool scratchpad_as_prefix = false,
        const bool no_graph = false,
        const bool is_flat_model = true,
        const bool align_prefix_front_pad = false,
        const py::kwargs &kwargs = py::kwargs()) {

    checks_and_sets(min_num_nodes, max_num_nodes, min_vocab, max_vocab, batch_size);

    Args args(task_type, scratchpad_type, "random_tree", min_vocab, max_vocab,
              is_causal, is_direct_ranking, query_at_end, no_graph,
              concat_edges, duplicate_edges, include_nodes_in_graph_tokenization,
              num_thinking_tokens, scratchpad_as_prefix, is_flat_model, align_prefix_front_pad,
              kwargs);

    // sample different paths but lose that it is exactly k-hops
    bool start_at_root = true;
    if (kwargs.contains("start_at_root")) {
        start_at_root = kwargs["start_at_root"].cast<bool>();
    }
    bool end_at_leaf = true;
    if (kwargs.contains("end_at_leaf")) {
        end_at_leaf = kwargs["end_at_leaf"].cast<bool>();
    }
    optional<vector<float>> probs = nullopt;  // override default probabilities for branching in random tree generator
    if (kwargs.contains("probs")) {
        if (!kwargs["probs"].is_none() and !kwargs["probs"].cast<py::list>().empty()) {
            probs = kwargs["probs"].cast<vector<float> >();
        }
    }

    int attempts = 0;
    auto batched_instances = BatchedInstances<boost::undirectedS>(args);
    while (batched_instances.size() < batch_size && attempts < max_attempts) {

        int sample_depth = max_depth;  // this is because topology and task mix.  Path length is defined by tree struct.
        if (args.task.task_sample_dist.has_value()) {
            std::discrete_distribution<int> d(args.task.task_sample_dist.value().begin(), args.task.task_sample_dist.value().end());
            sample_depth = d(gen);
        }

        GraphWrapper<boost::undirectedS> graph(min_num_nodes, max_num_nodes, min_vocab, max_vocab,
                                               shuffle_edges, shuffle_nodes);
        graph.make_random_tree(gen, max_degree, sample_depth, max_depth, bernoulli_p, probs);
        if (const auto a = graph.attempt_check(max_edges, attempts, max_attempts)) {
            attempts += a;
            continue;
        }

        auto instance = Instance<boost::undirectedS>(gen, graph, args, dictionary, pos_dictionary);
        batched_instances.add(instance);
    }
    return batched_instances.package_for_model(dictionary, pos_dictionary);
}

inline py::dict path_star_n(
        const int min_num_arms, const int max_num_arms, const int min_arm_length, const int max_arm_length,
        const string &task_type = "shortest_path",
        const string &task_type = "shortest_path",
        const string &scratchpad_type = "none",
        const bool shuffle_edges = false, const bool shuffle_nodes = false,
        int min_vocab = -1, int max_vocab = -1,
        const int batch_size = 256, const int max_edges = 512, int max_attempts = 1000,
        const bool is_causal = false,
        const bool is_direct_ranking = false,
        const bool concat_edges = true,
        const bool duplicate_edges = false,
        const bool include_nodes_in_graph_tokenization = false,
        const bool query_at_end = true,
        const int num_thinking_tokens = 0,
        const bool scratchpad_as_prefix = false,
        const bool no_graph = false,
        const bool is_flat_model = true,
        const bool align_prefix_front_pad = false,
        const py::kwargs &kwargs = py::kwargs()) {

    checks_and_sets(min_num_nodes, max_num_nodes, min_vocab, max_vocab, batch_size);


    Args args(task_type, scratchpad_type, "path_star", min_vocab, max_vocab,
              is_causal, is_direct_ranking, query_at_end, no_graph,
              concat_edges, duplicate_edges, include_nodes_in_graph_tokenization,
              num_thinking_tokens, scratchpad_as_prefix, is_flat_model, align_prefix_front_pad,
              kwargs);

    int attempts = 0;
    auto batched_instances = BatchedInstances<boost::directedS>(args);
    while (batched_instances.size() < batch_size && attempts < max_attempts) {
        GraphWrapper<boost::directedS> graph(min_num_nodes, max_num_nodes, min_vocab, max_vocab,
                                               shuffle_edges, shuffle_nodes);
        graph.make_path_star(gen, min_num_arms, max_num_arms, min_arm_length, max_arm_length);
        if (const auto a = graph.attempt_check(max_edges, attempts, max_attempts)) {
            attempts += a;
            continue;
        }
        auto instance = Instance<boost::directedS>(gen, graph, args, dictionary, pos_dictionary);
        );
        batched_instances.add(instance);
    }
    return batched_instances.package_for_model(dictionary, pos_dictionary);
}


inline py::dict balanced_n(
        const int min_num_nodes, int max_num_nodes, const int min_lookahead, const int max_lookahead,
        const int min_noise_reserve = 0, const int max_num_parents = 4, int max_noise = -1,
        const string &task_type = "shortest_path",
        const string &scratchpad_type = "none",
        const bool shuffle_edges = false, const bool shuffle_nodes = false,
        int min_vocab = -1, int max_vocab = -1,
        const int batch_size = 256, const int max_edges = 512, int max_attempts = 1000,
        const bool is_causal = false,
        const bool is_direct_ranking = false,
        const bool concat_edges = true,
        const bool duplicate_edges = false,
        const bool include_nodes_in_graph_tokenization = false,
        const bool query_at_end = true,
        const int num_thinking_tokens = 0,
        const bool scratchpad_as_prefix = false,
        const bool no_graph = false,
        const bool is_flat_model = true,
        const bool align_prefix_front_pad = false,
        const py::kwargs &kwargs = py::kwargs()) {

    if (min_lookahead <= 0) { throw std::invalid_argument("Invalid arguments: min_lookahead <= 0"); }
    if (max_lookahead <= 0) { throw std::invalid_argument("Invalid arguments: max_lookahead <= 0"); }
    if (max_num_parents <= 0) { throw std::invalid_argument("Invalid arguments: max_num_parents <= 0"); }
    if (!balanced_graph_size_check(min_num_nodes, max_lookahead, min_noise_reserve)) {
        throw std::invalid_argument("Invalid arguments: balanced_graph_size_check failed");
    }
    checks_and_sets(min_num_nodes, max_num_nodes, min_vocab, max_vocab, batch_size);

    Args args(task_type, scratchpad_type, "balanced", min_vocab, max_vocab,
              is_causal, is_direct_ranking, query_at_end, no_graph,
              concat_edges, duplicate_edges, include_nodes_in_graph_tokenization,
              num_thinking_tokens, scratchpad_as_prefix, is_flat_model, align_prefix_front_pad,
              kwargs);

    int attempts = 0;
    auto batched_instances = BatchedInstances<boost::undirectedS>(args);
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
        GraphWrapper<boost::directedS> graph(num_nodes, num_nodes, min_vocab, max_vocab,
                                             shuffle_edges, shuffle_nodes);
        graph.make_balanced(gen, lookahead, min_noise_reserve, max_num_parents, max_noise_sample);
        if (const auto a = graph.attempt_check(max_edges, attempts, max_attempts)) {
            attempts += a;
            continue;
        }
        auto instance = Instance<boost::undirectedS>(gen, graph, args, dictionary, pos_dictionary);
        );
        batched_instances.add(instance);
    }
    return batched_instances.package_for_model(dictionary, pos_dictionary);
}


inline py::dict khops_n(const int min_khops, const int max_khops,
                            const int min_prefix_length, const int max_prefix_length,
                            const bool right_side_connect=true,
                            const bool permutation_version=false,
                            const bool mask_to_vocab_size=false,
                            int min_vocab = -1, int max_vocab = -1,
                            const int batch_size = 256,
                            const int num_thinking_tokens = 0,
                            const string &scratchpad_type = "none",
                            const bool scratchpad_as_prefix = false,
                            const bool is_flat_model = true,
                            const bool align_prefix_front_pad = false,
                            const py::kwargs &kwargs = py::kwargs()) {

    // we always use the full vocabulary, this is another difference from graphs where the number of nodes can vary
    // if we do not do this then we need to use a node mapping which is extra complexity for little gain
    if (min_vocab == -1 and max_vocab == -1) {  // this is bad in general as the sequence length will be large
        min_vocab = dictionary_num_special;
        max_vocab = dictionary_max_vocab - 1;
    } else if (max_vocab == -1) {
        throw std::invalid_argument("Invalid arguments: max_vocab == -1 and min_vocab != -1");
    }

    optional<vector<float>> task_sample_dist = nullopt;
    if (kwargs.contains("task_sample_dist")) {
        if (!kwargs["task_sample_dist"].is_none() and !kwargs["task_sample_dist"].cast<py::list>().empty()) {
            task_sample_dist = kwargs["task_sample_dist"].cast<vector<float> >();
        }
    }

    vector<int> fake_segment_lengths;
    auto batched_instances = KHopsBatchedInstances("khops", min_vocab, max_vocab, num_thinking_tokens, is_flat_model, align_prefix_front_pad);

    for (int b = 0; b < batch_size; b++) {
        int k;
        int max_k;
        if (task_sample_dist.has_value()){
            k = std::discrete_distribution<int>(task_sample_dist->begin(), task_sample_dist->end())(gen) + min_khops;
            max_k = task_sample_dist.value()[task_sample_dist.value().size() - 1] + min_khops;
        } else {
            k = uniform_int_distribution<int>(min_khops, max_khops)(gen) + 1;  // this is [min, max]
            max_k = max_khops + 1;
        }
        int prefix_length = uniform_int_distribution<int>(min_prefix_length, max_prefix_length)(gen);


        auto instance = KHopsInstance(gen, dictionary, k, max_k , min_vocab, max_vocab,
                                      "khops", scratchpad_type,
                                      prefix_length, fake_segment_lengths,
                                      right_side_connect, permutation_version, mask_to_vocab_size,
                                      scratchpad_as_prefix);
        batched_instances.add(instance);
    }
    return batched_instances.package_for_model(dictionary, pos_dictionary);
}


inline py::dict khops_gen_n(const int min_khops, const int max_khops,
                            const int min_prefix_length, const int max_prefix_length,
                            const bool right_side_connect = true, const string &partition_method = "uniform",
                            int min_vocab = -1, int max_vocab = -1,
                            const int batch_size = 256,
                            const int num_thinking_tokens = 0,
                            const string &scratchpad_type = "none",
                            const bool scratchpad_as_prefix = false,
                            const bool is_flat_model = true,
                            const bool align_prefix_front_pad = false,
                            const py::kwargs &kwargs = py::kwargs()) {

    // we always use the full vocabulary, this is another difference from graphs where the number of nodes can vary
    // if we do not do this then we need to use a node mapping which is extra complexity for little gain
    if (min_vocab == -1 and max_vocab == -1) {  // this is bad in general as the sequence length will be large
        min_vocab = dictionary_num_special;
        max_vocab = dictionary_max_vocab - 1;
    } else if (max_vocab == -1) {
        throw std::invalid_argument("Invalid arguments: max_vocab == -1 and min_vocab != -1");
    }

    optional<vector<float>> task_sample_dist = nullopt;
    if (kwargs.contains("task_sample_dist")) {
        if (!kwargs["task_sample_dist"].is_none() and !kwargs["task_sample_dist"].cast<py::list>().empty()) {
            task_sample_dist = kwargs["task_sample_dist"].cast<vector<float> >();
        }
    }

    auto batched_instances = KHopsBatchedInstances("khops_gen", min_vocab, max_vocab, num_thinking_tokens, is_flat_model, align_prefix_front_pad);

    for (int b = 0; b < batch_size; b++) {
        int k;
        if (task_sample_dist.has_value()){
            k = std::discrete_distribution<int>(task_sample_dist->begin(), task_sample_dist->end())(gen) + min_khops;
        } else {
            k = uniform_int_distribution<int>(min_khops, max_khops)(gen) + 1;  // this is [min, max]
        }
        int prefix_length = uniform_int_distribution<int>(min_prefix_length, max_prefix_length)(gen);
        vector<int> segment_lengths;
        if (partition_method == "uniform") {
            segment_lengths = sample_int_partition.uniform_random_partition(prefix_length - k, k, gen, true);
            if (static_cast<int>(segment_lengths.size()) != k) {
                throw runtime_error("Error in uniform partitioning");
                segment_lengths = sample_int_partition.non_uniform_random_partition(prefix_length - k, k, gen, true);
            }
        } else {
            segment_lengths = sample_int_partition.non_uniform_random_partition(prefix_length - k, k, gen, true);
        }
        auto instance = KHopsInstance(gen, dictionary, -1, -1,
                                      min_vocab, max_vocab,
                                      "khops_gen", scratchpad_type,
                                      -1, segment_lengths, right_side_connect, false, false,
                                      scratchpad_as_prefix);
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
          py::arg("distance"), py::arg("start"), py::arg("end"), py::arg("gen"),
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
          py::arg("scratchpad_type") = "none",
          py::arg("shuffle_edges") = false,
          py::arg("shuffle_nodes") = false,
          py::arg("min_vocab") = -1,
          py::arg("max_vocab") = -1,
          py::arg("batch_size") = 256,
          py::arg("max_edges") = 512,
          py::arg("max_attempts") = 1000,
          py::arg("is_causal") = false,
          py::arg("is_direct_ranking") = false,
          py::arg("concat_edges") = true,
          py::arg("duplicate_edges") = false,
          py::arg("include_nodes_in_graph_tokenization") = false,
          py::arg("query_at_end") = true,
          py::arg("num_thinking_tokens") = 0,
          py::arg("scratchpad_as_prefix") = false,
          py::arg("no_graph") = false,
          py::arg("is_flat_model") = true,
          py::arg("align_prefix_front_pad") = false;

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
          py::arg("scratchpad_type") = "none",
          py::arg("shuffle_edges") = false,
          py::arg("shuffle_nodes") = false,
          py::arg("min_vocab") = -1,
          py::arg("max_vocab") = -1,
          py::arg("batch_size") = 256,
          py::arg("max_edges") = 512,
          py::arg("max_attempts") = 1000,
          py::arg("is_causal") = false,
          py::arg("is_direct_ranking") = false,
          py::arg("concat_edges") = true,
          py::arg("duplicate_edges") = false,
          py::arg("include_nodes_in_graph_tokenization") = false,
          py::arg("query_at_end") = true,
          py::arg("num_thinking_tokens") = 0,
          py::arg("scratchpad_as_prefix") = false,
          py::arg("no_graph") = false,
          py::arg("is_flat_model") = true,
          py::arg("align_prefix_front_pad") = false;

    m.def("random_tree_n", &random_tree_n,
          "Generate a batch of random tree graphs\nGraph specific parameters:\n\t"
          "probs: list of probs of branching.\n\t"
          "max_depth: max branching factor.\n\t",

          py::arg("min_num_nodes"),
          py::arg("max_num_nodes"),
          py::arg("max_degree") = 3,
          py::arg("max_depth") = 7,
          py::arg("bernoulli_p") = 0.5,
          py::arg("c_min") = 75,
          py::arg("c_max") = 125,
          py::arg("task_type") = "shortest_path",
          py::arg("scratchpad_type") = "none",
          py::arg("shuffle_edges") = false,
          py::arg("shuffle_nodes") = false,
          py::arg("min_vocab") = -1,
          py::arg("max_vocab") = -1,
          py::arg("batch_size") = 256,
          py::arg("max_edges") = 512,
          py::arg("max_attempts") = 1000,
          py::arg("is_causal") = false,
          py::arg("is_direct_ranking") = false,
          py::arg("concat_edges") = true,
          py::arg("duplicate_edges") = false,
          py::arg("include_nodes_in_graph_tokenization") = false,
          py::arg("query_at_end") = true,
          py::arg("num_thinking_tokens") = 0,
          py::arg("scratchpad_as_prefix") = false,
          py::arg("no_graph") = false,
          py::arg("is_flat_model") = true,
          py::arg("align_prefix_front_pad") = false;

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
          py::arg("scratchpad_type") = "none",
          py::arg("shuffle_edges") = false,
          py::arg("shuffle_nodes") = false,
          py::arg("min_vocab") = -1,
          py::arg("max_vocab") = -1,
          py::arg("batch_size") = 256,
          py::arg("max_edges") = 512,
          py::arg("max_attempts") = 1000,
          py::arg("is_causal") = false,
          py::arg("is_direct_ranking") = false,
          py::arg("concat_edges") = true,
          py::arg("duplicate_edges") = false,
          py::arg("include_nodes_in_graph_tokenization") = false,
          py::arg("query_at_end") = true,
          py::arg("num_thinking_tokens") = 0,
          py::arg("scratchpad_as_prefix") = false,
          py::arg("no_graph") = false,
          py::arg("is_flat_model") = true,
          py::arg("align_prefix_front_pad") = false;

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
          py::arg("scratchpad_type") = "none",
          py::arg("shuffle_edges") = false,
          py::arg("shuffle_nodes") = false,
          py::arg("min_vocab") = -1,
          py::arg("max_vocab") = -1,
          py::arg("batch_size") = 256,
          py::arg("max_edges") = 512,
          py::arg("max_attempts") = 1000,
          py::arg("is_causal") = false,
          py::arg("is_direct_ranking") = false,
          py::arg("concat_edges") = true,
          py::arg("duplicate_edges") = false,
          py::arg("include_nodes_in_graph_tokenization") = false,
          py::arg("query_at_end") = true,
          py::arg("num_thinking_tokens") = 0,
          py::arg("scratchpad_as_prefix") = false,
          py::arg("no_graph") = false,
          py::arg("is_flat_model") = true,
          py::arg("align_prefix_front_pad") = false;

    m.def("khops_n", &khops_n,
          "Generate a batch of k-hops tasks\nParameters:\n\t"
          "min_k: min number of hops.\n\t"
          "max_k: max number of hops.\n\t"
          "min_prefix_length: min prefix length.\n\t"
          "max_prefix_length: max prefix length.\n\t"
          "right_side_connect: whether to connect from right side.\n\t"
          "permutation_version:  \n\t"
          "mask_to_vocab_size:  \n\t",

          py::arg("min_khops"),
          py::arg("max_khops"),
          py::arg("min_prefix_length"),
          py::arg("max_prefix_length"),
          py::arg("right_side_connect") = true,
          py::arg("permutation_version") = true,
          py::arg("mask_to_vocab_size") = false,
          py::arg("min_vocab") = -1,
          py::arg("max_vocab") = -1,
          py::arg("batch_size") = 256,
          py::arg("num_thinking_tokens") = 0,
          py::arg("scratchpad_type") = "none",
          py::arg("scratchpad_as_prefix") = false,
          py::arg("is_flat_model") = true,
          py::arg("align_prefix_front_pad") = false);

    m.def("khops_gen_n", &khops_gen_n,
          "Generate a batch of k-hops generation tasks\nParameters:\n\t"
          "min_k: min number of hops.\n\t"
          "max_k: max number of hops.\n\t"
          "min_prefix_length: min prefix length.\n\t"
          "max_prefix_length: max prefix length.\n\t"
          "right_side_connect: whether to connect from right side.\n\t"
          "partition_method: method to partition the prefix length ('uniform' or 'non_uniform').\n\t",

          py::arg("min_khops"),
          py::arg("max_khops"),
          py::arg("min_prefix_length"),
          py::arg("max_prefix_length"),
          py::arg("right_side_connect") = true,
          py::arg("partition_method") = "uniform",
          py::arg("min_vocab") = -1,
          py::arg("max_vocab") = -1,
          py::arg("batch_size") = 256,
          py::arg("num_thinking_tokens") = 0,
          py::arg("scratchpad_type") = "none",
          py::arg("scratchpad_as_prefix") = false,
          py::arg("is_flat_model") = true,
          py::arg("align_prefix_front_pad") = false);

    m.def("verify_khop_gens", &KHopsGenTask::verify_khop_gens<int>,
          "Batch verify khops is correct",
          py::arg("prefixes"), py::arg("prefix_lengths"),
          py::arg("ground_truths"), py::arg("ground_truth_lengths"),
          py::arg("right_side_connect") = true);
}
