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

#include "undirected_graphs.h"
#include "directed_graphs.h"
#include "utils.h"

using namespace std;

namespace py = pybind11;
using namespace py::literals;

/* ************************************************
 *  Seeding, needed for datastream in python
 *  ***********************************************/

inline unsigned int seed_ = std::random_device{}();
static thread_local auto gen = std::mt19937(seed_);  // so each thread in the dataloader is different

inline unsigned int get_seed() {
    return seed_;
}

inline void set_seed(const unsigned int seed = 0) {
    gen.seed(seed);
    seed_ = seed;
}

/* ************************************************
 *  Constructing inputs and targets for model
 *  Single graph generation
 *  ***********************************************/

vector<int> get_node_shuffle_map(const int N, const int min_vocab, int max_vocab, const bool shuffle = false) {
    // Shuffle nodes and map to the new range [min_vocab, max_vocab)
    if ( max_vocab > 0 ) {
        assert( (max_vocab - min_vocab) >= N && max_vocab - min_vocab > 0 && min_vocab >= 0);
    } else {
        assert ( min_vocab == 0 );
        max_vocab = N;
    }
    auto m = std::vector<int>(max_vocab - min_vocab);
    std::iota(m.begin(), m.end(), min_vocab);
    if (shuffle) {
        std::shuffle(m.begin(), m.end(), gen);
    }
    return std::vector<int>(m.begin(), m.begin() + N);  // Only return the first N elements of the new range
}

vector<int> get_edge_shuffle_map(const int E, const bool shuffle = false) {
    // shuffle the edges around, this will be the shuffled order given to the model
    auto m = std::vector<int>(E);
    std::iota(m.begin(), m.end(), 0);
    if (shuffle) {
        std::shuffle(m.begin(), m.end(), gen);
    }
    return m;
}

template <typename D>
vector<pair<int, int>> get_edge_list(unique_ptr<Graph<D>> &g_ptr, vector<int> &shuffle_map) {
    // Get the edge list of the graph in the shuffled order
    auto edge_list =vector<pair<int, int>>(num_edges(*g_ptr), make_pair(-1, -1));
    typename boost::graph_traits<Graph<D>>::edge_iterator ei, ei_end;
    int cur = 0;
    for (boost::tie(ei, ei_end) = boost::edges(*g_ptr); ei != ei_end; ++ei) {
        edge_list[shuffle_map[cur]] = make_pair(source(*ei, *g_ptr), target(*ei, *g_ptr));
        cur += 1;
    }
    return edge_list;
}


inline vector<int> sample_path(const unique_ptr<vector<vector<int>>> &distances_ptr,
    const int max_length = 10, const int min_length = 1, int start = -1, int end = -1) {
    /*
     * Uniform sample paths of length between min_length and max_length
     * return length as vector of node ids
     * This is hardcoded for checking for distances of 1 as a connection
     */
    uniform_int_distribution<int> d1(min_length, max_length);
    pair<int, int> start_end;
    if (start != -1 && end != -1) {
        start_end = make_pair(start, end);
    } else {
        int attempts = 0;
        // could avoid while loop by making set of sets of paths and sampling that but may not as fast?
        while ( true ) { // sample a path of length between min_length and max_length
            auto len_ = d1(gen);
            // get all paths of that length
            auto set_of_paths = vector<pair<int, int>>();
            for (int i = 0; i < (*distances_ptr).size(); i++) {
                for (int j = 0; j < (*distances_ptr)[i].size(); j++) {
                    if ((*distances_ptr)[i][j] == len_) {
                        set_of_paths.push_back(make_pair(i, j));
                    }
                }
            }
            if (set_of_paths.size() > 0) { // sample a path from the set
                uniform_int_distribution<int> d2(0, set_of_paths.size() - 1);
                start_end = set_of_paths[d2(gen)];
                break;
            }
            attempts += 1;
            if ( attempts > 10 ) {
                // pick a random path after too many attempts
                uniform_int_distribution<int> d3(0, (*distances_ptr).size() - 1);
                auto i = d3(gen);
                auto j = d3(gen);
                start_end = make_pair(i, j);
                break;
            }
        }
    }
    // reconstruct path
    vector<int> path;
    path.push_back(start_end.first);
    int cur = start_end.first;
    // print_matrix(distances_ptr, (*distances_ptr).size(), (*distances_ptr)[0].size(), true, 100000, "n");
    while (cur != start_end.second) {
        // for all neighbors of cur, find the one with the shortest distance to end
        vector<pair<int, int>> neighbors;
        for (int i = 0; i < distances_ptr->size(); i++) {
            if ((*distances_ptr)[cur][i] == 1 &&  // hardcoded, should pass in graph and get edges
                (*distances_ptr)[i][start_end.second] < (*distances_ptr)[cur][start_end.second]) {
                neighbors.push_back(make_pair(i, (*distances_ptr)[i][start_end.second]));
            }
        }
        // shuffle neighbors and then sort by distance to end, dumb way to do this
        if (neighbors.size() == 0) {
            // print_matrix(distances_ptr, (*distances_ptr).size(), (*distances_ptr)[0].size(), true, 100000, " ");
            assert (neighbors.size() > 0);
        }
        std::shuffle(neighbors.begin(), neighbors.end(), gen);
        std::sort(neighbors.begin(), neighbors.end(), [](const pair<int, int> &a, const pair<int, int> &b) {
            return a.second < b.second;
        });
        // pick the first neighbor
        cur = neighbors[0].first;
        path.push_back(cur);
    }
    return path;
}

template <typename T>
void non_causal_ground_truths(unique_ptr<vector<vector<T>>> &distance, unique_ptr<vector<vector<T>>> &ground_truths_ptr,
    vector<pair<int, int>> &edge_list) {
    // Makes a [E, N] matrix of ground truths where each row is the distance from the edge.first to all other nodes
    auto N = distance->size();
    auto E = edge_list.size();
    ground_truths_ptr = make_unique<vector<vector<int>>>(E, vector<int>(N, -1));
    for (int t = 0; t < E; t++) {
        for (int i = 0; i < N; i++) {
            (*ground_truths_ptr)[t][i] = (*distance)[edge_list[t].first][i];
        }
    }
}

template <typename D>
py::dict package_for_python(unique_ptr<Graph<D>> &g_ptr,
                            const int max_length = 10, const int min_length = 1, int start = -1, int end = -1,
                            const bool is_causal = false,
                            const bool shuffle_edges = false,
                            const bool shuffle_nodes = false, const int min_vocab = 0, int max_vocab = -1) {

	const auto N = num_vertices(*g_ptr);
    const auto E = num_edges(*g_ptr);

    unique_ptr<DistanceMatrix<boost::undirectedS>> distances_ptr;
    // floyd_warshall(g_ptr, distances_ptr, false);  // much slower for sparse graphs
    johnson<D>(g_ptr, distances_ptr, false);

    auto edge_shuffle_map = get_edge_shuffle_map(E, shuffle_edges);  // just range if no shuffle
    auto edge_list = get_edge_list(g_ptr, edge_shuffle_map);

    unique_ptr<vector<vector<int>>> distances_ptr2;
    unique_ptr<vector<vector<int>>> ground_truths_ptr;
    if ( is_causal ) {
        floyd_warshall_frydenlund(g_ptr, distances_ptr2, ground_truths_ptr, edge_list, false);
    } else {
        convert_boost_matrix(distances_ptr, distances_ptr2, N, N);
        non_causal_ground_truths(distances_ptr2, ground_truths_ptr, edge_list);
    }

    auto path = sample_path(distances_ptr2, max_length, min_length, start, end);

    auto node_shuffle_map = get_node_shuffle_map(N, min_vocab, max_vocab, shuffle_nodes);
    auto new_N = *max_element(node_shuffle_map.begin(), node_shuffle_map.end()) + 1;

    auto original_distances = convert_distance_matrix<int, DistanceMatrix<boost::undirectedS>>(distances_ptr, node_shuffle_map, N, new_N);
    auto distances = convert_distance_matrix<int, vector<vector<int>>>(distances_ptr2, node_shuffle_map, N, new_N);
    auto ground_truths = convert_ground_truths<int, vector<vector<int>>>(ground_truths_ptr, node_shuffle_map, E, N, new_N);

    py::dict d;
    d["edge_list"] = convert_edge_list(edge_list, node_shuffle_map);
    d["original_distances"] = original_distances;
    d["distances"] = distances;
    d["ground-truths"] = ground_truths;
    d["path"] = convert_path(path, node_shuffle_map);
    d["node_map"] = convert_vector(node_shuffle_map);
    return d;
}


inline py::dict erdos_renyi(const int num_nodes, float p = -1.0, const int c_min = 75, const int c_max = 125,
    const int max_path_length = 10, const int min_path_length = 1,
    const bool is_causal = false, const bool shuffle_edges = false,
    const bool shuffle_nodes = false, const int min_vocab = 0, int max_vocab = -1) {

    assert (p < 1.0);  //  boost fails at p = 1.0, way to go boost
    unique_ptr<Graph<boost::undirectedS>> g_ptr;
    erdos_renyi_generator(g_ptr,  num_nodes, gen, p, c_min, c_max, false);
    return package_for_python(g_ptr, max_path_length, min_path_length, -1, -1,
        is_causal, shuffle_edges, shuffle_nodes, min_vocab, max_vocab);
}

inline py::dict euclidian(const int num_nodes, const int dim = 2, float radius = -1.0, const int c_min = 75, const int c_max = 125,
    const int max_path_length = 10, const int min_path_length = 1,
    const bool is_causal = false, const bool shuffle_edges = false,
    const bool shuffle_nodes = false, const int min_vocab = 0, int max_vocab = -1) {

    unique_ptr<Graph<boost::undirectedS>> g_ptr;
    unique_ptr<vector<vector<float>>> positions_ptr;
    euclidean_generator(g_ptr, positions_ptr, num_nodes, gen, dim, radius, c_min, c_max, false);
    auto d = package_for_python(g_ptr, max_path_length, min_path_length, -1, -1,
        is_causal, shuffle_edges, shuffle_nodes, min_vocab, max_vocab);
    // Only return valid positions (otherwise nodes have been mapped and I don't want to convert positions)
    // Positions are only for plotting, when mapping should not be done.
    // This may not be true if I ever use positions as neural net features
    if ( !shuffle_nodes && max_vocab == -1 ) {
        auto N = num_vertices(*g_ptr);
        constexpr size_t M = 2;
        auto positions = py::array_t<float, py::array::c_style>({N, M});
        auto ra = positions.mutable_unchecked();
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < 2; j++) {
                constexpr float r = 10000;
                ra(i, j) =  ceil((*positions_ptr)[i][j] * r) / r;;
            }
        }
        d["positions"] = positions;
    }
    return d;
}

inline py::dict path_star(const int min_num_arms, const int max_num_arms, const int min_arm_length, const int max_arm_length,
    const bool is_causal = false, const bool shuffle_edges = false,
    const bool shuffle_nodes = false, const int min_vocab = 0, int max_vocab = -1) {

    unique_ptr<Graph<boost::directedS>> g_ptr;
    auto start_end = path_star_generator(g_ptr,  min_num_arms, max_num_arms, min_arm_length,max_arm_length, gen, false);
    auto d = package_for_python(g_ptr, -1, -1, start_end.first, start_end.second,
        is_causal, shuffle_edges, shuffle_nodes, min_vocab, max_vocab);
    d["start"] = start_end.first;
    d["end"] = start_end.second;
    return d;
}

inline py::dict balanced(const int num_nodes, int lookahead, const int min_noise_reserve = 0, const int max_num_parents = 4, int max_noise = -1,
    const bool is_causal = false, const bool shuffle_edges = false,
    const bool shuffle_nodes = false, const int min_vocab = 0, int max_vocab = -1) {
    unique_ptr<Graph<boost::directedS>> g_ptr;
    auto start_end = balanced_generator(g_ptr, num_nodes, gen, lookahead, min_noise_reserve, max_num_parents, false);

    auto d = package_for_python(g_ptr, -1, -1, start_end.first, start_end.second,
        is_causal, shuffle_edges, shuffle_nodes, min_vocab, max_vocab);
    d["start"] = start_end.first;
    d["end"] = start_end.second;
    return d;
}


/* ************************************************
 *  Constructing inputs and targets for model
 *  Batched graph generation
 *  ***********************************************/


template <typename D>
void push_back_data(unique_ptr<Graph<D>> &g_ptr,
                    vector<int> &edge_shuffle_map,
                    const bool is_causal, const bool sample_target_paths,
                    list<unique_ptr<vector<pair<int, int>>>> &batched_edge_list,
                    list<int> &batched_edge_list_lengths,
                    list<unique_ptr<vector<vector<int>>>> &batched_distances,
                    list<unique_ptr<vector<vector<int>>>> &batched_ground_truths,
                    list<unique_ptr<vector<int>>> &batched_paths,
                    list<int> &batched_path_lengths,
                    const int max_length = 10, const int min_length = 1, int start = -1, int end = -1) {

    const auto E = num_edges(*g_ptr);
    auto edge_list = get_edge_list<D>(g_ptr, edge_shuffle_map);
    batched_edge_list.push_back(make_unique<vector<pair<int, int>>>(edge_list));
    batched_edge_list_lengths.push_back(E);
    unique_ptr<vector<vector<int>>> distances_ptr;
    unique_ptr<vector<vector<int>>> ground_truths_ptr;
    if ( is_causal ) {
        // auto path_d = time_before();
        floyd_warshall_frydenlund(g_ptr, distances_ptr, ground_truths_ptr, edge_list, false);
        // time_after(path_d, "floyd_warshall_frydenlund");
        if ( sample_target_paths ) {  // needs to be here because of unique_ptr scope
            // auto path_t = time_before();
            auto path = sample_path(distances_ptr, max_length, min_length, start, end);
            // time_after(path_t, "sample_path");
            batched_path_lengths.push_back(path.size());
            batched_paths.push_back(make_unique<vector<int>>(path));
        }
        batched_distances.push_back(move(distances_ptr));
        batched_ground_truths.push_back(move(ground_truths_ptr));
    } else {
        auto N = num_vertices(*g_ptr);
        unique_ptr<DistanceMatrix<D>> boost_distances_ptr;
        // auto path_d = time_before();
        // floyd_warshall<D>(g_ptr, distances_ptr, false);  // much slower for sparse graphs
        johnson<D>(g_ptr, boost_distances_ptr, false);
        // time_after(path_d, "floyd_warshall");
        convert_boost_matrix<int, DistanceMatrix<D>>(boost_distances_ptr, distances_ptr, N, N);
        non_causal_ground_truths(distances_ptr, ground_truths_ptr, edge_list);
        if ( sample_target_paths ) {
            // auto path_t = time_before();
            auto path = sample_path(distances_ptr, max_length, min_length, start, end);
            // time_after(path_t, "sample_path");
            batched_path_lengths.push_back(path.size());
            batched_paths.push_back(make_unique<vector<int>>(path));

        }
        batched_distances.push_back(move(distances_ptr));
        batched_ground_truths.push_back(move(ground_truths_ptr));
    }
}

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
    const int num_nodes, float p = -1.0, const int c_min = 75, const int c_max = 125,
    const int max_length = 10, const int min_length = 1, const bool sample_target_paths = true,
    const bool is_causal = false, const bool shuffle_edges = false,
    const bool shuffle_nodes = false, const int min_vocab = 0, int max_vocab = -1,
    const int batch_size = 256, const int max_edges = 512, int max_attempts = 1000) {

    assert ( p < 1.0);  //  boost fails at p = 1.0, way to go boost
    assert ( c_min <= c_max);
    assert ( batch_size > 0);
    assert ( min_length <= max_length);
    auto batched_edge_list = list<unique_ptr<vector<pair<int, int>>>>();
    auto batched_node_shuffle_map = list<unique_ptr<vector<int>>>();
    auto batched_distances = list<unique_ptr<vector<vector<int>>>>();
    auto batched_ground_truths = list<unique_ptr<vector<vector<int>>>>();
    auto batched_paths = list<unique_ptr<vector<int>>>();
    auto batched_edge_list_lengths = list<int>();
    auto batched_path_lengths = list<int>();

    int attempts = 0;
    int num = 0;
    while ( num < batch_size && attempts < max_attempts ) {
        unique_ptr<Graph<boost::undirectedS>> g_ptr;
        // auto graph_t = time_before();
        erdos_renyi_generator(g_ptr,  num_nodes, gen, p, c_min, c_max, false);
        // time_after(graph_t, "graph gen");
        const auto N = num_vertices(*g_ptr);
        const auto E = num_edges(*g_ptr);
        if (const auto a = attempt_check(E, max_edges, attempts, max_attempts) ) {
            attempts += a;
            continue;
        }
        auto edge_shuffle_map = get_edge_shuffle_map(E, shuffle_edges);
        auto node_shuffle_map = get_node_shuffle_map(N, min_vocab, max_vocab, shuffle_nodes);
        batched_node_shuffle_map.push_back(make_unique<vector<int>>(node_shuffle_map));
        // auto pack_t = time_before();
        push_back_data<boost::undirectedS>(g_ptr, edge_shuffle_map,  is_causal, sample_target_paths,
            batched_edge_list, batched_edge_list_lengths, batched_distances,
            batched_ground_truths, batched_paths,  batched_path_lengths,
            max_length, min_length);
        // time_after(pack_t, "pack");
        num += 1;
    }

    auto new_N = num_nodes;
    if ( max_vocab > 0 ) {
        new_N = max_vocab;
    }
    py::dict d;
    d["num_attempts"] = attempts;
    d["vocab_min_size"] = min_vocab;
    d["vocab_max_size"] = max_vocab;
    if ( attempts >= max_attempts ) {
        return d;
    }
    d["edge_list"] = batch_edge_list<int>(batched_edge_list, batched_node_shuffle_map);
    d["edge_list_lengths"] = batch_lengths<int>(batched_edge_list_lengths);
    auto bd = batch_distances<int>(batched_distances, batched_node_shuffle_map, new_N);
    d["distances"] = bd;
    d["hashs"] = hash_distance_matrix<int>(bd);
    d["ground-truths"] = batch_ground_truths<int>(batched_ground_truths, batched_node_shuffle_map, new_N);
    if ( sample_target_paths ) {
        d["paths"] = batch_paths<int>(batched_paths, batched_node_shuffle_map);
        d["path_lengths"] = batch_lengths<int>(batched_path_lengths);
    }
    return d;
}


inline py::dict euclidian_n(
    const int num_nodes, const int dim = 2, float radius = -1.0, const int c_min = 75, const int c_max = 125,
    const int max_length = 10, const int min_length = 1, const bool sample_target_paths = true,
    const bool is_causal = false, const bool shuffle_edges = false,
    const bool shuffle_nodes = false, const int min_vocab = 0, int max_vocab = -1,
    const int batch_size = 256, const int max_edges = 512, int max_attempts = 1000) {

    assert ( dim > 0);
    assert ( c_min <= c_max);
    assert ( batch_size > 0);
    assert ( min_length <= max_length);
    auto batched_edge_list = list<unique_ptr<vector<pair<int, int>>>>();
    auto batched_node_shuffle_map = list<unique_ptr<vector<int>>>();
    auto batched_distances = list<unique_ptr<vector<vector<int>>>>();
    auto batched_ground_truths = list<unique_ptr<vector<vector<int>>>>();
    auto batched_paths = list<unique_ptr<vector<int>>>();
    auto batched_edge_list_lengths = list<int>();
    auto batched_path_lengths = list<int>();

    auto batched_positions = list<unique_ptr<vector<vector<float>>>>();  // [batch_size, dim + 1] of node_id, x, y etc.

    int attempts = 0;
    int num = 0;
    while ( num < batch_size && attempts < max_attempts ) {
        unique_ptr<Graph<boost::undirectedS>> g_ptr;
        unique_ptr<vector<vector<float>>> positions_ptr;
        // auto graph_t = time_before();
        euclidean_generator(g_ptr, positions_ptr, num_nodes, gen, dim, radius, c_min, c_max, false);
        batched_positions.push_back(move(positions_ptr));
        // time_after(graph_t, "graph gen");
        const auto N = num_vertices(*g_ptr);
        const auto E = num_edges(*g_ptr);
        if (const auto a = attempt_check(E, max_edges, attempts, max_attempts) ) {
            attempts += a;
            continue;
        }
        auto edge_shuffle_map = get_edge_shuffle_map(E, shuffle_edges);
        auto node_shuffle_map = get_node_shuffle_map(N, min_vocab, max_vocab, shuffle_nodes);
        batched_node_shuffle_map.push_back(make_unique<vector<int>>(node_shuffle_map));
        // auto pack_t = time_before();
        push_back_data<boost::undirectedS>(g_ptr, edge_shuffle_map,  is_causal, sample_target_paths,
            batched_edge_list, batched_edge_list_lengths, batched_distances,
            batched_ground_truths, batched_paths,  batched_path_lengths,
            max_length, min_length);
        // time_after(pack_t, "pack");
        num += 1;
    }

    auto new_N = num_nodes;
    if ( max_vocab > 0 ) {
        new_N = max_vocab;
    }
    py::dict d;
    d["num_attempts"] = attempts;
    d["vocab_min_size"] = min_vocab;
    d["vocab_max_size"] = max_vocab;
    if ( attempts >= max_attempts ) {
        return d;
    }
    cout << "Generated " << num << " graphs of " << batch_size << endl;
    d["edge_list"] = batch_edge_list<int>(batched_edge_list, batched_node_shuffle_map);
    d["edge_list_lengths"] = batch_lengths<int>(batched_edge_list_lengths);
    auto bd = batch_distances<int>(batched_distances, batched_node_shuffle_map, new_N);
    d["distances"] = bd;
    d["hashs"] = hash_distance_matrix<int>(bd);
    d["ground-truths"] = batch_ground_truths<int>(batched_ground_truths, batched_node_shuffle_map, new_N);
    if ( sample_target_paths ) {
        d["paths"] = batch_paths<int>(batched_paths, batched_node_shuffle_map);
        d["path_lengths"] = batch_lengths<int>(batched_path_lengths);
    }
    cout << "Generated " << num << " graphs of " << batch_size << endl;
    d["positions"] = batch_positions<float>(batched_positions, batched_node_shuffle_map, dim);

    return d;

}


inline py::dict path_star_n(
    const int min_num_arms, const int max_num_arms, const int min_arm_length, const int max_arm_length,
    const bool sample_target_paths = true,
    const bool is_causal = false, const bool shuffle_edges = false,
    const bool shuffle_nodes = false, const int min_vocab = 0, int max_vocab = -1,
    const int batch_size = 256, const int max_edges = 512, int max_attempts = 1000) {
    assert ( min_num_arms > 0);
    assert ( min_arm_length > 0);
    assert ( batch_size > 0);
    auto batched_edge_list = list<unique_ptr<vector<pair<int, int>>>>();
    auto batched_node_shuffle_map = list<unique_ptr<vector<int>>>();
    auto batched_distances = list<unique_ptr<vector<vector<int>>>>();
    auto batched_ground_truths = list<unique_ptr<vector<vector<int>>>>();
    auto batched_paths = list<unique_ptr<vector<int>>>();
    auto batched_edge_list_lengths = list<int>();
    auto batched_path_lengths = list<int>();

    int attempts = 0;
    int num = 0;
    while ( num < batch_size && attempts < max_attempts ) {
        unique_ptr<Graph<boost::directedS>> g_ptr;
        // auto graph_t = time_before();
        auto start_end = path_star_generator(g_ptr,  min_num_arms, max_num_arms, min_arm_length,max_arm_length, gen, false);
        // time_after(graph_t, "graph gen");
        const auto N = num_vertices(*g_ptr);
        const auto E = num_edges(*g_ptr);
        if (const auto a = attempt_check(E, max_edges, attempts, max_attempts) ) {
            attempts += a;
            continue;
        }
        auto edge_shuffle_map = get_edge_shuffle_map(E, shuffle_edges);
        auto node_shuffle_map = get_node_shuffle_map(N, min_vocab, max_vocab, shuffle_nodes);
        batched_node_shuffle_map.push_back(make_unique<vector<int>>(node_shuffle_map));
        // auto pack_t = time_before();
        push_back_data<boost::directedS>(g_ptr, edge_shuffle_map,  is_causal, sample_target_paths,
            batched_edge_list, batched_edge_list_lengths, batched_distances,
            batched_ground_truths, batched_paths,  batched_path_lengths,
            -1, -1, start_end.first, start_end.second);
        // time_after(pack_t, "pack");
        num += 1;
    }
    auto new_N = max_num_arms * (max_arm_length - 1) + 1;
    if ( max_vocab > 0 ) {
        new_N = max_vocab;
    }
    py::dict d;
    d["num_attempts"] = attempts;
    d["vocab_min_size"] = min_vocab;
    d["vocab_max_size"] = max_vocab;
    if ( attempts >= max_attempts ) {
        return d;
    }
    d["edge_list"] = batch_edge_list<int>(batched_edge_list, batched_node_shuffle_map);
    d["edge_list_lengths"] = batch_lengths<int>(batched_edge_list_lengths);
    auto bd = batch_distances<int>(batched_distances, batched_node_shuffle_map, new_N);
    d["distances"] = bd;
    d["hashs"] = hash_distance_matrix<int>(bd);
    d["ground-truths"] = batch_ground_truths<int>(batched_ground_truths, batched_node_shuffle_map, new_N);
    if ( sample_target_paths ) {
        d["paths"] = batch_paths<int>(batched_paths, batched_node_shuffle_map);
        d["path_lengths"] = batch_lengths<int>(batched_path_lengths);
    }
    return d;
}


inline py::dict balanced_n(
    const int num_nodes, const int min_lookahead, const int max_lookahead, const int min_noise_reserve = 0, const int max_num_parents = 4, int max_noise = -1,
    const bool sample_target_paths = true,
    const bool is_causal = false, const bool shuffle_edges = false,
    const bool shuffle_nodes = false, const int min_vocab = 0, int max_vocab = -1,
    const int batch_size = 256, const int max_edges = 512, int max_attempts = 1000) {
    assert ( num_nodes > 0);
    assert ( min_lookahead > 0 && max_lookahead > 0 );
    assert ( batch_size > 0 );
    assert ( max_num_parents >= 0 );
    auto batched_edge_list = list<unique_ptr<vector<pair<int, int>>>>();
    auto batched_node_shuffle_map = list<unique_ptr<vector<int>>>();
    auto batched_distances = list<unique_ptr<vector<vector<int>>>>();
    auto batched_ground_truths = list<unique_ptr<vector<vector<int>>>>();
    auto batched_paths = list<unique_ptr<vector<int>>>();
    auto batched_edge_list_lengths = list<int>();
    auto batched_path_lengths = list<int>();

    int attempts = 0;
    int num = 0;

    if ( ! balanced_graph_size_check(num_nodes, max_lookahead, min_noise_reserve ) ) {
        assert( false && "Graph size check failed" );
    }

    while ( num < batch_size && attempts < max_attempts ) {
        unique_ptr<Graph<boost::directedS>> g_ptr;
        // this is the only method which samples here, this was done to keep the main function similar to the original code.
        auto lookahead = uniform_int_distribution<int>(min_lookahead, max_lookahead)(gen);
        int max_noise_sample;
        if ( max_noise > 0) {
            max_noise_sample = uniform_int_distribution<int>(0, max_noise)(gen);
        } else {
            max_noise_sample = uniform_int_distribution<int>(-1, lookahead)(gen);  // -1 means all remaining
        }
        // auto graph_t = time_before();
        auto start_end = balanced_generator(g_ptr, num_nodes, gen, lookahead, min_noise_reserve, max_num_parents, max_noise_sample);

        // time_after(graph_t, "graph gen");
        const auto N = num_vertices(*g_ptr);
        const auto E = num_edges(*g_ptr);
        if (const auto a = attempt_check(E, max_edges, attempts, max_attempts) ) {
            attempts += a;
            continue;
        }
        auto edge_shuffle_map = get_edge_shuffle_map(E, shuffle_edges);
        auto node_shuffle_map = get_node_shuffle_map(N, min_vocab, max_vocab, shuffle_nodes);
        batched_node_shuffle_map.push_back(make_unique<vector<int>>(node_shuffle_map));
        // auto pack_t = time_before();
        push_back_data<boost::directedS>(g_ptr, edge_shuffle_map,  is_causal, sample_target_paths,
            batched_edge_list, batched_edge_list_lengths, batched_distances,
            batched_ground_truths, batched_paths,  batched_path_lengths,
            -1, -1, start_end.first, start_end.second);
        // time_after(pack_t, "pack");
        num += 1;
    }

    auto new_N = num_nodes;
    if ( max_vocab > 0 ) {
        new_N = max_vocab;
    }
    py::dict d;
    d["num_attempts"] = attempts;
    d["vocab_min_size"] = min_vocab;
    d["vocab_max_size"] = max_vocab;
    if ( attempts >= max_attempts ) {
        return d;
    }
    d["edge_list"] = batch_edge_list<int>(batched_edge_list, batched_node_shuffle_map);
    d["edge_list_lengths"] = batch_lengths<int>(batched_edge_list_lengths);
    auto bd = batch_distances<int>(batched_distances, batched_node_shuffle_map, new_N);
    d["distances"] = bd;
    d["hashs"] = hash_distance_matrix<int>(bd);
    d["ground-truths"] = batch_ground_truths<int>(batched_ground_truths, batched_node_shuffle_map, new_N);
    if ( sample_target_paths ) {
        d["paths"] = batch_paths<int>(batched_paths, batched_node_shuffle_map);
        d["path_lengths"] = batch_lengths<int>(batched_path_lengths);
    }
    return d;
}



PYBIND11_MODULE(generator, m) {
  	m.doc() = "Graph generation module"; // optional module docstring
    m.def("set_seed", &set_seed, "Sets random seed (unique to thread)", py::arg("seed") = 0);
    m.def("get_seed", &get_seed, "Gets random seed (unique to thread)");

    // single graph generation
    m.def("erdos_renyi", &erdos_renyi,
        "Generate a single Erdos Renyi graph\nParameters:\n\t"
        "num_nodes: number of nodes\n\t"
        "p: probability of edge creation.  If -1 then 1/num_nodes.\n\t"
        "c_min: min number of sampled edges to form a single connected component\n\t"
        "c_max: max number of of sampled edges to form a single connected component\n\t"
        "max_path_length: max length of path to sample\n\t"
        "min_path_length: min length of path to sample\n\t"
        "is_causal: if true then return causally masked ground-truths\n\t"
        "shuffle_edges: if true then shuffle edges\n\t"
        "shuffle_nodes: if true then shuffle nodes\n\t"
        "min_vocab: min vocab size to map nodes into i.e. to exclude special tokens\n\t"
        "max_vocab: max vocab size to map nodes into.\n"
        "Returns a dict with the following keys (N is the max_vocab):\n\t"
        "edge_list: numpy [E, 2] of edges\n\t"
        "original_distances: numpy [N, N] of original distances (boost calc)\n\t"
        "distances: numpy [N, N] of distances (my calc, for sanity checking)\n\t"
        "ground-truths: numpy [E, N] of ground truths\n\t"
        "path: numpy [L] of path\n\t"
        "node_map: numpy [N] of node map\n\t"
        "hashs: numpy [N] of uint64_t hash of distances\n\t",

        py::arg("num_nodes"),
        py::arg("p") = -1.0, py::arg("c_min") = 75, py::arg("c_max") = 125,
        py::arg("max_path_length") = 10, py::arg("min_path_length") = 3,
        py::arg("is_causal") = false, py::arg("shuffle_edges") = false,
        py::arg("shuffle_nodes") = false, py::arg("min_vocab") = 0, py::arg("max_vocab") = -1);

    m.def("euclidian", &euclidian,
        "Generate a single Euclidian graph.\nParameters:\n\t"
        "num_nodes: number of nodes\n\t"
        "dims: number of dimensions\n\t"
        "radius: radius of graph.  If -1 then 1/sqrt(num_nodes).\n\t"
        "c_min: min number of sampled edges to form a single connected component\n\t"
        "c_max: max number of sampled edges to form a single connected component\n\t"
        "max_path_length: max length of path to sample\n\t"
        "min_path_length: min length of path to sample\n\t"
        "is_causal: if true then return causally masked ground-truths\n\t"
        "shuffle_edges: if true then shuffle edges\n\t"
        "shuffle_nodes: if true then shuffle nodes\n\t"
        "min_vocab: min vocab size to map nodes into i.e. to exclude special tokens\n\t"
        "max_vocab: max vocab size to map nodes into.\n"
        "Returns a dict with the following keys (N is the max_vocab):\n\t"
        "edge_list: numpy [E, 2] of edges\n\t"
        "original_distances: numpy [N, N] of original distances (boost calc)\n\t"
        "distances: numpy [N, N] of distances (my calc, for sanity checking)\n\t"
        "ground-truths: numpy [E, N] of ground truths\n\t"
        "path: numpy [L] of path\n\t"
        "node_map: numpy [N] of node map\n\t"
        "hashs: numpy [N] of uint64_t hash of distances\n\t",
        "positions: numpy [N, dims] of node positions.  Note: node positions are not returned if using node shuffle or vocab range (and these are needed for plotting).\n\t",

        py::arg("num_nodes"),
        py::arg("dims") = 2, py::arg("radius") = -1.0, py::arg("c_min") = 75, py::arg("c_max") = 125,
        py::arg("max_path_length") = 10, py::arg("min_path_length") = 3,
        py::arg("is_causal") = false, py::arg("shuffle_edges") = false,
        py::arg("shuffle_nodes") = false, py::arg("min_vocab") = 0, py::arg("max_vocab") = -1);

    m.def("path_star", &path_star, "Generate a single path star graph",
        py::arg("min_num_arms"), py::arg("max_num_arms"), py::arg("min_arm_length"), py::arg("max_arm_length"),
        py::arg("is_causal") = false, py::arg("shuffle_edges") = false,
        py::arg("shuffle_nodes") = false, py::arg("min_vocab") = 0, py::arg("max_vocab") = -1);

    m.def("balanced", &balanced, "Generate a single balanced graph.  Note these are done slightly differently from original paper.",
        py::arg("num_nodes"), py::arg("lookahead"), py::arg("min_noise_reserve") = 0, py::arg("max_num_parents") = 4, py::arg("max_noise") = -1,
        py::arg("is_causal") = false, py::arg("shuffle_edges") = false,
        py::arg("shuffle_nodes") = false, py::arg("min_vocab") = 0, py::arg("max_vocab") = -1);

    m.def("balanced_graph_size_check", &balanced_graph_size_check, "Check that the balanced graph size is valid.  Will fail assert otherwise.",
        py::arg("num_nodes"), py::arg("lookahead"), py::arg("min_noise_reserve") = 0);

    // batched graph generation
    m.def("erdos_renyi_n", &erdos_renyi_n, "Generate a batch of Erdos Renyi graphs",
        py::arg("num_nodes"), py::arg("p") = -1.0, py::arg("c_min") = 75, py::arg("c_max") = 125,
        py::arg("max_length") = 10, py::arg("min_length") = 1, py::arg("sample_target_paths") = true,
        py::arg("is_causal") = false,  py::arg("shuffle_edges") = false,
        py::arg("shuffle_nodes") = false, py::arg("min_vocab") = 0, py::arg("max_vocab") = -1,
        py::arg("batch_size") = 256, py::arg("max_edges") = 512,  py::arg("max_attempts") = 1000);

    m.def("euclidian_n", &euclidian_n, "Generate a batch of Euclidian graphs",
        py::arg("num_nodes"), py::arg("dims") = 2, py::arg("radius") = -1.0, py::arg("c_min") = 75, py::arg("c_max") = 125,
        py::arg("max_length") = 10, py::arg("min_length") = 1, py::arg("sample_target_paths") = true,
        py::arg("is_causal") = false, py::arg("shuffle_edges") = false,
        py::arg("shuffle_nodes") = false, py::arg("min_vocab") = 0, py::arg("max_vocab") = -1,
        py::arg("batch_size") = 256, py::arg("max_edges") = 512,  py::arg("max_attempts") = 1000);

    m.def("path_star_n", &path_star_n, "Generate a batch of path star graphs",
        py::arg("min_num_arms"), py::arg("max_num_arms"), py::arg("min_arm_length"), py::arg("max_arm_length"),
        py::arg("sample_target_paths") = true,
        py::arg("is_causal") = false,  py::arg("shuffle_edges") = false,
        py::arg("shuffle_nodes") = false, py::arg("min_vocab") = 0, py::arg("max_vocab") = -1,
        py::arg("batch_size") = 256,  py::arg("max_edges") = 512, py::arg("max_attempts") = 1000);

    m.def("balanced_n", &balanced_n, "Generate a batch of balanced graphs",
        py::arg("num_nodes"), py::arg("min_lookahead"), py::arg("max_lookahead"), py::arg("min_noise_reserve") = 0, py::arg("max_num_parents") = 4, py::arg("max_noise") = -1,
        py::arg("sample_target_paths") = true,
        py::arg("is_causal") = false,  py::arg("shuffle_edges") = false,
        py::arg("shuffle_nodes") = false, py::arg("min_vocab") = 0, py::arg("max_vocab") = -1,
        py::arg("batch_size") = 256,  py::arg("max_edges") = 512, py::arg("max_attempts") = 1000);


}