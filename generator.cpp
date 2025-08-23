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
#include <limits>
#include <cmath>

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
    py::array_t<bool, py::array::c_style> arr({static_cast<int>(hashes.size())});
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
    py::array_t<bool, py::array::c_style> arr({static_cast<int>(hashes.size())});
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
    py::array_t<bool, py::array::c_style> arr({static_cast<int>(hashes.size())});
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
 *  Dictionary for mapping to models vocabulary
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

inline void set_default_dictionary(const int max_vocab = 100) {
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
        {"|", 4},
        {"!", 5},
        {"=", 6},
        {".", 7},
        {"t1", 8},
        {"t2", 9},
        {"t3", 10},
        {"t4", 11},
        {"t5", 12},
        {"/", 13},
        {"?", 14},
        {"@", 15},
        {"#", 16},
        {"s1", 17},
        {"s2", 18},
        {"s3", 19},
        {"s4", 20},
        {"s5", 21}
    };

    if (max_vocab > 0) {
        auto num_special = static_cast<int>(dictionary.size());
        assert(max_vocab >=  num_special);
        for (int i = num_special; i < max_vocab; i++) {
            dictionary[std::to_string(i - num_special)] = i;
        }
    }
}

map<std::string, int> get_dictionary() {
    return dictionary;
}


/* ************************************************
 *  Constructing inputs and targets for model
 *  Single graph generation
 *  ***********************************************/

vector<int> get_node_shuffle_map(const int N, const int min_vocab, int max_vocab, const bool shuffle = false) {
    // Shuffle nodes and map to the new range [min_vocab, max_vocab)
    if (max_vocab > 0) {
        // asserts do not work on python side, use throws
        assert((max_vocab - min_vocab) >= N && max_vocab - min_vocab > 0 && min_vocab >= 0);
        if (max_vocab - min_vocab < N) { throw std::invalid_argument("max_vocab - min_vocab < N"); }
    } else {
        assert(min_vocab == 0);
        max_vocab = N;
    }
    auto m = std::vector<int>(max_vocab - min_vocab);
    std::iota(m.begin(), m.end(), min_vocab);
    if (shuffle) {
        std::shuffle(m.begin(), m.end(), gen);
    }
    return std::vector<int>(m.begin(), m.begin() + N); // Only return the first N elements of the new range
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

template<typename D>
vector<pair<int, int> > get_edge_list(unique_ptr<Graph<D> > &g_ptr, vector<int> &shuffle_map) {
    // Get the edge list of the graph in the shuffled order
    auto edge_list = vector<pair<int, int> >(num_edges(*g_ptr), make_pair(-1, -1));
    typename boost::graph_traits<Graph<D> >::edge_iterator ei, ei_end;
    int cur = 0;
    for (boost::tie(ei, ei_end) = boost::edges(*g_ptr); ei != ei_end; ++ei) {
        edge_list[shuffle_map[cur]] = make_pair(source(*ei, *g_ptr), target(*ei, *g_ptr));
        cur += 1;
    }
    return edge_list;
}


inline vector<int> sample_path(const unique_ptr<vector<vector<int> > > &distances_ptr,
                               const int max_path_length = 10, const int min_path_length = 1, int start = -1, int end = -1) {
    /*
     * Uniform sample paths of length between min_path_length and max_path_length
     * return length as vector of node ids
     * This is hardcoded for checking for distances of 1 as a connection
     */
    uniform_int_distribution<int> d1(min_path_length, max_path_length);
    pair<int, int> start_end;
    if (start != -1 && end != -1) {
        start_end = make_pair(start, end);
    } else {
        int attempts = 0;
        // could avoid while loop by making set of sets of paths and sampling that but may not as fast?
        while (true) {
            // sample a path of length between min_path_length and max_path_length
            auto len_ = d1(gen);
            // get all paths of that length
            auto set_of_paths = vector<pair<int, int> >();
            for (int i = 0; i < static_cast<int>((*distances_ptr).size()); i++) {
                for (int j = 0; j < static_cast<int>((*distances_ptr)[i].size()); j++) {
                    if ((*distances_ptr)[i][j] == len_) {
                        set_of_paths.push_back(make_pair(i, j));
                    }
                }
            }
            if (set_of_paths.size() > 0) {
                // sample a path from the set
                uniform_int_distribution<int> d2(0, set_of_paths.size() - 1);
                start_end = set_of_paths[d2(gen)];
                break;
            }
            attempts += 1;
            if (attempts > 10) {
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
        vector<pair<int, int> > neighbors;
        for (int i = 0; i < static_cast<int>(distances_ptr->size()); i++) {
            if ((*distances_ptr)[cur][i] == 1 && // hardcoded, should pass in graph and get edges
                (*distances_ptr)[i][start_end.second] < (*distances_ptr)[cur][start_end.second]) {
                neighbors.push_back(make_pair(i, (*distances_ptr)[i][start_end.second]));
            }
        }
        // shuffle neighbors and then sort by distance to end, dumb way to do this
        if (neighbors.size() == 0) {
            // print_matrix(distances_ptr, (*distances_ptr).size(), (*distances_ptr)[0].size(), true, 100000, " ");
            assert(neighbors.size() > 0);
            throw std::invalid_argument("No neighbors found.  This should never happen.");
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


template<typename T>
int varify_path(py::array_t<T, py::array::c_style> &distances, vector<int> &path) {
    // -1 if not a valid path, 0 if valid path but not a shortest path, 1 if valid path but is a shortest path
    auto start = path[0];
    auto end = path[path.size() - 1];
    auto shortest_distance = distances.at(start, end);
    if (shortest_distance < 0) {
        return -1;
    }
    // validate path
    auto cur = start;
    for (int i = 1; i < path.size(); i++) {
        auto next = path[i];
        if (distances.at(cur, next) != 1) {
            // hardcoded for distance of 1
            return -1;
        }
        cur = next;
    }
    if (path.size() > shortest_distance) {
        return 0;
    }
    return 1;
}


template<typename T>
py::array_t<int, py::array::c_style> verify_paths(py::array_t<T, py::array::c_style> &distances,
                                                  py::array_t<T, py::array::c_style> &paths,
                                                  py::array_t<T, py::array::c_style> &lengths) {
    // batch version [batch_size, vocab_size, vocab_size]
    auto batch_size = paths.shape(0);
    auto out = py::array_t<int, py::array::c_style>({static_cast<int>(batch_size)});
    out[py::make_tuple(py::ellipsis())] = 1; // initialize array to true
    auto ra = out.mutable_unchecked();
    for (auto b = 0; b < batch_size; b++) {
        auto start = paths.at(b, 0);
        auto end = paths.at(b, lengths.at(b) - 1);
        auto shortest_distance = distances.at(b, start, end);
        if (shortest_distance < 0 || shortest_distance > inf - 1) {
            ra(b) = -1;
            break;
        }
        // validate path
        auto path_dist = 0.0;
        auto cur = start;
        for (int j = 1; j < lengths.at(b); j++) {
            auto next = paths.at(b, j);
            auto cur_d = distances.at(b, cur, next);
            if (cur_d <= 0.0 || shortest_distance > inf - 1) {
                ra(b) = -1;
                break;
            }
            path_dist += cur_d;
            cur = next;
        }
        if (ra(b) != -1 && path_dist > shortest_distance) {
            ra(b) = 0;
        }
    }
    return out;
}


bool float_equality(double a, double b) {
    return std::fabs(a - b) < std::numeric_limits<double>::epsilon();
}

inline pair<vector<int>, vector<int> > sample_center_centroid(const unique_ptr<vector<vector<int> > > &distances_ptr,
                                                              vector<int> &given_query,
                                                              int max_query_size = -1, const int min_query_size = 2,
                                                              const bool is_center = true) {
    auto N = static_cast<int>(distances_ptr->size());
    if (max_query_size == -1 || max_query_size > N) {
        max_query_size = N;
    }
    auto new_query = vector<int>();
    if (given_query.empty()) {
        //sample query
        // stackoverflow.com/questions/33802205/how-to-sample-without-replacement-using-c-uniform-int-distribution
        uniform_int_distribution<int> d1(min_query_size, max_query_size);
        auto query_length = d1(gen);
        auto gen = std::mt19937{std::random_device{}()};
        auto nodes = std::vector<int>(N);
        std::iota(nodes.begin(), nodes.end(), 0);
        sample(nodes.begin(), nodes.end(), std::back_inserter(new_query), query_length, gen);
        // std::ranges::shuffle(new_query, gen);  // so that nodes are out of order, this doesn't matter with permute
    } else {
        // copy over elements from given query
        for (auto i: given_query) {
            new_query.push_back(i);
        }
    }
    auto Q = static_cast<int>(new_query.size());
    // calculate center or centroid of graph given queries
    auto values = vector<float>(N, static_cast<float>(inf));
    for (int v = 0; v < N; v++) {
        auto d = vector<float>(Q, 0.0);
        for (int q = 0; q < Q; q++) {
            d[q] = static_cast<float>((*distances_ptr)[v][new_query[q]]);
        }
        if (is_center) {
            // get max of d
            values[v] = *std::max_element(d.begin(), d.end());
        } else {
            // get average
            values[v] = static_cast<float>(std::accumulate(d.begin(), d.end(), 0.0)); // / Q);  avoid float
        }
    }

    auto outputs = vector<int>();
    auto min_value = *std::min_element(values.begin(), values.end());
    for (int i = 0; i < N; i++) {
        if (float_equality(values[i], min_value)) {
            outputs.push_back(i);
        }
    }
    if (outputs.empty()) {
        // this will be due to float equality and means function needs to be reimplemented
        auto s = "Center/centroid error no outputs found for min value " + std::to_string(min_value);
        throw std::invalid_argument(s);
    }
    return make_pair(new_query, outputs);
}

inline pair<vector<int>, vector<int> > sample_center_from_graph(const unique_ptr<vector<vector<int> > > &distances_ptr,
                                                                vector<int> &given_query, int max_query_size = -1,
                                                                const int min_query_size = 2) {
    return sample_center_centroid(distances_ptr, given_query, max_query_size, min_query_size, true);
}

inline pair<vector<int>, vector<int> > sample_centroid_from_graph(
    const unique_ptr<vector<vector<int> > > &distances_ptr,
    vector<int> &given_query, int max_query_size = -1, const int min_query_size = 2) {
    return sample_center_centroid(distances_ptr, given_query, max_query_size, min_query_size, false);
}


template<typename T>
void non_causal_ground_truths(unique_ptr<vector<vector<T> > > &distance,
                              unique_ptr<vector<vector<T> > > &ground_truths_ptr,
                              vector<pair<int, int> > &edge_list) {
    // Makes a [E, N] matrix of ground truths where each row is the distance from the edge.first to all other nodes
    auto N = distance->size();
    auto E = edge_list.size();
    ground_truths_ptr = make_unique<vector<vector<int> > >(E, vector<int>(N, -1));
    for (int t = 0; t < static_cast<int>(E); t++) {
        for (int i = 0; i < static_cast<int>(N); i++) {
            (*ground_truths_ptr)[t][i] = (*distance)[edge_list[t].first][i];
        }
    }
}

template<typename D>
py::dict package_for_python(unique_ptr<Graph<D> > &g_ptr,
                            const int max_path_length = 10, const int min_path_length = 1, int start = -1, int end = -1,
                            int max_query_size = -1, const int min_query_size = 2, const bool is_center = true,
                            const bool is_causal = false,
                            const bool shuffle_edges = false,
                            const bool shuffle_nodes = false, const int min_vocab = 0, int max_vocab = -1) {
    const auto N = num_vertices(*g_ptr);
    const auto E = num_edges(*g_ptr);

    unique_ptr<DistanceMatrix<boost::undirectedS> > distances_ptr;
    // floyd_warshall(g_ptr, distances_ptr, false);  // much slower for sparse graphs
    johnson<D>(g_ptr, distances_ptr, false);

    auto edge_shuffle_map = get_edge_shuffle_map(E, shuffle_edges); // just range if no shuffle
    auto edge_list = get_edge_list(g_ptr, edge_shuffle_map);

    unique_ptr<vector<vector<int> > > distances_ptr2;
    unique_ptr<vector<vector<int> > > ground_truths_ptr;
    if (is_causal) {
        floyd_warshall_frydenlund(g_ptr, distances_ptr2, ground_truths_ptr, edge_list, false);
    } else {
        convert_boost_matrix(distances_ptr, distances_ptr2, N, N);
        non_causal_ground_truths(distances_ptr2, ground_truths_ptr, edge_list);
    }

    auto path = sample_path(distances_ptr2, max_path_length, min_path_length, start, end);
    auto blank_query = vector<int>();
    auto center = sample_center_centroid(distances_ptr2, blank_query, max_query_size, min_query_size, is_center);
    // print center
    auto center_query = center.first;
    auto center_center = center.second;
    for (auto i: center_query) {
        cout << i << " ";
    }
    cout << endl;
    for (auto i: center_center) {
        cout << i << " ";
    }
    cout << endl;

    auto node_shuffle_map = get_node_shuffle_map(N, min_vocab, max_vocab, shuffle_nodes);
    auto new_N = *max_element(node_shuffle_map.begin(), node_shuffle_map.end()) + 1;

    auto original_distances = convert_distance_matrix<int, DistanceMatrix<boost::undirectedS> >(
        distances_ptr, node_shuffle_map, N, new_N);
    auto distances = convert_distance_matrix<int, vector<vector<int> > >(distances_ptr2, node_shuffle_map, N, new_N);
    auto ground_truths = convert_ground_truths<int, vector<vector<int> > >(
        ground_truths_ptr, node_shuffle_map, E, N, new_N);


    py::dict d;
    d["edge_list"] = convert_edge_list(edge_list, node_shuffle_map);
    d["original_distances"] = original_distances;
    d["distances"] = distances;
    d["ground_truths"] = ground_truths;
    d["path"] = convert_path(path, node_shuffle_map);
    auto c_q = convert_center(center, node_shuffle_map);
    d["center_query"] = c_q.first;
    d["center_center"] = c_q.second;
    d["is_center"] = is_center;
    d["node_map"] = convert_vector(node_shuffle_map);
    return d;
}


inline py::dict erdos_renyi(const int num_nodes, float p = -1.0, const int c_min = 75, const int c_max = 125,
                            const int max_path_length = 10, const int min_path_length = 1,
                            int max_query_size = -1, const int min_query_size = 2, const bool is_center = true,
                            const bool is_causal = false, const bool shuffle_edges = false,
                            const bool shuffle_nodes = false, const int min_vocab = 0, int max_vocab = -1) {
    if (p >= 1.0) { throw::invalid_argument("p >= 1.0"); } //  boost fails at p = 1.0, way to go boost
    unique_ptr<Graph<boost::undirectedS> > g_ptr;
    erdos_renyi_generator(g_ptr, num_nodes, gen, p, c_min, c_max, false);
    return package_for_python(g_ptr, max_path_length, min_path_length, -1, -1,
                              max_query_size, min_query_size, is_center,
                              is_causal, shuffle_edges, shuffle_nodes, min_vocab, max_vocab);
}

inline py::dict euclidean(const int num_nodes, const int dim = 2, float radius = -1.0, const int c_min = 75,
                          const int c_max = 125,
                          const int max_path_length = 10, const int min_path_length = 1,
                          int max_query_size = -1, const int min_query_size = 2, const bool is_center = true,
                          const bool is_causal = false, const bool shuffle_edges = false,
                          const bool shuffle_nodes = false, const int min_vocab = 0, int max_vocab = -1) {
    unique_ptr<Graph<boost::undirectedS> > g_ptr;
    unique_ptr<vector<vector<float> > > positions_ptr;
    euclidean_generator(g_ptr, positions_ptr, num_nodes, gen, dim, radius, c_min, c_max, false);
    auto d = package_for_python(g_ptr, max_path_length, min_path_length, -1, -1,
                                max_query_size, min_query_size, is_center,
                                is_causal, shuffle_edges, shuffle_nodes, min_vocab, max_vocab);
    // Only return valid positions (otherwise nodes have been mapped and I don't want to convert positions)
    // Positions are only for plotting, when mapping should not be done.
    // This may not be true if I ever use positions as neural net features
    if (!shuffle_nodes && max_vocab == -1) {
        auto N = num_vertices(*g_ptr);
        constexpr size_t M = 2;
        auto positions = py::array_t<float, py::array::c_style>({N, M});
        auto ra = positions.mutable_unchecked();
        for (int i = 0; i < static_cast<int>(N); i++) {
            for (int j = 0; j < 2; j++) {
                constexpr float r = 10000;
                ra(i, j) = ceil((*positions_ptr)[i][j] * r) / r;;
            }
        }
        d["positions"] = positions;
        // d["positions_reminder"] = "Remember that these are not the mapped positions, but the original positions";
        // TODO fix
    }
    return d;
}

inline py::dict path_star(const int min_num_arms, const int max_num_arms, const int min_arm_length,
                          const int max_arm_length,
                          int max_query_size = -1, const int min_query_size = 2, const bool is_center = true,
                          const bool is_causal = false, const bool shuffle_edges = false,
                          const bool shuffle_nodes = false, const int min_vocab = 0, int max_vocab = -1) {
    unique_ptr<Graph<boost::directedS> > g_ptr;
    auto start_end = path_star_generator(g_ptr, min_num_arms, max_num_arms, min_arm_length, max_arm_length, gen, false);
    auto d = package_for_python(g_ptr, -1, -1, start_end.first, start_end.second,
                                max_query_size, min_query_size, is_center,
                                is_causal, shuffle_edges, shuffle_nodes, min_vocab, max_vocab);
    d["start"] = start_end.first;
    d["end"] = start_end.second;
    return d;
}

inline py::dict balanced(const int num_nodes, int lookahead, const int min_noise_reserve = 0,
                         const int max_num_parents = 4, int max_noise = -1,
                         int max_query_size = -1, const int min_query_size = 2, const bool is_center = true,
                         const bool is_causal = false, const bool shuffle_edges = false,
                         const bool shuffle_nodes = false, const int min_vocab = 0, int max_vocab = -1) {
    unique_ptr<Graph<boost::directedS> > g_ptr;
    auto start_end = balanced_generator(g_ptr, num_nodes, gen, lookahead, min_noise_reserve, max_num_parents, false);

    auto d = package_for_python(g_ptr, -1, -1, start_end.first, start_end.second,
                                max_query_size, min_query_size, is_center,
                                is_causal, shuffle_edges, shuffle_nodes, min_vocab, max_vocab);
    d["start"] = start_end.first;
    d["end"] = start_end.second;
    return d;
}


/* ************************************************
 *  Constructing inputs and targets for model
 *  Batched graph generation
 *  ***********************************************/


template<typename D>
void push_back_data(unique_ptr<Graph<D> > &g_ptr,
                    vector<int> &edge_shuffle_map,
                    list<unique_ptr<vector<pair<int, int> > > > &batched_edge_list,
                    list<int> &batched_edge_list_lengths,
                    list<unique_ptr<vector<vector<int> > > > &batched_distances,
                    list<unique_ptr<vector<vector<int> > > > &batched_ground_truths,
                    const bool is_causal,
                    const string &task_type,
                    list<unique_ptr<vector<int> > > &batched_paths,
                    list<int> &batched_path_lengths,
                    const int max_path_length, const int min_path_length, int start, int end,
                    list<unique_ptr<pair<vector<int>, vector<int> > > > &batched_centers,
                    list<pair<int, int> > &batched_center_lengths,
                    int max_query_size, const int min_query_size
) {
    const auto E = num_edges(*g_ptr);
    auto edge_list = get_edge_list<D>(g_ptr, edge_shuffle_map);
    batched_edge_list.push_back(make_unique<vector<pair<int, int> > >(edge_list));
    batched_edge_list_lengths.push_back(E);
    unique_ptr<vector<vector<int> > > distances_ptr;
    unique_ptr<vector<vector<int> > > ground_truths_ptr;
    if (is_causal) {
        // auto path_d = time_before();
        floyd_warshall_frydenlund(g_ptr, distances_ptr, ground_truths_ptr, edge_list, false);
        // time_after(path_d, "floyd_warshall_frydenlund");
        if (task_type == "shortest_path" || task_type == "path") {
            // needs to be here because of unique_ptr scope
            // auto path_t = time_before();
            auto path = sample_path(distances_ptr, max_path_length, min_path_length, start, end);
            // time_after(path_t, "sample_path");
            batched_path_lengths.push_back(path.size());
            batched_paths.push_back(make_unique<vector<int> >(path));
        } else if (task_type == "center" || task_type == "centroid") {  // can only be one of these
            auto given_query = vector<int>();
            // auto center_t = time_before();
            auto q_c_pair = sample_center_centroid(distances_ptr, given_query, max_query_size, min_query_size,
                                                   task_type == "center");
            // time_after(center_t, "sample_center");
            batched_center_lengths.push_back(make_pair(q_c_pair.first.size(), q_c_pair.second.size()));
            batched_centers.push_back(make_unique<pair<vector<int>, vector<int> > >(q_c_pair));
        }
        batched_distances.push_back(move(distances_ptr));
        batched_ground_truths.push_back(move(ground_truths_ptr));
    } else {
        auto N = num_vertices(*g_ptr);
        unique_ptr<DistanceMatrix<D> > boost_distances_ptr;
        // auto path_d = time_before();
        // floyd_warshall<D>(g_ptr, distances_ptr, false);  // much slower for sparse graphs
        johnson<D>(g_ptr, boost_distances_ptr, false);
        // time_after(path_d, "floyd_warshall");
        convert_boost_matrix<int, DistanceMatrix<D> >(boost_distances_ptr, distances_ptr, N, N);
        non_causal_ground_truths(distances_ptr, ground_truths_ptr, edge_list);
        if (task_type == "shortest_path" || task_type == "path") {
            // auto path_t = time_before();
            auto path = sample_path(distances_ptr, max_path_length, min_path_length, start, end);
            // time_after(path_t, "sample_path");
            batched_path_lengths.push_back(path.size());
            batched_paths.push_back(make_unique<vector<int> >(path));
        } else if (task_type == "center" || task_type == "centroid") {  // can only be one of these
            auto given_query = vector<int>();
            // auto center_t = time_before();
            auto q_c_pair = sample_center_centroid(distances_ptr, given_query, max_query_size, min_query_size,
                                                   task_type == "center");
            // time_after(center_t, "sample_center");
            batched_center_lengths.push_back(make_pair(q_c_pair.first.size(), q_c_pair.second.size()));
            batched_centers.push_back(make_unique<pair<vector<int>, vector<int> > >(q_c_pair));
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
    const int min_num_nodes, int max_num_nodes, float p = -1.0, const int c_min = 75, const int c_max = 125,
    const string &task_type = "shortest_path",
    const int max_path_length = 10, const int min_path_length = 1,
    int max_query_size = -1, const int min_query_size = 2,
    const bool is_causal = false, const bool shuffle_edges = false,
    const bool shuffle_nodes = false, const int min_vocab = 0, int max_vocab = -1,
    const int batch_size = 256, const int max_edges = 512, int max_attempts = 1000,
    const bool concat_edges = true,
    const bool query_at_end = true,
    const int num_thinking_tokens = 0,
    const bool is_flat_model = true,
    const bool for_plotting = false,
    const py::kwargs& kwargs = py::kwargs()) {
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
    if (p > 1.0) { throw std::invalid_argument("Invalid arguments: p > 1.0"); }
    if (c_min > c_max) { throw std::invalid_argument("Invalid arguments: c_min > c_max"); }
    if (batch_size <= 0) { throw std::invalid_argument("Invalid arguments: batch_size <= 0"); }
    if (min_path_length > max_path_length) { throw std::invalid_argument("Invalid arguments: min_path_length > max_path_length"); }
    if (num_thinking_tokens < 0) {
        throw std::invalid_argument("Invalid arguments: num_thinking_tokens < 0");
    }
    if (!for_plotting && dictionary.empty()) {
        throw std::invalid_argument("Invalid arguments: dictionary is empty.  Please set it first.");
    }

    auto batched_node_shuffle_map = list<unique_ptr<vector<int> > >();
    auto batched_edge_list = list<unique_ptr<vector<pair<int, int> > > >();
    auto batched_edge_list_lengths = list<int>();
    auto batched_distances = list<unique_ptr<vector<vector<int> > > >();
    auto batched_ground_truths = list<unique_ptr<vector<vector<int> > > >();
    auto batched_paths = list<unique_ptr<vector<int> > >();
    auto batched_path_lengths = list<int>();
    auto batched_centers = list<unique_ptr<pair<vector<int>, vector<int> > > >();
    auto batched_center_lengths = list<pair<int, int> >();

    int attempts = 0;
    int num = 0;
    while (num < batch_size && attempts < max_attempts) {
        int num_nodes;
        if (max_num_nodes == min_num_nodes) {
            num_nodes = max_num_nodes;
        } else {
            uniform_int_distribution<int> d(min_num_nodes, max_num_nodes);
            num_nodes = d(gen);
        }
        unique_ptr<Graph<boost::undirectedS> > g_ptr;
        // auto graph_t = time_before();
        erdos_renyi_generator(g_ptr, num_nodes, gen, p, c_min, c_max, false);
        // time_after(graph_t, "graph gen");
        const auto N = num_vertices(*g_ptr);
        const auto E = num_edges(*g_ptr);
        if (const auto a = attempt_check(E, max_edges, attempts, max_attempts)) {
            attempts += a;
            continue;
        }
        auto edge_shuffle_map = get_edge_shuffle_map(E, shuffle_edges);
        auto node_shuffle_map = get_node_shuffle_map(N, min_vocab, max_vocab, shuffle_nodes);
        batched_node_shuffle_map.push_back(make_unique<vector<int> >(node_shuffle_map));
        // auto pack_t = time_before();
        push_back_data<boost::undirectedS>(g_ptr, edge_shuffle_map,
                                           batched_edge_list, batched_edge_list_lengths,
                                           batched_distances, batched_ground_truths,
                                           is_causal, task_type,
                                           batched_paths, batched_path_lengths,
                                           max_path_length, min_path_length, -1, -1,
                                           batched_centers, batched_center_lengths,
                                           max_query_size, min_query_size
        );
        // time_after(pack_t, "pack");
        num += 1;
    }

    if (for_plotting) {
        return package_for_plotting("erdos_renyi", task_type,
                                    attempts, max_attempts,
                                    min_vocab, max_vocab,
                                    batched_node_shuffle_map,
                                    batched_edge_list,
                                    batched_edge_list_lengths,
                                    batched_distances,
                                    batched_ground_truths,
                                    batched_paths,
                                    batched_path_lengths
        );
    }
    return package_for_model("erdos_renyi", task_type,
                             attempts, max_attempts,
                             min_vocab, max_vocab, dictionary,
                             batched_node_shuffle_map,
                             batched_edge_list,
                             batched_edge_list_lengths,
                             batched_distances,
                             batched_ground_truths,
                             batched_paths,
                             batched_path_lengths,
                             batched_centers,
                             batched_center_lengths,
                             concat_edges, query_at_end, num_thinking_tokens,
                             is_flat_model
    );
}


inline py::dict euclidean_n(
    const int min_num_nodes, int max_num_nodes, const int dim = 2, float radius = -1.0,
    const int c_min = 75, const int c_max = 125,
    const string &task_type = "shortest_path",
    const int max_path_length = 10, const int min_path_length = 1,
    int max_query_size = -1, const int min_query_size = 2,
    const bool is_causal = false, const bool shuffle_edges = false,
    const bool shuffle_nodes = false, const int min_vocab = 0, int max_vocab = -1,
    const int batch_size = 256, const int max_edges = 512, int max_attempts = 1000,
    const bool concat_edges = true,
    const bool query_at_end = true,
    const int num_thinking_tokens = 0,
    const bool is_flat_model = true,
    const bool for_plotting = false,
    const py::kwargs& kwargs = py::kwargs()) {
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
    if (dim < 0) { throw std::invalid_argument("Invalid arguments: dim < 0"); }
    if (c_min > c_max) { throw std::invalid_argument("Invalid arguments: c_min > c_max"); }
    if (batch_size <= 0) { throw std::invalid_argument("Invalid arguments: batch_size <= 0"); }
    if (min_path_length > max_path_length) { throw std::invalid_argument("Invalid arguments: min_path_length > max_path_length"); }
    if (num_thinking_tokens < 0) {
        throw std::invalid_argument("Invalid arguments: num_thinking_tokens < 0");
    }
    if (!for_plotting && dictionary.empty()) {
        throw std::invalid_argument("Invalid arguments: dictionary is empty.  Please set it first.");
    }

    auto batched_node_shuffle_map = list<unique_ptr<vector<int> > >();
    auto batched_edge_list = list<unique_ptr<vector<pair<int, int> > > >();
    auto batched_edge_list_lengths = list<int>();
    auto batched_distances = list<unique_ptr<vector<vector<int> > > >();
    auto batched_ground_truths = list<unique_ptr<vector<vector<int> > > >();
    auto batched_paths = list<unique_ptr<vector<int> > >();
    auto batched_path_lengths = list<int>();
    auto batched_centers = list<unique_ptr<pair<vector<int>, vector<int> > > >();
    auto batched_center_lengths = list<pair<int, int> >();
    // this one is different from the rest
    auto batched_positions = list<unique_ptr<vector<vector<float> > > >();
    // [batch_size, dim + 1] of node_id, x, y etc.

    int attempts = 0;
    int num = 0;
    while (num < batch_size && attempts < max_attempts) {
        int num_nodes;
        if (max_num_nodes == min_num_nodes) {
            num_nodes = max_num_nodes;
        } else {
            uniform_int_distribution<int> d(min_num_nodes, max_num_nodes);
            num_nodes = d(gen);
        }
        unique_ptr<Graph<boost::undirectedS> > g_ptr;
        unique_ptr<vector<vector<float> > > positions_ptr;
        // auto graph_t = time_before();
        euclidean_generator(g_ptr, positions_ptr, num_nodes, gen, dim, radius, c_min, c_max, false);
        batched_positions.push_back(move(positions_ptr));
        // time_after(graph_t, "graph gen");
        const auto N = num_vertices(*g_ptr);
        const auto E = num_edges(*g_ptr);
        if (const auto a = attempt_check(E, max_edges, attempts, max_attempts)) {
            attempts += a;
            continue;
        }
        auto edge_shuffle_map = get_edge_shuffle_map(E, shuffle_edges);
        auto node_shuffle_map = get_node_shuffle_map(N, min_vocab, max_vocab, shuffle_nodes);
        batched_node_shuffle_map.push_back(make_unique<vector<int> >(node_shuffle_map));
        // auto pack_t = time_before();

        push_back_data<boost::undirectedS>(g_ptr, edge_shuffle_map,
                                           batched_edge_list, batched_edge_list_lengths,
                                           batched_distances, batched_ground_truths,
                                           is_causal, task_type,
                                           batched_paths, batched_path_lengths,
                                           max_path_length, min_path_length, -1, -1,
                                           batched_centers, batched_center_lengths,
                                           max_query_size, min_query_size
        );
        // time_after(pack_t, "pack");
        num += 1;
    }

    if (for_plotting) {
        auto d = package_for_plotting("euclidean", task_type,
                                      attempts, max_attempts, min_vocab, max_vocab,
                                      batched_node_shuffle_map,
                                      batched_edge_list,
                                      batched_edge_list_lengths,
                                      batched_distances,
                                      batched_ground_truths,
                                      batched_paths,
                                      batched_path_lengths
        );
        d["positions"] = batch_positions<float>(batched_positions, batched_node_shuffle_map, dim);
        return d;
    }
    auto d = package_for_model("euclidean", task_type,
                               attempts, max_attempts,
                               min_vocab, max_vocab, dictionary,
                               batched_node_shuffle_map,
                               batched_edge_list,
                               batched_edge_list_lengths,
                               batched_distances,
                               batched_ground_truths,
                               batched_paths,
                               batched_path_lengths,
                               batched_centers,
                               batched_center_lengths,
                               concat_edges, query_at_end, num_thinking_tokens,
                               is_flat_model);
    d["positions"] = batch_positions<float>(batched_positions, batched_node_shuffle_map, dim);
    return d;
}


inline py::dict path_star_n(
    const int min_num_arms, const int max_num_arms, const int min_arm_length, const int max_arm_length,
    const string &task_type = "shortest_path",
    int max_query_size = -1, const int min_query_size = 2,
    const bool is_causal = false, const bool shuffle_edges = false,
    const bool shuffle_nodes = false, const int min_vocab = 0, int max_vocab = -1,
    const int batch_size = 256, const int max_edges = 512, int max_attempts = 1000,
    const bool concat_edges = true,
    const bool query_at_end = true,
    const int num_thinking_tokens = 0,
    const bool is_flat_model = true,
    const bool for_plotting = false,
    const py::kwargs& kwargs = py::kwargs()) {
    if (min_num_arms <= 0) { throw std::invalid_argument("Invalid arguments: min_num_arms <= 0"); }
    if (min_arm_length <= 0) { throw std::invalid_argument("Invalid arguments: min_arm_length <= 0"); }
    if (batch_size <= 0) { throw std::invalid_argument("Invalid arguments: batch_size <= 0"); }
    auto batched_node_shuffle_map = list<unique_ptr<vector<int> > >();
    auto batched_edge_list = list<unique_ptr<vector<pair<int, int> > > >();
    auto batched_edge_list_lengths = list<int>();
    auto batched_distances = list<unique_ptr<vector<vector<int> > > >();
    auto batched_ground_truths = list<unique_ptr<vector<vector<int> > > >();
    auto batched_paths = list<unique_ptr<vector<int> > >();
    auto batched_path_lengths = list<int>();
    auto batched_centers = list<unique_ptr<pair<vector<int>, vector<int> > > >();
    auto batched_center_lengths = list<pair<int, int> >();

    int attempts = 0;
    int num = 0;
    while (num < batch_size && attempts < max_attempts) {
        unique_ptr<Graph<boost::directedS> > g_ptr;
        // auto graph_t = time_before();
        auto start_end = path_star_generator(g_ptr, min_num_arms, max_num_arms, min_arm_length, max_arm_length, gen,
                                             false);
        // time_after(graph_t, "graph gen");
        const auto N = num_vertices(*g_ptr);
        const auto E = num_edges(*g_ptr);
        if (const auto a = attempt_check(E, max_edges, attempts, max_attempts)) {
            attempts += a;
            continue;
        }
        auto edge_shuffle_map = get_edge_shuffle_map(E, shuffle_edges);
        auto node_shuffle_map = get_node_shuffle_map(N, min_vocab, max_vocab, shuffle_nodes);
        batched_node_shuffle_map.push_back(make_unique<vector<int> >(node_shuffle_map));
        // auto pack_t = time_before();
        push_back_data<boost::directedS>(g_ptr, edge_shuffle_map,
                                         batched_edge_list, batched_edge_list_lengths,
                                         batched_distances, batched_ground_truths,
                                         is_causal, task_type,
                                         batched_paths, batched_path_lengths,  -1, -1,
                                         start_end.first, start_end.second,
                                         batched_centers, batched_center_lengths,
                                         max_query_size, min_query_size
        );
        // time_after(pack_t, "pack");
        num += 1;
    }

    if (for_plotting) {
        return package_for_plotting("path_star", task_type,
                                    attempts, max_attempts, min_vocab, max_vocab,
                                    batched_node_shuffle_map,
                                    batched_edge_list,
                                    batched_edge_list_lengths,
                                    batched_distances,
                                    batched_ground_truths,
                                    batched_paths,
                                    batched_path_lengths
        );
    }
    auto d = package_for_model("path_star", task_type,
                               attempts, max_attempts,
                               min_vocab, max_vocab, dictionary,
                               batched_node_shuffle_map,
                               batched_edge_list,
                               batched_edge_list_lengths,
                               batched_distances,
                               batched_ground_truths,
                               batched_paths,
                               batched_path_lengths,
                               batched_centers,
                               batched_center_lengths,
                               concat_edges, query_at_end, num_thinking_tokens,
                               is_flat_model);
    return d;
}


inline py::dict balanced_n(
    const int min_num_nodes, int max_num_nodes, const int min_lookahead, const int max_lookahead,
    const int min_noise_reserve = 0, const int max_num_parents = 4, int max_noise = -1,
    const string &task_type = "shortest_path",
    int max_query_size = -1, const int min_query_size = 2,
    const bool is_causal = false, const bool shuffle_edges = false,
    const bool shuffle_nodes = false, const int min_vocab = 0, int max_vocab = -1,
    const int batch_size = 256, const int max_edges = 512, int max_attempts = 1000,
    const bool concat_edges = true,
    const bool query_at_end = true,
    const int num_thinking_tokens = 0,
    const bool is_flat_model = true,
    const bool for_plotting = false,
    const py::kwargs& kwargs = py::kwargs()) {
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
    if (min_lookahead <= 0) { throw std::invalid_argument("Invalid arguments: min_lookahead <= 0"); }
    if (max_lookahead <= 0) { throw std::invalid_argument("Invalid arguments: max_lookahead <= 0"); }
    if (max_num_parents <= 0) { throw std::invalid_argument("Invalid arguments: max_num_parents <= 0"); }
    if (!balanced_graph_size_check(min_num_nodes, max_lookahead, min_noise_reserve)) {
        throw std::invalid_argument("Invalid arguments: balanced_graph_size_check failed");
    }
    if (batch_size <= 0) { throw std::invalid_argument("Invalid arguments: batch_size <= 0"); }

    auto batched_node_shuffle_map = list<unique_ptr<vector<int> > >();
    auto batched_edge_list = list<unique_ptr<vector<pair<int, int> > > >();
    auto batched_edge_list_lengths = list<int>();
    auto batched_distances = list<unique_ptr<vector<vector<int> > > >();
    auto batched_ground_truths = list<unique_ptr<vector<vector<int> > > >();
    auto batched_paths = list<unique_ptr<vector<int> > >();
    auto batched_path_lengths = list<int>();
    auto batched_centers = list<unique_ptr<pair<vector<int>, vector<int> > > >();
    auto batched_center_lengths = list<pair<int, int> >();

    int attempts = 0;
    int num = 0;

    while (num < batch_size && attempts < max_attempts) {
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

        // time_after(graph_t, "graph gen");
        const auto N = num_vertices(*g_ptr);
        const auto E = num_edges(*g_ptr);
        if (const auto a = attempt_check(E, max_edges, attempts, max_attempts)) {
            attempts += a;
            continue;
        }
        auto edge_shuffle_map = get_edge_shuffle_map(E, shuffle_edges);
        auto node_shuffle_map = get_node_shuffle_map(N, min_vocab, max_vocab, shuffle_nodes);
        batched_node_shuffle_map.push_back(make_unique<vector<int> >(node_shuffle_map));
        // auto pack_t = time_before();
        push_back_data<boost::directedS>(g_ptr, edge_shuffle_map,
                                         batched_edge_list, batched_edge_list_lengths,
                                         batched_distances, batched_ground_truths,
                                         is_causal, task_type,
                                         batched_paths, batched_path_lengths, -1, -1,
                                         start_end.first, start_end.second,
                                         batched_centers, batched_center_lengths,
                                         max_query_size, min_query_size
        );
        // time_after(pack_t, "pack");
        num += 1;
    }

    if (for_plotting) {
        return package_for_plotting("balanced", task_type,
                                    attempts, max_attempts, min_vocab, max_vocab,
                                    batched_node_shuffle_map,
                                    batched_edge_list,
                                    batched_edge_list_lengths,
                                    batched_distances,
                                    batched_ground_truths,
                                    batched_paths,
                                    batched_path_lengths
        );
    }
    auto d = package_for_model("balanced", task_type,
                               attempts, max_attempts,
                               min_vocab, max_vocab, dictionary,
                               batched_node_shuffle_map,
                               batched_edge_list,
                               batched_edge_list_lengths,
                               batched_distances,
                               batched_ground_truths,
                               batched_paths,
                               batched_path_lengths,
                               batched_centers,
                               batched_center_lengths,
                               concat_edges, query_at_end, num_thinking_tokens,
                               is_flat_model);
    return d;
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
          py::arg("max_vocab") = 100);

    m.def("get_dictionary", &get_dictionary,
          "Gets the dictionary/vocabulary of token to token_idx.\n"
          "Parameters:\n\t"
          "None\n"
          "Returns:\n\t"
          "dictionary: of str -> int\n");

    m.def("verify_paths", &verify_paths<int>,
          "Batch varies the that any predicted paths are valid given the distance matrices.\n"
          "Parameters:\n\t"
          "distances: [batch_size, vocab_size, vocab_size]\n\t"
          "paths: [batch_size, max_path_length]\n\t"
          "path_lengths: [batch_size]\n\t"
          "Returns:\n\t"
          "is_valid [batch_size], int, -1 if not valid, 0 if valid but not shortest, 1 if valid and shortest.\n",
          py::arg("distances"), py::arg("paths"), py::arg("path_lengths"));

    // single graph generation
    m.def("erdos_renyi", &erdos_renyi,
          "Generate a single Erdos Renyi graph\nParameters:\n\t"
          "num_nodes: number of nodes.\n\t"
          "p: probability of edge creation.  If -1 then 1/num_nodes.\n\t"
          "c_min: min number of sampled edges to form a single connected component\n\t"
          "c_max: max number of of sampled edges to form a single connected component\n\t"
          "max_path_length: max length of path to sample\n\t"
          "min_path_length: min length of path to sample\n\t"
          "is_causal: if true then return causally masked ground_truths\n\t"
          "shuffle_edges: if true then shuffle edges\n\t"
          "shuffle_nodes: if true then shuffle nodes\n\t"
          "min_vocab: min vocab size to map nodes into i.e. to exclude special tokens\n\t"
          "max_vocab: max vocab size to map nodes into.\n"
          "Returns a dict with the following keys (N is the max_vocab):\n\t"
          "edge_list: numpy [E, 2] of edges\n\t"
          "original_distances: numpy [N, N] of original distances (boost calc)\n\t"
          "distances: numpy [N, N] of distances (my calc, for sanity checking)\n\t"
          "ground_truths: numpy [E, N] of ground truths\n\t"
          "path: numpy [L] of path\n\t"
          "node_map: numpy [N] of node map\n\t"
          "hashes: numpy [N] of uint64_t hash of distances\n\t",

          py::arg("num_nodes"),
          py::arg("p") = -1.0, py::arg("c_min") = 75, py::arg("c_max") = 125,
          py::arg("max_path_length") = 10, py::arg("min_path_length") = 3,
          py::arg("max_query_size") = -1, py::arg("min_query_size") = 2, py::arg("is_center") = true,
          py::arg("is_causal") = false, py::arg("shuffle_edges") = false,
          py::arg("shuffle_nodes") = false, py::arg("min_vocab") = 0, py::arg("max_vocab") = -1);

    m.def("euclidean", &euclidean,
          "Generate a single euclidean graph.\nParameters:\n\t"
          "num_nodes: number of nodes.\n\t"
          "dims: number of dimensions\n\t"
          "radius: radius of graph.  If -1 then 1/sqrt(num_nodes).\n\t"
          "c_min: min number of sampled edges to form a single connected component\n\t"
          "c_max: max number of sampled edges to form a single connected component\n\t"
          "max_path_length: max length of path to sample\n\t"
          "min_path_length: min length of path to sample\n\t"
          "is_causal: if true then return causally masked ground_truths\n\t"
          "shuffle_edges: if true then shuffle edges\n\t"
          "shuffle_nodes: if true then shuffle nodes\n\t"
          "min_vocab: min vocab size to map nodes into i.e. to exclude special tokens\n\t"
          "max_vocab: max vocab size to map nodes into.\n"
          "Returns a dict with the following keys (N is the max_vocab):\n\t"
          "edge_list: numpy [E, 2] of edges\n\t"
          "original_distances: numpy [N, N] of original distances (boost calc)\n\t"
          "distances: numpy [N, N] of distances (my calc, for sanity checking)\n\t"
          "ground_truths: numpy [E, N] of ground truths\n\t"
          "path: numpy [L] of path\n\t"
          "node_map: numpy [N] of node map\n\t"
          "hashes: numpy [N] of uint64_t hash of distances\n\t"
          "positions: numpy [N, dims] of node positions.  Note: node positions are not returned if using node shuffle or vocab range (and these are needed for plotting).\n\t",

          py::arg("num_nodes"),
          py::arg("dims") = 2, py::arg("radius") = -1.0, py::arg("c_min") = 75, py::arg("c_max") = 125,
          py::arg("max_path_length") = 10, py::arg("min_path_length") = 3,
          py::arg("max_query_size") = -1, py::arg("min_query_size") = 2, py::arg("is_center") = true,
          py::arg("is_causal") = false, py::arg("shuffle_edges") = false,
          py::arg("shuffle_nodes") = false, py::arg("min_vocab") = 0, py::arg("max_vocab") = -1);

    m.def("path_star", &path_star,
          "Generate a single path star graph\nParameters:\n\t"
          "min_num_arms: min number of arms.\n\t"
          "max_num_arms: max number of arms.\n\t"
          "min_arm_length: min arm length.\n\t"
          "max_arm_length: max arm length.\n\t"
          "is_causal: if true then return causally masked ground_truths\n\t"
          "shuffle_edges: if true then shuffle edges\n\t"
          "shuffle_nodes: if true then shuffle nodes\n\t"
          "min_vocab: min vocab size to map nodes into i.e. to exclude special tokens\n\t"
          "max_vocab: max vocab size to map nodes into.\n"
          "Returns a dict with the following keys (N is the max_vocab):\n\t"
          "edge_list: numpy [E, 2] of edges\n\t"
          "original_distances: numpy [N, N] of original distances (boost calc)\n\t"
          "distances: numpy [N, N] of distances (my calc, for sanity checking)\n\t"
          "ground_truths: numpy [E, N] of ground truths\n\t"
          "path: numpy [L] of path\n\t"
          "node_map: numpy [N] of node map\n\t"
          "hashes: numpy [N] of uint64_t hash of distances\n\t",

          py::arg("min_num_arms"), py::arg("max_num_arms"), py::arg("min_arm_length"), py::arg("max_arm_length"),
          py::arg("max_query_size") = -1, py::arg("min_query_size") = 2, py::arg("is_center") = true,
          py::arg("is_causal") = false, py::arg("shuffle_edges") = false,
          py::arg("shuffle_nodes") = false, py::arg("min_vocab") = 0, py::arg("max_vocab") = -1);

    m.def("balanced", &balanced,
          "Generate a single balanced graph.  Note these are done slightly differently from original paper.\nParameters:\n\t"
          "num_nodes: number of nodes.\n\t"
          "lookahead: number of lookahead nodes.\n\t"
          "min_noise_reserve: min noise reserve.\n\t"
          "max_num_parents: max number of parents.\n\t"
          "max_noise: max noise.\n\t"
          "is_causal: if true then return causally masked ground_truths\n\t"
          "shuffle_edges: if true then shuffle edges\n\t"
          "shuffle_nodes: if true then shuffle nodes\n\t"
          "min_vocab: min vocab size to map nodes into i.e. to exclude special tokens\n\t"
          "max_vocab: max vocab size to map nodes into.\n"
          "Returns a dict with the following keys (N is the max_vocab):\n\t"
          "edge_list: numpy [E, 2] of edges\n\t"
          "original_distances: numpy [N, N] of original distances (boost calc)\n\t"
          "distances: numpy [N, N] of distances (my calc, for sanity checking)\n\t"
          "ground_truths: numpy [E, N] of ground truths\n\t"
          "path: numpy [L] of path\n\t"
          "node_map: numpy [N] of node map\n\t"
          "hashes: numpy [N] of uint64_t hash of distances\n\t",

          py::arg("num_nodes"), py::arg("lookahead"), py::arg("min_noise_reserve") = 0, py::arg("max_num_parents") = 4,
          py::arg("max_query_size") = -1, py::arg("min_query_size") = 2, py::arg("is_center") = true,
          py::arg("max_noise") = -1,
          py::arg("is_causal") = false, py::arg("shuffle_edges") = false,
          py::arg("shuffle_nodes") = false, py::arg("min_vocab") = 0, py::arg("max_vocab") = -1);

    m.def("balanced_graph_size_check", &balanced_graph_size_check,
          "Check that the balanced graph size is valid.  Will fail assert otherwise.",
          py::arg("num_nodes"), py::arg("lookahead"), py::arg("min_noise_reserve") = 0);

    // batched graph generation
    m.def("erdos_renyi_n", &erdos_renyi_n,
          "Generate a batch of Erdos Renyi graphs\nParameters:\n\t"
          "min_num_nodes: min number of nodes. We strongly recommend using shuffle_nodes and a vocab range map.\n\t"
          "max_num_nodes: min number of nodes.  If -1 use min only.\n\t"
          "p: probability of edge creation.  If -1 then 1/num_nodes.\n\t"
          "c_min: min number of sampled edges to form a single connected component\n\t"
          "c_max: max number of sampled edges to form a single connected component\n\t"
          "task_type: type of task to sample.\n\t"
          "max_path_length: max length of path to sample\n\t"
          "min_path_length: min length of path to sample\n\t"
          "max_query_size: max number of query nodes to sample \n\t"
          "min_query_size: min number of query nodes to sample\n\t"
          "is_causal: if true then return causally masked ground_truths\n\t"
          "shuffle_edges: if true then shuffle edges\n\t"
          "shuffle_nodes: if true then shuffle nodes\n\t"
          "min_vocab: min vocab size to map nodes into i.e. to exclude special tokens\n\t"
          "max_vocab: max vocab size to map nodes into.\n"
          "Returns a dict with the following keys (N is the max_vocab):\n\t"
          "edge_list: numpy [B, E, 2] of edges\n\t"
          "edge_list_lengths: numpy [B] of edge list lengths\n\t"
          "original_distances: numpy [B, N, N] of original distances (boost calc)\n\t"
          "distances: numpy [B, N, N] of distances (my calc, for sanity checking)\n\t"
          "ground_truths: numpy [B, E, N] of ground truths\n\t"
          "paths: numpy [B, L] of paths\n\t"
          "path_lengths: numpy [B] of path lengths\n\t"
          "node_map: numpy [B, N] of node map\n\t"
          "hashes: numpy [B, N] of uint64_t hash of distances\n\t"
          "num_attempts: int of number of attempts to generate the graph\n\t"
          "vocab_min_size: int of min vocab size\n\t"
          "vocab_max_size: int of max vocab size\n\t",

          py::arg("min_num_nodes"),
          py::arg("max_num_nodes"),
          py::arg("p") = -1.0,
          py::arg("c_min") = 75,
          py::arg("c_max") = 125,
          py::arg("task_type") = "shortest_path",
          py::arg("max_path_length") = 10,
          py::arg("min_path_length") = 1,
          py::arg("max_query_size") = -1,
          py::arg("min_query_size") = 2,
          py::arg("is_causal") = false,
          py::arg("shuffle_edges") = false,
          py::arg("shuffle_nodes") = false,
          py::arg("min_vocab") = 0,
          py::arg("max_vocab") = -1,
          py::arg("batch_size") = 256,
          py::arg("max_edges") = 512,
          py::arg("max_attempts") = 1000,
          py::arg("concat_edges") = true,
          py::arg("query_at_end") = true,
          py::arg("num_thinking_tokens") = 0,
          py::arg("is_flat_model") = true,
          py::arg("for_plotting") = false);

    m.def("euclidean_n", &euclidean_n,
          "Generate a batch of euclidean graphs\nParameters:\n\t"
          "min_num_nodes: min number of nodes. We strongly recommend using shuffle_nodes and a vocab range map.\n\t"
          "max_num_nodes: min number of nodes.  If -1 use min only.\n\t"
          "dims: number of dimensions\n\t"
          "radius: radius of graph.  If -1 then 1/sqrt(num_nodes).\n\t"
          "c_min: min number of sampled edges to form a single connected component\n\t"
          "c_max: max number of sampled edges to form a single connected component\n\t"
          "task_type: type of task to sample.\n\t"
          "max_path_length: max length of path to sample\n\t"
          "min_path_length: min length of path to sample\n\t"
          "max_query_size: max number of query nodes to sample \n\t"
          "min_query_size: min number of query nodes to sample\n\t"
          "is_causal: if true then return causally masked ground_truths\n\t"
          "shuffle_edges: if true then shuffle edges\n\t"
          "shuffle_nodes: if true then shuffle nodes\n\t"
          "min_vocab: min vocab size to map nodes into i.e. to exclude special tokens\n\t"
          "max_vocab: max vocab size to map nodes into.\n"
          "Returns a dict with the following keys (N is the max_vocab):\n\t"
          "edge_list: numpy [B, E, 2] of edges\n\t"
          "edge_list_lengths: numpy [B] of edge list lengths\n\t"
          "original_distances: numpy [B, N, N] of original distances (boost calc)\n\t"
          "distances: numpy [B, N, N] of distances (my calc, for sanity checking)\n\t"
          "ground_truths: numpy [B, E, N] of ground truths\n\t"
          "paths: numpy [B, L] of paths\n\t"
          "path_lengths: numpy [B] of path lengths\n\t"
          "node_map: numpy [B, N] of node map\n\t"
          "hashes: numpy [B, N] of uint64_t hash of distances\n\t"
          "positions: numpy [B, N, dims] of node positions\n\t"
          "num_attempts: int of number of attempts to generate the graph\n\t"
          "vocab_min_size: int of min vocab size\n\t"
          "vocab_max_size: int of max vocab size\n\t",

          py::arg("min_num_nodes"),
          py::arg("max_num_nodes"),
          py::arg("dims") = 2,
          py::arg("radius") = -1.0,
          py::arg("c_min") = 75,
          py::arg("c_max") = 125,
          py::arg("task_type") = "shortest_path",
          py::arg("max_path_length") = 10,
          py::arg("min_path_length") = 1,
          py::arg("max_query_size") = -1,
          py::arg("min_query_size") = 2,
          py::arg("is_causal") = false,
          py::arg("shuffle_edges") = false,
          py::arg("shuffle_nodes") = false,
          py::arg("min_vocab") = 0,
          py::arg("max_vocab") = -1,
          py::arg("batch_size") = 256,
          py::arg("max_edges") = 512,
          py::arg("max_attempts") = 1000,
          py::arg("concat_edges") = true,
          py::arg("query_at_end") = true,
          py::arg("num_thinking_tokens") = 0,
          py::arg("is_flat_model") = true,
          py::arg("for_plotting") = false);

    m.def("path_star_n", &path_star_n,
          "Generate a batch of path star graphs\nParameters:\n\t"
          "min_num_arms: min number of arms.\n\t"
          "max_num_arms: max number of arms.\n\t"
          "min_arm_length: min arm length.\n\t"
          "max_arm_length: max arm length.\n\t"
          "task_type: type of task to sample.\n\t"
          "max_query_size: max number of query nodes to sample \n\t"
          "min_query_size: min number of query nodes to sample\n\t"
          "is_causal: if true then return causally masked ground_truths\n\t"
          "shuffle_edges: if true then shuffle edges\n\t"
          "shuffle_nodes: if true then shuffle nodes\n\t"
          "min_vocab: min vocab size to map nodes into i.e. to exclude special tokens\n\t"
          "max_vocab: max vocab size to map nodes into.\n"
          "Returns a dict with the following keys (N is the max_vocab):\n\t"
          "edge_list: numpy [B, E, 2] of edges\n\t"
          "edge_list_lengths: numpy [B] of edge list lengths\n\t"
          "original_distances: numpy [B, N, N] of original distances (boost calc)\n\t"
          "distances: numpy [B, N, N] of distances (my calc, for sanity checking)\n\t"
          "ground_truths: numpy [B, E, N] of ground truths\n\t"
          "paths: numpy [B, L] of paths\n\t"
          "path_lengths: numpy [B] of path lengths\n\t"
          "node_map: numpy [B, N] of node map\n\t"
          "hashes: numpy [B, N] of uint64_t hash of distances\n\t"
          "num_attempts: int of number of attempts to generate the graph\n\t"
          "vocab_min_size: int of min vocab size\n\t"
          "vocab_max_size: int of max vocab size\n\t",

          py::arg("min_num_arms"),
          py::arg("max_num_arms"),
          py::arg("min_arm_length"),
          py::arg("max_arm_length"),
          py::arg("task_type") = "shortest_path",
          py::arg("max_query_size") = -1,
          py::arg("min_query_size") = 2,
          py::arg("is_causal") = false,
          py::arg("shuffle_edges") = false,
          py::arg("shuffle_nodes") = false,
          py::arg("min_vocab") = 0,
          py::arg("max_vocab") = -1,
          py::arg("batch_size") = 256,
          py::arg("max_edges") = 512,
          py::arg("max_attempts") = 1000,
          py::arg("concat_edges") = true,
          py::arg("query_at_end") = true,
          py::arg("num_thinking_tokens") = 0,
          py::arg("is_flat_model") = true,
          py::arg("for_plotting") = false);

    m.def("balanced_n", &balanced_n,
          "Generate a batch of balanced graphs\nParameters:\n\t"
          "min_num_nodes: min number of nodes. We strongly recommend using shuffle_nodes and a vocab range map.\n\t"
          "max_num_nodes: min number of nodes.  If -1 use min only.\n\t"
          "min_lookahead: min number of lookahead nodes.\n\t"
          "max_lookahead: max number of lookahead nodes.\n\t"
          "min_noise_reserve: min noise reserve.\n\t"
          "max_num_parents: max number of parents.\n\t"
          "max_noise: max noise.\n\t"
          "task_type: type of task to sample.\n\t"
          "max_query_size: max number of query nodes to sample \n\t"
          "min_query_size: min number of query nodes to sample\n\t"
          "sample_target_paths: if true then sample target paths\n\t"
          "is_causal: if true then return causally masked ground_truths\n\t"
          "shuffle_edges: if true then shuffle edges\n\t"
          "shuffle_nodes: if true then shuffle nodes\n\t"
          "min_vocab: min vocab size to map nodes into i.e. to exclude special tokens\n\t"
          "max_vocab: max vocab size to map nodes into.\n"
          "Returns a dict with the following keys (N is the max_vocab):\n\t"
          "edge_list: numpy [B, E, 2] of edges\n\t"
          "edge_list_lengths: numpy [B] of edge list lengths\n\t"
          "original_distances: numpy [B, N, N] of original distances (boost calc)\n\t"
          "distances: numpy [B, N, N] of distances (my calc, for sanity checking)\n\t"
          "ground_truths: numpy [B, E, N] of ground truths\n\t"
          "paths: numpy [B, L] of paths\n\t"
          "path_lengths: numpy [B] of path lengths\n\t"
          "node_map: numpy [B, N] of node map\n\t"
          "hashes: numpy [B, N] of uint64_t hash of distances\n\t"
          "num_attempts: int of number of attempts to generate the graph\n\t"
          "vocab_min_size: int of min vocab size\n\t"
          "vocab_max_size: int of max vocab size\n\t",

          py::arg("min_num_nodes"),
          py::arg("max_num_nodes"),
          py::arg("min_lookahead"),
          py::arg("max_lookahead"),
          py::arg("min_noise_reserve") = 0,
          py::arg("max_num_parents") = 4,
          py::arg("max_noise") = -1,
          py::arg("task_type") = "shortest_path",
          py::arg("max_query_size") = -1,
          py::arg("min_query_size") = 2,
          py::arg("is_causal") = false,
          py::arg("shuffle_edges") = false,
          py::arg("shuffle_nodes") = false,
          py::arg("min_vocab") = 0,
          py::arg("max_vocab") = -1,
          py::arg("batch_size") = 256,
          py::arg("max_edges") = 512,
          py::arg("max_attempts") = 1000,
          py::arg("concat_edges") = true,
          py::arg("query_at_end") = true,
          py::arg("num_thinking_tokens") = 0,
          py::arg("is_flat_model") = true,
          py::arg("for_plotting") = false);
}
