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

#include "undirected_graphs.h"
#include "directed_graphs.h"

using namespace std;

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

namespace py = pybind11;
using namespace py::literals;

static const py::bool_ py_true(true);

inline unsigned int seed_ = std::random_device{}();
static thread_local auto gen = std::mt19937(seed_);  // so each thread in the dataloader is different

chrono::time_point<high_resolution_clock> time_before() {
    return high_resolution_clock::now();

}

void time_after(chrono::time_point<high_resolution_clock> t1, const string &msg = "") {
    auto t2 = high_resolution_clock::now();
    duration<double, std::milli> ms_double = t2 - t1;
    std::cout << msg << ": " << ms_double.count() << "ms, " << ms_double.count() * 0.001 << "s"  << std::endl;
}


inline unsigned int get_seed() {
    return seed_;
}

inline void set_seed(const unsigned int seed = 0) {
    gen.seed(seed);
    seed_ = seed;
}


template <typename T>
void print_np(py::array_t<T, py::array::c_style> arr, bool full, const int cutoff = 100000) {
    auto ra = arr.mutable_unchecked();
     // std::cout << "Shape: " << arr.ndim() << std::endl;
    for (int i = 0; i < arr.ndim(); i++) {
        std::cout << "Dim " << i << ": " << arr.shape(i) << " ";
    }
    std::cout << std::endl;
    if ( arr.ndim() == 1 ) {
        for (int i = 0; i < arr.shape(0); i++) {
            std::cout << ra(i) << " ";
        }
    } else if ( arr.ndim() == 2) {
        for (int i = 0; i < arr.shape(0); i++) {
            for (int j = (full) ? 0 : i; j < arr.shape(1); j++) {
                if (ra(i, j) >= cutoff) {
                    std::cout << "inf " << std::endl;
                } else {
                	std::cout << ra(i, j) << " ";
                }
            }
            std::cout << std::endl;
        }
    }
}


template <typename D>
int get_node_list(unique_ptr<Graph<D>> &g_ptr, unique_ptr<vector<int>> &node_list_ptr) {
    node_list_ptr = make_unique<vector<int>>();
    typename boost::graph_traits<Graph<D>>::vertex_iterator vi, vi_end;
    for (boost::tie(vi, vi_end) = boost::vertices(*g_ptr); vi != vi_end; ++vi) {
        node_list_ptr->push_back(*vi);
    }
    return 0;
}

template <typename D>
py::array_t<int, py::array::c_style> get_node_list_np(unique_ptr<Graph<D>> &g_ptr) {
    const size_t N = num_vertices(*g_ptr);
    py::array_t<int, py::array::c_style> arr({N});
    auto ra = arr.mutable_unchecked();
    typename boost::graph_traits<Graph<D>>::vertex_iterator vi, vi_end;
    int cur = 0;
    for (boost::tie(vi, vi_end) = boost::vertices(*g_ptr); vi != vi_end; ++vi) {
        ra(cur) = *vi;
        cur += 1;
    }
    return arr;
}

vector<int> get_node_shuffle_map(const int N, const int min_vocab, const int max_vocab, const bool shuffle = false) {
    assert ( N >= max_vocab - min_vocab );
    auto m = std::vector<int>(max_vocab - min_vocab);
    std::iota(m.begin(), m.end(), min_vocab);
    if (shuffle) {
        std::shuffle(m.begin(), m.end(), gen);
    }
    // return on first N elements of m
    return std::vector<int>(m.begin(), m.begin() + N);
}

vector<int> get_edge_shuffle_map(const int E, const bool shuffle = false) {
    auto m = std::vector<int>(E);
    std::iota(m.begin(), m.end(), 0);
    if (shuffle) {
        std::shuffle(m.begin(), m.end(), gen);
    }
    return m;
}


template <typename D>
vector<pair<int, int>> get_edge_list(unique_ptr<Graph<D>> &g_ptr, vector<int> &shuffle_map) {
    auto edge_list =vector<pair<int, int>>(num_edges(*g_ptr), make_pair(-1, -1));
    typename boost::graph_traits<Graph<D>>::edge_iterator ei, ei_end;

    int cur = 0;
    for (boost::tie(ei, ei_end) = boost::edges(*g_ptr); ei != ei_end; ++ei) {
        // edge_list.push_back(make_pair(source(*ei, *g_ptr), target(*ei, *g_ptr)));
        edge_list[shuffle_map[cur]] = make_pair(source(*ei, *g_ptr), target(*ei, *g_ptr));
        cur += 1;
    }
    return edge_list;
}

template<typename D>
py::array_t<int, py::array::c_style> get_edge_list_np(unique_ptr<Graph<D>> &g_ptr, vector<int> &shuffle_map) {
    const size_t N = num_edges(*g_ptr);
    constexpr size_t M = 2;
    py::array_t<int, py::array::c_style> arr({N, M});
    auto ra = arr.mutable_unchecked();
    typename boost::graph_traits<Graph<D>>::edge_iterator ei, ei_end;

    int cur = 0;
    for (boost::tie(ei, ei_end) = boost::edges(*g_ptr); ei != ei_end; ++ei) {
        const auto i = source(*ei, *g_ptr);
        const auto j = target(*ei, *g_ptr);
        ra(shuffle_map[cur], 0) = i;
        ra(shuffle_map[cur], 1) = j;
        cur += 1;
    }
    return arr;
}


template <typename T, typename D>
py::array_t<T, py::array::c_style> convert_matrix(unique_ptr<D> &matrix_ptr,
                                                    const int N, const int M, T cuttoff = 100000, T max_value = -1) {
    // distances are in node order so just copy them to the array
    py::array_t<T, py::array::c_style>  arr = py::array_t<T, py::array::c_style>({N, M});
    auto ra = arr.mutable_unchecked();
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            if (cuttoff > 0 && (*matrix_ptr)[i][j] >= cuttoff) {
                ra(i, j) = max_value;
            } else {
                ra(i, j) = (*matrix_ptr)[i][j];
            }
        }
    }
    return arr;
}

template <typename T, typename D>
void convert_matrix(unique_ptr<D> &matrix_ptr, unique_ptr<vector<vector<T>>> &arr_ptr,
    const int N, const int M, T cuttoff = 100000, T max_value = -1) {
    arr_ptr = make_unique<vector<vector<T>>>(N, vector<T>(M));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            if (cuttoff > 0 && (*matrix_ptr)[i][j] >= cuttoff) {
               (*arr_ptr)[i][j] = max_value;
            } else {
                (*arr_ptr)[i][j] = (*matrix_ptr)[i][j];
            }
        }
    }
}

template <typename T>
py::array_t<T, py::array::c_style> convert_vector(vector<T> &vec, const int N, T cuttoff = 100000, T max_value = -1) {
    py::array_t<T, py::array::c_style>  arr = py::array_t<T, py::array::c_style>({N});
    auto ra = arr.mutable_unchecked();
    for (int i = 0; i < N; i++) {
        if (cuttoff > 0 && vec[i] >= cuttoff) {
            ra(i) = max_value;
        } else {
            ra(i) = vec[i];
        }
    }
    return arr;
}


inline vector<int> sample_path(const unique_ptr<vector<vector<int>>> &distances_ptr,
    const int max_length = 10, const int min_length = 1, int start = -1, int end = -1) {
    /*
     * Uniform sample paths of length between min_length and max_length
     * return length as vector of nodes
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
    while (cur != start_end.second) {
        // for all neighbors of cur, find the one with the shortest distance to end
        vector<pair<int, int>> neighbors;
        for (int i = 0; i < (*distances_ptr).size(); i++) {
            if ((*distances_ptr)[cur][i] == 1 && (*distances_ptr)[i][start_end.second] < (*distances_ptr)[cur][start_end.second]) {
                neighbors.push_back(make_pair(i, (*distances_ptr)[i][start_end.second]));
            }
        }
        // shuffle neighbors and then sort by distance to end, dumb way to do this
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


template <typename D>
py::dict package_for_python(unique_ptr<Graph<D>> &g_ptr,
                            const bool is_causal = false,
                            const bool return_full = false,
                            const bool shuffle_edges = false) {

	const auto N = num_vertices(*g_ptr);
    const auto E = num_edges(*g_ptr);

    unique_ptr<DistanceMatrix<boost::undirectedS>> distances_ptr;
    floyd_warshall(g_ptr, distances_ptr, false);
    // print_matrix(distances_ptr, N, N, true, 100000, " ");
    auto original_distances = convert_matrix<int, DistanceMatrix<boost::undirectedS>>(distances_ptr, N, N);

    auto shuffle_map = get_edge_shuffle_map(E, shuffle_edges);  // just range if no shuffle
    auto edge_list = get_edge_list(g_ptr, shuffle_map);

    unique_ptr<vector<vector<int>>> distances_ptr2;
    unique_ptr<vector<vector<int>>> ground_truths_ptr;
    floyd_warshall_frydenlund(g_ptr, distances_ptr2, ground_truths_ptr, edge_list, false);
    // print_matrix(distances_ptr2, N, N, true, 100000, " ");
    // print_matrix(ground_truths_ptr, E, N, true);

    auto distances = convert_matrix<int, vector<vector<int>>>(distances_ptr2, N, N);
    auto ground_truths = convert_matrix<int, vector<vector<int>>>(ground_truths_ptr, E, N);
    // print_np(distances, true);

    auto path = sample_path(distances_ptr2, 10, 3, -1, -1);

    py::dict d;
    d["edge_list"] = get_edge_list_np(g_ptr, shuffle_map);
    d["original_distances"] = original_distances;
    d["distances"] = distances;
    d["ground-truths"] = ground_truths;
    d["path"] = convert_vector(path, path.size());
    return d;
}


inline py::dict erdos_renyi(const int num_nodes, float p = -1.0, const int c_min = 75, const int c_max = 125,
    const bool is_causal = false, const bool return_full = false, const bool shuffle_edges = false) {
    /*
	* Expose generation to python
    */
    assert (p < 1.0);  //  boost fails at p = 1.0, way to go boost
    unique_ptr<Graph<boost::undirectedS>> g_ptr;
    erdos_renyi_generator(g_ptr,  num_nodes, gen, p, c_min, c_max, false);
    return package_for_python(g_ptr, is_causal, return_full, shuffle_edges);
}

inline py::dict euclidian(const int num_nodes, const int dim = 2, float radius = -1.0,
                          const int c_min = 75, const int c_max = 125,
    const bool is_causal = false, const bool return_full = false, const bool shuffle_edges = false) {
    /*
     */
    unique_ptr<Graph<boost::undirectedS>> g_ptr;
    unique_ptr<vector<vector<float>>> positions_ptr;
    euclidean_generator(g_ptr, positions_ptr, num_nodes, gen, dim, radius, c_min, c_max, false);
    auto d = package_for_python(g_ptr, is_causal, return_full, shuffle_edges);

    //print_matrix(positions_ptr, num_vertices(*g_ptr), 2, true);
    auto positions = convert_matrix<float, vector<vector<float>>>(positions_ptr, num_vertices(*g_ptr), 2);
    d["positions"] = positions;
    return d;
}

inline py::dict path_star(const int min_num_arms, const int max_num_arms, const int min_arm_length, const int max_arm_length,
    const bool is_causal = false, const bool return_full = false, const bool shuffle_edges = false) {
    /*
     *
     */
    unique_ptr<Graph<boost::directedS>> g_ptr;
    auto start_end = path_star_generator(g_ptr,  min_num_arms, max_num_arms, min_arm_length,max_arm_length, gen, false);
    auto d = package_for_python(g_ptr, is_causal, return_full, shuffle_edges);
    d["start"] = start_end.first;
    d["end"] = start_end.second;
    return d;
}

inline py::dict balanced(const int num_nodes, int lookahead, const int min_noise_reserve = 0, const int max_num_parents = 4,
    const bool is_causal = false, const bool return_full = false, const bool shuffle_edges = false) {
    /*
     *
     */
    unique_ptr<Graph<boost::directedS>> g_ptr;
    auto start_end = balanced_generator(g_ptr, num_nodes, gen, lookahead, min_noise_reserve, max_num_parents, false);
    auto d = package_for_python(g_ptr, is_causal, return_full, shuffle_edges);
    d["start"] = start_end.first;
    d["end"] = start_end.second;
    return d;
}

template <typename T>
py::array_t<T, py::array::c_style> batch_matrix(const list<unique_ptr<vector<vector<T>>>> &in_matrices,  T pad = -1) {
    int N = 0;
    int M = 0;
    for (auto &m : in_matrices) {
        if ((*m).size() > N) {
            N = (*m).size();
        }
        if ((*m)[0].size() > M) {
            M = (*m)[0].size();
        }
    }
    pad = static_cast<T>(pad);
    py::array_t<T, py::array::c_style> arr({static_cast<int>(in_matrices.size()), N, M});
    auto ra = arr.mutable_unchecked();
    int cur = 0;
    for (auto &m : in_matrices) {
        for (int j = 0; j < (*m).size(); j++) {
            for (int k = 0; k < (*m)[j].size(); k++) {
                ra(cur, j, k) = (*m)[j][k];
            }
            for (int k = (*m)[j].size(); k < M; k++) {
                ra(cur, j, k) = pad;
            }
        }
        for (int j = (*m).size(); j < N; j++) {
            for (int k = 0; k < M; k++) {
                ra(cur, j, k) = pad;
            }
        }
        cur += 1;
    }
    return arr;
}

// what a mess, refactored so this is not needed
template <typename T, typename T2>
py::array_t<T, py::array::c_style> batch_matrix(const list<unique_ptr<T2>> &in_matrices,
    const list<pair<int, int>> &batched_sizes, int pad = -1) {
    int N = 0;
    int M = 0;
    for (auto &m : batched_sizes) {
        if (m.first > N) {
            N = m.first;
        }
        if (m.second > M) {
            M = m.second;
        }
    }
    py::array_t<T, py::array::c_style> arr({static_cast<int>(in_matrices.size()), N, M});
    auto ra = arr.mutable_unchecked();
    int cur = 0;
    auto it1 = in_matrices.begin();
    auto it2 = batched_sizes.begin();
    for(; it1 != in_matrices.end() && it2 != batched_sizes.end(); ++it1, ++it2) {
        auto n = it2->first;
        auto m = it2->second;
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < m; k++) {
                ra(cur, j, k) = (**it1)[j][k];
            }
            for (int k = m; k < M; k++) {
                ra(cur, j, k) = pad;
            }
        }
        cur += 1;
    }
    return arr;
}


template <typename T>
py::array_t<T, py::array::c_style> batch_edge_list(const list<vector<pair<int, int>>> &batched_edge_list, int pad = -1) {
    int E = 0;
    for (auto &m : batched_edge_list) {
        if (m.size() > E) {
            E = m.size();
        }
    }
    py::array_t<T, py::array::c_style> arr({static_cast<int>(batched_edge_list.size()), E, 2});
    auto ra = arr.mutable_unchecked();
    auto cur = 0;
    for (auto &m : batched_edge_list) {
        for (int j = 0; j < m.size(); j++) {
            ra(cur, j, 0) = m[j].first;
            ra(cur, j, 1) = m[j].second;
        }
        for (int j = m.size(); j < E; j++) {
            ra(cur, j, 0) = pad;
            ra(cur, j, 1) = pad;
        }
        cur += 1;
    }
    return arr;
}

template <typename T>
py::array_t<T, py::array::c_style> batch_paths(const list<vector<int>> &batched_paths, int pad = -1) {
    int L = 0;
    for (auto &m : batched_paths) {
        if (m.size() > L) {
            L = m.size();
        }
    }
    py::array_t<T, py::array::c_style> arr({static_cast<int>(batched_paths.size()), L});
    auto ra = arr.mutable_unchecked();
    auto cur = 0;
    for (auto &m : batched_paths) {
        for (int j = 0; j < m.size(); j++) {
            ra(cur, j) = m[j];
        }
        for (int j = m.size(); j < L; j++) {
            ra(cur, j) = pad;
        }
        cur += 1;
    }
    return arr;
}

template <typename T>
py::array_t<T, py::array::c_style> batch_lengths(const list<int> &batched_lengths) {

    py::array_t<T, py::array::c_style> arr({static_cast<int>(batched_lengths.size())});
    auto ra = arr.mutable_unchecked();
    int cur = 0;
    for (auto &m : batched_lengths) {
        ra(cur) = m;
        cur += 1;
    }
    return arr;
}


template <typename D>
void push_back_data(unique_ptr<Graph<D>> &g_ptr,
                    list<vector<pair<int, int>>> &batched_edge_list,
                    list<unique_ptr<vector<vector<int>>>> &batched_distances,
                    list<pair<int, int>> &batched_sizes,
                    list<unique_ptr<vector<vector<int>>>> &batched_ground_truths,
                    list<vector<int>> &batched_paths,
                    list<int> &batched_edge_list_lengths,
                    list<int> &batched_path_lengths,
                    const int max_length = 10, const int min_length = 1,
                    const bool is_causal = false, const bool return_full = false, const bool shuffle_edges = false) {

    const auto E = num_edges(*g_ptr);
    auto shuffle_map = get_edge_shuffle_map(E, shuffle_edges);
    auto edge_list = get_edge_list<D>(g_ptr, shuffle_map);
    batched_edge_list.push_back(edge_list);
    batched_edge_list_lengths.push_back(E);
    if ( is_causal ) {
        unique_ptr<vector<vector<int>>> distances_ptr;
        unique_ptr<vector<vector<int>>> ground_truths_ptr;
        auto path_d = time_before();
        floyd_warshall_frydenlund(g_ptr, distances_ptr, ground_truths_ptr, edge_list, false);
        time_after(path_d, "floyd_warshall_frydenlund");
        auto path_t = time_before();
        auto path = sample_path(distances_ptr, max_length, min_length, -1, -1);
        time_after(path_t, "sample_path");
        batched_paths.push_back(path);
        batched_path_lengths.push_back(path.size());
        batched_distances.push_back(move(distances_ptr));
        batched_ground_truths.push_back(move(ground_truths_ptr));
    } else {
        auto N = num_vertices(*g_ptr);
        unique_ptr<DistanceMatrix<D>> distances_ptr;
        auto path_d = time_before();
        floyd_warshall<D>(g_ptr, distances_ptr, false);
        time_after(path_d, "floyd_warshall");
        unique_ptr<vector<vector<int>>> distances_ptr2;
        convert_matrix<int, DistanceMatrix<D>>(distances_ptr, distances_ptr2, N, N);
        auto path_t = time_before();
        auto path = sample_path(distances_ptr2, max_length, min_length, -1, -1);
        time_after(path_t, "sample_path");
        batched_paths.push_back(path);
        batched_path_lengths.push_back(path.size());
        batched_distances.push_back(move(distances_ptr2));
        batched_sizes.push_back(make_pair(N, N));
    }
}


inline py::dict erdos_renyi_n(
    const int num_nodes, float p = -1.0, const int c_min = 75, const int c_max = 125,
    const bool is_causal = false, const bool return_full = false, const bool shuffle_edges = false,
    const int max_length = 10, const int min_length = 1,
    const int batch_size = 256, const int max_edges = 512, const bool sample_target_paths = true, int max_attempts = 1000) {

    assert ( p < 1.0);  //  boost fails at p = 1.0, way to go boost
    assert ( c_min <= c_max);
    assert ( batch_size > 0);
    assert ( min_length <= max_length);
    auto batched_edge_list = list<vector<pair<int, int>>>();
    // dumb code
    auto batched_distances = list<unique_ptr<vector<vector<int>>>>();
    auto batched_sizes = list<pair<int, int>>();
    auto batched_ground_truths = list<unique_ptr<vector<vector<int>>>>();
    auto batched_paths = list<vector<int>>(batch_size);
    auto batched_edge_list_lengths = list<int>();
    auto batched_path_lengths = list<int>();
    int attempts = 0;
    int num = 0;
    while ( num < batch_size ) {
        unique_ptr<Graph<boost::undirectedS>> g_ptr;
        auto graph_t = time_before();
        erdos_renyi_generator(g_ptr,  num_nodes, gen, p, c_min, c_max, false);
        time_after(graph_t, "graph gen");
        const auto E = num_edges(*g_ptr);
        if ( E > max_edges ) {
            attempts += 1;
            if (attempts > max_attempts) {
                cout << "Failed to generate graph after " << attempts << " attempts" << endl;
                break;
            }
            continue;
        }
        auto pack_t = time_before();
        push_back_data<boost::undirectedS>(g_ptr, batched_edge_list, batched_distances,  batched_sizes,
            batched_ground_truths, batched_paths, batched_edge_list_lengths, batched_path_lengths,
            max_length, min_length,
            is_causal, return_full, shuffle_edges);
        time_after(pack_t, "pack");


        num += 1;
    }

    py::dict d;
    d["num_attempts"] = attempts;
    if ( attempts >= max_attempts ) {
        return d;
    }
    d["edge_list"] = batch_edge_list<int>(batched_edge_list);
    d["distances"] = batch_matrix<int>(batched_distances, -1);
    if ( is_causal ) {
        d["ground-truths"] = batch_matrix<int>(batched_ground_truths, -1);
    } else {

    }
    d["paths"] = batch_paths<int>(batched_paths, -1);
    d["edge_list_lengths"] = batch_lengths<int>(batched_edge_list_lengths);
    d["path_lengths"] = batch_lengths<int>(batched_path_lengths);
    return d;
}



PYBIND11_MODULE(generator, m) {
  	m.doc() = "Graph generation module"; // optional module docstring
    m.def("set_seed", &set_seed, "Sets random seed (unique to thread)", py::arg("seed") = 0);
    m.def("get_seed", &get_seed, "Gets random seed (unique to thread)");

    // single graph generation
    m.def("erdos_renyi", &erdos_renyi, "Generate a single Erdos Renyi graph", py::arg("num_nodes"),
        py::arg("p") = -1.0, py::arg("c_min") = 75, py::arg("c_max") = 125,
        py::arg("is_causal") = false, py::arg("return_full") = false, py::arg("shuffle_edges") = false);


    m.def("euclidian", &euclidian, "Generate a single Euclidian graph", py::arg("num_nodes"),
        py::arg("dims") = 2, py::arg("radius") = -1.0, py::arg("c_min") = 75, py::arg("c_max") = 125,
        py::arg("is_causal") = false, py::arg("return_full") = false, py::arg("shuffle_edges") = false);


    m.def("path_star", &path_star, "Generate a single path star graph",
        py::arg("min_num_arms"), py::arg("max_num_arms"), py::arg("min_arm_length"), py::arg("max_arm_length"),
        py::arg("is_causal") = false, py::arg("return_full") = false, py::arg("shuffle_edges") = false);


    m.def("balanced", &balanced, "Generate a single balanced graph.  Note these are done slightly differently from original paper.",
        py::arg("num_nodes"), py::arg("lookahead"), py::arg("min_noise_reserve") = 0, py::arg("max_num_parents") = 4,
        py::arg("is_causal") = false, py::arg("return_full") = false, py::arg("shuffle_edges") = false);

    // batched graph generation
    m.def("erdos_renyi_n", &erdos_renyi_n, "Generate a batch of Erdos Renyi graphs",
        py::arg("num_nodes"), py::arg("p") = -1.0, py::arg("c_min") = 75, py::arg("c_max") = 125,
        py::arg("max_length") = 10, py::arg("min_length") = 1,
        py::arg("is_causal") = false, py::arg("return_full") = false, py::arg("shuffle_edges") = false,
        py::arg("batch_size") = 256, py::arg("max_edges") = 512, py::arg("sample_target_paths") = true,
        py::arg("max_attempts") = 1000);


}