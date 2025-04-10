//
// Created by arvie on 4/6/25.
//

#ifndef PY_BINDINGS_H
#define PY_BINDINGS_H

#include <iostream>
// #include <python3.11/Python.h>
#include <Python.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>


/*
import pybind11;
print(pybind11.get_include())

https://medium.com/@ahmedfgad/pybind11-tutorial-binding-c-code-to-python-337da23685dc
*/

using namespace std;
namespace py = pybind11;
using namespace py::literals;

static const py::bool_ py_true(true);

inline unsigned int seed_ = std::random_device{}();
static thread_local auto gen = std::mt19937(seed_);  // so each thread in the dataloader is different


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

vector<int> get_shuffle_map(const int E, const bool shuffle = false) {
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
py::array_t<D, py::array::c_style> convert_matrix(unique_ptr<T> &matrix_ptr,
                                                    const int N, const int M, D cuttoff = 100000, D max_value = -1) {
    // distances are in node order so just copy them to the array
    py::array_t<D, py::array::c_style>  arr = py::array_t<D, py::array::c_style>({N, M});
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
    auto original_distances = convert_matrix<DistanceMatrix<boost::undirectedS>, int>(distances_ptr, N, N);

    auto shuffle_map = get_shuffle_map(E, shuffle_edges);  // just range if no shuffle
    auto edge_list = get_edge_list(g_ptr, shuffle_map);

    unique_ptr<vector<vector<int>>> distances_ptr2;
    unique_ptr<vector<vector<int>>> ground_truths_ptr;
    floyd_warshall_frydenlund(g_ptr, distances_ptr2, ground_truths_ptr, edge_list, false);
    // print_matrix(distances_ptr2, N, N, true, 100000, " ");
    // print_matrix(ground_truths_ptr, E, N, true);

    auto distances = convert_matrix<vector<vector<int>>, int>(distances_ptr2, N, N);
    auto ground_truths = convert_matrix<vector<vector<int>>, int>(ground_truths_ptr, E, N);
    // print_np(distances, true);

    py::dict d;
    // d["edge_list"] = 1; //edge_list;
    d["original_distances"] = original_distances;
    d["distances"] = distances;
    d["ground-truths"] = ground_truths;
    return d;
}


inline py::dict erdos_renyi(const int num_nodes, float p = -1.0, const int c_min = 75, const int c_max = 125,
    const bool is_causal = false, const bool return_full = false, const bool shuffle_edges = false) {
    /*
	* Expose generation to python
	* Notes:
    * Do shuffle nodes during copy to arrays? note shuffle only needed when learning a fixed vocabulary.
    * Actually just do this once data is on gpu?  Issue is if vocab is too large, then pushing lots of data to gpu
    * Shuffle for erdos_renyi and euclidean graphs not needed due to randomness
    * 'Shuffle' for path_star and balanced graphs is already done, just not past the number of nodes in graph.
     */
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

    // print_matrix(positions_ptr, num_vertices(*g_ptr), 2, true);
    auto positions = convert_matrix<vector<vector<float>>, float>(positions_ptr, num_vertices(*g_ptr), 2);
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


PYBIND11_MODULE(generator, m) {
  	m.doc() = "Graph generation module"; // optional module docstring
    m.def("set_seed", &set_seed, "Sets random seed (unique to thread)", py::arg("seed") = 0);
    m.def("get_seed", &get_seed, "Gets random seed (unique to thread)");

    m.def("erdos_renyi", &erdos_renyi, "TODO");
    m.def("euclidian", &euclidian, "TODO");
    m.def("path_star", &path_star, "TODO");
    m.def("balanced", &balanced, "TODO");

}



#endif //PY_BINDINGS_H
