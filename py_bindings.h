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

#include "data_gen.h"

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


inline py::dict return_dict_test(){  // example code, delete
    py::dict d;
    d["a"] = py::none();
    d["b"] = 2;
    for (auto item : d)
    {
        std::cout << "key: " << item.first << ", value=" << item.second << std::endl;
    };
    cout << endl;
    return d;
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


template <typename T>
py::array_t<int, py::array::c_style> convert_matrix(unique_ptr<T> &matrix_ptr,
                                                    const int N, const int M) {
    // distances are in node order so just copy them to the array
    py::array_t<int, py::array::c_style>  arr = py::array_t<int, py::array::c_style>({N, M});
    auto ra = arr.mutable_unchecked();
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            ra(i, j) = (*matrix_ptr)[i][j];
        }
    }
    return arr;
}

template <typename D>
py::dict package_for_python(unique_ptr<Graph<D>> &g_ptr,
                 const bool is_causal = false, const bool full = false, const bool shuffle = false){

	const auto N = num_vertices(*g_ptr);
    const auto E = num_edges(*g_ptr);

    unique_ptr<DistanceMatrix<boost::undirectedS>> distances_ptr;
    floyd_warshall(g_ptr, distances_ptr, false);
    print_matrix(distances_ptr, N, N, true);



    auto shuffle_map = get_shuffle_map(E, shuffle);  // just range if no shuffle
    auto edge_list = get_edge_list(g_ptr, shuffle_map);

    unique_ptr<vector<vector<int>>> distances_ptr2;
    unique_ptr<vector<vector<int>>> ground_truths_ptr;

    floyd_warshall_frydenlund(g_ptr, distances_ptr2, ground_truths_ptr, edge_list, false);
    print_matrix(distances_ptr2, N, N, true);
    print_matrix(ground_truths_ptr, E, N, true);

    auto distances = convert_matrix(distances_ptr2, N, N);
    print_np(distances, true);


    // auto distances = convert_distance<boost::undirectedS>(distances_ptr, N, E, shuffle_map, edge_list, is_causal, full);
    // print_np(edge_list);
    // auto node_list = get_node_list_np(g_ptr);
    // print_np(node_list);

    py::dict d;
    d["edge_list"] = 1; //edge_list;
    d["distances"] = 1; //distances;
    return d;
}


inline py::dict erdos_renyi(const int num_nodes, float p = -1.0, const int c_min = 75, const int c_max = 125,
    const bool is_causal = false, const bool full = false, const bool shuffle = false) {
    /*
	* Expose generation to python
	* Notes:
    * Do shuffle nodes during copy to arrays? note shuffle only needed when learning a fixed vocabulary.
    * Actually just do this once data is on gpu?  Issue is if vocab is too large, then pushing lots of data to gpu
    * Shuffle for erdos_renyi and euclidean graphs not needed due to randomness
    * 'Shuffle' for path_star and balanced graphs is already done, just not past the number of nodes in graph.
    *
    * Now shuffle means to just shuffle edge list
     */
    unique_ptr<Graph<boost::undirectedS>> g_ptr;
    erdos_renyi_generator(g_ptr,  num_nodes, gen, p, c_min, c_max, false);
    return package_for_python(g_ptr, is_causal, full, shuffle);
}


PYBIND11_MODULE(generator, m) {
  	m.doc() = "Graph generation module"; // optional module docstring
    m.def("erdos_renyi", &erdos_renyi);
    m.def("set_seed", &set_seed, "Sets random seed (unique to thread)", py::arg("seed") = 0);
    m.def("get_seed", &get_seed, "Gets random seed (unique to thread)");
}



#endif //PY_BINDINGS_H
