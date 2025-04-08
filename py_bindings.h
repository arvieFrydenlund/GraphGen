//
// Created by arvie on 4/6/25.
//

#ifndef PY_BINDINGS_H
#define PY_BINDINGS_H

#include <iostream>
#include <python3.11/Python.h>
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

inline unsigned int seed = std::random_device{}();
static thread_local std::mt19937 gen = std::mt19937(seed);


inline unsigned int get_seed() {
    return seed;
}


inline void set_seed(unsigned int new_seed) {
    gen.seed(new_seed);
    seed = new_seed;
}


py::dict return_dict_test(){
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

template <typename D>
int get_edge_list(unique_ptr<Graph<D>> &g_ptr, unique_ptr<vector<pair<int, int>>> &edge_list_ptr) {
    edge_list_ptr = make_unique<vector<pair<int, int>>>();
    typename boost::graph_traits<Graph<D>>::edge_iterator ei, ei_end;
    for (boost::tie(ei, ei_end) = boost::edges(*g_ptr); ei != ei_end; ++ei) {
        edge_list_ptr->push_back(make_pair(source(*ei, *g_ptr), target(*ei, *g_ptr)));
    }
    return 0;
}


template<typename D>
py::array_t<int, py::array::c_style> get_edge_list_np(unique_ptr<Graph<D>> &g_ptr) {

    const size_t N = num_edges(*g_ptr);
    constexpr size_t M = 2;
    py::array_t<int, py::array::c_style> arr({N, M});
    auto ra = arr.mutable_unchecked();
    typename boost::graph_traits<Graph<D>>::edge_iterator ei, ei_end;

    int cur = 0;
    for (boost::tie(ei, ei_end) = boost::edges(*g_ptr); ei != ei_end; ++ei) {
        const auto i = source(*ei, *g_ptr);
        const auto j = target(*ei, *g_ptr);
        ra(cur, 0) = i;
        ra(cur, 1) = j;
        cur += 1;
    }
    return arr;
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

template <typename T>
void print_np(py::array_t<T, py::array::c_style> arr) {
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
            for (int j = 0; j < arr.shape(1); j++) {
                std::cout << ra(i, j) << " ";
            }
            std::cout << std::endl;
        }
    }
}


template <typename D>
py::array_t<int, py::array::c_style> convert_distance(unique_ptr<DistanceMatrix<D>> &distances_ptr,
    const bool is_causal = false, const bool full = false) {

    const size_t N = sizeof((*distances_ptr)[0]);
    py::array_t<int, py::array::c_style> arr({N, N});
    if ( is_causal ) {
        // convert to full distance matrix
        // TODO
    } else if ( full ) {
        // convert to full distance matrix
        // TODO
    } else {

        auto ra = arr.mutable_unchecked();
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                ra(i, j) = (*distances_ptr)[i][j];
            }
        }
    }
    return arr;
}


inline py::dict erdos_renyi(const int num_nodes, float p = -1.0, const int c_min = 75, const int c_max = 125,
    const bool is_causal = false, const bool full = false, const bool shuffle = false) {

    unique_ptr<Graph<boost::undirectedS>> g_ptr;
    erdos_renyi_generator(g_ptr,  num_nodes, gen, p, c_min, c_max, false);

    unique_ptr<DistanceMatrix<boost::undirectedS>> distances_ptr;
    get_distances(g_ptr, distances_ptr, is_causal, false);
    print_distances<boost::undirectedS>(distances_ptr, num_vertices(*g_ptr));

    //auto edge_list = get_edge_list_np(g_ptr);
    //auto distances = convert_distance<boost::undirectedS>(distances_ptr, is_causal, full);
    // print_np(edge_list);
    // auto node_list = get_node_list_np(g_ptr);
    // print_np(node_list);

    py::dict d;
    d["edge_list"] = 1; //edge_list;
    d["distances"] = 1; //distances;

    return d;

}



#endif //PY_BINDINGS_H
