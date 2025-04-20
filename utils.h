//
// Created by arvie on 17/04/25.
//

#ifndef UTILS_H
#define UTILS_H

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

// Timing
chrono::time_point<high_resolution_clock> time_before() {
    return high_resolution_clock::now();

}

void time_after(chrono::time_point<high_resolution_clock> t1, const string &msg = "") {
    auto t2 = high_resolution_clock::now();
    duration<double, std::milli> ms_double = t2 - t1;
    std::cout << msg << ": " << ms_double.count() << "ms, " << ms_double.count() * 0.001 << "s"  << std::endl;
}

// printing
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


/* ************************************************
 *  Converting utils
 *  Single graph generation
 *  ***********************************************/

inline py::array_t<int, py::array::c_style> convert_edge_list(vector<pair<int, int>> &edge_list, vector<int>& node_shuffle_map) {
    // Convert a edge_list [E,2] (which has already been shuffled) to a numpy array and map node ids
    auto E = edge_list.size();
    constexpr size_t M = 2;
    py::array_t<int, py::array::c_style> arr({E, M});
    auto ra = arr.mutable_unchecked();
    int cur = 0;
    for (auto &e : edge_list) {
        const auto i = e.first;
        const auto j = e.second;
        ra(cur, 0) = node_shuffle_map[i];
        ra(cur, 1) = node_shuffle_map[j];
        cur += 1;
    }
    return arr;
}


template <typename T, typename D>
void convert_boost_matrix(unique_ptr<D> &matrix_ptr, unique_ptr<vector<vector<T>>> &arr_ptr,
    const int N, const int M) {
    // convert a boost distance matrix to a c++ matrix
    arr_ptr = make_unique<vector<vector<T>>>(N, vector<T>(M));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            (*arr_ptr)[i][j] = (*matrix_ptr)[i][j];
        }
    }
}


template <typename T, typename D>
py::array_t<T, py::array::c_style> convert_distance_matrix(unique_ptr<D> &matrix_ptr, vector<int>& node_shuffle_map,
    const int N, const int new_N, T cuttoff = 100000, T max_value = -1, T mask_value = -1) {
    // Convert a distance matrix [N, N] to a numpy array [new_N, new_N] by mapping node ids
    // indices are nodes, values are distances
    py::array_t<T, py::array::c_style>  arr = py::array_t<T, py::array::c_style>({new_N, static_cast<int>(new_N)});
    arr[py::make_tuple(py::ellipsis())] = mask_value;  // initialize array
    auto ra = arr.mutable_unchecked();
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (cuttoff > 0 && (*matrix_ptr)[i][j] >= cuttoff) {
                ra(node_shuffle_map[i], node_shuffle_map[j]) = max_value;
            } else {
                ra(node_shuffle_map[i], node_shuffle_map[j]) = (*matrix_ptr)[i][j];
            }
        }
    }
    return arr;
}


template <typename T, typename D>
py::array_t<T, py::array::c_style> convert_ground_truths(unique_ptr<D> &matrix_ptr, vector<int>& node_shuffle_map,
    const int E, const int N, const int new_N, T cuttoff = 100000, T max_value = -1, T mask_value = -1) {
    // indices are nodes, values are distances
    auto new_M = *max_element(node_shuffle_map.begin(), node_shuffle_map.end()) + 1;
    py::array_t<T, py::array::c_style>  arr = py::array_t<T, py::array::c_style>({E, new_N});
    arr[py::make_tuple(py::ellipsis())] = mask_value;  // initialize array
    auto ra = arr.mutable_unchecked();
    for (int i = 0; i < E; i++) {
        for (int j = 0; j < N; j++) {
            if (cuttoff > 0 && (*matrix_ptr)[i][j] >= cuttoff) {
                ra(i, node_shuffle_map[j]) = max_value;
            } else {
                ra(i, node_shuffle_map[j]) = (*matrix_ptr)[i][j];
            }
        }
    }
    return arr;
}


template <typename T>
py::array_t<T, py::array::c_style> convert_path(vector<T> &vec, vector<int>& node_shuffle_map) {
    int N = vec.size();
    py::array_t<T, py::array::c_style>  arr = py::array_t<T, py::array::c_style>({N});
    auto ra = arr.mutable_unchecked();
    for (int i = 0; i < N; i++) {
        ra(i) = node_shuffle_map[vec[i]];  // value is mapped
    }
    return arr;
}

template <typename T>
py::array_t<T, py::array::c_style> convert_vector(vector<T> &vec) {
    int N = vec.size();
    py::array_t<T, py::array::c_style>  arr = py::array_t<T, py::array::c_style>({N});
    auto ra = arr.mutable_unchecked();
    for (int i = 0; i < N; i++) {
        ra(i) = vec[i];
    }
    return arr;
}


/* ************************************************
 *  Converting utils
 *  Batched graph generation
 *  ***********************************************/

template <typename T>
py::array_t<T, py::array::c_style> batch_edge_list(const list<unique_ptr<vector<pair<int, int>>>> &batched_edge_list,
                                                   list<unique_ptr<vector<int>>> &batched_node_shuffle_map, int pad = -1) {
    int E = 0;
    for (auto &m : batched_edge_list) {
        if ((*m).size() > E) {
            E = (*m).size();
        }
    }
    py::array_t<T, py::array::c_style> arr({static_cast<int>(batched_edge_list.size()), E, 2});
    arr[py::make_tuple(py::ellipsis())] = static_cast<T>(pad);  // initialize array
    auto ra = arr.mutable_unchecked();
    auto cur = 0;
    // parallel iterate over batched_edge_list and batched_node_shuffle_map
    auto it1 = batched_edge_list.begin();
    auto it2 = batched_node_shuffle_map.begin();
    for (; it1 != batched_edge_list.end() && it2 != batched_node_shuffle_map.end(); ++it1, ++it2) {
    // for (; it1 != batched_edge_list.end(); ++it1) {
        for (int j = 0; j < (**it1).size(); j++) {
            ra(cur, j, 0) = (**it2)[(**it1)[j].first];
            ra(cur, j, 1) = (**it2)[(**it1)[j].second];
            // ra(cur, j, 0) = (*it1)[j].first;
            // ra(cur, j, 1) = (*it1)[j].second;
        }
        cur += 1;
    }
    return arr;
}

template <typename T>
py::array_t<T, py::array::c_style> batch_distances(const list<unique_ptr<vector<vector<T>>>> &batched_distances,
                                                   list<unique_ptr<vector<int>>> &batched_node_shuffle_map,
                                                   const int new_N, T cuttoff = 100000, T max_value = -1, T pad = -1) {

    py::array_t<T, py::array::c_style> arr({static_cast<int>(batched_distances.size()), new_N, new_N});
    arr[py::make_tuple(py::ellipsis())] = static_cast<T>(pad);  // initialize array
    auto ra = arr.mutable_unchecked();

    auto it1 = batched_distances.begin();
    auto it2 = batched_node_shuffle_map.begin();
    int cur = 0;
    for (; it1 != batched_distances.end() && it2 != batched_node_shuffle_map.end(); ++it1, ++it2) {
        for (int j = 0; j < (**it1).size(); j++) {
            for (int k = 0; k < (**it1)[j].size(); k++) {
                auto mapped_j = (**it2)[j];
                auto mapped_k = (**it2)[k];
                if (cuttoff > 0 && (**it1)[j][k] >= cuttoff) {
                    ra(cur, mapped_j, mapped_k) = max_value;
                } else {
                    ra(cur, mapped_j, mapped_k) = (**it1)[j][k];
                }
            }
        }
        cur += 1;
    }
    return arr;
}


template <typename T>
py::array_t<T, py::array::c_style> batch_ground_truths(const list<unique_ptr<vector<vector<T>>>> &batched_ground_truths,
                                                   list<unique_ptr<vector<int>>> &batched_node_shuffle_map,
                                                   const int new_N, T cuttoff = 100000, T max_value = -1, T pad = -1) {
    // indices are nodes, values are distances
    auto max_E = 0;
    for (auto &m : batched_ground_truths) {
        if ((*m).size() > max_E) {
            max_E = (*m).size();
        }
    }

    py::array_t<T, py::array::c_style> arr({static_cast<int>(batched_ground_truths.size()), max_E, new_N});
    arr[py::make_tuple(py::ellipsis())] = static_cast<T>(pad);  // initialize array
    auto ra = arr.mutable_unchecked();
    auto it1 = batched_ground_truths.begin();
    auto it2 = batched_node_shuffle_map.begin();
    int cur = 0;
    for (; it1 != batched_ground_truths.end() && it2 != batched_node_shuffle_map.end(); ++it1, ++it2) {
        for (int j = 0; j < (**it1).size(); j++) {
            for (int k = 0; k < (**it1)[j].size(); k++) {
                if (cuttoff > 0 && (**it1)[j][k] >= cuttoff) {
                    ra(cur, j, (**it2)[k]) = max_value;
                } else {
                    ra(cur, j, (**it2)[k]) = (**it1)[j][k];
                }
            }
        }
        cur += 1;
    }
    return arr;
}


template <typename T>
py::array_t<T, py::array::c_style> batch_paths(const list<unique_ptr<vector<int>>> &batched_paths,
                                               list<unique_ptr<vector<int>>> &batched_node_shuffle_map, int pad = -1) {

    int N = 0;
    for (auto &m : batched_paths) {
        if ((*m).size() > N) {
            N = (*m).size();
        }
    }
    py::array_t<T, py::array::c_style> arr({static_cast<int>(batched_paths.size()), N});
    arr[py::make_tuple(py::ellipsis())] = static_cast<T>(pad); // initialize array
    auto ra = arr.mutable_unchecked();
    auto it1 = batched_paths.begin();
    auto it2 = batched_node_shuffle_map.begin();
    int cur = 0;
    for (; it1 != batched_paths.end() && it2 != batched_node_shuffle_map.end(); ++it1, ++it2) {
        for (int j = 0; j < (**it1).size(); j++) {
            ra(cur, j) = (**it2)[(**it1)[j]];
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


template <typename T>
py::array_t<T, py::array::c_style> batch_positions(const list<unique_ptr<vector<vector<T>>>> &batched_positions,
                                                   list<unique_ptr<vector<int>>> &batched_node_shuffle_map,
                                                   const int dim,
                                                   int pad = -1) {
    int N = 0;
    for (auto &m : batched_positions) {
        if ((*m).size() > N) {
            N = (*m).size();
        }
    }
    py::array_t<T, py::array::c_style> arr({static_cast<int>(batched_positions.size()), N, dim + 1});
    arr[py::make_tuple(py::ellipsis())] = static_cast<T>(pad);  // initialize array
    auto ra = arr.mutable_unchecked();
    auto it1 = batched_positions.begin();
    auto it2 = batched_node_shuffle_map.begin();
    int cur = 0;
    for (; it1 != batched_positions.end() && it2 != batched_node_shuffle_map.end(); ++it1, ++it2) {
        for (int j = 0; j < (**it1).size(); j++) {
            cout << "j: " << j << endl;
            ra(cur, j, 0) = static_cast<T>((**it2)[j]);  // map node id
            for (int d = 0; d < dim; d++) { // positions
                constexpr float r = 10000;
                ra(cur, j, d + 1) =  ceil((**it1)[j][d] * r) / r;;
            }
        }
        cur += 1;
    }
    return arr;
}


// Hashing
// has each distance matrix as a string, return the hashes as a numpy array
template <typename T>
py::array_t<std::uint64_t, py::array::c_style> hash_distance_matrix(const py::array_t<T, py::array::c_style> &batched_distances) {
    // Convert a distance matrix [N, N] to a numpy array [new_N, new_N] by mapping node ids
    auto shape = batched_distances.shape();
    py::array_t<std::uint64_t, py::array::c_style> arr({static_cast<int>(shape[0])});
    auto ra = arr.mutable_unchecked();
    auto bd = batched_distances.unchecked();
    for (int b = 0; b < shape[0]; b++) {
      	// make string from distance matrix
        std::string str = "";
        for (int i = 0; i < shape[1]; i++) {
            for (int j = 0; j < shape[2]; j++) {
                str += std::to_string(bd(b, i, j));
            }
        }
        auto hash = std::hash<std::string>{}(str);
        // auto has2 = static_cast<std::uint64_t>(hash);
        // cout << "hash: " << hash << " has2: " << has2 << endl;
        ra(b) = static_cast<std::uint64_t>(hash);
    }
    return arr;
}






#endif //UTILS_H
