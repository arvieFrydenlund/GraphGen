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




#endif //UTILS_H
