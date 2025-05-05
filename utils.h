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
    std::cout << msg << ": " << ms_double.count() << "ms, " << ms_double.count() * 0.001 << "s" << std::endl;
}

// printing
template<typename T>
void print_np(py::array_t<T, py::array::c_style> arr, bool full, const int cutoff = 100000) {
    auto ra = arr.mutable_unchecked();
    // std::cout << "Shape: " << arr.ndim() << std::endl;
    for (int i = 0; i < arr.ndim(); i++) {
        std::cout << "Dim " << i << ": " << arr.shape(i) << " ";
    }
    std::cout << std::endl;
    if (arr.ndim() == 1) {
        for (int i = 0; i < arr.shape(0); i++) {
            std::cout << ra(i) << " ";
        }
    } else if (arr.ndim() == 2) {
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

inline py::array_t<int, py::array::c_style> convert_edge_list(vector<pair<int, int> > &edge_list,
                                                              vector<int> &node_shuffle_map) {
    // Convert a edge_list [E,2] (which has already been shuffled) to a numpy array and map node ids
    auto E = edge_list.size();
    constexpr size_t M = 2;
    py::array_t<int, py::array::c_style> arr({E, M});
    auto ra = arr.mutable_unchecked();
    int cur = 0;
    for (auto &e: edge_list) {
        const auto i = e.first;
        const auto j = e.second;
        ra(cur, 0) = node_shuffle_map[i];
        ra(cur, 1) = node_shuffle_map[j];
        cur += 1;
    }
    return arr;
}


template<typename T, typename D>
void convert_boost_matrix(unique_ptr<D> &matrix_ptr, unique_ptr<vector<vector<T> > > &arr_ptr,
                          const int N, const int M) {
    // convert a boost distance matrix to a c++ matrix
    arr_ptr = make_unique<vector<vector<T> > >(N, vector<T>(M));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            (*arr_ptr)[i][j] = (*matrix_ptr)[i][j];
        }
    }
}


template<typename T, typename D>
py::array_t<T, py::array::c_style> convert_distance_matrix(unique_ptr<D> &matrix_ptr, vector<int> &node_shuffle_map,
                                                           const int N, const int new_N, T cuttoff = 100000,
                                                           T max_value = -1, T mask_value = -1) {
    // Convert a distance matrix [N, N] to a numpy array [new_N, new_N] by mapping node ids
    // indices are nodes, values are distances
    cout << "in convert_distance_matrix " << new_N << endl;
    auto arr = py::array_t<T, py::array::c_style>({new_N, new_N});
    cout << "arr shape: " << arr.shape(0) << " " << arr.shape(1) << endl;
    arr[py::make_tuple(py::ellipsis())] = mask_value; // initialize array
    auto ra = arr.mutable_unchecked();
    cout << "starting convert_distance_matrix" << endl;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (cuttoff > 0 && (*matrix_ptr)[i][j] >= cuttoff) {
                ra(node_shuffle_map[i], node_shuffle_map[j]) = max_value;
            } else {
                ra(node_shuffle_map[i], node_shuffle_map[j]) = (*matrix_ptr)[i][j];
            }
        }
    }
    cout << "out convert_distance_matrix" << endl;
    return arr;
}


template<typename T, typename D>
py::array_t<T, py::array::c_style> convert_ground_truths(unique_ptr<D> &matrix_ptr, vector<int> &node_shuffle_map,
                                                         const int E, const int N, const int new_N, T cuttoff = 100000,
                                                         T max_value = -1, T mask_value = -1) {
    // indices are nodes, values are distances
    // auto new_M = *max_element(node_shuffle_map.begin(), node_shuffle_map.end()) + 1;
    py::array_t<T, py::array::c_style> arr = py::array_t<T, py::array::c_style>({E, new_N});
    arr[py::make_tuple(py::ellipsis())] = mask_value; // initialize array
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


template<typename T>
py::array_t<T, py::array::c_style> convert_path(vector<T> &vec, vector<int> &node_shuffle_map) {
    int N = vec.size();
    py::array_t<T, py::array::c_style> arr = py::array_t<T, py::array::c_style>({N});
    auto ra = arr.mutable_unchecked();
    for (int i = 0; i < N; i++) {
        ra(i) = node_shuffle_map[vec[i]]; // value is mapped
    }
    return arr;
}

template<typename T>
pair<py::array_t<T, py::array::c_style>, py::array_t<T, py::array::c_style>> convert_center(pair<vector<T>, vector<T>> &q_c, vector<int> &node_shuffle_map) {
    // center is a pair of vectors, first is the query, second is the centroid
    auto q = q_c.first;
    auto c = q_c.second;;
    cout << q.size() << " " << c.size() << endl;
    auto arr_q = py::array_t<T, py::array::c_style>(q.size());
    auto ra_q = arr_q.mutable_unchecked();
    for (int i = 0; i < static_cast<int>(q.size()); i++) {
        ra_q(i) = node_shuffle_map[q[i]];
    }
    auto arr_c = py::array_t<T, py::array::c_style>(c.size());
    auto ra_c = arr_c.mutable_unchecked();
    for (int i = 0; i < static_cast<int>(c.size()); i++) {
        ra_c(i) = node_shuffle_map[c[i]];
    }
    return make_pair(arr_q, arr_c);
}

template<typename T>
py::array_t<T, py::array::c_style> convert_vector(vector<T> &vec) {
    int N = vec.size();
    py::array_t<T, py::array::c_style> arr = py::array_t<T, py::array::c_style>({N});
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

template<typename T>
py::array_t<T, py::array::c_style> batch_edge_list(const list<unique_ptr<vector<pair<int, int> > > > &batched_edge_list,
                                                   const list<unique_ptr<vector<int> > > &batched_node_shuffle_map,
                                                   int pad = -1) {
    int E = 0;
    for (auto &m: batched_edge_list) {
        if (static_cast<int>((*m).size()) > E) {
            E = (*m).size();
        }
    }
    py::array_t<T, py::array::c_style> arr({static_cast<int>(batched_edge_list.size()), E, 2});
    arr[py::make_tuple(py::ellipsis())] = static_cast<T>(pad); // initialize array
    auto ra = arr.mutable_unchecked();
    auto cur = 0;
    // parallel iterate over batched_edge_list and batched_node_shuffle_map
    auto it1 = batched_edge_list.begin();
    auto it2 = batched_node_shuffle_map.begin();
    for (; it1 != batched_edge_list.end() && it2 != batched_node_shuffle_map.end(); ++it1, ++it2) {
        // for (; it1 != batched_edge_list.end(); ++it1) {
        for (int j = 0; j < static_cast<int>((**it1).size()); j++) {
            ra(cur, j, 0) = (**it2)[(**it1)[j].first];
            ra(cur, j, 1) = (**it2)[(**it1)[j].second];
            // ra(cur, j, 0) = (*it1)[j].first;
            // ra(cur, j, 1) = (*it1)[j].second;
        }
        cur += 1;
    }
    return arr;
}

template<typename T>
py::array_t<T, py::array::c_style> batch_distances(const list<unique_ptr<vector<vector<T> > > > &batched_distances,
                                                   const list<unique_ptr<vector<int> > > &batched_node_shuffle_map,
                                                   const int new_N, T cuttoff = 100000, T max_value = -1, T pad = -1) {
    py::array_t<T, py::array::c_style> arr({static_cast<int>(batched_distances.size()), new_N, new_N});
    arr[py::make_tuple(py::ellipsis())] = static_cast<T>(pad); // initialize array
    auto ra = arr.mutable_unchecked();

    auto it1 = batched_distances.begin();
    auto it2 = batched_node_shuffle_map.begin();
    int cur = 0;
    for (; it1 != batched_distances.end() && it2 != batched_node_shuffle_map.end(); ++it1, ++it2) {
        for (int j = 0; j < static_cast<int>((**it1).size()); j++) {
            for (int k = 0; k < static_cast<int>((**it1)[j].size()); k++) {
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


template<typename T>
py::array_t<T, py::array::c_style> batch_ground_truths(
    const list<unique_ptr<vector<vector<T> > > > &batched_ground_truths,
    const list<unique_ptr<vector<int> > > &batched_node_shuffle_map,
    const int new_N, T cuttoff = 100000, T max_value = -1, T pad = -1) {
    // indices are nodes, values are distances
    auto max_E = 0;
    for (auto &m: batched_ground_truths) {
        if (static_cast<int>((*m).size()) > max_E) {
            max_E = (*m).size();
        }
    }

    py::array_t<T, py::array::c_style> arr({static_cast<int>(batched_ground_truths.size()), max_E, new_N});
    arr[py::make_tuple(py::ellipsis())] = static_cast<T>(pad); // initialize array
    auto ra = arr.mutable_unchecked();
    auto it1 = batched_ground_truths.begin();
    auto it2 = batched_node_shuffle_map.begin();
    int cur = 0;
    for (; it1 != batched_ground_truths.end() && it2 != batched_node_shuffle_map.end(); ++it1, ++it2) {
        for (int j = 0; j < static_cast<int>((**it1).size()); j++) {
            for (int k = 0; k < static_cast<int>((**it1)[j].size()); k++) {
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


template<typename T>
py::array_t<T, py::array::c_style> batch_paths(const list<unique_ptr<vector<int> > > &batched_paths,
                                               const list<unique_ptr<vector<int> > > &batched_node_shuffle_map,
                                               int pad = -1) {
    int N = 0;
    for (auto &m: batched_paths) {
        if (static_cast<int>((*m).size()) > N) {
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
        for (int j = 0; j < static_cast<int>((**it1).size()); j++) {
            ra(cur, j) = (**it2)[(**it1)[j]];
        }
        cur += 1;
    }
    return arr;
}

template<typename T>
py::array_t<T, py::array::c_style> batch_lengths(const list<int> &batched_lengths) {
    py::array_t<T, py::array::c_style> arr({static_cast<int>(batched_lengths.size())});
    auto ra = arr.mutable_unchecked();
    int cur = 0;
    for (auto &m: batched_lengths) {
        ra(cur) = m;
        cur += 1;
    }
    return arr;
}


template<typename T>
py::array_t<T, py::array::c_style> batch_positions(const list<unique_ptr<vector<vector<T> > > > &batched_positions,
                                                   const list<unique_ptr<vector<int> > > &batched_node_shuffle_map,
                                                   const int dim,
                                                   int pad = -1) {
    int N = 0;
    for (auto &m: batched_positions) {
        if (static_cast<int>((*m).size()) > N) {
            N = (*m).size();
        }
    }
    py::array_t<T, py::array::c_style> arr({static_cast<int>(batched_positions.size()), N, dim + 1});
    arr[py::make_tuple(py::ellipsis())] = static_cast<T>(pad); // initialize array
    auto ra = arr.mutable_unchecked();
    auto it1 = batched_positions.begin();
    auto it2 = batched_node_shuffle_map.begin();
    int cur = 0;
    for (; it1 != batched_positions.end() && it2 != batched_node_shuffle_map.end(); ++it1, ++it2) {
        for (int j = 0; j < static_cast<int>((**it1).size()); j++) {
            ra(cur, j, 0) = static_cast<T>((**it2)[j]); // map node id
            for (int d = 0; d < dim; d++) {
                // positions
                constexpr float r = 10000;
                ra(cur, j, d + 1) = ceil((**it1)[j][d] * r) / r;;
            }
        }
        cur += 1;
    }
    return arr;
}


// Hashing
// has each distance matrix as a string, return the hashes as a numpy array
template<typename T>
py::array_t<std::uint64_t, py::array::c_style> hash_distance_matrix(
    const py::array_t<T, py::array::c_style> &batched_distances) {
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


/* ************************************************
 *  Converting utils
 *  Batched graph generation for input into network
 *  ***********************************************/

inline vector<vector<int> >
label_smooth_path(const vector<vector<int> > &distances_ptr, vector<int> &path) {
    /* Return a vector of labels for each node in the path if they are alternative valid shortest paths
     * The labels for at labels[:][0] are just the original path
     */
    auto labels = vector<vector<int> >(path.size(), vector<int>());
    labels[0].push_back(path[0]); // start
    auto end = path[path.size() - 1];
    labels[path.size() - 1].push_back(end); // end
    for (int i = 1; i < static_cast<int>(path.size() - 1); i++) {
        auto prev = path[i - 1];
        auto path_node = path[i];
        labels[i].push_back(path_node); // add true path
        for (int j = 0; j < static_cast<int>(distances_ptr.size()); j++) {
            if (distances_ptr[prev][j] == 1 &&  j != path_node &&
                distances_ptr[j][end] == distances_ptr[path_node][end]) {
                labels[i].push_back(j);
                }
        }
    }
    return labels;
}


inline void add_arguments_to_dict(py::dict &d,
    const int attempts, const int max_attempts,
    const int min_vocab, const int max_vocab,
    const bool concat_edges,
    const bool query_at_end,
    const int num_thinking_tokens,
    const bool is_flat_model,
    const bool for_plotting) {

    d["num_attempts"] = attempts;
    d["vocab_min_size"] = min_vocab;
    d["vocab_max_size"] = max_vocab;
    if (attempts >= max_attempts) {
        d["success"] = false;
    } else {
        d["success"] = true;
    }
    d["concat_edges"] = concat_edges;
    d["query_at_end"] = query_at_end;
    d["num_thinking_tokens"] = num_thinking_tokens;
    d["is_flat_model"] = is_flat_model;
    d["for_plotting"] = for_plotting;
}


inline py::dict package_for_plotting(const int attempts, const int max_attempts,
                                     const int min_vocab, const int max_vocab,
                                     const list<unique_ptr<vector<int> > > &batched_node_shuffle_map,
                                     const list<unique_ptr<vector<pair<int, int> > > > &batched_edge_list,
                                     const list<int> &batched_edge_list_lengths,
                                     const list<unique_ptr<vector<vector<int> > > > &batched_distances,
                                     const list<unique_ptr<vector<vector<int> > > > &batched_ground_truths,
                                     const list<unique_ptr<vector<int> > > &batched_paths,
                                     const list<int> &batched_path_lengths
) {
    py::dict d;
    add_arguments_to_dict(d, attempts, max_attempts, min_vocab, max_vocab,
        false, false, 0, true, true);
    if (attempts >= max_attempts) {
        return d;
    }
    d["edge_list"] = batch_edge_list<int>(batched_edge_list, batched_node_shuffle_map);
    d["edge_list_lengths"] = batch_lengths<int>(batched_edge_list_lengths);
    auto bd = batch_distances<int>(batched_distances, batched_node_shuffle_map, max_vocab);
    d["distances"] = bd;
    d["hashes"] = hash_distance_matrix<int>(bd);
    d["ground_truths"] = batch_ground_truths<int>(batched_ground_truths, batched_node_shuffle_map, max_vocab);
    if (!batched_paths.empty()) {
        d["paths"] = batch_paths<int>(batched_paths, batched_node_shuffle_map);
        d["path_lengths"] = batch_lengths<int>(batched_path_lengths);
        // TODO label smoothing for paths
    }
    // todo center
    return d;
}


inline py::dict package_for_flat_model(const int attempts, const int max_attempts,
                                  const int min_vocab, const int max_vocab, map<std::string, int> &dictionary,
                                  const list<unique_ptr<vector<int> > > &batched_node_shuffle_map,
                                  const list<unique_ptr<vector<pair<int, int> > > > &batched_edge_list,
                                  const list<int> &batched_edge_list_lengths,
                                  const list<unique_ptr<vector<vector<int> > > > &batched_distances,
                                  const list<unique_ptr<vector<vector<int> > > > &batched_ground_truths,
                                  const list<unique_ptr<vector<int> > > &batched_paths,
                                  const list<int> &batched_path_lengths,
                                  const list<unique_ptr<pair<vector<int>, vector<int> > > > &batched_centers,
                                  const list<pair<int, int> > &batched_center_lengths,
                                  const bool concat_edges = true,
                                  const bool query_at_end = true,
                                  const int num_thinking_tokens = 0) {
    /*
     *  Package the data for a flat encoder- or decoder-only model i.e. same layers over src and tgt
     *   // or non-flat encoder-encoder and encoder-decoder model i.e. different layers over src and tgt
     *
     *  src_tokens [batch_size, seq_len, num_input_tokens]
     *  src_lengths [batch_size] in range [0, seq_len]
     *  prev_output_tokens [batch_size, task_len, max_k], i.e. targets, where k is for label smoothing
     *  graph_start_indices [batch_size]
     *  graph_length [batch_size]
     *  query_start_indices [batch_size]
     *  query_length [batch_size]
     *  task_start_indices [batch_size]
     *  task_length [batch_size]
     *  distances [batch_size, vocab_size, vocab_size]
     *  ground_truths [batch_size, num_edges, vocab_size], graph reconstruction
     *
     */

    py::dict d;
    add_arguments_to_dict(d, attempts, max_attempts, min_vocab, max_vocab,
        concat_edges, query_at_end, num_thinking_tokens, true, false);
    if (attempts >= max_attempts) {
        return d;
    }

    auto padding = dictionary["<pad>"];
    auto start_marker = dictionary["<s>"];
    auto end_marker = dictionary["</s>"];
    auto edge_marker = dictionary["|"];
    auto query_start_marker = dictionary["/"];
    auto query_end_marker = dictionary["?"];
    // auto task_1_marker = dictionary["t1"];
    // auto task_2_marker = dictionary["t2"];
    auto task_start_marker = dictionary["="];
    auto task_end_marker = dictionary["."];

    auto batch_size = static_cast<int>(batched_edge_list.size());

    vector<vector<int> > query;
    py::array_t<int, py::array::c_style> query_lengths(batch_size);
    py::array_t<int, py::array::c_style> task; // tgt-side ground-truths
    py::array_t<int, py::array::c_style> task_lengths(batch_size);
    //calculate lengths as  edge_len + query_length + num_thinking_tokens + task_length + 2 per instance in batch
    auto src_lengths = py::array_t<int, py::array::c_style>(batch_size);

    auto src_len_ra = src_lengths.mutable_unchecked();
    if (not batched_path_lengths.empty()) {
        if (not batched_centers.empty()) {  // can only do one task
            throw std::runtime_error("Cannot do both path and center tasks");
        }
        // The issue here is that we need to know the size of max label before making tensor
        // thus the smoothing values need to be calculated first
        auto batched_label_smoothing = vector<vector<vector<int> > >(batch_size);
        auto max_path_length = 0;
        auto max_k = 0;

        int cur = 0;
        auto it1 = batched_paths.begin();
        auto it2 = batched_distances.begin();
        for (; it1 != batched_paths.end() && it2 != batched_distances.end(); ++it1, ++it2) {
            auto labels = label_smooth_path(**it2, **it1);
            if (static_cast<int>(labels.size()) > max_path_length) {
                max_path_length = labels.size();
            }
            for (auto &m: labels) {
                if (static_cast<int>(m.size()) > max_k) {
                    max_k = m.size();
                }
            }
            batched_label_smoothing[cur] = labels;
            cur += 1;
        }

        task = py::array_t<int, py::array::c_style>({batch_size, max_path_length + 2, max_k});
        task[py::make_tuple(py::ellipsis())] = padding; // initialize array
        query_lengths[py::make_tuple(py::ellipsis())] = 4; // initialize array
        query = vector<vector<int> >(batch_size, vector<int>(4));
        auto ra = task.mutable_unchecked();
        auto ra_t_lengths = task_lengths.mutable_unchecked();

        cur = 0;
        auto it3 = batched_node_shuffle_map.begin();
        for (; it3 != batched_node_shuffle_map.end(); ++it3) {
            auto labels = batched_label_smoothing[cur];
            auto path_length = static_cast<int>(labels.size());
            query[cur][0] = query_start_marker;
            query[cur][1] = (**it3)[labels[0][0]]; // start node
            query[cur][2] = (**it3)[labels[path_length - 1][0]]; // end node
            query[cur][3] = query_end_marker;

            ra(cur, 0, 0) = task_start_marker;
            for (int j = 0; j < path_length; j++) {
                for (int k = 0; k < static_cast<int>(labels[j].size()); k++) {
                    ra(cur, j + 1, k) = (**it3)[labels[j][k]];
                }
            }
            ra(cur, path_length + 1, 0) = task_end_marker;
            ra_t_lengths(cur) = path_length + 2;
            src_len_ra(cur) = path_length + 2 + static_cast<int>(query[cur].size());
            cur += 1;
        }
    } else if (not batched_center_lengths.empty()) {
        // batched_center_lengths is a list of queries and tasks
        auto max_center_task_len = 0;
        for (auto &m: batched_center_lengths) {
            if (static_cast<int>(m.second) > max_center_task_len) {
                max_center_task_len = m.second;
            }
        }
        // only one token but has label smoothing, so 3 is special tokens
        task = py::array_t<int, py::array::c_style>({batch_size, 3, max_center_task_len + 2});
        task[py::make_tuple(py::ellipsis())] = padding; // initialize array
        task_lengths[py::make_tuple(py::ellipsis())] = 1; // initialize array
        query = vector<vector<int> >(batch_size);
        auto ra = task.mutable_unchecked();
        auto ra_q_lengths = query_lengths.mutable_unchecked();
        auto it1 = batched_centers.begin();
        auto it2 = batched_node_shuffle_map.begin();

        int cur = 0;
        for (; it1 != batched_centers.end() && it2 != batched_node_shuffle_map.end(); ++it1, ++it2) {
            // auto query_length = static_cast<int>((*it1)->first.size());
            auto task_length = static_cast<int>((*it1)->second.size());
            auto cur_query = (*it1)->first;
            cur_query.insert(cur_query.begin(), query_start_marker);
            cur_query.push_back(query_end_marker);
            query[cur] = cur_query;

            ra(cur, 0, 0) = task_start_marker;
            for (int j = 0; j < task_length; j++) {
                auto node = (*it1)->second[j];
                ra(cur, 1, j) = (**it2)[node];
            }
            ra(cur, 2, 0) = task_end_marker;
            ra_q_lengths(cur) = task_length + 2;
            src_len_ra(cur) = task_length + 2 + static_cast<int>(query[cur].size());
            cur += 1;
        }
    }

    // add edge sizes and thinking tokens to src_lengths
    auto it0 = batched_edge_list_lengths.begin();
    for (int b = 0; b < batch_size; b++) {
        if (concat_edges) {
            src_len_ra(b) += static_cast<int>(*it0) + num_thinking_tokens + 2;
        } else {
            src_len_ra(b) += static_cast<int>(*it0) * 3 + num_thinking_tokens + 2;
        }
        ++it0;
    }

    auto num_input_tokens = (concat_edges) ? 2 : 1;
    int max_seq_len = 0;
    for (int b = 0; b < batch_size; b++) {
        if (src_len_ra(b) > max_seq_len) {
            max_seq_len = src_len_ra(b);
        }
    }
    auto src_tokens = py::array_t<int, py::array::c_style>({batch_size, max_seq_len, num_input_tokens});
    py::array_t<int, py::array::c_style> query_start_indices;
    py::array_t<int, py::array::c_style> graph_start_indices;
    py::array_t<int, py::array::c_style> graph_lengths;
    py::array_t<int, py::array::c_style> task_start_indices;

    // tgt-side ground-truths
    src_tokens[py::make_tuple(py::ellipsis())] = padding; // initialize array
    auto ra = src_tokens.mutable_unchecked();

    for (auto b = 0; b < batch_size; b++) {
        ra(b, 0, 0) = start_marker;
        if (concat_edges) {  // non-graph tokens get duplicated
            ra(b, 0, 1) = start_marker;
        }
    }
    auto curs = vector<int>(batch_size, 1); // current position in src_tokens

    if (!query_at_end) {  // write in query before graph
        query_start_indices = py::array_t<int, py::array::c_style>(py::cast(curs));
        for (auto b = 0; b < batch_size; b++) {
            auto cur = curs[b];
            for (int j = 0; j < static_cast<int>(query[b].size()); j++) {
                ra(b, cur + j, 0) = query[b][j];
                if (concat_edges) {
                    ra(b, cur + j, 1) = query[b][j];
                }
                curs[b] += 1;
            }
        }
    }

    auto it1 = batched_edge_list.begin();
    auto it2 = batched_node_shuffle_map.begin();
    graph_start_indices = py::array_t<int, py::array::c_style>(py::cast(curs));
    if (concat_edges) {  // note: there are no graph markers
        graph_lengths = py::array_t<int, py::array::c_style>(py::cast(batched_edge_list_lengths));
        for (auto b = 0; b < batch_size; b++) {
            auto cur = curs[b];
            for (int j = 0; j < static_cast<int>((*it1)->size()); j++) {
                ra(b, cur + j, 0) = (**it2)[(*it1)->at(j).first];
                ra(b, cur + j, 1) =(**it2)[(*it1)->at(j).second];
                curs[b] += 1;

            }
            ++it1;
            ++it2;
        }
    } else {
        auto ra_gl = graph_lengths.mutable_unchecked();
        for (auto b = 0; b < batch_size; b++) {
            ra_gl[b] = static_cast<int>((*it1)->size()) * 3;
            auto cur = curs[b];
            for (int j = 0; j < static_cast<int>((*it1)->size()); j++) {
                ra(b, cur + j, 0) = (*it1)->at(j).first;
                ra(b, cur + j + 1, 0) = (*it1)->at(j).second;
                ra(b, cur + j + 2, 0) = edge_marker;
                curs[b] += 3;
            }
            ++it1;
            ++it2;
        }
    }

    if (query_at_end) {  // write in query if after graph
        query_start_indices = py::array_t<int, py::array::c_style>(py::cast(curs));
        for (auto b = 0; b < batch_size; b++) {
            auto cur = curs[b];
            for (int j = 0; j < static_cast<int>(query[b].size()); j++) {
                ra(b, cur + j, 0) = query[b][j];
                if (concat_edges) {
                    ra(b, cur + j, 1) = query[b][j];
                }
                curs[b] += 1;
            }
        }
    }

    if (num_thinking_tokens > 0) { // write in thinking tokens between query and task
        auto thinking_token = dictionary["!"];
        for (auto b = 0; b < batch_size; b++) {
            auto cur = curs[b];
            for (int j = 0; j < num_thinking_tokens; j++) {
                ra(b, cur + j, 0) = thinking_token;
                if (concat_edges) {
                    ra(b, cur + j, 1) = thinking_token;
                }
                curs[b] += 1;
            }
        }
    }

    // write in the task
    if (not batched_path_lengths.empty() || not batched_center_lengths.empty()) {
        task_start_indices = py::array_t<int, py::array::c_style>(py::cast(curs));
        for (auto b = 0; b < batch_size; b++) {
            auto cur = curs[b];
            for (int j = 0; j < task_lengths.at(b); j++) {  // includes special tokens already
                ra(b, cur + j, 0) = task.at(b, j, 0);
                if (concat_edges) {
                    ra(b, cur + j, 1) = task.at(b, j, 0);
                }
                curs[b] += 1;
            }
            ra(b, curs[b], 0) = end_marker; // end marker
            if (concat_edges) {
                ra(b, curs[b], 1) = end_marker; // end marker
            }
            // curs[b]++; // for correct length
        }
    }

    d["src_tokens"] = src_tokens;
    d["src_lengths"] = src_lengths;
    d["prev_output_tokens"] = task;
    d["task_lengths"] = task_lengths;

    d["query_start_indices"] = query_start_indices;
    d["query_lengths"] = query_lengths;
    d["graph_start_indices"] = graph_start_indices;
    d["graph_lengths"] = graph_lengths;
    d["task_start_indices"] = task_start_indices;

    auto bd = batch_distances<int>(batched_distances, batched_node_shuffle_map, max_vocab);
    d["distances"] = bd;
    d["hashes"] = hash_distance_matrix<int>(bd);
    d["ground_truths"] = batch_ground_truths<int>(batched_ground_truths, batched_node_shuffle_map, max_vocab);
    return d;
}

inline py::dict package_for_non_flat_model(const int attempts, const int max_attempts,
                                  const int min_vocab, const int max_vocab, map<std::string, int> &dictionary,
                                  const list<unique_ptr<vector<int> > > &batched_node_shuffle_map,
                                  const list<unique_ptr<vector<pair<int, int> > > > &batched_edge_list,
                                  const list<int> &batched_edge_list_lengths,
                                  const list<unique_ptr<vector<vector<int> > > > &batched_distances,
                                  const list<unique_ptr<vector<vector<int> > > > &batched_ground_truths,
                                  const list<unique_ptr<vector<int> > > &batched_paths,
                                  const list<int> &batched_path_lengths,
                                  const list<unique_ptr<pair<vector<int>, vector<int> > > > &batched_centers,
                                  const list<pair<int, int> > &batched_center_lengths,
                                  const bool concat_edges = true,
                                  const bool query_at_end = true,
                                  const int num_thinking_tokens = 0) {

    py::dict d;
    return d;
}


inline py::dict package_for_model() {
    py::dict d;
    return d;
}

inline py::dict package_for_model(const int attempts, const int max_attempts,
                                  const int min_vocab, const int max_vocab, map<std::string, int> &dictionary,
                                  const list<unique_ptr<vector<int> > > &batched_node_shuffle_map,
                                  const list<unique_ptr<vector<pair<int, int> > > > &batched_edge_list,
                                  const list<int> &batched_edge_list_lengths,
                                  const list<unique_ptr<vector<vector<int> > > > &batched_distances,
                                  const list<unique_ptr<vector<vector<int> > > > &batched_ground_truths,
                                  const list<unique_ptr<vector<int> > > &batched_paths,
                                  const list<int> &batched_path_lengths,
                                  const list<unique_ptr<pair<vector<int>, vector<int> > > > &batched_centers,
                                  const list<pair<int, int> > &batched_center_lengths,
                                  const bool concat_edges = true,
                                  const bool query_at_end = true,
                                  const int num_thinking_tokens = 0,
                                  const bool is_flat_model = true) {

    if (is_flat_model) {
        return package_for_flat_model(attempts, max_attempts, min_vocab, max_vocab, dictionary,
                                batched_node_shuffle_map, batched_edge_list, batched_edge_list_lengths,
                                batched_distances, batched_ground_truths, batched_paths, batched_path_lengths,
                                batched_centers, batched_center_lengths, concat_edges, query_at_end,
                                num_thinking_tokens);
    }
    return package_for_non_flat_model(attempts, max_attempts, min_vocab, max_vocab, dictionary,
                                batched_node_shuffle_map, batched_edge_list, batched_edge_list_lengths,
                                batched_distances, batched_ground_truths, batched_paths, batched_path_lengths,
                                batched_centers, batched_center_lengths, concat_edges, query_at_end,
                                num_thinking_tokens);
}


#endif //UTILS_H
