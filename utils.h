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
    // auto new_M = *max_element(node_shuffle_map.begin(), node_shuffle_map.end()) + 1;
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
                                                   const list<unique_ptr<vector<int>>> &batched_node_shuffle_map,
                                                   int pad = -1) {
    int E = 0;
    for (auto &m : batched_edge_list) {
        if (static_cast<int>((*m).size()) > E) {
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

template <typename T>
py::array_t<T, py::array::c_style> batch_distances(const list<unique_ptr<vector<vector<T>>>> &batched_distances,
                                                   const list<unique_ptr<vector<int>>> &batched_node_shuffle_map,
                                                   const int new_N, T cuttoff = 100000, T max_value = -1, T pad = -1) {

    py::array_t<T, py::array::c_style> arr({static_cast<int>(batched_distances.size()), new_N, new_N});
    arr[py::make_tuple(py::ellipsis())] = static_cast<T>(pad);  // initialize array
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


template <typename T>
py::array_t<T, py::array::c_style> batch_ground_truths(const list<unique_ptr<vector<vector<T>>>> &batched_ground_truths,
                                                       const list<unique_ptr<vector<int>>> &batched_node_shuffle_map,
                                                       const int new_N, T cuttoff = 100000, T max_value = -1, T pad = -1) {
    // indices are nodes, values are distances
    auto max_E = 0;
    for (auto &m : batched_ground_truths) {
        if (static_cast<int>((*m).size()) > max_E) {
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


template <typename T>
py::array_t<T, py::array::c_style> batch_paths(const list<unique_ptr<vector<int>>> &batched_paths,
                                               const list<unique_ptr<vector<int>>> &batched_node_shuffle_map,
                                               int pad = -1) {

    int N = 0;
    for (auto &m : batched_paths) {
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
                                                   const list<unique_ptr<vector<int>>> &batched_node_shuffle_map,
                                                   const int dim,
                                                   int pad = -1) {
    int N = 0;
    for (auto &m : batched_positions) {
        if (static_cast<int>((*m).size()) > N) {
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
        for (int j = 0; j < static_cast<int>((**it1).size()); j++) {
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


/* ************************************************
 *  Converting utils
 *  Batched graph generation for input into network
 *  ***********************************************/

inline py::dict package_for_plotting(const int attempts, const int max_attempts,
                                     const int min_vocab, const int max_vocab,
    								 const list<unique_ptr<vector<int>>> &batched_node_shuffle_map,
                                     const list<unique_ptr<vector<pair<int, int>>>> &batched_edge_list,
                                     const list<int> &batched_edge_list_lengths,
                                     const list<unique_ptr<vector<vector<int>>>> &batched_distances,
                                     const list<unique_ptr<vector<vector<int>>>> &batched_ground_truths,
                                     const list<unique_ptr<vector<int>>> &batched_paths,
                                     const list<int> &batched_path_lengths
                                     ){
	py::dict d;
    d["num_attempts"] = attempts;
    d["vocab_min_size"] = min_vocab;
    d["vocab_max_size"] = max_vocab;
    if ( attempts >= max_attempts ) {
        return d;
    }
    d["edge_list"] = batch_edge_list<int>(batched_edge_list, batched_node_shuffle_map);
    d["edge_list_lengths"] = batch_lengths<int>(batched_edge_list_lengths);
    auto bd = batch_distances<int>(batched_distances, batched_node_shuffle_map, max_vocab);
    d["distances"] = bd;
    d["hashes"] = hash_distance_matrix<int>(bd);
    d["ground_truths"] = batch_ground_truths<int>(batched_ground_truths, batched_node_shuffle_map, max_vocab);
    if ( ! batched_paths.empty() ) {
        d["paths"] = batch_paths<int>(batched_paths, batched_node_shuffle_map);
        d["path_lengths"] = batch_lengths<int>(batched_path_lengths);
    }
    return d;

}



inline py::dict package_for_model(const int attempts, const int max_attempts,
                                  const int min_vocab, const int max_vocab, map<std::string, int> &dictionary,
    							  const list<unique_ptr<vector<int>>> &batched_node_shuffle_map,
                                  const list<unique_ptr<vector<pair<int, int>>>> &batched_edge_list,
                                  const list<int> &batched_edge_list_lengths,
                                  const list<unique_ptr<vector<vector<int>>>> &batched_distances,
                                  const list<unique_ptr<vector<vector<int>>>> &batched_ground_truths,
                                  const list<unique_ptr<vector<int>>> &batched_paths,
                                  const list<int> &batched_path_lengths,
                                  const list<unique_ptr<pair<vector<int>, vector<int>>>> &batched_centers,
                                  const list<pair<int, int>> &batched_center_lengths,
                                  const bool is_flat_model = true,
                                  const bool concat_edges = true,
                                  const bool query_at_end = true,
                                  const int num_thinking_tokens = 0){
  /*
   *  Package the data for either a flat encoder- or decoder-only model i.e. same layers over src and tgt
   *  or non-flat encoder-encoder and encoder-decoder model i.e. different layers over src and tgt
   *
   *  src_tokens [batch_size, seq_len, num_input_tokens]
   *  src_lengths [batch_size] in range [0, seq_len]
   *  prev_output_tokens [batch_size, task_len, max_k], where k is fore label smooth, i.e. task_inputs and tagets
   *  src_ground_truths [batch_size, num_edges, vocab_size]
   *  graph_start_index [batch_size]
   *  graph_length [batch_size]
   *  query_start_index [batch_size]
   *  query_length [batch_size]
   *  task_start_index [batch_size]
   *  task_length [batch_size]
   *  distances [batch_size, vocab_size, vocab_size]
   *
   * dictionary symbols:
   * '<s>', '<pad>', '</s>', '<unk>', '|', '!', '=', '.',
   * 't1', 't2', 't3', 't4', 't5',
   * '/', '?', '@', '#',
   * 's1', 's2', 's3', 's4', 's5',
   * '0', '1', '2', ....
   */

    auto padding = static_cast<int>(dictionary["<pad>"]);
    auto start_marker = static_cast<int>(dictionary["<s>"]);
    auto end_marker = static_cast<int>(dictionary["</s>"]);
    auto edge_marker = static_cast<int>(dictionary["|"]);
    auto query_start_marker = dictionary["/"];
	auto query_end_marker = dictionary["?"];
    auto thinking_token = dictionary["!"];
    auto task_start_marker = dictionary["="];
    auto task_end_marker = dictionary["."];


	auto batch_size = static_cast<int>(batched_edge_list.size());

  	vector<vector<int>> query;
    py::array_t<int, py::array::c_style> query_lengths(batch_size);
    py::array_t<int, py::array::c_style> task;  // tgt-side groud-truths
    py::array_t<int, py::array::c_style> task_lengths(batch_size);

  	if ( not batched_path_lengths.empty() ) {
          auto max_path_length = *max_element(batched_path_lengths.begin(), batched_path_lengths.end());
          task = py::array_t<int, py::array::c_style>({batch_size, max_path_length + 2, 1});
          task[py::make_tuple(py::ellipsis())] = padding;  // initialize array
          query_lengths[py::make_tuple(py::ellipsis())] = 4;  // initialize array
          query = vector<vector<int>>(batch_size, vector<int>(4));
          auto ra = task.mutable_unchecked();
          auto ra_t_lengths = task_lengths.mutable_unchecked();
    	  auto it1 = batched_paths.begin();
    	  auto it2 = batched_node_shuffle_map.begin();

          int cur = 0;
    	  for (; it1 != batched_paths.end() && it2 != batched_node_shuffle_map.end(); ++it1, ++it2) {
             auto path_length = static_cast<int>((**it1).size());
             query[cur][0] = query_start_marker;
             query[cur][1] = (**it2)[(**it1)[0]];  // startnode
             query[cur][2] = (**it2)[(**it1)[path_length - 1]];  // end node
             query[cur][3] = query_end_marker;

             ra(cur, 0, 0) = task_start_marker;
             for (int j = 0; j < path_length; j++) {
          		ra(cur, j + 1, 0) = (**it2)[(**it1)[j]];
          	 }
             ra(cur, path_length + 1, 0) = task_end_marker;
             ra_t_lengths(cur) = path_length + 2;
          	 cur += 1;
          }
  	} else if ( not batched_center_lengths.empty() ) {
          // batched_center_lengths is a list of queries and tasks
          auto max_center_task_len = 0;
          for (auto &m : batched_center_lengths) {
              if (static_cast<int>(m.second) > max_center_task_len) {
                  max_center_task_len = m.second;
              }
          }
          // only one token but has label smoothing, so 3 is special tokens
          task = py::array_t<int, py::array::c_style>({batch_size, 3, max_center_task_len});
  		  task[py::make_tuple(py::ellipsis())] = padding;  // initialize array
          task_lengths[py::make_tuple(py::ellipsis())] = 1;  // initialize array
          query = vector<vector<int>>(batch_size);
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
              cur += 1;
          }
  	}

    // src_tokens [batch_size, seq_len, num_input_tokens] if concat_edges num_input_tokens = 2, else 1
    auto max_query_length = 0;
    for (auto &m : query) {
        if (static_cast<int>(m.size()) > max_query_length) {
            max_query_length = static_cast<int>(m.size());
        }
    }
    auto max_task_length = 0;
    auto E = static_cast<int>(*max_element(batched_edge_list_lengths.begin(), batched_edge_list_lengths.end()));
    int num_input_tokens = (concat_edges) ? 2 : 1;
    int E_len = (concat_edges) ? E : E * 3;
    int seq_len = E_len + max_query_length + num_thinking_tokens + max_task_length + 2; // 2 for start and end markers
    auto src_tokens = py::array_t<int, py::array::c_style>({batch_size, seq_len, num_input_tokens});  // tgt-side groud-truths
    src_tokens[py::make_tuple(py::ellipsis())] = padding;  // initialize array
    auto ra = src_tokens.mutable_unchecked();

    for ( auto b = 0; b < batch_size; b++ ) {
        ra(b, 0, 0) = task_start_marker;
    }
    auto curs = vector<int>(batch_size, 1); // current position in src_tokens

    if ( ! query_at_end ) { // write in query if at start
      for ( auto b = 0; b < batch_size; b++ ) {
        auto cur = curs[b];
        for ( int j = 0; j < static_cast<int>(query[b].size()); j++ ) {
            ra(b, cur, 0) = query[b][j];
            curs[b] += 1;
        }
      }
    }



	py::dict d;
    return d;

}




#endif //UTILS_H
