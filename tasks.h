//
// Created by arvie on 11/4/25.
//

#ifndef GRAPHGEN_TASKS_H
#define GRAPHGEN_TASKS_H

#include <iostream>
#include <random>
#include <queue>
#include <map>
#include "undirected_graphs.h"
#include "directed_graphs.h"

#include <Python.h>
#include <chrono>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

using namespace std;

namespace py = pybind11;
using namespace py::literals;

class Task {
public:
    vector<int> tokenized_query_inputs;
    vector<int> tokenized_task_inputs;
    vector<vector<int> > tokenized_task_targets;

    void tokenize(const bool use_unique_depth_markers = true) {
        throw std::invalid_argument("Not implemented yet");
    };

    int get_max_num_labels() {
        // max size of tokenized_task_targets
        int max_labels = 0;
        for (size_t i = 0; i < tokenized_task_targets.size(); i++) {
            if (static_cast<int>(tokenized_task_targets[i].size()) > max_labels) {
                max_labels = static_cast<int>(tokenized_task_targets[i].size());
            }
        }
        return max_labels;
    }
};


class ShortestPathTask : public Task {

    vector<int> path;
    vector<vector<int> > label_smoothed_path;

    ShortestPathTask(std::mt19937 &gen,
                 const unique_ptr<vector<vector<int> > > &distances_ptr,
                 const int max_path_length = 10, const int min_path_length = 1, int start = -1, int end = -1,
                 const vector<float> &task_sample_dist = vector<float>()) {
        /*
         * This is hardcoded for integer path lengths
         * Uniform sample paths of length between min_path_length and max_path_length
         * return length as vector of node ids
         * This is hardcoded for checking for distances of 1 as a connection
         */

        // define d1
        std::discrete_distribution<int> d1;
        if (task_sample_dist.empty()) {
            // + 1 for inclusive i.e (3, 70) = 3, 4, 5, 6, 7 = five possible lengths
            vector<float> uweights(max_path_length - min_path_length + 1, 1.0); // uniform distribution
            d1 = std::discrete_distribution<int>(uweights.begin(), uweights.end());
        } else {
            d1 = std::discrete_distribution<int>(task_sample_dist.begin(), task_sample_dist.end());
        }

        pair<int, int> start_end;
        if (start != -1 && end != -1) {
            start_end = make_pair(start, end);
        } else {
            int attempts = 0;
            // could avoid while loop by making set of sets of paths and sampling that but may not as fast?
            while (true) {
                // sample a path of length between min_path_length and max_path_length
                auto sampled_path_length = d1(gen) + min_path_length;
                // get all paths of that length
                auto set_of_paths = vector<pair<int, int> >();
                if (start != -1 && end == -1) {
                    // known start
                    for (int j = 0; j < static_cast<int>((*distances_ptr)[start].size()); j++) {
                        // +1 because distance is path length - 1
                        if ((*distances_ptr)[start][j] + 1 == sampled_path_length) {
                            set_of_paths.push_back(make_pair(start, j));
                        }
                    }
                } else {
                    for (int i = 0; i < static_cast<int>((*distances_ptr).size()); i++) {
                        for (int j = 0; j < static_cast<int>((*distances_ptr)[i].size()); j++) {
                            if ((*distances_ptr)[i][j] + 1 == sampled_path_length) {
                                set_of_paths.push_back(make_pair(i, j));
                            }
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
                    while (true) {
                        uniform_int_distribution<int> d3(0, (*distances_ptr).size() - 1);
                        auto i = d3(gen);
                        auto j = d3(gen);
                        if ((*distances_ptr)[i][j] < inf && (*distances_ptr)[i][j] > 0) {
                            start_end = make_pair(i, j);
                            break;
                        }
                        attempts += 1;
                        if (attempts > 1000) {
                            throw std::invalid_argument(
                                "Could not find a path in 1000 attempts.  This should never happen.");
                        }
                    }
                    break;
                }
            }
        }
        // reconstruct path
        path.push_back(start_end.first);
        int cur = start_end.first;
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
        label_smooth_path(distances_ptr);
    }

    void label_smooth_path(const unique_ptr<vector<vector<int> > > &distances_ptr) {
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
            for (int j = 0; j < static_cast<int>(distances_ptr->size()); j++) {
                if ((*distances_ptr)[prev][j] == 1 && j != path_node &&
                    (*distances_ptr)[j][end] == (*distances_ptr)[path_node][end]) {
                    labels[i].push_back(j);
                }
            }
        }
        this->label_smoothed_path = labels;
    }

    void tokenize(
        const map<std::string, int> &dictionary,
        const unique_ptr<vector<int> > &node_shuffle_map,
        std::mt19937& gen) {
        auto query_start_marker = dictionary["/"];
        auto query_end_marker = dictionary["?"];
        auto task_start_marker = dictionary["="];
        auto task_end_marker = dictionary["."];

        //  query tokenization
        tokenized_query_inputs.push_back(query_end_marker);
        tokenized_query_inputs.push_back(node_shuffle_map->at(path[0])); // start
        tokenized_query_inputs.push_back(node_shuffle_map->at(path[path.size() - 1])); // end node
        tokenized_query_inputs.push_back(query_end_marker);

        //  task tokenization
        tokenized_task_inputs.push_back(task_start_marker);
        tokenized_task_targets.push_back(vector<int>({task_start_marker}));
        // shuffle map new path and label_smoothed
        for (size_t i = 0; i < path.size(); i++) {
            tokenized_task_inputs.push_back(node_shuffle_map->at(path[i]));
            auto labels = vector<int>();
            for (size_t j = 0; j < label_smoothed_path[i].size(); j++) {
                labels.push_back(node_shuffle_map->at(label_smoothed_path[i][j]));
            }
            tokenized_task_targets.push_back(labels);
        }
        tokenized_task_inputs.push_back(task_end_marker);
        tokenized_task_targets.push_back(vector<int>({task_end_marker}));
    }


    template<typename T>
    static int varify_path(py::array_t<T, py::array::c_style> &distances, vector<int> &path) {
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
    static py::array_t<int, py::array::c_style> verify_paths(py::array_t<T, py::array::c_style> &distances,
                                                             py::array_t<T, py::array::c_style> &queries,
                                                             py::array_t<T, py::array::c_style> &paths,
                                                             py::array_t<T, py::array::c_style> &lengths) {
        // batch version [batch_size, vocab_size, vocab_size]
        auto batch_size = paths.shape(0);
        auto out = py::array_t<int, py::array::c_style>(static_cast<int>(batch_size));
        out[py::make_tuple(py::ellipsis())] = 1; // initialize array to true
        auto ra = out.mutable_unchecked();
        for (auto b = 0; b < batch_size; b++) {
            auto start = queries.at(b, 0); //paths.at(b, 0);
            auto end = queries.at(b, 1); //paths.at(b, lengths.at(b) - 1);
            // check start and end of path match query
            if (start != paths.at(b, 0) || end != paths.at(b, lengths.at(b) - 1)) {
                ra(b) = -1;
                continue;
            }
            // check there exists a path between start and end
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
};


class CenterTask : public Task {
    vector<int> path;
    vector<vector<int> > label_smoothed_path;

    CenterTask (std::mt19937 &gen,
                 const unique_ptr<vector<vector<int> > > &distances_ptr,
                 const int max_path_length = 10, const int min_path_length = 1, int start = -1, int end = -1,
                 const vector<float> &task_sample_dist = vector<float>()) {
    }

#endif //GRAPHGEN_TASKS_H
