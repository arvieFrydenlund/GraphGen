//
// Created by arvie on 11/4/25.
//

#ifndef GRAPHGEN_TASKS_H
#define GRAPHGEN_TASKS_H

#include <iostream>
#include <random>
#include <queue>
#include <map>
#include "matrix.h"
#include "undirected_graphs.h"
#include "directed_graphs.h"
#include "scratch_pads.h"

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
    /*
     A task is split into a query and a task component,
     with the query being part of the prompt and the task being the expected output

     For graphs, targets are often multi-label,
     thus we separate the 1-D input and (potentially) 2-D targets
     */
    Matrix<int> tokenized_query_inputs;
    Matrix<int> tokenized_task_inputs;  // input part
    Matrix<int> tokenized_task_targets;  // defines multi-label targets
    Matrix<int> tokenized_query_pos;
    // tokenized_task_pos; these get made in the instance due to scratchpad being part of task

    bool use_query_invariance;

    virtual ~Task() = default;  // Polymorphic base classes should declare virtual destructors

    // use polymorphism for different task types when passing them around
    virtual void tokenize(const map<std::string, int> &dictionary,
                          const vector<int> &node_shuffle_map,
                          const map<std::string, int> pos_dictionary,
                          std::mt19937 &gen) {
        throw std::invalid_argument("Not implemented yet");
    };

    void set_tokenized_pos(const int query_size, const int task_size, const map<std::string, int> pos_dictionary){

        auto query_invariance_marker = pos_dictionary.at("query_invariance");
        auto query_start = pos_dictionary.at("query_start");
        auto query_end = pos_dictionary.at("query_end");
        auto task_start = pos_dictionary.at("task_start");
        auto task_end = pos_dictionary.at("task_end");

        if (query_size > query_end - query_start + 1) {
            throw std::invalid_argument("Query size exceeds available position tokens.");
        }
        if (task_size > task_end - task_start + 1) {
            throw std::invalid_argument("Task size exceeds available position tokens.");
        }

        tokenized_query_pos.resize(query_size);
        for (size_t i = 0; i < static_cast<size_t>(query_size); i++) {
            if (use_query_invariance) {
                tokenized_query_pos(i) = query_invariance_marker;
            } else {
                tokenized_query_pos(i) = query_start + static_cast<int>(i);
            }
        }
    }

};


class ShortestPathTask : public Task {
public:
    int start, end = -1;
    vector<int> path;
    vector<vector<int> > label_smoothed_path; // multi-labels for alternative valid paths from start to end
    int max_num_labels = 1;

    static pair<int, int> sample_start_end(std::mt19937 &gen,
                                    const unique_ptr<vector<vector<int> > > &distances_ptr,
                                    const int max_path_length, const int min_path_length,
                                    std::discrete_distribution<int> &d1, int start = -1) {

        pair<int, int> start_end;
        int attempts = 0;
        // could avoid while loop by making set of sets of paths and sampling that but may not as fast?
        while (true) {
            // sample a path of length between min_path_length and max_path_length
            auto sampled_path_length = d1(gen) + min_path_length;
            // get all paths of that length
            auto set_of_paths = vector<pair<int, int> >();
            if (start != -1) {  // known start
                for (int j = 0; j < static_cast<int>((*distances_ptr)[start].size()); j++) {
                    // + 1 because distance is path length - 1
                    if ((*distances_ptr)[start][j] + 1 == sampled_path_length) {
                        set_of_paths.push_back(make_pair(start, j));
                    }
                }
            } else {
                for (int i = 0; i < static_cast<int>((*distances_ptr).size()); i++) {
                    for (int j = 0; j < static_cast<int>((*distances_ptr)[i].size()); j++) {
                        // + 1 because distance is path length - 1
                        if ((*distances_ptr)[i][j] + 1 == sampled_path_length) {
                            set_of_paths.push_back(make_pair(i, j));
                        }
                    }
                }
            }
            if (!set_of_paths.empty()) { // sample a path from the set
                uniform_int_distribution<int> d2(0, static_cast<int>(set_of_paths.size()) - 1);
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
        return start_end;
    }

    ShortestPathTask(std::mt19937 &gen,
                     const unique_ptr<vector<vector<int> > > &distances_ptr,
                     const int max_path_length = 10, const int min_path_length = 3, int start = -1, int end = -1,
                     const optional<vector<float>> &task_sample_dist = nullopt,
                     const bool use_query_invariance = false, const bool sample_path = true) {
        /*
         * A) This is hardcoded for integer path lengths
         * Uniform sample paths of length between min_path_length and max_path_length
         * return length as vector of node ids
         * B) This is hardcoded for checking for distances of 1 as a connection
         */

        this->use_query_invariance = use_query_invariance;

        // define d1
        std::discrete_distribution<int> d1;
        if (!task_sample_dist.has_value()) {
            // + 1 for inclusive i.e (3, 70) = 3, 4, 5, 6, 7 = five possible lengths
            vector<float> uweights(max_path_length - min_path_length + 1, 1.0); // uniform distribution
            d1 = std::discrete_distribution<int>(uweights.begin(), uweights.end());
        } else {
            d1 = std::discrete_distribution<int>(task_sample_dist->begin(), task_sample_dist->end());
        }

        pair<int, int> start_end;
        if (start != -1 && end != -1) {
            start_end = make_pair(start, end);
        } else {
            start_end = sample_start_end(gen, distances_ptr, max_path_length, min_path_length, d1, start);
        }
        this->start = start_end.first;
        this->end = start_end.second;
        if (sample_path) {
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
            label_smooth_path(distances_ptr);  // get the smoothing values while we have distances
        }
    }

    void label_smooth_path(const unique_ptr<vector<vector<int> > > &distances_ptr) {
        /* Return a vector of labels for each node in the path if they are alternative valid shortest paths
         * The labels for at labels[:][0] are just the original path
         */
        label_smoothed_path = vector<vector<int> >(path.size(), vector<int>());
        label_smoothed_path[0].push_back(path[0]); // start
        auto end = path[path.size() - 1];
        label_smoothed_path[path.size() - 1].push_back(end); // end
        for (int i = 1; i < static_cast<int>(path.size() - 1); i++) {
            auto prev = path[i - 1];
            auto path_node = path[i];
            label_smoothed_path[i].push_back(path_node); // add true path
            for (int j = 0; j < static_cast<int>(distances_ptr->size()); j++) {
                if ((*distances_ptr)[prev][j] == 1 && j != path_node &&
                    (*distances_ptr)[j][end] == (*distances_ptr)[path_node][end]) {
                    label_smoothed_path[i].push_back(j);
                }
            }
            if (label_smoothed_path[i].size() > static_cast<size_t>(max_num_labels)) {
                max_num_labels = static_cast<int>(label_smoothed_path[i].size());
            }
        }
    }

    void set_path(const vector<int> &path, const unique_ptr<vector<vector<int> > > &distances_ptr, const bool use_label_smoothing=true){
        // some scratchpads may generate their own paths, and we need to respect that
        this->path = vector<int>(path);
        if (use_label_smoothing) {
            label_smooth_path(distances_ptr);
        } else {  // no label smoothing, just single labels
            label_smoothed_path = vector<vector<int> >(this->path.size(), vector<int>());
            for (size_t i = 0; i < this->path.size(); i++) {
                label_smoothed_path[i].push_back(this->path[i]);
            }
        }
    }

    void tokenize(
            const map<std::string, int> &dictionary,
            const vector<int> &node_shuffle_map,
            const map<std::string, int> pos_dictionary,
            std::mt19937 &gen) {

        /*  Ex. if the path is 0 -> 1 -> 2 -> 3 and there is an alternative path 0 -> 4 -> 2 -> 3
         * query: / 0 3 ?
         * task input: = 0 1 2 3 .
         * task target: = [0] [1,4] [2] [3] .
         */

        auto pad = dictionary.at("<pad>");
        auto query_start_marker = dictionary.at("/");
        auto query_end_marker = dictionary.at("?");
        auto task_start_marker = dictionary.at("=");
        auto task_end_marker = dictionary.at(".");

        auto query_size = 4; // q_start, start_node, end_node, q_end
        auto task_size = static_cast<int>(path.size()) + 2; // t_start, path nodes, t_end


        tokenized_query_inputs.resize(query_size);
        tokenized_task_inputs.resize(task_size);
        tokenized_task_targets.resize(task_size, max_num_labels, pad);

        // query tokenization
        tokenized_query_inputs(0) = query_start_marker;
        tokenized_query_inputs(1) = node_shuffle_map.at(path[0]); // start
        tokenized_query_inputs(2) = node_shuffle_map.at(path[path.size() - 1]); // end node
        tokenized_query_inputs(3) = query_end_marker;

        // task tokenization
        tokenized_task_inputs(0) = task_start_marker;
        tokenized_task_targets(0, 0) = task_start_marker;
        for (size_t i = 0; i < path.size(); i++) {  // shuffle map new path and label_smoothed
            tokenized_task_inputs(i + 1) = node_shuffle_map.at(path[i]);  // + 1 for start marker
            for (size_t j = 0; j < label_smoothed_path[i].size(); j++) {
                tokenized_task_targets(i + 1, j) = node_shuffle_map.at(label_smoothed_path[i][j]);
            }
        }
        tokenized_task_inputs(static_cast<int>(tokenized_task_inputs.shape()[0] - 1)) = task_end_marker;
        tokenized_task_targets(static_cast<int>(tokenized_task_targets.shape()[0] - 1), 0) = task_end_marker;
        set_tokenized_pos(query_size, task_size, pos_dictionary);
    }

    template<typename T>
    static int verify_path(py::array_t<T, py::array::c_style> &distances, vector<int> &path) {
        // -1 if not a valid path, 0 if valid path but not a shortest path, 1 if valid path but is a shortest path
        auto start = path[0];
        auto end = path[path.size() - 1];
        auto shortest_distance = distances.at(start, end);
        if (shortest_distance < 0) {
            return -1;
        }
        // validate path
        auto cur = start;
        for (int i = 1; i < static_cast<int>(path.size()); i++) {
            auto next = path[i];
            if (distances.at(cur, next) != 1) { // hardcoded for distance of 1
                return -1;
            }
            cur = next;
        }
        if (static_cast<int>(path.size()) > shortest_distance) {
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


class BFSTask : public Task {
    /*
     * The task is to generate the scratchpad-only
     */
public:
    int start, end = -1;
    vector<int> path;
    unique_ptr<BFSScratchPad> scratchpad;

    template<typename D>
    BFSTask(std::mt19937 &gen,
            const unique_ptr<Graph<D> > &g_ptr,
            const vector<int> &node_shuffle_map,  // needed if sorting adjacency lists
                     const unique_ptr<vector<vector<int> > > &distances_ptr,
                     const int max_path_length = 10, const int min_path_length = 3, int start = -1, int end = -1,
                     const optional<vector<float>> &task_sample_dist = nullopt,
                     const bool use_query_invariance = false,
            const bool sort_adjacency_lists = false,
            const bool use_unique_depth_markers = true) {

        // define d1
        std::discrete_distribution<int> d1;
        if (!task_sample_dist.has_value()) {
            // + 1 for inclusive i.e (3, 70) = 3, 4, 5, 6, 7 = five possible lengths
            vector<float> uweights(max_path_length - min_path_length + 1, 1.0); // uniform distribution
            d1 = std::discrete_distribution<int>(uweights.begin(), uweights.end());
        } else {
            d1 = std::discrete_distribution<int>(task_sample_dist->begin(), task_sample_dist->end());
        }

        auto start_end = ShortestPathTask::sample_start_end(gen, distances_ptr, max_path_length, min_path_length, d1, start);

        this->start = start_end.first;
        this->end = start_end.second;

        auto scratchpad = BFSScratchPad(this->start, this->end, g_ptr,
                                node_shuffle_map, sort_adjacency_lists, use_unique_depth_markers);
        this->scratchpad = make_unique<BFSScratchPad>(scratchpad );
        auto path(this->scratchpad->path);
        this->path = path;
    }


    void tokenize(
            const map<std::string, int> &dictionary,
            const vector<int> &node_shuffle_map,
            const map<std::string, int> pos_dictionary,
            std::mt19937 &gen) {

        scratchpad->tokenize(dictionary, node_shuffle_map, pos_dictionary, gen);
        // copy over tokenized values
        Matrix<int> tokenized_task_inputs(scratchpad->tokenized_inputs);  // input part
        Matrix<int> tokenized_task_targets(scratchpad->tokenized_targets);  // defines multi-label targets
        this->tokenized_task_inputs = tokenized_task_inputs;
        this->tokenized_task_targets = tokenized_task_targets;

        auto query_start_marker = dictionary.at("/");
        auto query_end_marker = dictionary.at("?");

        auto query_size = 4; // q_start, start_node, end_node, q_end
        auto task_size = static_cast<int>(path.size()) + 2; // t_start, path nodes, t_end
        tokenized_query_inputs.resize(query_size);

        // query tokenization
        tokenized_query_inputs(0) = query_start_marker;
        tokenized_query_inputs(1) = node_shuffle_map.at(path[0]); // start
        tokenized_query_inputs(2) = node_shuffle_map.at(path[path.size() - 1]); // end node
        tokenized_query_inputs(3) = query_end_marker;
        set_tokenized_pos(query_size, task_size, pos_dictionary);
    }
};


class CenterTask : public Task {
public:
    vector<int> new_query;
    vector<int> outputs;
    bool is_center = true;  // otherwise centroid
    bool use_query_invariance;

    CenterTask(std::mt19937 &gen,
               const unique_ptr<vector<vector<int> > > &distances_ptr,
               optional<vector<int>> &given_query,
               int max_query_size = -1, const int min_query_size = 2,
               const bool is_center = true, const bool use_query_invariance = false) {
        this->is_center = is_center;
        this->use_query_invariance = use_query_invariance;

        auto N = static_cast<int>(distances_ptr->size());
        if (max_query_size == -1 || max_query_size > N) {
            max_query_size = N;
        }
        new_query = vector<int>();
        if (given_query.has_value()) {
            //sample query
            // stackoverflow.com/questions/33802205/how-to-sample-without-replacement-using-c-uniform-int-distribution
            uniform_int_distribution<int> d1(min_query_size, max_query_size);
            auto query_length = d1(gen);
            auto nodes = std::vector<int>(N);
            std::iota(nodes.begin(), nodes.end(), 0);
            sample(nodes.begin(), nodes.end(), std::back_inserter(new_query), query_length, gen);
            // std::ranges::shuffle(new_query, gen);  // so that nodes are out of order, this doesn't matter with permute
        } else {
            // copy over elements from given query
            for (auto i: given_query.value()) {
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
            if (is_center) {  // get max of d
                values[v] = *std::max_element(d.begin(), d.end());
            } else {  // get average
                values[v] = static_cast<float>(std::accumulate(d.begin(), d.end(), 0.0)); // / Q);  avoid float div
            }
        }

        outputs = vector<int>();
        auto min_value = *std::min_element(values.begin(), values.end());
        for (int i = 0; i < N; i++) {
            if (float_equality(values[i], min_value)) {
                outputs.push_back(i);
            }
        }
        if (outputs.empty()) {
            // this will be due to float equality and means the function needs to be reimplemented better
            auto s = "Center/centroid error no outputs found for min value " + std::to_string(min_value);
            throw std::invalid_argument(s);
        }

    }

    bool float_equality(double a, double b) {
        return std::fabs(a - b) < std::numeric_limits<double>::epsilon();
    }

    void tokenize(
            const map<std::string, int> &dictionary,
            const vector<int> &node_shuffle_map,
            const map<std::string, int> pos_dictionary,
            std::mt19937 &gen) {

        /* Ex if the query is 0,1,2 and the center is 3,4
         * query : / 0 1 2 ?
         * task input: = 3 4 .
         * task target: = [3 4] [4] .
         */

        auto query_start_marker = dictionary.at("/");
        auto query_end_marker = dictionary.at("?");
        auto task_start_marker = dictionary.at("=");
        auto task_end_marker = dictionary.at(".");

        auto query_size = static_cast<int>(new_query.size()) + 2; // q_start, num_nodes q_end
        auto task_size = static_cast<int>(outputs.size()) + 2; // t_start, path nodes, t_end

        tokenized_query_inputs.resize(query_size);
        tokenized_task_inputs.resize(task_size);
        tokenized_task_targets.resize(task_size, static_cast<int>(outputs.size()), dictionary.at("<pad>"));

        // query tokenization
        tokenized_query_inputs(0) = query_start_marker;
        for (size_t i = 0; i < new_query.size(); i++) {
            tokenized_query_inputs(i + 1) = node_shuffle_map.at(new_query[i]);
        }
        tokenized_query_inputs(tokenized_query_inputs.shape()[0] - 1) = query_end_marker;

        // task tokenization
        tokenized_task_inputs(0) = task_start_marker;
        tokenized_task_targets(0, 0) = task_start_marker;
        for (size_t i = 0; i < outputs.size(); i++) {
            tokenized_task_inputs(i + 1) = node_shuffle_map.at(outputs[i]);
            for (size_t j = i; j < outputs.size(); j++) {  // label smoothing outputs since they have no order
                tokenized_task_targets(i + 1, j) = node_shuffle_map.at(outputs[j]);
            }
        }
        tokenized_task_inputs(tokenized_task_inputs.shape()[0] - 1) = task_end_marker;
        tokenized_task_targets(tokenized_task_targets.shape()[0] - 1, 0) = task_end_marker;

        set_tokenized_pos(query_size, task_size, pos_dictionary);
    }

};


class KHopsTask : public Task {
public:
    int sampled_k;
    int max_tokens;
    bool is_generative;

    KHopsTask() {
    }


};

#endif //GRAPHGEN_TASKS_H
