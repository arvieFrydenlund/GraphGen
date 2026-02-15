//
// Created by arvie on 28/10/25.
//

#ifndef SCRATCH_PADS_H
#define SCRATCH_PADS_H

#include <iostream>
#include <random>
#include <queue>
#include <stack>
#include <map>
#include <unordered_set>
#include "matrix.h"
#include "undirected_graphs.h"
#include "directed_graphs.h"
#include "dictionaries.h"

using namespace std;

class ScratchPad {
public:
    Matrix<int> tokenized_inputs;
    Matrix<int> tokenized_targets;
    // Matrix<int> tokenized_pos;  these can not exist since they are really part of the task

    virtual ~ScratchPad() = default;  // Polymorphic base classes should declare virtual destructors

    virtual void tokenize(const map<std::string, int> &dictionary,
                          const vector<int> &node_shuffle_map,
                          const map<std::string, int> pos_dictionary,
                          std::mt19937 &gen) {
        throw std::invalid_argument("Not implemented yet");
    };

};


class BFSScratchPad : public ScratchPad {
public:
    vector<map<int, vector<int> > > levels;
    /*
    *  Have a vector for each level of the BFS
    *  these then need to be a vmap of a node to its adjacency list
    *  Then  I can random shuffle the adjacency lists within each pair
    *  or sort them by id to get a deterministic order (this only works if I apply the node_shuffle_map first)
    */

    bool stop_once_found = true;
    bool use_unique_depth_markers = true;
    vector<int> path;

    template<typename D>
    BFSScratchPad(int start, int end,
                  const unique_ptr<Graph<D> > &g_ptr,
                  const vector<pair<int, int> > &edge_list,  // sort by edge list order to avoid semantics leak
                  const bool use_unique_depth_markers = true
    ) {
        this->use_unique_depth_markers = use_unique_depth_markers;

        auto node_order = vector<int>( static_cast<int>(boost::num_vertices(*g_ptr)), -1);
        // for each edge look at both nodes and add them to the node order if not already there,
        // this is the order they will be seen in the adjacency lists
        for (size_t i = 0; i < edge_list.size(); i++) {
            auto u = edge_list[i].first;
            auto v = edge_list[i].second;
            if (node_order[u] == -1) {
                node_order[u] = static_cast<int>(i);
            }
            if (node_order[v] == -1) {
                node_order[v] = static_cast<int>(i);
            }
        }

        auto visited = map<int, bool>();
        visited[start] = true;
        auto q = queue<int>();
        q.push(start);
        bool found = false;
        while (!found) {
            if (q.empty()) {
                throw std::invalid_argument("BFS ScratchPad: could not find end node from start node");
            }
            auto current_level_nodes = map<int, vector<int> >();
            auto next_level_nodes = vector<int>();
            // process beyond end node found or stop once found
            auto should_stop = q.empty();
            while (!should_stop){ // process the current level
                vector<int> cur_neighbors;
                auto cur = q.front();
                q.pop();
                // get all adjacency nodes not in visited
                auto neighbors_boost = boost::adjacent_vertices(cur, *g_ptr);
                auto neighbors = vector<int>();  // convert to vector
                for (auto nbr = neighbors_boost.first; nbr != neighbors_boost.second; ++nbr) {
                    neighbors.push_back(*nbr);
                }
                // sort neighbors by node order to get the order
                std::sort(neighbors.begin(), neighbors.end(), [&node_order](const int &a, const int &b) {
                    return node_order[a] < node_order[b];
                });
                for (auto nbr = neighbors.begin(); nbr != neighbors.end(); ++nbr) {
                    if (visited.find(*nbr) == visited.end()) {
                        cur_neighbors.push_back(*nbr);
                        visited[*nbr] = true;
                        if (*nbr == end) {
                            found = true;
                        }
                    }
                }
                //if (cur_neighbors.size() == 0) { // put this here if you want to prevent empty neighbor lists
                //    continue;  // no new neighbors, i.e. no '[ ]' in scratchpad output
                //}
                current_level_nodes[cur] = cur_neighbors;
                for (auto n: cur_neighbors) {
                    next_level_nodes.push_back(n);
                }
                should_stop = q.empty();
                if (stop_once_found and found) {
                    should_stop = true;
                }
            }
            levels.push_back(current_level_nodes);  // this should always be non empty
            for (auto n: next_level_nodes) {
                q.push(n);
            }
        }

        // reconstruct path, backwards to be consistent with khops
        path.push_back(end);
        // for each level backwards
        for (int i = static_cast<int>(levels.size()) - 1; i >= 0; i--) {
            // find which node in level i has end as neighbor
            for (const auto &pair: levels[i]) {
                auto node = pair.first;
                auto nbrs = pair.second;
                if (std::find(nbrs.begin(), nbrs.end(), path.back()) != nbrs.end()) {
                    path.push_back(node);
                    break;
                }
            }
        }
        std::reverse(path.begin(), path.end());
    }

    void tokenize(
            const map<std::string, int> &dictionary,
            const vector<int> &node_shuffle_map,
            const map<std::string, int> pos_dictionary,
            std::mt19937 &gen
    ) {
        // tokenize the BFS levels into a single sequence
        // pair(inputs, targets)
        // where targets can be multiple tokens due to label smoothing over order, ex.
        // D0: 20 [6 7 10 ] D1: 6 [8 9 18 ] 7 [3 15 ] 10 [1 4 ]
        // [6, 7, 10] can be in any order, so targets are [6, 7, 10], then [7, 10], then [10]
        // Note that at D1, 6 must be first since it is first in the adjacency list of D0's first node
        // real tokenization is D0 20 [6 7 10 ] D1 6 [8 9 18 ] 7 [3 15 ] 10 [1 4 ] if using unique depth markers
        // else, D 20 [6 7 10 ] D 6 [8 9 18 ] 7 [3 15 ] 10 [1 4 ]
        // note this has changed by forcing sort byt edge list

        int num_tokens = 1;  // start of scratchpad
        int max_targets = 1;  // if we want to do label smoothing over the adjacency list
        for (size_t i = 0; i < levels.size(); i++) {
            num_tokens += 1; // depth marker
            for (auto &p: levels[i]) {
                auto nbrs = p.second;
                num_tokens += 1 + 2 + nbrs.size(); //head node + both [ ] + nbrs
                if (nbrs.size() > static_cast<size_t>(max_targets)) {
                    max_targets = static_cast<int>(nbrs.size());
                }
            }
        }
        tokenized_inputs = Matrix<int>(num_tokens, 1);
        // tokenized_targets = Matrix<int>(num_tokens, max_targets, dictionary.at("<pad>"));  // if we want to do label smoothing over the adjacency list
        tokenized_targets = Matrix<int>(num_tokens, 1, dictionary.at("<pad>"));
        int cur = 0;
        tokenized_inputs(cur, 0) = dictionary.at("#");
        tokenized_targets(cur, 0) = dictionary.at("#");
        cur += 1;

        // Get the initial adjacency list order
        // start node as only item in new_levels[0]
        auto prior_adj_orders = vector<int>{levels[0].begin()->first};  // start node
        for (size_t i = 0; i < levels.size(); i++) {
            string marker = "D";
            if (use_unique_depth_markers) {
                marker = "D" + to_string(i);;
            }
            tokenized_inputs(cur, 0) = dictionary.at(marker);
            tokenized_targets(cur, 0) = dictionary.at(marker);
            cur += 1;

            auto next_adj_orders = vector<int>{};
            // cur < num_tokens needed for when stop_once_found is true
            for (size_t j = 0; j < prior_adj_orders.size() and cur < num_tokens; j++) {  // process each node in level
                auto node = prior_adj_orders[j];
                auto nbrs = levels[i][node];
                // write in node
                tokenized_inputs(cur, 0) = node_shuffle_map[node];
                tokenized_targets(cur, 0) = node_shuffle_map[node];
                cur++;  // and node's adjacency list
                tokenized_inputs(cur, 0) = dictionary.at("[");
                tokenized_targets(cur, 0) = dictionary.at("[");
                cur++;
                for (size_t k = 0; k < nbrs.size(); k++, cur++) {
                    tokenized_inputs(cur, 0) = node_shuffle_map[nbrs[k]];
                    // if we want to do label smoothing over the adjacency list
                    // for (size_t t = k; t < nbrs.size(); t++) {
                    //     tokenized_targets(cur, t - k) = node_shuffle_map[nbrs[t]];
                    // }
                    tokenized_targets(cur, 0) = node_shuffle_map[nbrs[k]];
                    next_adj_orders.push_back(nbrs[k]);
                }
                tokenized_inputs(cur, 0) = dictionary.at("]");
                tokenized_targets(cur, 0) = dictionary.at("]");
                cur++;
            }
            prior_adj_orders = next_adj_orders;
        }
    }

    static bool depth_check(const int id, const int cut_depth, const map<int, std::string> reverse_dict){
        const auto &pred = reverse_dict.at(id);
        const auto &extra_after_symbol = get_dictionary_extra_after_symbol();
        if (pred == extra_after_symbol || pred == extra_after_symbol + std::to_string(cut_depth)){
            return true;
        }
        return false;
    }

    template<typename T>
    static int verify_bfs_gen(const py::array_t<T, py::array::c_style> &distances,
                              const int start, const int end, const vector<int> &gen,
                              const bool check_special_tokens = true) {
        /* This should work regardless of order of adjacency list
         *
         * -1 if special tokens are wrong, 0 if not a valid BFS gen, 1 if valid BFS gen
         *
         * D0    7     [     24    20    ]     D1    24    [     16    ]     20    [     5     ]
         * D2    16    [     ]     5     [     1     15    ]     D3    1     [     14    ]     15    [     6     ]
         * D4    14    [     12    ]     6     [     18    ]     D5    12    [     23    ]     18    [     3     ]
         * D6    23    [     10    ]     3     [     ]     D7    10    [     2     ]     D8    2     [     11    ]
         * D9    11    [     17    8     ]     D10   17    [     9     22    ]     8     [     ]
         */

        map<int, std::string> reverse_dict;
        int adjacency_start_idx = -1;
        int adjacency_end_idx = -1;
        if (check_special_tokens){
            auto dict = get_dictionary();
            for (const auto &item : dict) {
                reverse_dict[item.second] = item.first;
            }
            adjacency_start_idx = dict.at("[");
            adjacency_end_idx =  dict.at("]");
        }

        auto cur_depth = 0;
        auto cur_q = queue<int>();
        cur_q.push(start);
        auto next_q = queue<int>();
        bool found = false;

        int cur = 0;
        while (cur < static_cast<int>(gen.size()) and !found and !(cur_q.empty() and next_q.empty())) {
            if (check_special_tokens and !depth_check(gen[cur], cur_depth, reverse_dict)) {
                return -4;  // depth is incorrect
            }
            cur++;
            while (cur < static_cast<int>(gen.size()) and !found and !cur_q.empty()) {
                auto cur_node = cur_q.front();
                cur_q.pop();
                if (cur < static_cast<int>(gen.size()) and gen[cur] != cur_node) {
                    return -2;  // adjacency head is incorrect
                }
                cur++;
                if (cur < static_cast<int>(gen.size()) and check_special_tokens and gen[cur] != adjacency_start_idx) {
                    return -3;  // adjacent start is incorrect
                }
                cur++;
                while (cur < static_cast<int>(gen.size()) and gen[cur] != adjacency_end_idx) {
                    auto nbr = gen[cur];
                    if (cur_node >= distances.shape(0) || nbr >= distances.shape(1) ||
                        distances.at(cur_node, nbr) != 1) {
                        return -1;  // distance is not adjacent
                    }
                    next_q.push(nbr);
                    if (nbr == end) {
                        found = true;
                    }
                    cur++;
                }
                if (cur < static_cast<int>(gen.size()) and check_special_tokens and gen[cur] != adjacency_end_idx) {
                    return -3;  // adjacency end is incorrect
                }
                cur++;
            }
            cur_depth++;
            std::swap(cur_q, next_q);
        }
        return static_cast<int>(found);  // 0 not found, 1 found
    }

    template<typename T>
    static py::array_t<int, py::array::c_style> verify_bfs_gens(py::array_t<T, py::array::c_style> &distances,
                                                                py::array_t<T, py::array::c_style> &queries,
                                                                py::array_t<T, py::array::c_style> &gens,
                                                                py::array_t<T, py::array::c_style> &lengths,
                                                                const bool check_special_tokens = true) {
        auto batch_size = gens.shape(0);
        auto out = py::array_t<int, py::array::c_style>(static_cast<int>(batch_size));
        out[py::make_tuple(py::ellipsis())] = -5; // initialize array
        auto ra = out.mutable_unchecked();
        for (auto b = 0; b < batch_size; b++) {
            auto start = queries.at(b, 0); //paths.at(b, 0);
            auto end = queries.at(b, 1); //paths.at(b, lengths.at(b) - 1);
            auto gen = vector<int>();
            for (int i = 0; i < lengths.at(b); i++) {
                gen.push_back(gens.at(b, i));
            }
            // get distance slice
            auto distances_slice = py::array_t<T, py::array::c_style>({distances.shape(1), distances.shape(2)});
            auto rd = distances_slice.mutable_unchecked();
            auto bd = distances.unchecked();
            for (int i = 0; i < distances.shape(1); i++) {
                for (int j = 0; j < distances.shape(2); j++) {
                    rd(i, j) = bd(b, i, j);
                }
            }
            auto res = verify_bfs_gen(distances_slice, start, end, gen, check_special_tokens);
            ra(b) = res;
        }
        return out;
    }

};

class DFSScratchPad : public ScratchPad {
    /*
     * Note that DFS does not guarantee getting a shortest path, only a path.
     * Be careful with use_unique_depth_markers because of this, since you may need a lot of extra markers
     *
     *
     * There are two versions: non-cheating and cheating version,
     * the latter means that no multi-hop reasoning needs to be done
     */
public:

    bool sort_adjacency_lists = false;
    bool use_unique_depth_markers = true;
    bool is_partial_redundant = false;  // copy over choice node, with modified adjacency list
    bool is_full_redundant = false;  // copy over full DFS prefix at each choice node
    vector<int> path;

    vector<tuple<int, int, int, vector<int> > > dfs_steps; // (current_node, parent_node, depth, neighbors)

    template<typename D>
    bool _dfs_helper(const unique_ptr<Graph<D> > &g_ptr, map<int, bool> &visted, int cur_node, int end, int cur_level,
                     const vector<int> &node_shuffle_map, const vector<int> &reverse_node_shuffle_map) {
        visted[cur_node] = true;
        // get all adjacency nodes not in visited
        auto neighbors_boost = boost::adjacent_vertices(cur_node, *g_ptr);
        auto neighbors = vector<int>();  // convert to vector
        for (auto nbr = neighbors_boost.first; nbr != neighbors_boost.second; ++nbr) {
            neighbors.push_back(*nbr);
        }
        if (this->sort_adjacency_lists) { // otherwise they are shuffled randomly on account of the map
            // map them to shuffled ids and sort, then map back, so silly  :(
            vector<int> mapped_neighbors(neighbors.size());
            for (size_t i = 0; i < neighbors.size(); i++) {
                mapped_neighbors[i] = node_shuffle_map[neighbors[i]];
            }
            sort(mapped_neighbors.begin(), mapped_neighbors.end());
            for (size_t i = 0; i < mapped_neighbors.size(); i++) {
                neighbors[i] = reverse_node_shuffle_map[mapped_neighbors[i]];
            }
        }

        auto unseen_neighbors = vector<int>();
        for (auto n: neighbors) {
            if (visted.find(n) == visted.end()) {
                unseen_neighbors.push_back(n);
            }
        }
        for (int i = 0; i < static_cast<int>(unseen_neighbors.size()); i++) {
            auto nbr = unseen_neighbors[i];
            auto others = vector<int>();  // for label smoothing
            for (int j = i; j < static_cast<int>(unseen_neighbors.size()); j++) {
                others.push_back(unseen_neighbors[j]);
            }
            dfs_steps.push_back(make_tuple(nbr, cur_node, cur_level, others));
            if (nbr == end) {
                return true;
            }
            auto is_found = _dfs_helper(g_ptr, visted, nbr, end, cur_level + 1,
                                        node_shuffle_map, reverse_node_shuffle_map);
            if (is_found) {
                return true;
            }
        }
        return false;
    }

    template<typename D>
    DFSScratchPad(int start, int end,
                  const unique_ptr<Graph<D> > &g_ptr,
                  const vector<int> &node_shuffle_map,  // needed if sorting adjacency lists
                  const bool sort_adjacency_lists = false,
                  const bool use_unique_depth_markers = true,
                  const bool is_partial_redundant = false,
                  const bool is_full_redundant = false
    ) {

        this->sort_adjacency_lists = sort_adjacency_lists;
        this->use_unique_depth_markers = use_unique_depth_markers;
        this->is_partial_redundant = is_partial_redundant;
        this->is_full_redundant = is_full_redundant;

        auto reverse_node_shuffle_map = vector<int>(node_shuffle_map.size(), -1);
        if (sort_adjacency_lists) {
            for (size_t i = 0; i < node_shuffle_map.size(); i++) {
                reverse_node_shuffle_map[node_shuffle_map[i]] = static_cast<int>(i);
            }
        }

        auto visited = map<int, bool>();
        _dfs_helper(g_ptr, visited, start, end, 0, node_shuffle_map, reverse_node_shuffle_map);

        // reconstruct path, backwards to be consistent with khops -- could just do this in the helper, whatever
        path.push_back(end);
        for (int i = static_cast<int>(dfs_steps.size()) - 1; i >= 0; i--) {
            auto step = dfs_steps[i];
            auto cur_node = get<0>(step);
            if (cur_node == path.back()) {
                auto parent = get<1>(step);
                path.push_back(parent);
            }
        }
        std::reverse(path.begin(), path.end());
    }

    void tokenize(
            const map<std::string, int> &dictionary,
            const vector<int> &node_shuffle_map,
            const map<std::string, int> pos_dictionary,
            std::mt19937 &gen
    ) {
        // tokenize the BFS levels into a single sequence
        // where targets can be multiple tokens due to label smoothing over order, ex.
        if (this->use_unique_depth_markers) {
            if (!is_valid_extra_dictionary_symbol(static_cast<int>(path.size()))) {
                throw std::invalid_argument(
                        "DFS ScratchPad: use_unique_depth_markers is true but there are not enough unique depth markers in the dictionary for the max depth of the DFS path");
            }
        }

        int num_tokens = 1;  // start of scratchpad
        int max_targets = 1;
        for (size_t i = 0; i < dfs_steps.size(); i++) {
            auto step = dfs_steps[i];
            auto nbrs = get<3>(step);
            num_tokens += 3; // marker, node, and neighbor
            if (nbrs.size() > static_cast<size_t>(max_targets)) {
                max_targets = static_cast<int>(nbrs.size());
            }
        }

        tokenized_inputs = Matrix<int>(num_tokens, 1);
        tokenized_targets = Matrix<int>(num_tokens, max_targets, dictionary.at("<pad>"));
        int cur = 0;
        tokenized_inputs(cur, 0) = dictionary.at("#");
        tokenized_targets(cur, 0) = dictionary.at("#");
        cur += 1;

        for (size_t i = 0; i < dfs_steps.size(); i++) {
            num_tokens += 1; // depth marker
            auto step = dfs_steps[i];
            auto cur_node = get<0>(step);
            auto parent = get<1>(step);
            auto depth = get<2>(step);
            auto nbrs = get<3>(step);
            string marker = "D";
            if (use_unique_depth_markers) {
                marker = "D" + to_string(depth);;
            }
            tokenized_inputs(cur, 0) = dictionary.at(marker);
            tokenized_targets(cur, 0) = dictionary.at(marker);
            cur += 1;
            tokenized_inputs(cur, 0) = node_shuffle_map[parent];
            tokenized_targets(cur, 0) = node_shuffle_map[parent];
            cur += 1;
            tokenized_inputs(cur, 0) = node_shuffle_map[cur_node];
            for (size_t t = 0; t < nbrs.size(); t++) {
                tokenized_targets(cur, t) = node_shuffle_map[nbrs[t]];
            }
            cur += 1;
        }
    }

    template<typename T>
    static int verify_dfs_gen(const py::array_t<T, py::array::c_style> &distances,
                              const int start, const int end, const vector<int> &gen,
                              const bool check_special_tokens = true,
                              const bool is_cheat = false) {
        return -1;
    }

};

// TODO bredth-first ordering (full graph) as a distance (u, v, dist_id), this fully encodes distance matrix as SP
// always can evaluate task given SP, then SP generation and task generation.  First says if SP resolves task (if SP can be generated)
// this is tree-traversal with (breadth-first) level-ordering, and to do this we need to output N full tree traversals so N^2 nodes

#endif //SCRATCH_PADS_H
