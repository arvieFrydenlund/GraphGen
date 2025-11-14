//
// Created by arvie on 28/10/25.
//

#ifndef SCRATCH_PADS_H
#define SCRATCH_PADS_H

#include <iostream>
#include <random>
#include <queue>
#include <map>
#include "matrix.h"
#include "undirected_graphs.h"
#include "directed_graphs.h"


using namespace std;

class ScratchPad {
public:
    Matrix<int> tokenized_inputs;
    Matrix<int> tokenized_targets;
    // Matrix<int> tokenized_pos;  these can not exist since they are really part of the task

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

    bool sort_adjacency_lists = false;
    bool use_unique_depth_markers = true;
    vector<int> path;

    template<typename D>
    BFSScratchPad(int start, int end,
                  const unique_ptr<Graph<D> > &g_ptr,
                  const vector<int> &node_shuffle_map,  // needed if sorting adjacency lists
                  const bool sort_adjacency_lists = false,
                  const bool use_unique_depth_markers = true
    ) {

        this->sort_adjacency_lists = sort_adjacency_lists;
        this->use_unique_depth_markers = use_unique_depth_markers;

        auto reverse_node_shuffle_map = vector<int>(node_shuffle_map.size(), -1);
        if (sort_adjacency_lists) {
            for (size_t i = 0; i < node_shuffle_map.size(); i++) {
                reverse_node_shuffle_map[node_shuffle_map[i]] = static_cast<int>(i);
            }
        }

        auto visited = map<int, bool>();
        visited[start] = true;
        auto q = queue<int>();
        q.push(start);
        bool not_found = true;

        while (not_found) {
            if (q.empty()) {
                throw std::invalid_argument("BFS ScratchPad: could not find end node from start node");
            }
            auto current_level_nodes = map<int, vector<int> >();
            auto next_level_nodes = vector<int>();
            while (!q.empty()) {  // process the current level
                vector<int> cur_neighbors;
                auto cur = q.front();
                q.pop();
                // get all adjacency nodes not in visited
                auto neighbors_boost = boost::adjacent_vertices(cur, *g_ptr);
                auto neighbors = vector<int>();  // convert to vector
                for (auto nbr = neighbors_boost.first; nbr != neighbors_boost.second; ++nbr) {
                    neighbors.push_back(*nbr);
                }
                if (sort_adjacency_lists){ // otherwise they are shuffled randomly on account of the map
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
                for (auto nbr = neighbors.begin(); nbr != neighbors.end(); ++nbr) {
                    if (visited.find(*nbr) == visited.end()) {
                        cur_neighbors.push_back(*nbr);
                        visited[*nbr] = true;
                        if (*nbr == end) {
                            not_found = false;
                        }
                    }
                }
                //if (cur_neighbors.size() == 0) { // put this here if you want to prevent empty neighbor lists
                //    continue;  // no new neighbors
                //}
                current_level_nodes[cur] = cur_neighbors;
                for (auto n: cur_neighbors) {
                    next_level_nodes.push_back(n);
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

        int num_tokens = 1;  // start of scratchpad
        int max_targets = 1;
        for (size_t i = 0; i < levels.size(); i++) {
            num_tokens += 1; // depth marker
            for (auto &p: levels[i]) {
                auto nbrs = p.second;
                num_tokens += 1 + 2 + nbrs.size(); //start of adjacency list, and node and end of adjacency list
                if (nbrs.size() > static_cast<size_t>(max_targets)) {
                    max_targets = static_cast<int>(nbrs.size());
                }
            }
        }

        tokenized_inputs = Matrix<int>(num_tokens, 1);
        tokenized_targets = Matrix<int>(num_tokens, max_targets, dictionary.at("<pad>"));
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
            for (size_t j = 0; j < prior_adj_orders.size(); j++) {  // process each node in level
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
                    for (size_t t = k; t < nbrs.size(); t++) {
                        tokenized_targets(cur, t - k) = node_shuffle_map[nbrs[t]];
                    }
                    next_adj_orders.push_back(nbrs[k]);
                }
                tokenized_inputs(cur, 0) = dictionary.at("]");
                tokenized_targets(cur, 0) = dictionary.at("]");
                cur++;
            }
            prior_adj_orders = next_adj_orders;
        }
    }

};

// todo DFS

// node list BFS -- complete node list in BFS order (and then what order after target is found?)


#endif //SCRATCH_PADS_H
