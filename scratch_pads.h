//
// Created by arvie on 28/10/25.
//

#ifndef SCRATCH_PADS_H
#define SCRATCH_PADS_H

#include <iostream>
#include <random>
#include <queue>
#include <map>
#include "undirected_graphs.h"
#include "directed_graphs.h"

using namespace std;

class ScratchPad {
public:
    vector<int> tokenized_inputs;
    vector<vector<int> > tokenized_targets;

    virtual void tokenize(const map<std::string, int> &dictionary,
                          const vector<int> &node_shuffle_map,
                          std::mt19937 &gen) {
        throw std::invalid_argument("Not implemented yet");
    };

    pair<vector<int>, vector<vector<int> > > get_tokenized() {
        return make_pair(this->tokenized_inputs, this->tokenized_targets);
    }
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


    template<typename D>
    BFSScratchPad(const int start, const int end,
                  const unique_ptr<Graph<D> > &g_ptr,
                  const bool sort_adjacency_lists = false,
                  const bool use_unique_depth_markers = true
    ) {
        // constructor, fill in levels
        this->sort_adjacency_lists = sort_adjacency_lists;
        this->use_unique_depth_markers = use_unique_depth_markers;

        bool is_finished = false; // don't stop at finding target but do entire level with target in it
        map<int, bool> visited;
        visited[start] = true;
        queue<int> q;
        q.push(start);

        while (!q.empty()) {
            map<int, vector<int> > current_level_nodes;
            vector<int> next_level_nodes;
            while (!q.empty()) {
                // process the current level
                vector<int> cur_neighbors;
                auto cur = q.front();
                q.pop();
                // get all adjacency nodes not in visited
                auto neighbors = boost::adjacent_vertices(cur, *g_ptr);
                for (auto nbr = neighbors.first; nbr != neighbors.second; ++nbr) {
                    if (static_cast<int>(*nbr) == end) {
                        is_finished = true;
                    }
                    if (visited.find(*nbr) == visited.end()) {
                        cur_neighbors.push_back(*nbr);
                        visited[*nbr] = true;
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
            if (!current_level_nodes.empty()) {
                levels.push_back(current_level_nodes);
            }
            if (!is_finished) {
                if (!next_level_nodes.empty()) {
                    for (auto n: next_level_nodes) {
                        q.push(n);
                        if (n == end) {
                            is_finished = true;
                        }
                    }
                }
            }
        }
    }

    static void _add(vector<int> &inputs,
              vector<vector<int> > &targets,
              const string &e,
              const map<std::string, int> &dictionary) {
        inputs.push_back(dictionary.at(e));
        targets.push_back(vector<int>{dictionary.at(e)});
    }

    static void _add(vector<int> &inputs,
              vector<vector<int> > &targets,
              const int e) {
        inputs.push_back(e);
        targets.push_back(vector<int>{e});
    }

    static void _add(vector<int> &inputs,
              vector<vector<int> > &targets,
              const int e,
              const vector<int> &ts,
              const int up_to) {
        inputs.push_back(e);
        // push back targets until up_to
        auto nt = vector<int>{};
        for (size_t i = 0; i < static_cast<size_t>(up_to); i++) {
            nt.push_back(ts[i]);
        }
        targets.push_back(nt);
    }

    void tokenize(
            const map<std::string, int> &dictionary,
            const vector<int> &node_shuffle_map,
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

        // note that order of neighbours [v1, v2] creates the order for the next level
        // so only shuffle neighbour order here, if sort, sort instead of shuffle

        // map all nodes, and either shuffle or sort adjacency lists
        vector<map<int, vector<int> > > new_levels; // copy to modify
        for (size_t i = 0; i < levels.size(); i++) {
            map<int, vector<int> > new_level;
            for (auto &p: levels[i]) {
                auto nbrs = p.second;
                vector<int> mapped_nbrs;
                for (size_t j = 0; j < nbrs.size(); j++) {
                    mapped_nbrs.push_back(node_shuffle_map.at(nbrs[j]));
                }
                if (sort_adjacency_lists) {
                    sort(mapped_nbrs.begin(), mapped_nbrs.end());
                } else {
                    std::shuffle(mapped_nbrs.begin(), mapped_nbrs.end(), gen);
                }
                new_level[node_shuffle_map.at(p.first)] = mapped_nbrs;
            }
            new_levels.push_back(new_level);
        }

        auto inputs = vector<int>{};
        auto targets = vector<vector<int> >{};
        // Get the initial adjacency list order
        // start node as only item in new_levels[0]
        auto start_node = new_levels[0].begin()->first;

        auto prior_adj_orders = vector<vector<int> >{vector<int>{start_node}};
        for (size_t i = 0; i < levels.size(); i++) {
            // process level i
            // other example
            // D0 14 [13 17 ] D1 13 [10 ] 17 [8 11 20 ] D2 10 [9 ] 8 [] 11 [7 ] 20 [12 18 ] D3 9 [0 ] 7 [] 12 [5 ] 18 [1 ] D4 0 [19 ] 5 [2 ] 1 [3 4 15 ]
            // prior_adj_orders = [[14]], then [[13, 17]], then [[10], [8, 11, 20]], or
            // prior_adj_orders = [[14]], then [[17, 13]], then [[8, 11, 20], [10]]
            // and [8, 11, 20] can be shuffled any way

            // add depth marker
            string marker = "D";
            if (use_unique_depth_markers) {
                marker = "D" + to_string(i);;
            }
            _add(inputs, targets, marker, dictionary);

            // process each node in level
            auto next_adj_orders = vector<vector<int> >{};
            for (size_t j = 0; j < prior_adj_orders.size(); j++) {
                auto next_order = vector<int>{};
                for (size_t k = 0; k < prior_adj_orders[j].size(); k++) {
                    auto cur = prior_adj_orders[j][k];
                    _add(inputs, targets, cur);
                    _add(inputs, targets, "[", dictionary);
                    auto neighbors = new_levels[i][cur];
                }
            }
        }

        this->tokenized_inputs = inputs;
        this->tokenized_targets = targets;
    }

    /*
    void pprint_levels() {
        // basically the tokenization, for debugging
        cout << "Num levels: " << levels.size() << endl;
        for (size_t i = 0; i < levels.size(); i++) {
            cout << "D" << i << ": ";
            for (size_t j = 0; j < levels[i].size(); j++) {
                cout << "" << levels[i][j].first << " [";
                for (size_t k = 0; k < levels[i][j].second.size(); k++) {
                    cout << levels[i][j].second[k] << " ";
                }
                cout << "] ";
            }
            // cout << endl;
        }
        cout << endl;
    }
    */

};


#endif //SCRATCH_PADS_H
