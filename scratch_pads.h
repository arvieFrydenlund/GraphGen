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
    vector<int> path;

    bool sort_adjacency_lists = false;
    bool use_unique_depth_markers = true;


    template<typename D>
    BFSScratchPad(vector<int> &path,
                  const unique_ptr<Graph<D> > &g_ptr,
                  const bool sort_adjacency_lists = false,
                  const bool use_unique_depth_markers = true
    ) {
        // constructor, fill in levels
        this->path = path;
        this->sort_adjacency_lists = sort_adjacency_lists;
        this->use_unique_depth_markers = use_unique_depth_markers;

        const int start = path.front(); // get task-specific info before casting to base class
        const int end = path.back();

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

        // note that order of neighbours [v1, v2] creates the order for the next level
        // so only shuffle neighbour order here, if sort, sort instead of shuffle

        // map all nodes, and either shuffle or sort adjacency lists

        // when there are multiple valid paths, we randomly choose one path, and randomly choose one BFS order
        // However to be content with khops we should make it that the path chosen int he khops one.
        // Solution:  just make sure path sorts to end
        vector<map<int, vector<int> > > new_levels; // copy to modify
        int num_tokens = 1;  // start of scratchpad
        int max_targets = 1;

        // print length of path and length of levels
        cout << "BFS ScratchPad: path length = " << path.size() << ", levels = " << levels.size() << endl;

        // length of levels should be 1 less than path (since target is not included)

        for (size_t i = 0; i < levels.size(); i++) {
            map<int, vector<int> > new_level;
            num_tokens += 1; // depth marker
            for (auto &p: levels[i]) {
                auto nbrs = p.second;
                num_tokens += 1 + 2 + nbrs.size(); //start of adjacency list, and node and end of adjacency list
                if (nbrs.size() > static_cast<size_t>(max_targets)) {
                    max_targets = static_cast<int>(nbrs.size());
                }
                vector<int> mapped_nbrs;
                for (size_t j = 0; j < nbrs.size(); j++) {
                    mapped_nbrs.push_back(node_shuffle_map.at(nbrs[j]));
                }
                if (sort_adjacency_lists) {
                    sort(mapped_nbrs.begin(), mapped_nbrs.end());
                } else {
                    std::shuffle(mapped_nbrs.begin(), mapped_nbrs.end(), gen);
                }
                // force path node to be last in its adjacency list


                new_level[node_shuffle_map.at(p.first)] = mapped_nbrs;
            }
            new_levels.push_back(new_level);
        }

        tokenized_inputs = Matrix<int>(num_tokens, 1);
        tokenized_targets = Matrix<int>(num_tokens, max_targets, dictionary.at("<pad>"));
        int cur = 0;
        tokenized_inputs(cur, 0) = dictionary.at("#");
        tokenized_targets(cur, 0) = dictionary.at("#");
        cur += 1;

        // Get the initial adjacency list order
        // start node as only item in new_levels[0]
        auto prior_adj_orders = vector<int>{new_levels[0].begin()->first};  // start node
        for (size_t i = 0; i < new_levels.size(); i++) {
            // process level i
            // other example
            // D0 14 [13 17 ] D1 13 [10 ] 17 [8 11 20 ] D2 10 [9 ] 8 [] 11 [7 ] 20 [12 18 ] D3 9 [0 ] 7 [] 12 [5 ] 18 [1 ] D4 0 [19 ] 5 [2 ] 1 [3 4 15 ]
            // prior_adj_orders = [14], then [13, 17], then [10, 8, 11, 20], or
            // and [8, 11, 20] can be shuffled any way

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
                auto nbrs = new_levels[i][node];
                // write in node and its adjacency list
                tokenized_inputs(cur, 0) = node;
                tokenized_targets(cur, 0) = node;
                cur++;
                tokenized_inputs(cur, 0) = dictionary.at("[");
                tokenized_targets(cur, 0) = dictionary.at("[");
                cur++;
                for (size_t k = 0; k < nbrs.size(); k++, cur++) {
                    tokenized_inputs(cur, 0) = nbrs[k];
                    for (size_t t = k; t < nbrs.size(); t++) {
                        tokenized_targets(cur, t - k) = nbrs[t];
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
