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

class ScratchPad { // For easy passing of common variables to different scratch pad types
public:
  pair<vector<int>, vector<vector<int>>> tokenize(const bool use_unique_depth_markers = true) {
    // targets, but input sequence is shifted by one
   throw std::invalid_argument("Not implemented yet");
  };
};


class BFSScratchPad : public ScratchPad {
public:
       vector<vector<pair<int, vector<int>>>> levels;
       /*
       *  Have a vector for each level of the BFS
       *  these then need to be a vector of pairs of vectors:  [(4, [8, 6]), (5, [7])].
       *  Then  I can random shuffle the pairs and the adjacency lists within each pair
       *  or sort them by id to get a deterministic order (this only works if I apply the node_shuffle_map first)
       */
       vector<int> tokenized_inputs;
       vector<vector<int>> tokenized_targets;


       	template<typename D>
        BFSScratchPad( const int start, const int end,
           const unique_ptr<Graph<D> > &g_ptr) {
                // constructor, fill in levels

            bool is_finished = false;  // don't stop at finding target but do entire level with target in it
            map<int, bool> visited;
       	    visited[start] = true;
            queue<int> q;
            q.push(start);

            while (!q.empty()){
                vector<pair<int, vector<int>>> current_level_nodes;
                vector<int> next_level_nodes;
                while (!q.empty()) { // process the current level
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
                    pair<int, vector<int>> p = make_pair(cur, cur_neighbors);
                    current_level_nodes.push_back(p);
                    for (auto n : cur_neighbors) {
                        next_level_nodes.push_back(n);
                    }
                }
                if (current_level_nodes.size() > 0) {
                    levels.push_back(current_level_nodes);
                }
                if (!is_finished) {
                    if (next_level_nodes.size() > 0) {
                        for (auto n : next_level_nodes) {
                            q.push(n);
                            if (n == end) {
                                is_finished = true;
                            }
                        }
                    }
                }
            }
       	    cout << "start " << start << " end " << end << endl;
       	    this->pprint_levels();
        }

       void shuffle_levels() {
       	    // note that order of neightbours [v1, v2] creates the order for the next level
       	    // so only shuffle neighbour order here

       	}

        void tokenize(
            const vector<int> &node_shuffle_map,
            const map<std::string, int> &dictionary,
            std::mt19937& gen,
            const bool use_unique_depth_markers = true) {
                // tokenize the BFS levels into a single sequence
       	        // pair(inputs, targets)
                // where targets can be multiple tokens due to label smoothing over order, ex.
       	        // D0: 20 [6 7 10 ] D1: 6 [8 9 18 ] 7 [3 15 ] 10 [1 4 ]
       	        // [6, 7, 10] can be in any order, so targets are [6, 7, 10], then [7, 10], then [10]
       	        // Note that at D1, 6 must be first since it is first in the adjacency list of D0's first node
                // real tokenization is D0 20 [6 7 10 ] D1 6 [8 9 18 ] 7 [3 15 ] 10 [1 4 ] if using unique depth markers
       	        // else, D 20 [6 7 10 ] D 6 [8 9 18 ] 7 [3 15 ] 10 [1 4 ]


       	    auto inputs = vector<int>{};
       	    auto targets = vector<vector<int>>{};

       	    this->tokenized_inputs = inputs;
       	    this->tokenized_targets = targets;
        }

    pair<vector<int>, vector<vector<int>>> get_tokenized() {
       	return make_pair(this->tokenized_inputs, this->tokenized_targets);
    }

    void pprint_levels() { // basically the tokenization, for debugging
       	cout << "Num levels: " << levels.size()  << endl;
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
};


inline void get_bfs() {

}

#endif //SCRATCH_PADS_H
