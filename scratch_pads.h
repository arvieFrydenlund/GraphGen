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
       vector<pair<int, vector<int>>> levels;
      /*
       *  Have a vector for each level of the BFS
       *  these then need to be a vector  of pairs of vectors:  [(4, [8, 6]), (5, [7])].
       *  Then  I can random shuffle the pairs and the adjacency lists within each pair
       *  or sort them by id to get a deterministic order (this only works if I apply the node_shuffle_map first)
       */

       	template<typename D>
        BFSScratchPad( const int start, const int end,
           const unique_ptr<Graph<D> > &g_ptr) {
                // constructor, fill in levels

            bool is_finished = false;  // don't stop at finding target but do entire level with target in it
            map<int, bool> visited;
       	    visited[start] = true;
            queue<int> q;
            q.push(start);

            while (!is_finished){
                while (!q.empty()) {
                   vector<int> current_level_nodes;
                    auto cur = q.front();
                    q.pop();
                    // get all adjacency nodes not in visited
                	auto neighbors = boost::adjacent_vertices(cur, *g_ptr);
                    for (auto nbr = neighbors.first; nbr != neighbors.second; ++nbr) {
                      	if (static_cast<int>(*nbr) == end) {
                        	is_finished = true;
                    	}
                        if (visited.find(*nbr) == visited.end()) {
                            current_level_nodes.push_back(*nbr);
                            visited[*nbr] = true;
                        }
                    }
                    cout << "BFS level node: " << cur << " with children: ";
                    for (auto n : current_level_nodes) {
                        cout << n << " ";
                    }
                    cout << endl << "At level: " << levels.size() << endl;

                    auto p = make_pair(cur, current_level_nodes);
                    levels.push_back(p);
                    if (cur == end) {
                        is_finished = true;
                    }
                    for (auto n : current_level_nodes) {
                        q.push(n);
                    }
                }
            }
        }

        pair<vector<int>, vector<vector<int>>> tokenize(
            const vector<int> &node_shuffle_map,
            const map<std::string, int> &dictionary,
            const bool use_unique_depth_markers = true) {
                // tokenize the BFS levels into a single sequence
                // targets, but input sequence is shifted by one

       	    auto pair = make_pair(vector<int>{}, vector<vector<int>>{});
       	    return pair;
        }
};


inline void get_bfs() {

}

#endif //SCRATCH_PADS_H
