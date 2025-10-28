//
// Created by arvie on 28/10/25.
//

#ifndef SCRATCH_PADS_H
#define SCRATCH_PADS_H

#include <iostream>
#include <random>
#include "undirected_graphs.h"
#include "directed_graphs.h"

using namespace std;

class ScratchPad {
  /*
   * For easy passing of common variables to different scratch pad types
   */
public:
  pair<vector<int>, vector<vector<int>>> tokenize() {
    // targets, but input sequence is shifted by one
   throw std::invalid_argument("Not implemented yet");
  };
};


class BFS : public ScratchPad {
   /*
    * BFS class to perform breadth first search on a graph
    */
public:
       vector<pair<int, vector<int>>> levels;
      /*
       *  Have a vector for each level of the BFS
       *  these then need to be a vector  of pairs of vectors:  [(4, [8, 6]), (5, [7])].
       *  Then  I can random shuffle the pairs and the adjacency lists within each pair
       *  or sort them by id to get a deterministic order (this only works if I apply the node_shuffle_map first)
       */

        BFS( const int start, const int end, const unique_ptr<vector<vector<int> > > &distances_ptr, const vector<int> &node_shuffle_map) {
                // constructor
        };


        pair<vector<int>, vector<vector<int>>> tokenize() {
            // tokenize the BFS levels into a single sequence
            // targets, but input sequence is shifted by one
        };


};


inline void get_bfs() {

}

#endif //SCRATCH_PADS_H
