//
// Created by arvie on 11/4/25.
//

#ifndef GRAPHGEN_INSTANCE_H
#define GRAPHGEN_INSTANCE_H

#include <iostream>
#include <random>
#include <queue>
#include <map>
#include "undirected_graphs.h"
#include "directed_graphs.h"

using namespace std;

/*
 * Instance class to hold graph and related data i.e. all info for a single instance
 */
template<typename D>
class Instance {
public:
    int N;
    int E;
    unique_ptr<Graph<D> > g_ptr;
    vector<int> node_shuffle_map;
    vector<int> edge_shuffle_map;
    vector<vector<int> > distances;
    vector<vector<int> > ground_truths;



    Instance(unique_ptr<Graph<D> > &g_ptr,
                    vector<int> &edge_shuffle_map
            ){

    }

    vector<int> get_node_shuffle_map(const int N, const int min_vocab, int max_vocab, const bool shuffle = false) {
        // Shuffle nodes and map to the new range [min_vocab, max_vocab)
        if (max_vocab > 0) {
            // asserts do not work on python side, use throws
            assert((max_vocab - min_vocab) >= N && max_vocab - min_vocab > 0 && min_vocab >= 0);
            if (max_vocab - min_vocab < N) { throw std::invalid_argument("max_vocab - min_vocab < N"); }
        } else {
            assert(min_vocab == 0);
            max_vocab = N;
        }
        auto m = std::vector<int>(max_vocab - min_vocab);
        std::iota(m.begin(), m.end(), min_vocab);
        if (shuffle) {
            std::shuffle(m.begin(), m.end(), gen);
        }
        // Only return the first N elements of the new range, thus this will be the length of the graph
        // since we use N = num_vertices(*g_ptr)
        return std::vector<int>(m.begin(), m.begin() + N);
    }

    vector<int> get_edge_shuffle_map(const int E, const bool shuffle = false) {
        // shuffle the edges around, this will be the shuffled order given to the model
        auto m = std::vector<int>(E);
        std::iota(m.begin(), m.end(), 0);
        if (shuffle) {
            std::shuffle(m.begin(), m.end(), gen);
        }
        return m;
    }

    template<typename D>
    vector<pair<int, int> > get_edge_list(unique_ptr<Graph<D> > &g_ptr, vector<int> &shuffle_map) {
        // Get the edge list of the graph in the shuffled order
        auto edge_list = vector<pair<int, int> >(num_edges(*g_ptr), make_pair(-1, -1));
        typename boost::graph_traits<Graph<D> >::edge_iterator ei, ei_end;
        int cur = 0;
        for (boost::tie(ei, ei_end) = boost::edges(*g_ptr); ei != ei_end; ++ei) {
            edge_list[shuffle_map[cur]] = make_pair(source(*ei, *g_ptr), target(*ei, *g_ptr));
            cur += 1;
        }
        return edge_list;
    }
};


class BatchedInstances {


};


#endif //GRAPHGEN_INSTANCE_H