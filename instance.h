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
#include "tasks.h"
#include "scratch_pads.h"
#include "graph_tokenizer.h"

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
    unique_ptr<GraphTokenizer> graph_tokenizer;
    unique_ptr<Task> task;
    unique_ptr<ScratchPad> scratch_pad;

    vector<int> get_node_shuffle_map(std::mt19937 &gen, const int min_vocab, int max_vocab, const bool shuffle = false) {
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

    vector<int> get_edge_shuffle_map(std::mt19937 &gen, const bool shuffle = false) {
        // shuffle the edges around, this will be the shuffled order given to the model
        auto m = std::vector<int>(E);
        std::iota(m.begin(), m.end(), 0);
        if (shuffle) {
            std::shuffle(m.begin(), m.end(), gen);
        }
        return m;
    }

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

    Instance(unique_ptr<Graph<D> > &g_ptr,
             std::mt19937 &gen,
             const map<std::string, int> &dictionary,
             const int min_vocab, int max_vocab, const bool shuffle_nodes, const bool shuffle_edges,
             // task parameters
             const string &task_type,  const string scratchpad_type,
             const int max_path_length, const int min_path_length, int start, int end,  // shortest path
             const bool sort_adjacency_lists, const bool use_unique_depth_markers, // DFS/BFS scratchpad
             int max_query_size, const int min_query_size, // center
             // tokenization parameters
             const bool is_causal,
             const bool is_direct_ranking,
             const bool concat_edges,
             const bool duplicate_edges,
             const bool include_nodes_in_graph_tokenization
            ){

        this->g_ptr = std::move(g_ptr); // take ownership of the graph pointer

        N = num_vertices(*g_ptr);
        E = num_edges(*g_ptr);
        edge_shuffle_map = get_edge_shuffle_map(gen, shuffle_edges);
        node_shuffle_map = get_node_shuffle_map(N, min_vocab, max_vocab, shuffle_nodes);

        auto edge_list = get_edge_list(g_ptr, edge_shuffle_map);
        graph_tokenizer = make_unique<GraphTokenizer>(edge_list, concat_edges, duplicate_edges, include_nodes_in_graph_tokenization);
        graph_tokenizer->tokenize(dictionary, node_shuffle_map, gen);

        if (include_nodes_in_graph_tokenization){
            graph_tokenizer->get_distances(g_ptr);
            graph_tokenizer->get_node_ground_truths(g_ptr, is_direct_ranking);
        } else {
            // this also gets the distances due to legacy code
            graph_tokenizer->get_edge_ground_truths(g_ptr, is_causal);
        }

        if (task_type == "shortest_path") {
            task = make_unique<ShortestPathTask>(start, end, max_path_length, min_path_length, g_ptr);
            if (scratchpad_type == "bfs" || scratchpad_type == "BFS"){
                scratch_pad = make_unique<BFSScratchPad>(start, end, g_ptr, sort_adjacency_lists, use_unique_depth_markers);
            } else if (scratchpad_type == "dfs" || scratchpad_type == "DFS"){
                // scratch_pad = make_unique<DFSScratchPad>(start, end, g_ptr, sort_adjacency_lists, use_unique_depth_markers);
            }
        } else if (task_type == "center") {
            task = make_unique<CenterTask>(max_query_size, min_query_size, g_ptr);
            // scratch_pad in future?
        }
        if (task) {
            task->tokenize(dictionary, node_shuffle_map, gen);
        }
        if (scratch_pad) {
            scratch_pad->tokenize(dictionary, node_shuffle_map, gen);
        }
    }


};

template<typename D>
class BatchedInstances {
    BatchedInstances(vector<Instance<D>> &instances,
                     const string &graph_type, const string &task_type,
                     const int attempts, const int max_attempts,
             const int min_vocab, int max_vocab,

             // tokenization parameters
             const bool query_at_end = false,
             const int num_thinking_tokens = 0,
             const bool is_flat_model = true,
             const bool align_prefix_front_pad = false

    ) {

    }



};


#endif //GRAPHGEN_INSTANCE_H