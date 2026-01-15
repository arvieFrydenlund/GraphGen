//
// Created by arvie on 11/8/25.
//

#ifndef GRAPHGEN_TESTS_H
#define GRAPHGEN_TESTS_H

#include <iostream>
#include "instance.h"


void test_khops_gen(const int min_khops = 1, const int max_khops = 7,
                    const int min_prefix_length = 70, const int max_prefix_length = 100,
                    const bool right_side_connect = true, const string &partition_method = "uniform",
                    int min_vocab = -1, int max_vocab = -1,
                    const int batch_size = 10,
                    const int num_thinking_tokens = 0,
                    const string &scratchpad_type = "none",
                    const bool scratchpad_as_prefix = false,
                    const bool is_flat_model = true,
                    const bool align_prefix_front_pad = false){
    auto d = khops_gen_n(
            min_khops, max_khops,
            min_prefix_length, max_prefix_length,
            right_side_connect, partition_method,
            min_vocab, max_vocab,
            batch_size,
            num_thinking_tokens,
            scratchpad_type,
            scratchpad_as_prefix,
            is_flat_model,
            align_prefix_front_pad);
}


void test_erdos_renyi_n( // basic one
        int min_num_nodes, int max_num_nodes,
        const string &task_type = "bfs", //"shortest_path",
        const int max_path_length = 12, const int min_path_length = 3,
        const bool sort_adjacency_lists = false, const bool use_unique_depth_markers = true,
        int max_query_size = -1, const int min_query_size = 2,
        const bool is_causal = false, const bool is_direct_ranking = false, const bool shuffle_edges = true,
        const bool shuffle_nodes = true, const int min_vocab = -1, int max_vocab = -1,
        const int batch_size = 3,
        const bool concat_edges = false,
        const bool duplicate_edges = false,
        const bool include_nodes_in_graph_tokenization = true,
        const bool query_at_end = false,
        const int num_thinking_tokens = 0,
        const string scratchpad_type = "bfs",
        const bool scratchpad_as_prefix = true,
        const bool no_graph = false,
        const bool is_flat_model = true,
        const bool align_prefix_front_pad = true,
        const bool use_edges_invariance = false,  // for concated edges this allows true permutation invariance
        const bool use_node_invariance = false,
        const bool use_graph_invariance = true,
        const bool use_query_invariance = false,
        const bool use_task_structure = false,  // divide positions by task structure
        const bool use_graph_structure = true) {

    auto d = erdos_renyi_n(
            min_num_nodes, max_num_nodes, -1.0, 75, 125,
            task_type,
            max_path_length, min_path_length,
            sort_adjacency_lists, use_unique_depth_markers,
            max_query_size, min_query_size,
            is_causal, is_direct_ranking, shuffle_edges,
            shuffle_nodes, min_vocab, max_vocab,
            batch_size, 512, 100,
            concat_edges,
            duplicate_edges,
            include_nodes_in_graph_tokenization,
            query_at_end,
            num_thinking_tokens,
            scratchpad_type,
            scratchpad_as_prefix, no_graph,
            is_flat_model,
            align_prefix_front_pad,
            use_edges_invariance,
            use_node_invariance,
            use_graph_invariance,
            use_query_invariance,
            use_task_structure,
            use_graph_structure
    );
}

#endif //GRAPHGEN_TESTS_H