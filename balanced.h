//
// Created by arvie on 3/26/25.
//

#ifndef BALANCED_H
#define BALANCED_H

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/graph/erdos_renyi_generator.hpp>
#include <boost/random/linear_congruential.hpp>
#include <boost/graph/directed_graph.hpp>
#include <boost/graph/exterior_property.hpp>
#include <boost/graph/floyd_warshall_shortest.hpp>
#include <boost/graph/random.hpp>
#include <random>
#include <iostream>
#include <string>
#include <vector>

#include "graph.h"


using namespace std;


inline int path_star_generator(unique_ptr<Graph<boost::directedS>> &g_ptr, unique_ptr<DistanceMatrix<boost::directedS>> &distances_ptr,
    const int min_num_arms, const int max_num_arms,
    const int min_arm_length, const int max_arm_length,
    std::mt19937 &gen, const bool verbose = false) {

    g_ptr = make_unique<Graph<boost::directedS>>();

    int num_arms = min_num_arms;
    if ( min_num_arms < max_num_arms) {
        num_arms = std::uniform_int_distribution<int>(min_num_arms, max_num_arms)(gen);
    }
    int arm_length = max_arm_length;
    if ( min_arm_length < max_arm_length) {
        arm_length = std::uniform_int_distribution<int>(min_arm_length, max_arm_length)(gen);
    }

    auto node_ids = vector<int>(num_arms * (arm_length - 1) + 1);
    for (int i = 0; i < num_arms * (arm_length - 1) + 1; i++) {
        node_ids[i] = i;
        boost::add_vertex(*g_ptr);
    }
    std::shuffle(node_ids.begin(), node_ids.end(), gen);

    // add start node
    int cur = 0;
    int start = node_ids[cur];
    cur++;
    for (int i = 0; i < num_arms; i++) {
        int prev_node = start;
        for (int j = 0; j < arm_length - 1; j++) {
            boost::add_edge(prev_node, node_ids[cur], *g_ptr);
            prev_node = node_ids[cur];
            cur++;
        }
    }
    make_edge_weights(*g_ptr, verbose);

    if (verbose) {
        cout << "Graph has " << num_vertices(*g_ptr) << " vertices and " << num_edges(*g_ptr) << " edges" << endl;
    }

    distances_ptr = make_unique<DistanceMatrix<boost::directedS>>(num_vertices(*g_ptr));
    auto distances = *distances_ptr.get();
    const int r = floyd_warshall(*g_ptr, distances, verbose);
    return r;
}


inline int balanced_generator(unique_ptr<Graph<boost::directedS>> &g_ptr, unique_ptr<DistanceMatrix<boost::directedS>> &distances_ptr,
    const int num_nodes, std::mt19937 &gen, const int max_num_parents = 4, const int max_prefix_vertices = -1, const bool verbose = false) {

    return 0;
}




#endif //BALANCED_H
