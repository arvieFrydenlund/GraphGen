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


typedef
  boost::adjacency_list<
    boost::vecS            // edge list
  , boost::vecS            // vertex list
  , boost::directedS     // directedness
  , boost::no_property     // property associated with vertices, float or int etc.
  , EdgeWeightProperty     // property associated with edges
  >
DirGraph;

typedef boost::property_map<DirGraph, boost::edge_weight_t>::type DirWeightMap;
typedef boost::exterior_vertex_property<DirGraph, t_weight> DirDistanceProperty;
typedef DirDistanceProperty::matrix_type DirDistanceMatrix;
typedef DirDistanceProperty::matrix_map_type DirDistanceMatrixMap;

inline int floyd_warshall(DirGraph &g, DirDistanceMatrix &distances, bool verbose = false) {
    // https://stackoverflow.com/questions/26855184/floyd-warshall-all-pairs-shortest-paths-on-weighted-undirected-graph-boost-g

    const DirWeightMap weight_pmap = boost::get(boost::edge_weight, g);
    DirDistanceMatrixMap dm(distances, g);

    bool valid = floyd_warshall_all_pairs_shortest_paths(g, dm, boost::weight_map(weight_pmap));

    if (!valid) {
        if (verbose) {
            std::cerr << "Error - Negative cycle in matrix" << std::endl;
        }
        return -1;
    }
    if (verbose) {
        std::cout << "Distance matrix: " << std::endl;
        for (std::size_t i = 0; i < num_vertices(g); ++i) {
            for (std::size_t j = i; j < num_vertices(g); ++j) {
                std::cout << "From vertex " << i+1 << " to " << j+1 << " : ";
                if(distances[i][j] > 100000)
                    std::cout << "inf" << std::endl;
                else
                    std::cout << distances[i][j] << std::endl;
            }
            std::cout << std::endl;
        }
    }
    return 0;
}


/* Path-star graphs and balanced graphs */
inline int path_star_generator(unique_ptr<DirGraph> &g_ptr, unique_ptr<DistanceMatrix> &distances_ptr,
    const int min_num_arms, const int max_num_arms,
    const int min_arm_length, const int max_arm_length,
    std::mt19937 &gen, const bool verbose = false) {

    g_ptr = make_unique<DirGraph>();
    auto g = *g_ptr.get();

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
        boost::add_vertex(g);
    }
    std::shuffle(node_ids.begin(), node_ids.end(), gen);

    // add start node
    int cur = 0;
    int start = node_ids[cur];
    cur++;
    for (int i = 0; i < num_arms; i++) {
        int prev_node = start;
        for (int j = 0; j < arm_length - 1; j++) {
            boost::add_edge(prev_node, node_ids[cur], g);
            prev_node = node_ids[cur];
            cur++;
        }
    }

    if (verbose) {
        cout << "Graph has " << num_vertices(g) << " vertices and " << num_edges(g) << " edges" << endl;
    }

    unique<DirDistanceMatrix>(num_vertices(g));
    //auto distances = *distances_ptr.get();
    //int r = floyd_warshall(g, distances, verbose);
    //return r;
    return 0;
}




#endif //BALANCED_H
