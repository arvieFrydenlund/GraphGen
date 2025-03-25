//
// Created by arvie on 3/24/25.
//

#ifndef CPP_GRAPH_H
#define

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/graph/erdos_renyi_generator.hpp>
#include <boost/random/linear_congruential.hpp>
#include <boost/graph/undirected_graph.hpp>
#include <boost/graph/exterior_property.hpp>
#include <boost/graph/floyd_warshall_shortest.hpp>
#include <boost/graph/random.hpp>
#include <random>
#include <iostream>
#include <string>
#include <vector>


using namespace std;

typedef int t_weight;
typedef boost::property<boost::edge_weight_t, t_weight> EdgeWeightProperty;

typedef
  boost::adjacency_list<
    boost::vecS            // edge list
  , boost::vecS            // vertex list
  , boost::undirectedS     // directedness
  , boost::no_property     // property associated with vertices, float or int etc.
  , EdgeWeightProperty     // property associated with edges
  >
UnDirGraph;

typedef boost::sorted_erdos_renyi_iterator<boost::minstd_rand, UnDirGraph> ERGen;

typedef boost::property_map<UnDirGraph, boost::edge_weight_t>::type WeightMap;
typedef boost::exterior_vertex_property<UnDirGraph, t_weight> DistanceProperty;
typedef DistanceProperty::matrix_type DistanceMatrix;
typedef DistanceProperty::matrix_map_type DistanceMatrixMap;


// template <typename Graph>
int floyd_warshall(UnDirGraph g, DistanceMatrix &distances, bool verbose = false) {

    const WeightMap weight_pmap = boost::get(boost::edge_weight, g);
    DistanceMatrixMap dm(distances, g);

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


#endif //CPP_GRAPH_H
