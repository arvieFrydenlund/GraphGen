//
// Created by arvie on 3/24/25.
//

#ifndef GRAPH_H
#define GRAPH_H

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

template <typename D>  // undirectedS  or directedS
using Graph = boost::adjacency_list<
                                    boost::vecS            // edge list
                                  , boost::vecS            // vertex list
                                  , D    // directedness
                                  , boost::no_property     // property associated with vertices, float or int etc.
                                  , EdgeWeightProperty     // property associated with edges
                                  >;

using ERGen = boost::sorted_erdos_renyi_iterator<std::mt19937, Graph<boost::undirectedS>>;
using ERGen2 = boost::sorted_erdos_renyi_iterator<boost::minstd_rand, Graph<boost::undirectedS>>;

template <typename D>
using Cluster = boost::adjacency_list<boost::listS, boost::listS, D>;

template <typename D>
using  WeightMap = typename boost::property_map<Graph<D>, boost::edge_weight_t>::type;

template <typename D>
using DistanceProperty = boost::exterior_vertex_property<Graph<D>, t_weight>;

template <typename D>
using DistanceMatrix = typename DistanceProperty<D>::matrix_type;
template <typename D>
using DistanceMatrixMap = typename DistanceProperty<D>::matrix_map_type;


template<typename T>
vector<T> list_to_vector(list<T> &l) {
    vector<T> v;
    v.reserve(l.size());
    std::copy(std::begin(l), std::end(l), std::back_inserter(v));
    return v;
}


inline map<int, list<int>> get_connected_components_map(Graph<boost::undirectedS> &g, bool verbose = false) {
    std::vector<int> component (num_vertices (g));
    size_t num_components = connected_components(g, &component[0]);
    if (verbose) {
        cout << "Number of connected components: " << num_components << " num edges " << num_edges(g) << endl;
    }
    // make map of connected components
    map<int, list<int>> component_map;
    for (size_t i = 0; i < num_vertices(g); i++) {
        component_map[component[i]].push_back(i);
    }
    return component_map;
}


inline int sample_num_connected(std::mt19937 &gen, const int num_nodes, const int c_min = 75, const int c_max = 125) {
    if ( num_nodes < c_min ) {
        return 1;
    }
    if ( num_nodes > c_max ) {
        return 2;
    }
    std::uniform_int_distribution dist(1, 2);
    return dist(gen);
}


template <typename D>
 WeightMap<D> make_edge_weights(Graph<D> &g, bool verbose = false) {

    // make all edge weights 1
    WeightMap<D> weight_map = boost::get(boost::edge_weight, g);
    typename boost::graph_traits<Graph<D>>::edge_iterator ei, ei_end;
    for (boost::tie(ei, ei_end) = boost::edges(g); ei != ei_end; ++ei) {
        weight_map[*ei] = 1;
        if (verbose) {
            cout << *ei << " " << weight_map[*ei] << endl;
        }
    }
    return weight_map;
}


template <typename D>
int floyd_warshall(Graph<D> &g, DistanceMatrix<D> &distances, bool verbose = false) {
    // https://stackoverflow.com/questions/26855184/floyd-warshall-all-pairs-shortest-paths-on-weighted-undirected-graph-boost-g

    const WeightMap<D> weight_pmap = boost::get(boost::edge_weight, g);
    DistanceMatrixMap<D> dm(distances, g);

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


template <typename D>
int floyd_warshall_frydenlund(Graph<D> &g, DistanceMatrix<D> &distances, bool verbose = false) {
    return 1;
}

template <typename D>
void pprint_distances(DistanceMatrix<D> &distances) {
    std::cout << "Distance matrix: " << std::endl;
    // for each connected component
    // make square matrix to console

}


#endif //GRAPH_H
