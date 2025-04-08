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
#include <memory>


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
vector<T> list_to_vector(list<T> &l) {  // convert list to vector
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
int floyd_warshall(Graph<D> &g, DistanceMatrix<D> &distances, WeightMap<D> &weight_pmap, bool verbose = false) {
    // https://stackoverflow.com/questions/26855184/floyd-warshall-all-pairs-shortest-paths-on-weighted-undirected-graph-boost-g

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
int floyd_warshall_frydenlund(Graph<D> &g, DistanceMatrix<D> &distances, WeightMap<D> &weight_pmap, bool verbose = false) {
    return 1;
}

template <typename D>
void print_distances(unique_ptr<DistanceMatrix<D>> &distances_ptr, int N) {
    // cant figure out the damn shape of the distance matrix
    // so just pass in the size, sigh
    auto distances = *distances_ptr;
    std::cout << "Distance matrix: " << std::endl;
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = i; j < N; ++j) {
            if(distances[i][j] > 100000)
                std::cout << "inf " << std::endl;
            else
                std::cout << distances[i][j] << " ";
        }
        std::cout << std::endl;
    }

}

template <typename D>
int get_distances(unique_ptr<Graph<D>> &g_ptr, unique_ptr<DistanceMatrix<D>> &distances_ptr,
    const bool is_casual = false, bool verbose = false) {

    distances_ptr = make_unique<DistanceMatrix<D>>(num_vertices(*g_ptr));
    auto weight_pmap = make_edge_weights(*g_ptr, false);
    if ( is_casual ) {
        return floyd_warshall_frydenlund(*g_ptr, *distances_ptr, weight_pmap, verbose);
    }
    return floyd_warshall(*g_ptr, *distances_ptr, weight_pmap, verbose);
}


inline int erdos_renyi_generator(unique_ptr<Graph<boost::undirectedS>> &g_ptr,  const int num_nodes, std::mt19937 &gen,
    float p = -1.0, const int c_min = 75, const int c_max = 125, const bool verbose = false) {

    if ( p < 0 ) {
        p = 1.0 / static_cast<float>(num_nodes);
    }

    g_ptr = make_unique<Graph<boost::undirectedS>>(ERGen(gen, num_nodes, p), ERGen(), num_nodes);


    if (verbose) {
        cout << "Graph has " << num_vertices(*g_ptr) << " vertices and " << num_edges(*g_ptr) << " edges " << p << " p" << endl;
    }

    // connect components
    auto component_map = get_connected_components_map(*g_ptr , verbose);
    while ( component_map.size() > 1 ) {
        for (size_t i = 0; i < component_map.size(); i++) {
            for (int _ = 0; _ < sample_num_connected(gen, num_nodes, c_min, c_max); _++) {
                uniform_int_distribution<int> dc(i, component_map.size() - 1);
                int j = dc(gen);
                // sample a node from each component
                vector<int> c1 = list_to_vector(component_map[i]);
                uniform_int_distribution<int> d1(0, c1.size() - 1);
                int v1 = c1[d1(gen)];
                vector<int> c2 = list_to_vector(component_map[j]);
                uniform_int_distribution<int> d2(0, c2.size() - 1);
                int v2 = c2[d2(gen)];
                boost::add_edge(v1, v2, *g_ptr);
            }
        }
        //cout << "Number of connected components: " << component_map.size() << endl;
        component_map = get_connected_components_map(*g_ptr, verbose);  // remake connected components
    }

    // component_map = get_connected_components_map(*g_ptr, verbose);
    //cout << "Number of connected components: " << component_map.size() << endl;

    return 0;
}


inline int euclidean_generator(unique_ptr<Graph<boost::undirectedS>> &g_ptr,
    unique_ptr<vector<vector<float>>> &positions_ptr, const int num_nodes, std::mt19937 &gen,
    const int dim = 2, float radius = -1.0, const int c_min = 75, const int c_max = 125, const bool verbose = true) {
    /*  These are the only graphs which have a true property which is the position a vector of length dim
     *  Because of this, we just keep this as a num_nodes x dim matrix
     */

    if ( radius <= 0 ) {
        radius = 1.0 / sqrt(static_cast<float>(num_nodes));
    }
    g_ptr = make_unique<Graph<boost::undirectedS>>(num_nodes);

    // uniformly generates num_nodes points in dim
    std::uniform_real_distribution<float> distr(0, 1);
    positions_ptr = make_unique<vector<vector<float>>>(num_nodes, vector<float>(dim));
    auto positions = *positions_ptr.get();
    for (int i = 0; i < num_nodes; i++) {
        for (int j = 0; j < dim; j++) {
            positions[i][j] = distr(gen);
        }
    }

    // add edges if the distance between two nodes is less than radius
    for (int i = 0; i < num_nodes; i++) {
        for (int j = i+1; j < num_nodes; j++) {
            float dist = 0;
            for (int k = 0; k < dim; k++) {
                dist += pow(positions[i][k] - positions[j][k], 2);
            }
            dist = sqrt(dist);
            if ( dist < radius ) {
                boost::add_edge(i, j, *g_ptr);
            }
        }
    }

    if ( verbose ) {
        for (int i = 0; i < num_nodes; i++) {
            cout << "Node " << i << " : ";
            for (int j = 0; j < dim; j++) {
                cout << positions[i][j] << " ";
            }
            // edges
            cout << "Edges: ";
            boost::graph_traits<Graph<boost::undirectedS>>::adjacency_iterator ai, ai_end;
            for (boost::tie(ai, ai_end) = boost::adjacent_vertices(i, *g_ptr); ai != ai_end; ++ai) {
                cout << *ai << " ";
            }
            cout << endl;
        }
    }

    // connect components
    auto component_map = get_connected_components_map(*g_ptr, verbose);
    while ( component_map.size() > 1 ) {
        for (size_t i = 0; i < component_map.size(); i++) {
            vector<tuple<int, int, float>> closest;
            for (size_t j = 0; j < component_map.size(); j++) {
                if ( i == j ) {
                    continue;
                }
                tuple<int, int, float> closest_in_component = make_tuple(-1, -1, 1000000.0);
                for (int u : component_map[i]) {
                    for (int v : component_map[j]) {
                        float distance = 0;
                        for (int k = 0; k < dim; k++) {
                            distance += pow(positions[u][k] - positions[v][k], 2);
                        }
                        distance = sqrt(distance);
                        if ( distance < get<2>(closest_in_component) ) {
                            closest_in_component = make_tuple(u, v, distance);
                        }
                    }
                }
                closest.push_back(closest_in_component);
            }
            sort(closest.begin(), closest.end(), [](auto &left, auto &right) {
                return get<2>(left) < get<2>(right);
            });
            for (size_t k = 0; k < sample_num_connected(gen, num_nodes, c_min, c_max); k++) {
                int u = get<0>(closest[k]);
                int v = get<1>(closest[k]);
                boost::add_edge(u, v, *g_ptr);
            }
        }
        component_map = get_connected_components_map(*g_ptr , verbose);  // remake connected components
    }

    make_edge_weights(*g_ptr, verbose);
    return 0;
}


#endif //GRAPH_H
