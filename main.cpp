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
#include <math.h>
#include <memory>

#include "graph.h"
#include "balanced.h"


using namespace std;

int erdos_renyi_generator(unique_ptr<Graph<boost::undirectedS>> &g_ptr, unique_ptr<DistanceMatrix<boost::undirectedS>> &distances_ptr, const int num_nodes,
    std::mt19937 &gen, float p = -1.0, const int c_min = 75, const int c_max = 125, const bool verbose = false) {

    if ( p < 0 ) {
        p = 1.0 / static_cast<float>(num_nodes);
    }

    g_ptr = make_unique<Graph<boost::undirectedS>>(ERGen(gen, num_nodes, p), ERGen(), num_nodes);
    make_edge_weights(*g_ptr, verbose);

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
        component_map = get_connected_components_map(*g_ptr, verbose);  // remake connected components
    }

    distances_ptr = make_unique<DistanceMatrix<boost::undirectedS>>(num_vertices(*g_ptr));
    auto distances = *distances_ptr.get();
    const int r = floyd_warshall(*g_ptr, distances, verbose);
    return r;
}


int euclidean_generator(unique_ptr<Graph<boost::undirectedS>> &g_ptr, unique_ptr<DistanceMatrix<boost::undirectedS>> &distances_ptr, unique_ptr<vector<vector<float>>> &positions_ptr, const int num_nodes, std::mt19937 &gen, const int dim = 2, float radius = -1.0, const int c_min = 75, const int c_max = 125, const bool verbose = true) {
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

    if (verbose) {
        cout << "Graph has " << num_vertices(*g_ptr) << " vertices and " << num_edges(*g_ptr) << " edges" << endl;
    }

    distances_ptr = make_unique<DistanceMatrix<boost::undirectedS>>(num_vertices(*g_ptr));
    auto distances = *distances_ptr.get();
    const int r = floyd_warshall(*g_ptr, distances, verbose);
    return r;
}

int main(){
    int num_nodes = 100;

    unique_ptr<Graph<boost::undirectedS>> g_ptr;
    unique_ptr<DistanceMatrix<boost::undirectedS>> distances_ptr;

    unsigned int seed = std::random_device{}();
    cout << "Seed: " << seed << endl;
    // seed = 2871693314;
    // seed = 2564846737;

    std::mt19937 gen(seed);

    // erdos_renyi_generator(g_ptr, distances_ptr, num_nodes, gen, -1.0, 75, 125, false);

    // unique_ptr<vector<vector<float>>> positions_ptr;
    // euclidean_generator(g_ptr, distances_ptr, positions_ptr, num_nodes, gen, 2, -1.0, 75, 125, false);

    // cout << "Graph has " << num_vertices(*g_ptr) << " vertices and " << num_edges(*g_ptr) << " edges " << endl;

    unique_ptr<Graph<boost::directedS>> dg_ptr;
    unique_ptr<DistanceMatrix<boost::directedS>> ddistances_ptr;

    // path_star_generator(dg_ptr, ddistances_ptr, 6, 6, 5, 6, gen, false);

    balanced_generator(dg_ptr, ddistances_ptr, 37, gen, 7, 5);

    if ( dg_ptr ) {
        cout << "Graph has " << num_vertices(*dg_ptr) << " vertices and " << num_edges(*dg_ptr) << " edges " << endl;
    }




    return 0;
};
// TIP See CLion help at <a
// href="https://www.jetbrains.com/help/clion/">jetbrains.com/help/clion/</a>.
//  Also, you can try interactive lessons for CLion by selecting
//  'Help | Learn IDE Features' from the main menu.