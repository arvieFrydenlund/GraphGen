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


using namespace std;

int erdos_renyi_generator(unique_ptr<UnDirGraph> &g_ptr, unique_ptr<DistanceMatrix> &distances_ptr, const int num_nodes,
    std::mt19937 &gen, float p = -1.0, const int c_min = 75, const int c_max = 125, const bool verbose = false) {

    if ( p < 0 ) {
        p = 1.0 / static_cast<float>(num_nodes);
    }

    g_ptr = make_unique<UnDirGraph>(ERGen(gen, num_nodes, p), ERGen(), num_nodes);
    auto g = *g_ptr.get();
    make_edge_weights(g, verbose);

    cout << "Graph has " << num_vertices(g) << " vertices and " << num_edges(g) << " edges " << p << " p" << endl;

    if (verbose) {
        cout << "Graph has " << num_vertices(g) << " vertices and " << num_edges(g) << " edges " << p << " p" << endl;
    }

    // connect components
    /*
    *components = list(nx.connected_components(self.G))
        num_round = 0
        while len(components) > 1:
            for i in range(len(components)-1):
                for _ in range(self.c_rule()):
                    j = self.rng.choice(range(i + 1, len(components)))
                    u = self.rng.choice(list(components[i]))
                    v = self.rng.choice(list(components[j]))
                    self.G.add_edge(u, v)
            num_round += 1
            components = list(nx.connected_components(self.G))
    */
    std::vector<int> component (num_vertices (g));
    size_t num_components = connected_components(g, &component[0]);
    cout << "Number of connected components: " << num_components << endl;

    // make map of connected components
    map<int, list<int>> component_map;
    for (size_t i = 0; i < num_vertices(g); i++) {
        component_map[component[i]].push_back(i);
    }

    while ( num_components > 1 ) {
        for (size_t i = 0; i < num_components; i++) {
            for (size_t j = i+1; j < num_components; j++) {
                for (int _ = 0; _ < sample_num_connected(gen, num_nodes, c_min, c_max); _++) {
                    // cout << "Connecting components " << i << " and " << j << endl;
                    // sample a node from each component
                    vector<int> c1 = list_to_vector(component_map[i]);
                    uniform_int_distribution<int> d1(0, c1.size() - 1);
                    int v1 = c1[d1(gen)];
                    vector<int> c2 = list_to_vector(component_map[j]);
                    uniform_int_distribution<int> d2(0, c2.size() - 1);
                    int v2 = c2[d2(gen)];
                    boost::add_edge(v1, v2, g);
                }
            }
        }
        // remake connected components
        component = vector<int>(num_vertices (g));
        num_components = connected_components(g, &component[0]);
        component_map.clear();
        for (size_t i = 0; i < num_vertices(g); i++) {
            component_map[component[i]].push_back(i);
        }
        cout << "Number of connected components: " << num_components << endl;
    }




    distances_ptr = make_unique<DistanceMatrix>(num_vertices(g));
    auto distances = *distances_ptr.get();
    int r = floyd_warshall(g, distances, verbose);
    return r;
}

int euclidean_generator(unique_ptr<UnDirGraph> &g_ptr, unique_ptr<DistanceMatrix> &distances_ptr, const int num_nodes, std::mt19937 &gen, const int dim = 2, float radius = -1.0, const bool verbose = true) {
    /*  These are the only graphs which have a true property which is the position a vector of length dim
     *  Because of this, we just keep this as a num_nodes x dim matrix
     */

    if ( radius <= 0 ) {
        radius = 1.0 / sqrt(static_cast<float>(num_nodes));
    }
    g_ptr = make_unique<UnDirGraph>(num_nodes);
    auto g = *g_ptr.get();

    // uniformly generates num_nodes points in dim
    std::uniform_real_distribution<float> distr(0, 1);
    vector<vector<float>> positions = vector(num_nodes, vector<float>(dim));
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
                boost::add_edge(i, j, g);
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
            boost::graph_traits<UnDirGraph>::adjacency_iterator ai, ai_end;
            for (boost::tie(ai, ai_end) = boost::adjacent_vertices(i, g); ai != ai_end; ++ai) {
                cout << *ai << " ";
            }
            cout << endl;
        }
    }

    make_edge_weights(g, verbose);

    if (verbose) {
        cout << "Graph has " << num_vertices(g) << " vertices and " << num_edges(g) << " edges" << endl;
    }

    distances_ptr = make_unique<DistanceMatrix>(num_vertices(g));
    auto distances = *distances_ptr.get();
    int r = floyd_warshall(g, distances, verbose);
    return r;
}

int main(){
    int num_nodes = 100;



    unique_ptr<UnDirGraph> g_ptr;
    unique_ptr<DistanceMatrix> distances_ptr;

    unsigned int seed = std::random_device{}();
    cout << "Seed: " << seed << endl;

    std::mt19937 gen(seed);

    erdos_renyi_generator(g_ptr, distances_ptr, num_nodes, gen, -1.0, 125, 125, false);



    // euclidean_generator(g_ptr, distances_ptr, num_nodes, gen, 2, -1.0, true);

    return 0;
};

// TIP See CLion help at <a
// href="https://www.jetbrains.com/help/clion/">jetbrains.com/help/clion/</a>.
//  Also, you can try interactive lessons for CLion by selecting
//  'Help | Learn IDE Features' from the main menu.