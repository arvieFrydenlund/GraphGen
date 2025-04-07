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

#include "undirected_graphs.h"
#include "directed_graphs.h"
#include "data_gen.h"
#include "py_bindings.h"

using namespace std;

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

    balanced_generator(dg_ptr,  37, gen, 7, 5);

    if ( dg_ptr ) {
        cout << "Graph has " << num_vertices(*dg_ptr) << " vertices and " << num_edges(*dg_ptr) << " edges " << endl;
    }




    return 0;
};
// TIP See CLion help at <a
// href="https://www.jetbrains.com/help/clion/">jetbrains.com/help/clion/</a>.
//  Also, you can try interactive lessons for CLion by selecting
//  'Help | Learn IDE Features' from the main menu.