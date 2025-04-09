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
#include "py_bindings.h"
#include <pybind11/embed.h>

using namespace std;
namespace py = pybind11;


void test_erdos_renyi(std::mt19937 &gen, const int num_nodes = 100, const bool verbose = false) {

    unique_ptr<Graph<boost::undirectedS>> g_ptr;
    unique_ptr<DistanceMatrix<boost::undirectedS>> distances_ptr;
    erdos_renyi_generator(g_ptr,  num_nodes, gen, -1.0, 75, 125, false);

    // get_distances(g_ptr, distances_ptr, false, verbose);

    if ( g_ptr ) {
        cout << "Graph has " << num_vertices(*g_ptr) << " vertices and " << num_edges(*g_ptr) << " edges " << endl;
    }

}

void test_euclidian(std::mt19937 &gen, const int num_nodes = 100, const bool verbose = false) {

    unique_ptr<Graph<boost::undirectedS>> g_ptr;
    unique_ptr<DistanceMatrix<boost::undirectedS>> distances_ptr;
    unique_ptr<vector<vector<float>>> positions_ptr;
    euclidean_generator(g_ptr, positions_ptr, num_nodes, gen, 2, -1.0, 75, 125, false);

    //get_distances(g_ptr, distances_ptr, false, verbose);

    if ( g_ptr ) {
        cout << "Graph has " << num_vertices(*g_ptr) << " vertices and " << num_edges(*g_ptr) << " edges " << endl;
    }

    /*
    unique_ptr<vector<pair<int, int>>> edge_list_ptr;
    get_edge_list(g_ptr, edge_list_ptr);

    if ( edge_list_ptr ) {
        cout << "Edge list has " << edge_list_ptr->size() << " edges " << endl;
        for (auto e : *edge_list_ptr.get()) {
            cout << e.first << " " << e.second << endl;
        }
    }

    unique_ptr<vector<int>> node_list_ptr;
    get_node_list(g_ptr, node_list_ptr);
    if ( node_list_ptr ) {
        cout << "Node list has " << node_list_ptr->size() << " nodes " << endl;
        for (auto n : *node_list_ptr.get()) {
            cout << n << " ";
        }
        cout << endl;
    }
    */


}

void test_path_star(std::mt19937 &gen, const int num_nodes = 100, const bool verbose = false) {

    unique_ptr<Graph<boost::directedS>> g_ptr;
    unique_ptr<DistanceMatrix<boost::directedS>> distances_ptr;
    path_star_generator(g_ptr, 6, 6, 5, 6, gen, false);

    //get_distances(g_ptr, distances_ptr, true, verbose);

    if ( g_ptr ) {
        cout << "Graph has " << num_vertices(*g_ptr) << " vertices and " << num_edges(*g_ptr) << " edges " << endl;
    }

}


void test_balanced(std::mt19937 &gen, const int num_nodes = 100, const bool verbose = false) {

    unique_ptr<Graph<boost::directedS>> g_ptr;
    unique_ptr<DistanceMatrix<boost::directedS>> distances_ptr;
    balanced_generator(g_ptr,  37, gen, 7, 5);

    //get_distances(g_ptr, distances_ptr, true, verbose);

    if ( g_ptr ) {
        cout << "Graph has " << num_vertices(*g_ptr) << " vertices and " << num_edges(*g_ptr) << " edges " << endl;
    }

}

void test_pybind(string graph_type = "erdos_renyi") {




    //unique_ptr<Graph<boost::undirectedS>> g_ptr;
    //unique_ptr<Graph<boost::undirectedS>> g_ptr;
    // unique_ptr<DistanceMatrix<boost::undirectedS>> distances_ptr;
    //erdos_renyi_generator(g_ptr,  25, gen, -1.0, 75, 125, false);
    //erdos_renyi_generator(g_ptr,  75, gen, -1.0, 75, 125, false);

    py::dict d;

    if ( graph_type == "erdos_renyi" ) {
        d = erdos_renyi(15, -1.0, 75, 125, false, false);

    } else if ( graph_type == "euclidian" ) {
        d = euclidian(15, 2, -1.0, 75, 125, false, false);
    } else if ( graph_type == "path_star" ) {
        d = path_star(6, 6, 5, 6, false, false);
    } else if ( graph_type == "balanced" ) {
        // d = balanced(37, 7, 5);
    } else {
        cout << "Unknown graph type: " << graph_type << endl;
        return;
    }

    // print the dict
    for (auto item : d)
    {
        std::cout << "key: " << item.first << ", value=" << item.second << std::endl;
    };

}



int main(){
    py::scoped_interpreter guard{}; // needed to run pybind11 code as a C++ program, not needed for module

    // unsigned int seed = std::random_device{}();
    // cout << "Seed: " << seed << endl;
    // seed = 2871693314;
    // seed = 2564846737;

    // std::mt19937 gen(seed);

    // test_erdos_renyi(gen, 25, false);
    // test_euclidian(gen, 25, false);

    //auto d = return_dict_test();
    // cout << "Dict: " << &d << endl;


    //test_pybind("erdos_renyi");

    // test_pybind("euclidian");

    test_pybind("path_star");





    return 0;
};
// TIP See CLion help at <a
// href="https://www.jetbrains.com/help/clion/">jetbrains.com/help/clion/</a>.
//  Also, you can try interactive lessons for CLion by selecting
//  'Help | Learn IDE Features' from the main menu.