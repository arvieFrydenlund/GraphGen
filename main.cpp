#include <boost/graph/adjacency_list.hpp>
#include <random>
#include <iostream>
#include <string>
#include <vector>
#include <math.h>
#include <memory>

#include "undirected_graphs.h"
#include "directed_graphs.h"
#include "utils.h"
#include "generator.cpp"
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

void test_pybind(string graph_type = "erdos_renyi",
  const int num_nodes = 15, const int batch_size = 7,
  const bool is_casual = true, const bool shuffle_edges = false,
  const bool shuffle_nodes = false, const int min_vocab = 0, int max_vocab = -1,
  const bool concat_edges = true, const bool query_at_end = true,
  const bool is_flat_model = true, const bool for_plotting = false,
  const int max_edges = 512) {

    py::dict d;
    if ( graph_type == "erdos_renyi" ) {
        cout << "Erdos Renyi Graph Test: " << endl;
        d = erdos_renyi(num_nodes, -1.0, 75, 125,
            10, 3,
            -1, 2, true,
            is_casual, shuffle_edges, shuffle_nodes, min_vocab, max_vocab);

    } else  if ( graph_type == "erdos_renyi_n" ) {

        cout << "Erdos Renyi Graph Test: " << endl;
        d = erdos_renyi_n(num_nodes, num_nodes + 5, -1.0, 75, 125,
            10, 3, true,
            -1, 2, false, false,
            is_casual, shuffle_edges, shuffle_nodes, min_vocab, max_vocab, batch_size, max_edges,
            concat_edges, query_at_end, 0, is_flat_model, for_plotting
            );

    } else if ( graph_type == "euclidian" ) {

        cout << "Euclidian Graph Test: " << endl;
        d = euclidian(num_nodes, 2, -1.0, 75, 125, 10, 3, -1, 2, true, is_casual, false, false);

    } else  if ( graph_type == "euclidian_n" ) {

        cout << "Euclidian Graph Test: " << endl;
        d = euclidian_n(num_nodes, num_nodes + 5, 2, -1.0, 75, 125,  10, 3, true,
            is_casual, true, true, 2,  num_nodes + 10, batch_size, max_edges);

    } else if ( graph_type == "path_star" ) {

        cout << "Path Star Graph Test: " << endl;
        d = path_star(3, 3, 5, 5, -1, 2, true, false, false);

    } else if ( graph_type == "path_star_n" ) {

        cout << "Path Star Graph Test: " << endl;
        d = path_star_n(2, 3, 5, 7, true, is_casual, true, true, 3, num_nodes + 10, batch_size);

    } else if ( graph_type == "balanced" ) {

        cout << "Balanced Graph Test: " << endl;
        d = balanced(num_nodes, 7, 5, 4, -1, -1, 2, true, is_casual, false, false, 0, -1);

    } else if ( graph_type == "balanced_n" ) {

        cout << "Balanced Graph Test: " << endl;
        d = balanced_n(num_nodes, num_nodes + 5,  3, 8, 4, 4, -1, true, is_casual, true, true, 3, num_nodes + 15, batch_size);

    } else {
        cout << "Unknown graph type: " << graph_type << endl;
        return;
    }

    // print the dict
    for (auto item : d)
    {
        std::cout << "key: " << item.first << ", value=" << item.second;
        // if numpy array print shape
        if (py::isinstance<py::array>(item.second)) {
            auto arr = item.second.cast<py::array>();
            std::cout << " Shape: [";
            for (size_t i = 0; i < static_cast<size_t>(arr.ndim()); i++) {
                std::cout << arr.shape(i) << " ";
            }
            std::cout << "]";
        }
        cout << endl;
    }
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

    cout << "Seed: " << get_seed() << endl;

    const int num_nodes = 15;
    const int batch_size = 7;
    const bool is_casual = true;
    const bool shuffle_edges = false;
    const bool shuffle_nodes = false;
    const int min_vocab = 0;
    int max_vocab = -1;
    const bool is_flat_model = true;
    const bool for_plotting = false;


    test_pybind("erdos_renyi", num_nodes, batch_size, is_casual, shuffle_edges, shuffle_nodes, min_vocab, max_vocab, is_flat_model, for_plotting);
    // test_pybind("euclidian");
    // test_pybind("path_star");
    // test_pybind("balanced", 25);

    auto t1 = time_before();
    // test_pybind("erdos_renyi_n", num_nodes, batch_size, is_casual, shuffle_edges, shuffle_nodes, min_vocab, max_vocab, is_flat_model, for_plotting);
    // test_pybind("erdos_renyi_n", 150, 256);
    // test_pybind("euclidian_n", 50, 256, false);
    // test_pybind("path_star_n", 50, 256, false);
    // test_pybind("balanced_n", 50, 256, false);
    time_after(t1, "Final");


    return 0;
};
