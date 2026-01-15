#include <boost/graph/adjacency_list.hpp>
#include <random>
#include <iostream>
#include <string>
#include <vector>
#include <math.h>
#include <memory>


#include "utils.h"
#include "generator.cpp"
#include "tests.h"
#include <pybind11/embed.h>

using namespace std;
namespace py = pybind11;


void test_pybind(string graph_type = "erdos_renyi",
                 const int num_nodes = 15, const int batch_size = 7,
                 const bool is_casual = true, const bool shuffle_edges = false,
                 const bool shuffle_nodes = false, const int min_vocab = 0,
                 int max_vocab = -1,
                 const bool concat_edges = true,
                 const bool query_at_end = false,
                 const bool is_flat_model = true,
                 const bool for_plotting = false,
                 const int max_edges = 512) {
    py::dict d;


    // print the dict
    for (auto item: d) {
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

// run with -fsanitize=address
// ASAN_OPTIONS=detect_leaks=1
int main() {
    py::scoped_interpreter guard{};
    // needed to run pybind11 code as a C++ program, not needed for module

    set_seed(44);
    cout << "Seed: " << get_seed() << endl;
    int max_num_nodes = 25;

    set_default_dictionary(max_num_nodes, 20);  // 10 extra tokens D0-D9
    set_default_pos_dictionary();

    auto t = time_before();
    // test_erdos_renyi_n(15, max_num_nodes);
    // time_after(t, "Final test_erdos_renyi_n");

    t = time_before();
    test_khops_gen();
    time_after(t, "Final test_khops_gen");

    cout << "Done!" << endl;

    return 0;
};