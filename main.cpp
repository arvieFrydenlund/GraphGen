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


int main() {
    py::scoped_interpreter guard{};
    // needed to run pybind11 code as a C++ program, not needed for module

    set_seed(44);
    cout << "Seed: " << get_seed() << endl;
    int min_num_nodes = 25;
    int max_num_nodes = 25;

    set_default_dictionary(max_num_nodes, 20);  // 10 extra tokens D0-D9
    set_default_pos_dictionary();

    // write in all other arguments for test_erdos_renyi_n
    string task_type = "shortest_path";
    int max_path_length = 12;
    int min_path_length = 3;
    bool sort_adjacency_lists = false;
    bool use_unique_depth_markers = true;
    int max_query_size = -1;
    int min_query_size = 2;
    bool is_causal = false;
    bool is_direct_ranking = false;
    bool shuffle_edges = true;
    bool shuffle_nodes = true;
    int min_vocab = -1;
    int max_vocab = -1;
    int batch_size = 3;
    bool concat_edges = false;
    bool duplicate_edges = false;
    bool include_nodes_in_graph_tokenization = true;
    bool query_at_end = false;
    int num_thinking_tokens = 0;
    string scratchpad_type = "bfs";
    bool scratchpad_as_prefix = true;
    bool no_graph = false;
    bool is_flat_model = true;
    bool align_prefix_front_pad = true;
    bool use_edges_invariance = false;  // for concated edges this allows true permutation invariance
    bool use_node_invariance = false;
    bool use_graph_invariance = true;
    bool use_query_invariance = false;
    bool use_task_structure = false;  // divide positions by task structure
    bool use_graph_structure = true;

    auto t = time_before();

    if (true) {  // khops
        t = time_before();
        test_khops();
        time_after(t, "Final test_khops");
    } else if (false) {  // khops gen
        t = time_before();
        test_khops_gen();
        time_after(t, "Final test_khops_gen");
    } else if (false) {  // bfs task
        t = time_before();
        task_type = "bfs";
        test_erdos_renyi_n(
                min_num_nodes, max_num_nodes,
                task_type,
                max_path_length, min_path_length,
                sort_adjacency_lists, use_unique_depth_markers,
                max_query_size, min_query_size,
                is_causal, is_direct_ranking, shuffle_edges,
                shuffle_nodes, min_vocab, max_vocab,
                batch_size,
                concat_edges,
                duplicate_edges,
                include_nodes_in_graph_tokenization,
                query_at_end,
                num_thinking_tokens,
                scratchpad_type,
                scratchpad_as_prefix, no_graph,
                is_flat_model,
                align_prefix_front_pad,
                use_edges_invariance,
                use_node_invariance,
                use_graph_invariance,
                use_query_invariance,
                use_task_structure,
                use_graph_structure
        );
        time_after(t, "Final test_erdos_renyi_n bfs task");
        } else if (true) {  //random_tree

           const int max_degree = 3;
           const int max_depth = 7;
           const float bernoulli_p = 0.5;

            t = time_before();
            task_type = "shortest_path";
            test_random_tree_n(
                    min_num_nodes, max_num_nodes,
                    max_degree, max_depth, bernoulli_p,
                    task_type,
                    max_path_length, min_path_length,
                    sort_adjacency_lists, use_unique_depth_markers,
                    max_query_size, min_query_size,
                    is_causal, is_direct_ranking, shuffle_edges,
                    shuffle_nodes, min_vocab, max_vocab,
                    batch_size,
                    concat_edges,
                    duplicate_edges,
                    include_nodes_in_graph_tokenization,
                    query_at_end,
                    num_thinking_tokens,
                    scratchpad_type,
                    scratchpad_as_prefix, no_graph,
                    is_flat_model,
                    align_prefix_front_pad,
                    use_edges_invariance,
                    use_node_invariance,
                    use_graph_invariance,
                    use_query_invariance,
                    use_task_structure,
                    use_graph_structure
            );
            time_after(t, "Final test_random_tree_n");
        }

    cout << "Done!" << endl;

    return 0;
};