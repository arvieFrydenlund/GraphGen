//
// Created by arvie on 11/4/25.
//

#ifndef GRAPHGEN_INSTANCE_H
#define GRAPHGEN_INSTANCE_H

#include <iostream>
#include <random>
#include <map>

#include <Python.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "matrix.h"
#include "undirected_graphs.h"
#include "directed_graphs.h"
#include "graph_tokenizer.h"
#include "tasks.h"
#include "scratch_pads.h"

using namespace std;

namespace py = pybind11;
using namespace py::literals;

static const py::bool_ py_true(true);

/*
 * Instance class to hold graph and related data i.e. all info for a single instance
 */
template<typename D>
class Instance {
public:
    bool use_task_structure;

    int N;
    int E;
    unique_ptr<Graph<D> > g_ptr;
    vector<int> node_shuffle_map;
    vector<int> edge_shuffle_map;
    unique_ptr<GraphTokenizer> graph_tokenizer;
    unique_ptr<Task> task;
    unique_ptr<ScratchPad> scratch_pad;
    unique_ptr<vector<vector<float> > > positions_ptr;

    Matrix<int> tokenized_inputs;
    Matrix<int> tokenized_targets;
    Matrix<int> tokenized_positions;
    // prefix
    int query_start_idx = -1; // index in the tokenized input where the query starts
    int query_length = 0;
    int graph_start_idx = -1; // index in the tokenized input where the graph starts
    int graph_length = 0;
    int graph_nodes_start_idx = -1; // index in the tokenized input where the graph nodes start
    int graph_nodes_length = 0; // should just be N again
    int thinking_tokens_start_idx = -1;
    int thinking_tokens_length = 0;
    // task
    int scratch_pad_start_idx = -1; // index in the tokenized input
    int scratch_pad_length = 0;
    int true_task_start_idx = -1; // index in the tokenized input where the task starts excluding any scratch pad
    int true_task_length = 0;
    // task is anything the model generates i.e scratch pad and actual task
    int task_start_idx = -1; // index in the tokenized input where the task starts
    int task_length = 0;


    void make_node_shuffle_map(std::mt19937 &gen, const int min_vocab, int max_vocab,
                               const bool shuffle = false) {
        // Shuffle nodes and map to the new range [min_vocab, max_vocab)
        if (max_vocab > 0) {
            // asserts do not work on python side, use throws
            assert((max_vocab - min_vocab) >= N && max_vocab - min_vocab > 0 && min_vocab >= 0);
            if (max_vocab - min_vocab < N) { throw std::invalid_argument("max_vocab - min_vocab < N"); }
        } else {
            assert(min_vocab == 0);
            max_vocab = N;
        }
        auto m = std::vector<int>(max_vocab - min_vocab);
        std::iota(m.begin(), m.end(), min_vocab);
        if (shuffle) {
            std::shuffle(m.begin(), m.end(), gen);
        }
        node_shuffle_map = std::vector<int>(m.begin(), m.begin() + N);
    }

    void make_edge_shuffle_map(std::mt19937 &gen, const bool shuffle = false) {
        // shuffle the edges around, this will be the shuffled order given to the model
        edge_shuffle_map = std::vector<int>(E);
        std::iota(edge_shuffle_map.begin(), edge_shuffle_map.end(), 0);
        if (shuffle) {
            std::shuffle(edge_shuffle_map.begin(), edge_shuffle_map.end(), gen);
        }
    }

    vector<pair<int, int> > get_edge_list(unique_ptr<Graph<D> > &g_ptr, vector<int> &shuffle_map) {
        // Get the edge list of the graph in the shuffled order
        auto edge_list = vector<pair<int, int> >(num_edges(*g_ptr), make_pair(-1, -1));
        typename boost::graph_traits<Graph<D> >::edge_iterator ei, ei_end;
        int cur = 0;
        for (boost::tie(ei, ei_end) = boost::edges(*g_ptr); ei != ei_end; ++ei) {
            edge_list[shuffle_map[cur]] = make_pair(source(*ei, *g_ptr), target(*ei, *g_ptr));
            cur += 1;
        }
        return edge_list;
    }

    Instance(unique_ptr<Graph<D> > &g_ptr,
             std::mt19937 &gen,
             const map<std::string, int> &dictionary,
             const int min_vocab, int max_vocab, const bool shuffle_nodes, const bool shuffle_edges,
            // task parameters
             const string &task_type, const string scratchpad_type,
             const int max_path_length, const int min_path_length, int start, int end,
             const optional<vector<float> > &task_sample_dist, // shortest path
             const bool sort_adjacency_lists, const bool use_unique_depth_markers, // DFS/BFS scratchpad
             optional<vector<int> > &given_query, int max_query_size, const int min_query_size,
            // center
            // tokenization parameters
             const bool is_causal,
             const bool is_direct_ranking,
             const bool concat_edges,
             const bool duplicate_edges,
             const bool include_nodes_in_graph_tokenization,
             const map<std::string, int> pos_dictionary,
             const bool use_edges_invariance, // for concated edges this allows true permutation invariance
             const bool use_node_invariance,
             const bool use_graph_invariance,
             const bool use_query_invariance,
             const bool use_task_structure, // divide positions by task structure
             const bool use_graph_structure, // 2d positions by graph structure
             optional<unique_ptr<vector<vector<float> > > > positions_ptr = nullopt
    ) {

        this->use_task_structure = use_task_structure;

        N = num_vertices(*g_ptr);
        E = num_edges(*g_ptr);
        make_node_shuffle_map(gen, min_vocab, max_vocab, shuffle_nodes);
        make_edge_shuffle_map(gen, shuffle_edges);


        auto edge_list = get_edge_list(g_ptr, edge_shuffle_map);
        graph_tokenizer = make_unique<GraphTokenizer>(edge_list, is_causal, is_direct_ranking,
                                                      concat_edges, duplicate_edges,
                                                      include_nodes_in_graph_tokenization,
                                                      use_edges_invariance, use_node_invariance,
                                                      use_graph_invariance, use_graph_structure);
        graph_tokenizer->tokenize(dictionary, node_shuffle_map, pos_dictionary, gen);

        // issue of when to convert these to node shuffle map order, it can not be before we make the task
        // since these need the original distance matrix in the original node order
        if (include_nodes_in_graph_tokenization) {
            graph_tokenizer->get_distances(g_ptr);
            graph_tokenizer->get_node_ground_truths();
        } else {
            // this also gets the distances due to legacy code
            graph_tokenizer->get_edge_ground_truths(g_ptr);
        }

        if (task_type == "shortest_path") {
            // the ShortestPathTask and BFSScratchPad are badly coupled
            // this is because we use the ShortestPathTask to sample a start and end point
            // however the BFSScratchPad needs to be able to construct the path since otherwise
            // the scratch pad could be inconsistent with the task path in the case when multiple shortest paths exist
            // thus we need to use the one first discovered by BFS.  Same goes for the DFS scratch pad.

            auto should_sample_path = scratchpad_type == "none" or scratchpad_type == "None";
            auto short_path = make_unique<ShortestPathTask>(gen, graph_tokenizer->distances_ptr,
                                                            max_path_length, min_path_length,
                                                            start, end, task_sample_dist, use_query_invariance,
                                                            should_sample_path);
            if (scratchpad_type == "bfs" || scratchpad_type == "BFS") {
                auto bfs_scratch_pad = make_unique<BFSScratchPad>(short_path->start, short_path->end, g_ptr,
                                                                  node_shuffle_map, sort_adjacency_lists,
                                                                  use_unique_depth_markers);
                short_path->set_path(bfs_scratch_pad->path, graph_tokenizer->distances_ptr, false);
                scratch_pad = std::move(bfs_scratch_pad);
            } else if (scratchpad_type == "dfs" || scratchpad_type == "DFS") {
                auto dfs_scratch_pad = make_unique<DFSScratchPad>(short_path->start, short_path->end, g_ptr,
                                                                  node_shuffle_map, sort_adjacency_lists,
                                                                  use_unique_depth_markers);
                short_path->set_path(dfs_scratch_pad->path, graph_tokenizer->distances_ptr, false);
                scratch_pad = std::move(dfs_scratch_pad);
            }
            task = std::move(short_path);
        } else if (task_type == "center" || task_type == "centroid") {
            auto is_center = (task_type == "center" ? true : false);
            task = make_unique<CenterTask>(gen, graph_tokenizer->distances_ptr, given_query, max_query_size,
                                           min_query_size, is_center);
            // scratch_pad in future?
        }
        if (task) {
            task->tokenize(dictionary, node_shuffle_map, pos_dictionary, gen);
        }
        if (scratch_pad) {
            scratch_pad->tokenize(dictionary, node_shuffle_map, pos_dictionary, gen);
        }
        this->g_ptr = std::move(g_ptr); // take ownership of the graph pointer
        if (positions_ptr.has_value()) {
            this->positions_ptr = std::move(positions_ptr.value());
        }
    }

    void tokenize(
            /*
             * Creates the input sequence, the target sequence, and the positional embeddings
             */
            const map<std::string, int> &dictionary,
            const map<std::string, int> pos_dictionary,
            bool query_at_end,
            int num_thinking_tokens,
            bool is_flat_model) {
        // set lengths
        int num_tokens = 2; // +2 for start and end sequence tokens
        if (task) {
            query_length = static_cast<int>(task->tokenized_query_inputs.shape()[0]);
            num_tokens += query_length;
        }
        graph_length = static_cast<int>(graph_tokenizer->tokenized_inputs.shape()[0]);
        if (graph_tokenizer->include_nodes_in_graph_tokenization) {
            graph_nodes_length = static_cast<int>(graph_tokenizer->node_list.size());
        }
        num_tokens += graph_length;
        thinking_tokens_length = num_thinking_tokens;
        num_tokens += thinking_tokens_length;
        auto max_labels = 0;
        if (is_flat_model) {
            if (task) {
                task_length = static_cast<int>(task->tokenized_task_inputs.shape()[0]);
                true_task_length = task_length;
                max_labels = static_cast<int>(task->tokenized_task_targets.shape()[1]);
            }
            if (scratch_pad) {
                scratch_pad_length = static_cast<int>(scratch_pad->tokenized_inputs.shape()[0]);
                task_length += scratch_pad_length;
                if (static_cast<int>(scratch_pad->tokenized_targets.shape()[1]) > max_labels) {
                    max_labels = static_cast<int>(scratch_pad->tokenized_targets.shape()[1]);
                }
            }
            num_tokens += task_length;
        } // todo later

        // init output matrices
        auto concat_edges = graph_tokenizer->concat_edges;
        auto use_graph_structure = graph_tokenizer->use_graph_structure;
        if (concat_edges) {
            tokenized_inputs = Matrix<int>(num_tokens, 2, dictionary.at("<pad>"));
            use_graph_structure = false; // cannot have both concat edges and graph structure
        } else {
            tokenized_inputs = Matrix<int>(num_tokens, 1, dictionary.at("<pad>"));
        }
        if (task) {
            // + 1 for end of sequence token
            tokenized_targets = Matrix<int>(task_length + 1, max_labels, dictionary.at("<pad>"));
        }
        if (use_graph_structure) {
            tokenized_positions = Matrix<int>(num_tokens, 2, pos_dictionary.at("pad"));
        } else {
            tokenized_positions = Matrix<int>(num_tokens, 1, pos_dictionary.at("pad"));
        }

        // write in values
        auto cur = 0;
        // start of sequence
        auto start_marker = dictionary.at("<s>");
        tokenized_inputs(cur, 0) = start_marker;
        if (concat_edges) {
            tokenized_inputs(cur, 1) = start_marker;
        }
        tokenized_positions(cur, 0) = pos_dictionary.at("misc_start");
        if (use_graph_structure) {
            tokenized_positions(cur, 1) = pos_dictionary.at("misc_start");
        }
        cur++;

        if (!query_at_end and task) {
            // write in query if before graph
            query_start_idx = cur;
            for (size_t i = 0; i < task->tokenized_query_inputs.shape()[0]; i++, cur++) {
                tokenized_inputs(cur, 0) = task->tokenized_query_inputs(i);
                if (concat_edges) {
                    // replicate query if concat edges in both dims
                    tokenized_inputs(cur, 1) = tokenized_inputs(cur, 0);
                }
                tokenized_positions(cur, 0) = task->tokenized_query_pos(i);
            }
        }
        // write in graph tokens
        graph_start_idx = cur;
        if (graph_tokenizer->include_nodes_in_graph_tokenization) {
            graph_nodes_start_idx = cur + static_cast<int>(graph_tokenizer->tokenized_inputs.shape()[0]) -
                                    static_cast<int>(graph_tokenizer->node_list.size());
        }
        for (size_t i = 0; i < graph_tokenizer->tokenized_inputs.shape()[0]; i++, cur++) {
            tokenized_inputs(cur, 0) = graph_tokenizer->tokenized_inputs(i, 0);
            if (concat_edges) {
                tokenized_inputs(cur, 1) = graph_tokenizer->tokenized_inputs(i, 1);
            }
            tokenized_positions(cur, 0) = graph_tokenizer->tokenized_pos(i, 0);
            if (use_graph_structure) {
                tokenized_positions(cur, 1) = graph_tokenizer->tokenized_pos(i, 1);
            }
        }
        if (query_at_end and task) {
            // write in query if after graph
            query_start_idx = cur;
            for (size_t i = 0; i < task->tokenized_query_inputs.shape()[0]; i++, cur++) {
                tokenized_inputs(cur, 0) = task->tokenized_query_inputs(i);
                if (concat_edges) {
                    tokenized_inputs(cur, 1) = tokenized_inputs(cur, 0);
                }
                tokenized_positions(cur, 0) = task->tokenized_query_pos(i);
            }
        }

        if (num_thinking_tokens > 0) {
            // write in thinking tokens, these are part of the prefix!
            thinking_tokens_start_idx = cur;
            auto thinking_token_id = dictionary.at("!");
            auto thinking_start_marker = pos_dictionary.at("thinking_start");
            auto thinking_end_marker = pos_dictionary.at("thinking_end");
            if (num_thinking_tokens > thinking_end_marker - thinking_start_marker + 1) {
                throw std::invalid_argument("num_thinking_tokens exceeds available thinking position markers");
            }
            for (int i = 0; i < num_thinking_tokens; i++, cur++) {
                tokenized_inputs(cur, 0) = thinking_token_id;
                if (concat_edges) {
                    tokenized_inputs(cur, 1) = thinking_token_id;
                }
                tokenized_positions(cur, 0) = thinking_start_marker + i;
            }
        }

        // the task
        task_start_idx = cur; // this gets defined regardless since it is also prefix end index
        if (is_flat_model and task) {
            // write in task and scratchpad
            auto cur_task_pos = 0;
            auto task_start = pos_dictionary.at("task_start");
            auto task_end = pos_dictionary.at("task_end");
            if (task_length > task_end - task_start + 1) {
                throw std::invalid_argument("Task size exceeds available position tokens.");
            }

            if (scratch_pad) {
                // write in scratchpad tokens
                scratch_pad_start_idx = cur;
                for (size_t i = 0; i < scratch_pad->tokenized_inputs.shape()[0]; i++, cur++, cur_task_pos++) {
                    tokenized_inputs(cur, 0) = scratch_pad->tokenized_inputs(i);
                    if (concat_edges) {
                        tokenized_inputs(cur, 1) = tokenized_inputs(cur, 0);
                    }
                    tokenized_positions(cur, 0) = task_start + cur_task_pos;
                    // targets
                    for (size_t j = 0; j < static_cast<size_t>(scratch_pad->tokenized_targets.shape()[1]); j++) {
                        tokenized_targets(cur_task_pos, j) = scratch_pad->tokenized_targets(i, j);
                    }
                }
            }
            // write in task
            true_task_start_idx = cur;
            for (size_t i = 0; i < task->tokenized_task_inputs.shape()[0]; i++, cur++, cur_task_pos++) {
                tokenized_inputs(cur, 0) = task->tokenized_task_inputs(i);
                if (concat_edges) {
                    tokenized_inputs(cur, 1) = task->tokenized_task_inputs(i);
                }
                tokenized_positions(cur, 0) = task_start + cur_task_pos;
                // targets
                for (size_t j = 0; j < static_cast<size_t>(task->tokenized_task_targets.shape()[1]); j++) {
                    tokenized_targets(cur_task_pos, j) = task->tokenized_task_targets(i, j);
                }
            }
            // end of sequence
            auto end_marker = dictionary.at("</s>");
            tokenized_inputs(cur, 0) = end_marker;
            if (concat_edges) {
                tokenized_inputs(cur, 1) = end_marker;
            }
            tokenized_targets(cur_task_pos, 0) = end_marker; // target also end marker
            tokenized_positions(cur, 0) = task_start + cur_task_pos;
            if (use_graph_structure) {
                tokenized_positions(cur, 1) = task_start + cur_task_pos;
            }
            cur++;
            cur_task_pos++;
        } // not implemented for non-flat models yet
    }

    void pprint(const int pad_id = 1) const {
        auto max_num_digits = 0; // so we can align columns
        for (size_t i = 0; i < tokenized_inputs.shape()[0]; i++) {
            for (size_t j = 0; j < tokenized_inputs.shape()[1]; j++) {
                auto num_digits = static_cast<int>(to_string(tokenized_inputs(i, j)).size());
                if (num_digits > max_num_digits) {
                    max_num_digits = num_digits;
                }
            }
            for (size_t j = 0; j < tokenized_positions.shape()[1]; j++) {
                auto num_digits = static_cast<int>(to_string(tokenized_positions(i, j)).size());
                if (num_digits > max_num_digits) {
                    max_num_digits = num_digits;
                }
            }
        }

        string s = "Tokenized Inputs, Targets, and Positions:\n";
        s += "Pos:   ";
        for (size_t j = 0; j < tokenized_positions.shape()[1]; j++) {
            for (size_t i = 0; i < tokenized_positions.shape()[0]; i++) {
                s += to_string(tokenized_positions(i, j));
                s += string(max_num_digits - to_string(tokenized_positions(i, j)).size() + 1, ' ');
            }
            if (j == tokenized_positions.shape()[1] - 1) {
                s += "\n";
            } else {
                s += "\n       ";
            }
        }

        s += "Input: "; // 7
        for (size_t j = 0; j < tokenized_inputs.shape()[1]; j++) {
            for (size_t i = 0; i < tokenized_inputs.shape()[0]; i++) {
                s += to_string(tokenized_inputs(i, j));
                s += string(max_num_digits - to_string(tokenized_inputs(i, j)).size() + 1, ' ');
            }
            if (j == tokenized_inputs.shape()[1] - 1) {
                s += "\n";
            } else {
                s += "\n       ";
            }
        }

        s += "TGT:   "; // these are multi line, if pad_id then just print empty padded to length
        auto empty = string(max_num_digits + 1, ' ');
        for (size_t j = 0; j < tokenized_targets.shape()[1]; j++) {
            s += string(task_start_idx * (max_num_digits + 1), ' '); // pad to align with inputs
            for (size_t i = 0; i < tokenized_targets.shape()[0]; i++) {
                if (tokenized_targets(i, j) != pad_id) {
                    s += to_string(tokenized_targets(i, j));
                    s += string(max_num_digits - to_string(tokenized_targets(i, j)).size() + 1, ' ');
                } else {
                    s += empty;
                }
            }
            if (j == tokenized_targets.shape()[1] - 1) {
                s += "\n";
            } else {
                s += "\n       ";
            }
        }

        s += "Idx:   "; // just print range for reference
        for (size_t i = 0; i < tokenized_inputs.shape()[0]; i++) {
            s += to_string(i);
            s += string(max_num_digits - to_string(i).size() + 1, ' ');
        }
        s += "\n";
        cout << s << endl;
    }
};

template<typename D>
class BatchedInstances {
public:
    vector<Instance<D> > instances;
    string graph_type;
    string task_type;
    int min_vocab;
    int max_vocab;
    // tokenization parameters
    bool query_at_end;
    int num_thinking_tokens;
    bool is_flat_model;
    bool align_prefix_front_pad;

    BatchedInstances(const string &graph_type, const string &task_type,
                     const int min_vocab, int max_vocab,

            // tokenization parameters
                     const bool query_at_end = false,
                     const int num_thinking_tokens = 0,
                     const bool is_flat_model = true,
                     const bool align_prefix_front_pad = false
    ) {
        this->graph_type = graph_type;
        this->task_type = task_type;
        this->min_vocab = min_vocab;
        this->max_vocab = max_vocab;
        this->query_at_end = query_at_end;
        this->num_thinking_tokens = num_thinking_tokens;
        this->is_flat_model = is_flat_model;
        this->align_prefix_front_pad = align_prefix_front_pad;

        instances = vector<Instance<D> >();
    }

    void add(Instance<D> &instance) {
        instances.push_back(std::move(instance));
    }

    int size() {
        return static_cast<int>(instances.size());
    }

    py::dict package_for_model(const map<std::string, int> &dictionary, const map<std::string, int> pos_dictionary) {
        if (instances.size() == 0) {
            throw std::invalid_argument("No instances to package for model.");
        }
        py::dict d;

        // make all the info needed for the model input
        int max_tokenized_inputs_len = 0;
        int max_tokenized_targets_len = 0;
        int max_tokenized_targets_labels = 0;
        int max_tokenized_true_targets_len = 0;
        int max_tokenized_scratchpad_len = 0;
        int max_prefix_size = 0;
        int max_num_nodes = 0;
        int max_num_edges = 0;

        auto concat_edges = instances[0].graph_tokenizer->concat_edges;
        auto use_graph_structure = instances[0].graph_tokenizer->use_graph_structure;
        auto include_nodes_in_graph_tokenization = instances[0].graph_tokenizer->include_nodes_in_graph_tokenization;

        for (size_t i = 0; i < instances.size(); i++) {
            // create inputs, targets and positions, and component start_indices and lengths
            instances[i].tokenize(
                    dictionary,
                    pos_dictionary,
                    query_at_end,
                    num_thinking_tokens,
                    is_flat_model
            );
            // batch the info
            if (static_cast<int>(instances[i].tokenized_inputs.shape()[0]) > max_tokenized_inputs_len) {
                max_tokenized_inputs_len = static_cast<int>(instances[i].tokenized_inputs.shape()[0]);
            }
            if (instances[i].task_start_idx > max_prefix_size) {  // use task start index to get prefix size
                max_prefix_size = instances[i].task_start_idx;
            }
            if (instances[i].task) {
                if (static_cast<int>(instances[i].tokenized_targets.shape()[0]) > max_tokenized_targets_len) {
                    max_tokenized_targets_len = static_cast<int>(instances[i].tokenized_targets.shape()[0]);
                }
                if (static_cast<int>(instances[i].tokenized_targets.shape()[1]) > max_tokenized_targets_labels) {
                    max_tokenized_targets_labels = static_cast<int>(instances[i].tokenized_targets.shape()[1]);
                }
                if (static_cast<int>(instances[i].true_task_length) > max_tokenized_true_targets_len) {
                    max_tokenized_true_targets_len = static_cast<int>(instances[i].true_task_length);
                }
                if (instances[i].scratch_pad) {
                    if (static_cast<int>(instances[i].scratch_pad_length) > max_tokenized_scratchpad_len) {
                        max_tokenized_scratchpad_len = static_cast<int>(instances[i].scratch_pad_length);
                    }
                }
            }
            if (instances[i].N > max_num_nodes) {
                max_num_nodes = instances[i].N;
            }
            if (instances[i].E > max_num_edges) {
                max_num_edges = instances[i].E;
            }
            // instances[i].pprint();
        }

        // set up numpy arrays
        auto batch_size = static_cast<int>(instances.size());

        auto new_max_tokenized_inputs_len = max_tokenized_inputs_len;
        if (align_prefix_front_pad and is_flat_model) {  // only align if flat model since non-flat separate prefixes
            // align up to prefix size and then targets after
            new_max_tokenized_inputs_len = max_prefix_size + max_tokenized_targets_len;
        }
        auto src_tokens = py::array_t<int, py::array::c_style>(
                {batch_size, new_max_tokenized_inputs_len, concat_edges ? 2 : 1});
        src_tokens[py::make_tuple(py::ellipsis())] = dictionary.at("<pad>");
        auto src_lengths = py::array_t<int, py::array::c_style>({batch_size});

        auto task_targets = py::array_t<int, py::array::c_style>(
                {batch_size, max_tokenized_targets_len, max_tokenized_targets_labels});
        task_targets[py::make_tuple(py::ellipsis())] = dictionary.at("<pad>");
        auto task_lengths = py::array_t<int, py::array::c_style>({batch_size});

        auto positions = py::array_t<int, py::array::c_style>(
                {batch_size, new_max_tokenized_inputs_len, use_graph_structure ? 2 : 1});
        positions[py::make_tuple(py::ellipsis())] = pos_dictionary.at("pad");

        py::array_t<int, py::array::c_style> num_nodes(batch_size);
        py::array_t<int, py::array::c_style> num_edges(batch_size);

        py::array_t<int, py::array::c_style> query_start_indices(batch_size);
        py::array_t<int, py::array::c_style> query_lengths(batch_size);

        py::array_t<int, py::array::c_style> graph_start_indices(
                batch_size); // includes both edges and nodes if applicable
        py::array_t<int, py::array::c_style> graph_lengths(batch_size); // includes both edges and nodes if applicable
        py::array_t<int, py::array::c_style> graph_edge_start_indices(batch_size); // includes only edges
        py::array_t<int, py::array::c_style> graph_edge_lengths(batch_size); // includes only edges
        py::array_t<int, py::array::c_style> graph_node_start_indices(batch_size); // includes nodes if applicable
        py::array_t<int, py::array::c_style> graph_node_lengths(batch_size); // includes nodes if applicable

        py::array_t<int, py::array::c_style> thinking_tokens_start_idx(batch_size);
        py::array_t<int, py::array::c_style> thinking_tokens_length(batch_size);

        py::array_t<int, py::array::c_style> task_start_indices(batch_size);
        py::array_t<int, py::array::c_style> scratch_pad_start_indices(batch_size);
        py::array_t<int, py::array::c_style> scratch_pad_lengths(batch_size);
        py::array_t<int, py::array::c_style> true_task_start_indices(batch_size);
        py::array_t<int, py::array::c_style> true_task_lengths(batch_size);

        // gather_ids, these select hidden-states for computing loss(es)
        // these are just start_index + range(length) with padding, but we might as well precompute them here
        auto task_gather_indices = py::array_t<int, py::array::c_style>({batch_size, max_tokenized_targets_len});
        task_gather_indices[py::make_tuple(py::ellipsis())] = 0; // initialize to 0 since it needs to be a valid index
        auto true_task_gather_indices = py::array_t<int, py::array::c_style>({batch_size, max_tokenized_true_targets_len + 1});
        true_task_gather_indices[py::make_tuple(py::ellipsis())] = 0; // initialize to 0 since it needs to be a valid index
        auto scratch_pad_gather_indices = py::array_t<int, py::array::c_style>({batch_size, max_tokenized_scratchpad_len});
        scratch_pad_gather_indices[py::make_tuple(py::ellipsis())] = 0;
        auto graph_edge_gather_indices = py::array_t<int, py::array::c_style>({batch_size, max_num_edges});
        graph_edge_gather_indices[py::make_tuple(py::ellipsis())] = 0;
        auto graph_node_gather_indices = py::array_t<int, py::array::c_style>({batch_size, max_num_nodes});
        graph_node_gather_indices[py::make_tuple(py::ellipsis())] = 0;

        // write into numpy arrays
        auto src_tokens_ar = src_tokens.mutable_unchecked();
        auto src_lengths_ar = src_lengths.mutable_unchecked();
        auto positions_ar = positions.mutable_unchecked();
        auto task_targets_ar = task_targets.mutable_unchecked();
        auto task_lengths_ar = task_lengths.mutable_unchecked();

        auto num_nodes_ar = num_nodes.mutable_unchecked();
        auto num_edges_ar = num_edges.mutable_unchecked();

        auto query_start_indices_ar = query_start_indices.mutable_unchecked();
        auto query_lengths_ar = query_lengths.mutable_unchecked();
        auto graph_start_indices_ar = graph_start_indices.mutable_unchecked();
        auto graph_lengths_ar = graph_lengths.mutable_unchecked();
        auto graph_edge_start_indices_ar = graph_edge_start_indices.mutable_unchecked();
        auto graph_edge_lengths_ar = graph_edge_lengths.mutable_unchecked();
        auto graph_node_start_indices_ar = graph_node_start_indices.mutable_unchecked();
        auto graph_node_lengths_ar = graph_node_lengths.mutable_unchecked();
        auto thinking_tokens_start_idx_ar = thinking_tokens_start_idx.mutable_unchecked();
        auto thinking_tokens_length_ar = thinking_tokens_length.mutable_unchecked();
        auto task_start_indices_ar = task_start_indices.mutable_unchecked();
        auto scratch_pad_start_indices_ar = scratch_pad_start_indices.mutable_unchecked();
        auto scratch_pad_lengths_ar = scratch_pad_lengths.mutable_unchecked();
        auto true_task_start_indices_ar = true_task_start_indices.mutable_unchecked();
        auto true_task_lengths_ar = true_task_lengths.mutable_unchecked();

        auto task_gather_indices_ar = task_gather_indices.mutable_unchecked();
        auto true_task_gather_indices_ar = true_task_gather_indices.mutable_unchecked();
        auto scratch_pad_gather_indices_ar = scratch_pad_gather_indices.mutable_unchecked();
        auto graph_edge_gather_indices_ar = graph_edge_gather_indices.mutable_unchecked();
        auto graph_node_gather_indices_ar = graph_node_gather_indices.mutable_unchecked();

        for (size_t i = 0; i < instances.size(); i++) {
            auto offset = 0;
            if (align_prefix_front_pad and is_flat_model) {
                offset = max_prefix_size - instances[i].task_start_idx;
            }
            for (size_t j = 0; j < instances[i].tokenized_inputs.shape()[0]; j++) {
                for (size_t k = 0; k < instances[i].tokenized_inputs.shape()[1]; k++) {
                    src_tokens_ar(i, j + offset, k) = instances[i].tokenized_inputs(j, k);
                }
                for (size_t k = 0; k < instances[i].tokenized_positions.shape()[1]; k++) {
                    positions_ar(i, j + offset, k) = instances[i].tokenized_positions(j, k);
                }
            }
            if (instances[i].task) {
                for (size_t j = 0; j < instances[i].tokenized_targets.shape()[0]; j++) {
                    task_gather_indices_ar(i, j) = instances[i].task_start_idx + offset + j;
                    for (size_t k = 0; k < instances[i].tokenized_targets.shape()[1]; k++) {
                        task_targets_ar(i, j, k) = instances[i].tokenized_targets(j, k);
                    }
                }
                task_lengths_ar(i) = instances[i].task_length;
                for (int j = 0; j < instances[i].true_task_length + 1; j++) {  // +1 for end of seq token
                    true_task_gather_indices_ar(i, j) = instances[i].true_task_start_idx + offset + j;
                }
                if (instances[i].scratch_pad) {
                    for (int j = 0; j < instances[i].scratch_pad_length; j++) {
                        scratch_pad_gather_indices_ar(i, j) = instances[i].scratch_pad_start_idx + offset + j;
                    }
                }
            }
            for (size_t j = 0; j < instances[i].graph_tokenizer->edge_list.size(); j++) {
                if(concat_edges) {
                    graph_edge_gather_indices_ar(i, j) = instances[i].graph_start_idx + offset + j;
                } else { // need every third token since we just want the edge marker positions (which needs shift by 2)
                    graph_edge_gather_indices_ar(i, j) = instances[i].graph_start_idx + offset + (j * 3) + 2;
                }
            }
            if (include_nodes_in_graph_tokenization) {
                for (size_t j = 0; j < instances[i].graph_tokenizer->node_list.size(); j++) {
                    graph_node_gather_indices_ar(i, j) = instances[i].graph_nodes_start_idx + offset + j;
                }
            }

            // all others
            num_nodes_ar(i) = instances[i].N;
            num_edges_ar(i) = instances[i].E;
            src_lengths_ar(i) = instances[i].tokenized_inputs.shape()[0] + offset;
            query_start_indices_ar(i) = instances[i].query_start_idx + offset;
            query_lengths_ar(i) = instances[i].query_length;
            graph_start_indices_ar(i) = instances[i].graph_start_idx + offset;
            graph_lengths_ar(i) = instances[i].graph_length;
            graph_edge_start_indices_ar(i) = instances[i].graph_start_idx + offset;
            graph_edge_lengths_ar(i) = instances[i].graph_length;
            graph_node_start_indices_ar(i) = instances[i].graph_nodes_start_idx + offset;
            graph_node_lengths_ar(i) = instances[i].graph_nodes_length;
            thinking_tokens_start_idx_ar(i) = instances[i].thinking_tokens_start_idx + offset;
            thinking_tokens_length_ar(i) = instances[i].thinking_tokens_length;
            task_start_indices_ar(i) = instances[i].task_start_idx + offset;
            scratch_pad_start_indices_ar(i) = instances[i].scratch_pad_start_idx + offset;
            scratch_pad_lengths_ar(i) = instances[i].scratch_pad_length;
            true_task_start_indices_ar(i) = instances[i].true_task_start_idx + offset;
            true_task_lengths_ar(i) = instances[i].true_task_length;
        }

        d["num_nodes"] = num_nodes;
        d["num_edges"] = num_edges;
        d["src_tokens"] = src_tokens;
        d["src_lengths"] = src_lengths;

        if( instances[0].use_task_structure ) {
            d["positions"] = positions;
        } else{
            d["positions"] = py::none();  // just delete the made positions and model will use range per norm
        }
        // has task
        if (instances[0].task) {
            d["prev_output_tokens"] = task_targets;  // fairseq naming convention, yuck
            d["query_start_indices"] = query_start_indices;
            d["query_lengths"] = query_lengths;
        } else {
            d["prev_output_tokens"] = py::none();
            d["query_start_indices"] = py::none();
            d["query_lengths"] = py::none();
        }
        d["graph_start_indices"] = graph_start_indices;
        d["graph_lengths"] = graph_lengths;
        d["graph_edge_start_indices"] = graph_edge_start_indices;
        d["graph_edge_lengths"] = graph_edge_lengths;
        d["graph_edge_gather_indices"] = graph_edge_gather_indices;
        if (include_nodes_in_graph_tokenization) {
            d["graph_node_start_indices"] = graph_node_start_indices;
            d["graph_node_lengths"] = graph_node_lengths;
            d["graph_node_gather_indices"] = graph_node_gather_indices;
        } else {
            d["graph_node_start_indices"] = py::none();
            d["graph_node_lengths"] = py::none();
            d["graph_node_gather_indices"] = py::none();
        }
        d["thinking_tokens_start_idx"] = thinking_tokens_start_idx;
        d["thinking_tokens_length"] = thinking_tokens_length;

        if (instances[0].task) {
            d["task_start_indices"] = task_start_indices;
            d["task_lengths"] = task_lengths;
            d["task_gather_indices"] = task_gather_indices;

            if (instances[0].scratch_pad) {
                d["scratch_pad_start_indices"] = scratch_pad_start_indices;
                d["scratch_pad_lengths"] = scratch_pad_lengths;
                d["scratch_pad_gather_indices"] = scratch_pad_gather_indices;
            } else {
                d["scratch_pad_start_indices"] = py::none();
                d["scratch_pad_lengths"] = py::none();
                d["scratch_pad_gather_indices"] = py::none();
            }
            d["true_task_start_indices"] = true_task_start_indices;
            d["true_task_lengths"] = true_task_lengths;
            d["true_task_gather_indices"] = true_task_gather_indices;
        } else {
            d["task_start_indices"] = py::none();
            d["task_lengths"] = py::none();
            d["task_task_gather_indices"] = py::none();
            d["scratch_pad_start_indices"] = py::none();
            d["scratch_pad_lengths"] = py::none();
            d["true_task_start_indices"] = py::none();
            d["true_task_lengths"] = py::none();
            d["true_task_gather_indices"] = py::none();
        }

        // distances and ground truths batching

        auto bd = batch_distances<int>();  // do node shuffling inside
        d["distances"] = bd;  // (B, max_vocab, max_vocab), where max_vocab is just up to node range (not extra symbols)
        d["hashes"] = hash_distance_matrix<int>(bd);

        auto gt_gather_indices_and_distances = batch_ground_truth_gather_indices<int>();
        if (instances[0].graph_tokenizer->is_direct_ranking) {
            d["ground_truths_gather_indices"] = gt_gather_indices_and_distances.first;
        }else{
            d["ground_truths_gather_indices"] = py::none();
        }
        d["ground_truths_gather_distances"] = gt_gather_indices_and_distances.second;

        // if euclidean batch the node positions  TODO

        // arguments
        d["graph_type"] = graph_type;
        d["task_type"] = task_type;
        d["is_flat_model"] = is_flat_model;
        d["concat_edges"] = concat_edges;
        d["query_at_end"] = query_at_end;
        d["align_prefix_front_pad"] = align_prefix_front_pad;
        d["min_vocab"] = min_vocab;
        d["max_vocab"] = max_vocab;

        return d;
    }

    template<typename T>
    py::array_t<T, py::array::c_style> batch_distances(T cuttoff = 100000, T max_value = -1, T pad = -1) {
        auto batch_size = static_cast<int>(instances.size());
        py::array_t<T, py::array::c_style> arr({static_cast<int>(batch_size), this->max_vocab, this->max_vocab });
        arr[py::make_tuple(py::ellipsis())] = static_cast<T>(pad); // initialize array
        auto ra = arr.mutable_unchecked();

        for (int b = 0; b < batch_size; b++) {
            const auto &distances_ptr = instances[b].graph_tokenizer->distances_ptr;
            auto &node_shuffle_map = instances[b].node_shuffle_map;
            for (int j = 0; j < static_cast<int>(distances_ptr->size()); j++) {
                for (int k = 0; k < static_cast<int>((*distances_ptr)[j].size()); k++) {
                    auto mapped_j = node_shuffle_map[j];
                    auto mapped_k = node_shuffle_map[k];
                    if (cuttoff > 0 && (*distances_ptr)[j][k] >= cuttoff) {
                        ra(b, mapped_j, mapped_k) = max_value;
                    } else {
                        ra(b, mapped_j, mapped_k) = (*distances_ptr)[j][k];
                    }
                }
            }
        }
        return arr;
    }

    template<typename T>
    pair<py::array_t<T, py::array::c_style>, py::array_t<T, py::array::c_style> > batch_ground_truth_gather_indices(
            T cuttoff = 100000, T max_value = -1, T pad = -1) {
        // indices are nodes, values are distances
        // note that if the node shuffle map as a big range this will be very empty, so we cut it down to num nodes

        int max_d1 = 0;  // either edges or nodes
        int max_d2 = 0;  // nodes
        auto batch_size = static_cast<int>(instances.size());
        for (int b = 0; b < batch_size; b++) {
            const auto &graph_ground_truths_ptr = instances[b].graph_tokenizer->graph_ground_truths_ptr;
            if (static_cast<int>(graph_ground_truths_ptr->size()) > max_d1) {
                max_d1 = static_cast<int>(graph_ground_truths_ptr->size());
            }
            auto num_nodes = instances[b].N;
            if (num_nodes > max_d2) {
                max_d2 = num_nodes;
            }
        }

        py::array_t<T, py::array::c_style> arr_indices({static_cast<int>(batch_size), max_d1, max_d2});
        py::array_t<T, py::array::c_style> arr_distances({static_cast<int>(batch_size), max_d1, max_d2});
        arr_indices[py::make_tuple(py::ellipsis())] = 0;; // initialize array
        arr_distances[py::make_tuple(py::ellipsis())] = static_cast<T>(pad); // initialize array
        auto ra_i = arr_indices.mutable_unchecked();
        auto ra_d = arr_distances.mutable_unchecked();

        for (int b = 0; b < batch_size; b++) {
            const auto &graph_ground_truths_ptr = instances[b].graph_tokenizer->graph_ground_truths_ptr;
            auto node_shuffle_map = instances[b].node_shuffle_map;

            for (int j = 0; j < static_cast<int>(graph_ground_truths_ptr->size()); j++) {
                auto cur_gt = 0;
                for (int k = 0; k < static_cast<int>((*graph_ground_truths_ptr)[j].size()); k++) {
                    if (((*graph_ground_truths_ptr)[j][k] >= 0) && ((cuttoff <= 0) || ((*graph_ground_truths_ptr)[j][k] < cuttoff))) {
                        ra_i(b, j, cur_gt) = node_shuffle_map[k];
                        ra_d(b, j, cur_gt) = (*graph_ground_truths_ptr)[j][k];
                        cur_gt += 1;
                    }
                }
            }
        }
        return make_pair(arr_indices, arr_distances);
    }

    // Hashing
    // has each distance matrix as a string, return the hashes as a numpy array
    template<typename T>
    py::array_t<std::uint64_t, py::array::c_style> hash_distance_matrix(
            const py::array_t<T, py::array::c_style> &batched_distances) {
        // Convert a distance matrix [N, N] to a numpy array [new_N, new_N] by mapping node ids
        auto shape = batched_distances.shape();
        py::array_t<std::uint64_t, py::array::c_style> arr({static_cast<int>(shape[0])});
        auto ra = arr.mutable_unchecked();
        auto bd = batched_distances.unchecked();
        for (int b = 0; b < shape[0]; b++) {
            // make string from distance matrix
            std::string str = "";
            for (int i = 0; i < shape[1]; i++) {
                for (int j = 0; j < shape[2]; j++) {
                    str += std::to_string(bd(b, i, j));
                }
            }
            auto hash = std::hash<std::string>{}(str);
            // auto has2 = static_cast<std::uint64_t>(hash);
            // cout << "hash: " << hash << " has2: " << has2 << endl;
            ra(b) = static_cast<std::uint64_t>(hash);
        }
        return arr;
    }


};


template<typename T>
void print_np(py::array_t<T, py::array::c_style> arr, const string &title = "", bool full = true,
              const int cutoff = 100000) {
    if (!title.empty()) {
        std::cout << title << std::endl;
    }
    auto ra = arr.mutable_unchecked();
    // std::cout << "Shape: " << arr.ndim() << std::endl;
    for (int i = 0; i < arr.ndim(); i++) {
        std::cout << "Dim " << i << ": " << arr.shape(i) << " ";
    }
    std::cout << std::endl;
    if (arr.ndim() == 1) {
        for (int i = 0; i < arr.shape(0); i++) {
            std::cout << ra(i) << " ";
        }
    } else if (arr.ndim() == 2) {
        for (int i = 0; i < arr.shape(0); i++) {
            for (int j = (full) ? 0 : i; j < arr.shape(1); j++) {
                if (ra(i, j) >= cutoff) {
                    std::cout << "inf " << std::endl;
                } else {
                    std::cout << ra(i, j) << " ";
                }
            }
            std::cout << std::endl;
        }
    } else if (arr.ndim() == 3) {
        for (int b = 0; b < arr.shape(0); b++) {
            std::cout << "Batch " << b << ":" << std::endl;
            for (int i = 0; i < arr.shape(1); i++) {
                for (int j = (full) ? 0 : i; j < arr.shape(2); j++) {
                    if (ra(b, i, j) >= cutoff) {
                        std::cout << "inf " << std::endl;
                    } else {
                        std::cout << ra(b, i, j) << " ";
                    }
                }
                std::cout << std::endl << std::endl;
            }
        }
    }
}

void pprint_dict(const py::dict &d) {
    // not very useful, just do this this in python at this point
    auto src_tokens = d["src_tokens"];
    print_np(src_tokens.cast<py::array_t<int, py::array::c_style> >(), "src_tokens");


}


#endif //GRAPHGEN_INSTANCE_H
