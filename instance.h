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
#include "args.h"
#include "undirected_graphs.h"
#include "directed_graphs.h"
#include "graph_wrapper.h"
#include "graph_tokenizer.h"
#include "tasks.h"
#include "scratch_pads.h"

using namespace std;

namespace py = pybind11;
using namespace py::literals;


/*
 * Instance class to hold all info for a single instance
 */
template<typename D>
class Instance {
public:

    const Args &args;
    unique_ptr<GraphWrapper<D>> graph = nullptr;  // optional because of k_hops

    unique_ptr<GraphTokenizer> graph_tokenizer = nullptr;
    unique_ptr<Task> task = nullptr;
    unique_ptr<ScratchPad> scratch_pad = nullptr;
    unique_ptr<vector<vector<float> > > positions_ptr = nullptr;

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
    // task is anything the model generates i->E scratch pad and actual task
    int task_start_idx = -1; // index in the tokenized input where the task starts
    int task_length = 0;

    Instance(std::mt19937 &gen, unique_ptr<GraphWrapper<D>> ggraph, Args &args,
             const map<std::string, int> &dictionary,
             const map<std::string, int> &pos_dictionary
    ): args(args), graph(std::move(ggraph)) {

        if (!args.tok->no_graph) {  // for variant of BFS scratchpad task
            graph_tokenizer = make_unique<GraphTokenizer>(args.tok, args.pos, graph->edge_list, graph->node_list);
            graph_tokenizer->tokenize(dictionary, graph->node_shuffle_map, pos_dictionary, gen);
        }

        // issue of when to convert these to node shuffle map order, it can not be before we make the task
        // since these need the original distance matrix in the original node order
        if (graph->g_ptr) {
            if (args.tok->include_nodes_in_graph_tokenization) {
                graph->get_distances();
                graph->get_node_ground_truths(args.tok->is_direct_ranking);
            } else {
                // this also gets the distances due to legacy code
                graph->get_edge_ground_truths(args.tok->is_causal);
            }
        }

        if (args.task->task_type == "shortest_path") {
            // the ShortestPathTask and BFSScratchPad are badly coupled
            // this is because we use the ShortestPathTask to sample a start and end point
            // however the BFSScratchPad needs to be able to construct the path since otherwise
            // the scratch pad could be inconsistent with the task path in the case when multiple shortest paths exist
            // thus we need to use the one first discovered by BFS.  Same goes for the DFS scratch pad.

            auto should_sample_path = args.sp->scratchpad_type == "none" or args.sp->scratchpad_type == "None";
            auto short_path = make_unique<ShortestPathTask>(gen, args.task, args.pos,
                                                            graph->distances_ptr,
                                                            graph->start, graph->end,
                                                            should_sample_path);
            if (args.sp->scratchpad_type == "bfs" || args.sp->scratchpad_type == "BFS") {
                auto bfs_scratch_pad = make_unique<BFSScratchPad>(args.sp, args.pos,
                                                                  graph->g_ptr, graph->edge_list,
                                                                  short_path->start, short_path->end);
                short_path->set_path(bfs_scratch_pad->path, graph->distances_ptr, false);
                scratch_pad = std::move(bfs_scratch_pad);
            } else if (args.sp->scratchpad_type == "dfs" || args.sp->scratchpad_type == "DFS") {
                /*
                auto dfs_scratch_pad = make_unique<DFSScratchPad>(short_path->start, short_path->end, g_ptr,
                                                                  node_shuffle_map, sort_adjacency_lists,
                                                                  use_unique_depth_markers);
                short_path->set_path(dfs_scratch_pad->path, graph_tokenizer->distances_ptr, false);
                scratch_pad = std::move(dfs_scratch_pad);
                */
                throw std::invalid_argument("DFS SP not implemented yet");

            }
            task = std::move(short_path);
        } else if (args.task->task_type == "bfs" || args.task->task_type == "BFS") {  // the scratchpad generation as the main task
            task = make_unique<BFSTask>(gen, args.task, args.sp, args.pos,
                                        graph->g_ptr, graph->edge_list,
                                        graph->distances_ptr,
                                        graph->start, graph->end);
        } else if (args.task->task_type == "dfs" || args.task->task_type == "DFS") {
            throw std::invalid_argument("DFS task not implemented yet");

        } else if (args.task->task_type == "center" || args.task->task_type == "centroid") {
            //task = make_unique<CenterTask>(gen, dynamic_cast<const CenterCentroidTaskArgs&>(args.task), args.pos,
            //                               graph->distances_ptr);
            // scratch_pad in future?
        } else if (args.task->task_type == "khops_gen") {
            task = make_unique<KHopsGenTask>(gen, args.task, args.pos, graph->min_vocab, graph->max_vocab, graph->khops_segment_lengths);
        } else if (args.task->task_type == "khops") {
            task = make_unique<KHopsTask>(gen, args.task, args.pos, graph->min_vocab, graph->max_vocab,
                                          graph->khops_k, graph->khops_max_k, graph->khops_prefix_length);
        }
        if (task) {
            task->tokenize(dictionary, graph->node_shuffle_map, pos_dictionary, gen);
        }
        if (scratch_pad) {
            scratch_pad->tokenize(dictionary, graph->node_shuffle_map, pos_dictionary, gen);
        }
    }

    void tokenize(
            /*
             * Creates the input sequence, the target sequence, and the positional embeddings
             */
            const map<std::string, int> &dictionary,
            const map<std::string, int> pos_dictionary,
            const bool should_repeat = true) {

        // get length and size info before coping over tokenizations in correct places
        int num_tokens = 2; //  seq_length +2 for start and end sequence tokens
        int max_labels = 1;  //  target size
        int max_input_struct = 1; // for structure input
        int max_pos_struct = 0;  // max position token id structure
        if (task) {  // really query, which is part of task
            query_length = static_cast<int>(task->tokenized_query_inputs.shape()[0]);
            max_input_struct = max(max_input_struct, static_cast<int>(task->tokenized_query_inputs.shape()[1]));
            num_tokens += query_length;
            max_labels = static_cast<int>(task->tokenized_task_targets.shape()[1]);
            max_pos_struct = max(max_pos_struct, static_cast<int>(task->tokenized_query_pos.shape()[1]));
        }
        if (graph_tokenizer) {  // if graph
            graph_length = static_cast<int>(graph_tokenizer->tokenized_inputs.shape()[0]);
            if (args.tok->include_nodes_in_graph_tokenization) {
                graph_nodes_length = static_cast<int>(graph_tokenizer->node_list.size());
            }
            if (!args.tok->no_graph) {
                num_tokens += graph_length;
                max_input_struct = max(max_input_struct,
                                       static_cast<int>(graph_tokenizer->tokenized_inputs.shape()[1]));
                // has no target labels since those are done separately
                max_pos_struct = max(max_pos_struct, static_cast<int>(graph_tokenizer->tokenized_pos.shape()[1]));
            }
        }
        // if thinking tokens
        thinking_tokens_length = args.tok->num_thinking_tokens;
        num_tokens += thinking_tokens_length;
        if (args.tok->is_flat_model) {  // task
            if (task) {  // we set values in task so it must come before scratch pad since it can affect task length
                task_length = static_cast<int>(task->tokenized_task_inputs.shape()[0]);
                true_task_length = task_length;
                max_input_struct = max(max_input_struct, static_cast<int>(task->tokenized_task_inputs.shape()[1]));
                max_labels = max(max_labels, static_cast<int>(task->tokenized_task_targets.shape()[1]));
                // max_pos_struct = max(max_pos_struct, static_cast<int>(task->tokenized_task_pos.shape()[1]));
            }
            if (scratch_pad) {
                scratch_pad_length = static_cast<int>(scratch_pad->tokenized_inputs.shape()[0]);
                if (args.tok->scratchpad_as_prefix) {
                    num_tokens += scratch_pad_length;
                } else {
                    task_length += scratch_pad_length;
                }
                max_input_struct = max(max_input_struct, static_cast<int>(scratch_pad->tokenized_inputs.shape()[1]));
                max_labels = max(max_labels, static_cast<int>(scratch_pad->tokenized_targets.shape()[1]));
                // max_pos_struct = max(max_pos_struct, static_cast<int>(scratch_pad->tokenized_pos.shape()[1]));
            }
            num_tokens += task_length;
        } else {  // the task is separate
            throw std::invalid_argument("Non-flat model tokenization not implemented yet");
        }

        tokenized_inputs = Matrix<int>(num_tokens, max_input_struct, dictionary.at("<pad>"));
        if (task) {
            tokenized_targets = Matrix<int>(task_length + 1, max_labels, dictionary.at("<pad>"));
        }
        tokenized_positions = Matrix<int>(num_tokens, max_pos_struct, pos_dictionary.at("pad"));

        // write in values
        auto cur = 0;
        // start of sequence
        tokenized_inputs.set_tok(cur, dictionary.at("<s>"), should_repeat);
        tokenized_positions.set_tok(cur, pos_dictionary.at("misc_start"), should_repeat);
        cur++;

        if (task && !args.tok->query_at_end) {  // write in query if before graph
            query_start_idx = cur;
            for (size_t i = 0; i < task->tokenized_query_inputs.shape()[0]; i++, cur++) {
                tokenized_inputs.copy_tok(task->tokenized_query_inputs, cur, i,
                         should_repeat);  //tokenized_inputs(cur, 0) = task->tokenized_query_inputs(i);
                tokenized_positions.copy_tok(task->tokenized_query_pos, cur, i,
                         should_repeat);  // tokenized_positions(cur, 0) = task->tokenized_query_pos(i);
            }
        }
        if (graph_tokenizer) {  // write in graph tokens
            graph_start_idx = cur;
            if (args.tok->include_nodes_in_graph_tokenization) {
                graph_nodes_start_idx = cur + graph_tokenizer->graph_nodes_start_idx;
            }
            for (size_t i = 0; i < graph_tokenizer->tokenized_inputs.shape()[0]; i++, cur++) {
                tokenized_inputs.copy_tok(graph_tokenizer->tokenized_inputs, cur, i,
                         should_repeat);  // tokenized_inputs(cur, 0) = graph_tokenizer->tokenized_inputs(i, 0);
                tokenized_positions.copy_tok(graph_tokenizer->tokenized_pos, cur, i,
                         should_repeat);  // tokenized_positions(cur, 0) = graph_tokenizer->tokenized_pos(i, 0);
            }
        }
        if (task && args.tok->query_at_end) {
            // write in query if after graph
            query_start_idx = cur;
            for (size_t i = 0; i < task->tokenized_query_inputs.shape()[0]; i++, cur++) {
                tokenized_inputs.copy_tok(task->tokenized_query_inputs, cur, i,
                         should_repeat);  //tokenized_inputs(cur, 0) = task->tokenized_query_inputs(i);
                tokenized_positions.copy_tok(task->tokenized_query_pos, cur, i,
                         should_repeat);  // tokenized_positions(cur, 0) = task->tokenized_query_pos(i);
            }
        }
        if (args.tok->num_thinking_tokens > 0) {
            // write in thinking tokens, these are part of the prefix!
            thinking_tokens_start_idx = cur;
            auto thinking_token_id = dictionary.at("!");
            auto thinking_start_marker = pos_dictionary.at("thinking_start");
            auto thinking_end_marker = pos_dictionary.at("thinking_end");
            if (args.tok->num_thinking_tokens > thinking_end_marker - thinking_start_marker + 1) {
                throw std::invalid_argument("num_thinking_tokens exceeds available thinking position markers");
            }
            for (int i = 0; i < args.tok->num_thinking_tokens; i++, cur++) {
                tokenized_inputs.set_tok(cur,thinking_token_id,
                        should_repeat);  // tokenized_inputs(cur, 0) = thinking_token_id;
                tokenized_positions.set_tok(cur,thinking_start_marker + i,
                        should_repeat);  //  tokenized_positions(cur, 0) = thinking_start_marker + i;
            }
        }

        if (!args.tok->scratchpad_as_prefix) {  // the task
            task_start_idx = cur; // this gets defined regardless since it is also prefix end index
        }
        if (task) {
            if (args.tok->is_flat_model) {
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
                    for (size_t i = 0; i < scratch_pad->tokenized_inputs.shape()[0]; i++, cur++) {
                        tokenized_inputs.copy_tok(scratch_pad->tokenized_inputs, cur, i,
                                 should_repeat);  // tokenized_inputs(cur, 0) = scratch_pad->tokenized_inputs(i);
                        tokenized_positions.set_tok(cur, task_start +
                                                     static_cast<int>(cur_task_pos));  //  tokenized_positions(cur, 0) = task_start + static_cast<int>(i);
                        if (args.pos->use_full_structure) {
                            tokenized_positions(cur, tokenized_positions.shape()[1] - 1) = pos_dictionary.at("task_invariance");
                        }
                        if (!args.tok->scratchpad_as_prefix) {  // if part of targets
                            cur_task_pos++;
                        }
                    }
                }
                if (args.tok->scratchpad_as_prefix) {
                    task_start_idx = cur;
                }
                // write in task
                true_task_start_idx = cur;
                for (size_t i = 0; i < task->tokenized_task_inputs.shape()[0]; i++, cur++, cur_task_pos++) {
                    tokenized_inputs.copy_tok(task->tokenized_task_inputs, cur, i,
                             should_repeat);  // tokenized_inputs(cur, 0) = task->tokenized_task_inputs(i);
                    tokenized_targets.copy_tok(task->tokenized_task_targets, cur_task_pos, i,
                             should_repeat); // tokenized_targets(cur_task_pos, 0) = task->tokenized_task_targets(i, 0);
                    tokenized_positions.set_tok(cur, task_start + cur_task_pos); // tokenized_positions(cur, 0) = task_start + cur_task_pos;
                    if (args.pos->use_full_structure) {
                        tokenized_positions(cur, tokenized_positions.shape()[1] - 1) = pos_dictionary.at("task_invariance");
                    }
                }
                // end of sequence
                auto end_marker = dictionary.at("</s>");
                tokenized_inputs.set_tok(cur,end_marker, should_repeat);  // tokenized_inputs(cur, 0) = end_marker;
                tokenized_targets.set_tok(cur_task_pos,end_marker,
                        should_repeat); // tokenized_targets(cur_task_pos, 0) = end_marker; // target also end marker
                tokenized_positions.set_tok(cur, task_start + cur_task_pos,
                        should_repeat); // tokenized_positions(cur, 0) = task_start + cur_task_pos;
                if (args.pos->use_full_structure) {
                    tokenized_positions(cur, tokenized_positions.shape()[1] - 1) = pos_dictionary.at("task_invariance");
                }

                cur++;
                cur_task_pos++;
            } else {
                throw std::invalid_argument("Non-flat model tokenization not implemented yet");
            }
        }
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
    Args &args;
    vector<Instance<D> > instances;

    explicit BatchedInstances(Args &args) : args(args) {
        instances = vector<Instance<D> >();
    }

    void add(Instance<D> &instance) {
        instances.push_back(std::move(instance));
    }

    int size() {
        return static_cast<int>(instances.size());
    }

    void copy_to_numpy(py::array_t<int, py::array::c_style> &npm, Matrix<int> &mat, const int offset, const int batch_idx) const {
        return;
        auto npm_ar = npm.mutable_unchecked();
        for (size_t i = 0; i < mat.shape()[0]; i++) {
            for (size_t j = 0; j < mat.shape()[1]; j++) {
                npm_ar(batch_idx, i + offset, j) = mat(i, j);
            }
        }
    }

    void range_to_numpy(py::array_t<int, py::array::c_style> &npm,
                        const int start, const int length, const int stride,
                        const int batch_idx) const {
        auto npm_ar = npm.mutable_unchecked();
        for (size_t i = 0; i < length; i++) {
            npm_ar(batch_idx, i) = start + i * stride;
        }
    }

    py::dict package_for_model(const map<std::string, int> &dictionary, const map<std::string, int> pos_dictionary) {
        if (instances.size() == 0) {
            throw std::invalid_argument("No instances to package for model.");
        }
        py::dict d;

        // make all the info needed for the model input
        int max_tokenized_inputs_len = 0;
        int max_tokenized_inputs_struct = 0;
        int max_tokenized_targets_len = 0;
        int max_tokenized_targets_labels = 0;
        int max_tokenized_true_targets_len = 0;
        int max_tokenized_scratchpad_len = 0;
        int max_tokenized_positions_len = 0;
        int max_tokenized_positions_struct = 0;
        int max_prefix_size = 0;
        int max_num_nodes = 0;
        int max_num_edges = 0;

        for (size_t i = 0; i < instances.size(); i++) {
            // create inputs, targets and positions, and component start_indices and lengths
            instances[i].tokenize(dictionary, pos_dictionary);
            // batch the info
            max_tokenized_inputs_len = max(max_tokenized_inputs_len, static_cast<int>(instances[i].tokenized_inputs.shape()[0]));
            max_tokenized_inputs_struct = max(max_tokenized_inputs_struct, static_cast<int>(instances[i].tokenized_inputs.shape()[1]));
            max_prefix_size = max(max_prefix_size, instances[i].task_start_idx);

            max_tokenized_positions_len = max(max_tokenized_positions_len, static_cast<int>(instances[i].tokenized_positions.shape()[0]));
            max_tokenized_positions_struct = max(max_tokenized_positions_struct, static_cast<int>(instances[i].tokenized_positions.shape()[1]));

            if (instances[i].task) {
                max_tokenized_targets_len = max(max_tokenized_targets_len,
                                                static_cast<int>(instances[i].tokenized_targets.shape()[0]));
                max_tokenized_true_targets_len = max(max_tokenized_true_targets_len,
                                                     static_cast<int>(instances[i].true_task_length));
                max_tokenized_targets_labels = max(max_tokenized_targets_labels,
                                                   static_cast<int>(instances[i].tokenized_targets.shape()[1]));
                if (instances[i].scratch_pad) {
                    max_tokenized_scratchpad_len = max(max_tokenized_scratchpad_len,
                                                       static_cast<int>(instances[i].scratch_pad->tokenized_inputs.shape()[0]));
                }
            }

            max_num_nodes = max(max_num_nodes, instances[i].graph->N);
            max_num_edges = max(max_num_edges, instances[i].graph->E);
            // instances[i].pprint();
        }

        cout << "Shapes:" << " max_tokenized_inputs_len: " << max_tokenized_inputs_len
             << " max_tokenized_inputs_struct: " << max_tokenized_inputs_struct
             << " max_tokenized_targets_len: " << max_tokenized_targets_len
             << " max_tokenized_targets_labels: " << max_tokenized_targets_labels
             << " max_tokenized_positions_len: " << max_tokenized_positions_len
             << " max_tokenized_positions_struct: " << max_tokenized_positions_struct
             << " max_prefix_size: " << max_prefix_size
             << " max_num_nodes: " << max_num_nodes
             << " max_num_edges: " << max_num_edges
             << endl;

        // set up numpy arrays
        auto batch_size = static_cast<int>(instances.size());

        auto new_max_tokenized_inputs_len = max_tokenized_inputs_len;
        if (args.tok->align_prefix_front_pad and args.tok->is_flat_model) {  // only align if flat model since non-flat separate prefixes
            // align up to prefix size and then targets after
            new_max_tokenized_inputs_len = max_prefix_size + max_tokenized_targets_len;
        }
        auto src_tokens = py::array_t<int, py::array::c_style>({batch_size, new_max_tokenized_inputs_len, max_tokenized_inputs_struct});
        src_tokens[py::make_tuple(py::ellipsis())] = dictionary.at("<pad>");
        auto src_lengths = py::array_t<int, py::array::c_style>(batch_size);

        auto task_targets = py::array_t<int, py::array::c_style>({batch_size, max_tokenized_targets_len, max_tokenized_targets_labels});
        task_targets[py::make_tuple(py::ellipsis())] = dictionary.at("<pad>");
        auto task_lengths = py::array_t<int, py::array::c_style>(batch_size);

        auto positions = py::array_t<int, py::array::c_style>({batch_size, new_max_tokenized_inputs_len, max_tokenized_positions_struct});
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
        auto true_task_gather_indices = py::array_t<int, py::array::c_style>(
                {batch_size, max_tokenized_true_targets_len + 1});
        true_task_gather_indices[py::make_tuple(
                py::ellipsis())] = 0; // initialize to 0 since it needs to be a valid index
        auto scratch_pad_gather_indices = py::array_t<int, py::array::c_style>(
                {batch_size, max_tokenized_scratchpad_len});
        scratch_pad_gather_indices[py::make_tuple(py::ellipsis())] = 0;
        auto graph_edge_gather_indices = py::array_t<int, py::array::c_style>({batch_size, max_num_edges});
        graph_edge_gather_indices[py::make_tuple(py::ellipsis())] = 0;
        auto graph_node_gather_indices = py::array_t<int, py::array::c_style>({batch_size, max_num_nodes});
        graph_node_gather_indices[py::make_tuple(py::ellipsis())] = 0;

        // write into numpy arrays
        auto src_lengths_ar = src_lengths.mutable_unchecked();
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

        for (size_t i = 0; i < instances.size(); i++) {
            auto offset = 0;
            if (args.tok->align_prefix_front_pad and args.tok->is_flat_model) {
                offset = max_prefix_size - instances[i].task_start_idx;
            }

            copy_to_numpy(src_tokens, instances[i].tokenized_inputs, offset, i);  // src_tokens_ar(i, j + offset, k) = instances[i].tokenized_inputs(j, k);
            copy_to_numpy(positions, instances[i].tokenized_positions, offset, i); // positions_ar(i, j + offset, k) = instances[i].tokenized_positions(j, k);

            if (instances[i].task) {
                copy_to_numpy(task_targets, instances[i].tokenized_targets, 0, i); // task_targets_ar(i, j, k) = instances[i].tokenized_targets(j, k);  // never any offset
                range_to_numpy(task_gather_indices, instances[i].task_start_idx + offset, instances[i].tokenized_targets.shape()[0], 1, i); // task_gather_indices_ar(i, j) = instances[i].task_start_idx + offset + j;
                task_lengths_ar(i) = instances[i].task_length;
                range_to_numpy(true_task_gather_indices, instances[i].true_task_start_idx + offset, instances[i].true_task_length + 1, 1, i); //
                if (instances[i].scratch_pad) {
                    range_to_numpy(scratch_pad_gather_indices, instances[i].scratch_pad_start_idx + offset, instances[i].scratch_pad_length, 1, i); // scratch_pad_gather_indices_ar(i, j) = instances[i].scratch_pad_start_idx + offset + j;
                }
            }
            if (instances[0].graph_tokenizer) {
                if (args.tok->concat_edges) {
                   range_to_numpy(graph_edge_gather_indices, instances[i].graph_start_idx + offset, instances[i].graph_length, 1, i); // graph_edge_gather_indices_ar(i, j) = instances[i].graph_start_idx + offset + j;
                } else { // need every third token since we just want the edge marker positions (which needs shift by 2)
                    range_to_numpy(graph_edge_gather_indices, instances[i].graph_start_idx + offset + 2, instances[i].graph_tokenizer->edge_list.size(), 3, i); // graph_edge_gather_indices_ar(i, j) = instances[i].graph_start_idx + offset + (j * 3) + 2;
                }
                if (args.tok->include_nodes_in_graph_tokenization) {
                   range_to_numpy(graph_node_gather_indices, instances[i].graph_nodes_start_idx + offset, instances[i].graph_tokenizer->node_list.size(), 1, i); // graph_node_gather_indices_ar(i, j) = instances[i].graph_nodes_start_idx + offset + j;
                }
            }

            // all others
            num_nodes_ar(i) = instances[i].graph->N;
            num_edges_ar(i) = instances[i].graph->E;
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
            true_task_lengths_ar(i) = instances[i].true_task_length;
        }

        d["num_nodes"] = num_nodes;
        d["num_edges"] = num_edges;
        d["src_tokens"] = src_tokens;
        d["src_lengths"] = src_lengths;

        if (args.pos->return_pos_ids) {
            d["positions"] = positions;
        } else {
            d["positions"] = py::none();  // just delete the made positions and model will use range per norm
        }
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
        if (args.tok->include_nodes_in_graph_tokenization) {
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
                d["scratchpad_type"] = args.sp->scratchpad_type;
            } else {
                d["scratch_pad_start_indices"] = py::none();
                d["scratch_pad_lengths"] = py::none();
                d["scratch_pad_gather_indices"] = py::none();
                d["scratchpad_type"] = py::none();
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
            d["scratchpad_type"] = py::none();
        }

        // distances and ground truths batching

        if (instances[0].graph->g_ptr) {
            auto bd = batch_distances<int>();  // do node shuffling inside
            d["distances"] = bd;  // (B, max_vocab, max_vocab), where max_vocab is just up to node range (not extra symbols)
            d["hashes"] = hash_distance_matrix<int>(bd);

            auto gt_gather_indices_and_distances = batch_ground_truth_gather_indices<int>();
            if (args.tok->is_direct_ranking) {
                d["ground_truths_gather_indices"] = gt_gather_indices_and_distances.first;
            } else {
                d["ground_truths_gather_indices"] = py::none();
            }
            d["ground_truths_gather_distances"] = gt_gather_indices_and_distances.second;

            if (args.tok->no_graph) {
                d["graph_edge_start_indices"] = py::none();
                d["graph_edge_lengths"] = py::none();
                d["graph_edge_gather_indices"] = py::none();
                d["graph_node_start_indices"] = py::none();
                d["graph_node_lengths"] = py::none();
                d["graph_node_gather_indices"] = py::none();
                d["ground_truths_gather_indices"] = py::none();
            }
        } else {
            d["hashes"] = hash_src_tokens(src_tokens);

            d["graph_edge_start_indices"] = py::none();
            d["graph_edge_lengths"] = py::none();
            d["graph_edge_gather_indices"] = py::none();
            d["graph_node_start_indices"] = py::none();
            d["graph_node_lengths"] = py::none();
            d["graph_node_gather_indices"] = py::none();
            d["distances"] = py::none();
            d["ground_truths_gather_indices"] = py::none();
        }

        // if euclidean batch the node positions  TODO

        // arguments
        d["graph_type"] = args.graph_type;
        d["task_type"] = args.task->task_type;
        d["is_flat_model"] = args.tok->is_flat_model;
        d["concat_edges"] = args.tok->concat_edges;
        d["query_at_end"] = args.tok->query_at_end;
        d["scratchpad_as_prefix"] = args.tok->scratchpad_as_prefix;
        d["align_prefix_front_pad"] = args.tok->align_prefix_front_pad;
        d["min_vocab"] = args.min_vocab;
        d["max_vocab"] = args.max_vocab;

        return d;
    }

    template<typename T>
    py::array_t<T, py::array::c_style> batch_distances(T cuttoff = 100000, T max_value = -1, T pad = -1) {
        auto batch_size = static_cast<int>(instances.size());

        py::array_t<T, py::array::c_style> arr({static_cast<int>(batch_size), this->args.max_vocab, this->args.max_vocab});
        arr[py::make_tuple(py::ellipsis())] = static_cast<T>(pad); // initialize array
        auto ra = arr.mutable_unchecked();

        for (int b = 0; b < batch_size; b++) {
            const auto &distances_ptr = instances[b].graph->distances_ptr;
            auto &node_shuffle_map = instances[b].graph->node_shuffle_map;
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
            const auto &graph_ground_truths_ptr = instances[b].graph->graph_ground_truths_ptr;
            if (static_cast<int>(graph_ground_truths_ptr->size()) > max_d1) {
                max_d1 = static_cast<int>(graph_ground_truths_ptr->size());
            }
            auto num_nodes = instances[b].graph->N;
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
            const auto &graph_ground_truths_ptr = instances[b].graph->graph_ground_truths_ptr;
            auto node_shuffle_map = instances[b].graph->node_shuffle_map;

            for (int j = 0; j < static_cast<int>(graph_ground_truths_ptr->size()); j++) {
                auto cur_gt = 0;
                for (int k = 0; k < static_cast<int>((*graph_ground_truths_ptr)[j].size()); k++) {
                    if (((*graph_ground_truths_ptr)[j][k] >= 0) &&
                        ((cuttoff <= 0) || ((*graph_ground_truths_ptr)[j][k] < cuttoff))) {
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

    template<typename T>
    py::array_t<std::uint64_t, py::array::c_style> hash_src_tokens(
            const py::array_t<T, py::array::c_style> &src_tokens) {
        // Convert a distance matrix [N, N] to a numpy array [new_N, new_N] by mapping node ids
        auto shape = src_tokens.shape();
        py::array_t<std::uint64_t, py::array::c_style> arr({static_cast<int>(shape[0])});
        auto ra = arr.mutable_unchecked();
        auto bd = src_tokens.unchecked();
        for (int b = 0; b < shape[0]; b++) {
            std::string str = "";
            for (int i = 0; i < shape[1]; i++) {
                str += std::to_string(bd(b, i));
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
