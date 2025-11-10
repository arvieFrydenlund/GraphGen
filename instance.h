//
// Created by arvie on 11/4/25.
//

#ifndef GRAPHGEN_INSTANCE_H
#define GRAPHGEN_INSTANCE_H

#include <iostream>
#include <random>
#include <map>
#include "matrix.h"
#include "tasks.h"
#include "scratch_pads.h"
#include "graph_tokenizer.h"

using namespace std;

/*
 * Instance class to hold graph and related data i.e. all info for a single instance
 */
template<typename D>
class Instance {
public:
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
    Matrix<int> tokenized_positions;
    int query_start_idx = -1; // index in the tokenized input where the query starts
    int graph_start_idx = -1; // index in the tokenized input where the graph starts
    int graph_nodes_start_idx = -1; // index in the tokenized input where the graph nodes start
    int scratch_pad_start_idx = -1; // index in the tokenized input
    int task_start_idx = -1; // index in the tokenized input where the task starts

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
             optional<vector<int> > &given_query, int max_query_size, const int min_query_size, const bool is_center,
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
        N = num_vertices(*g_ptr);
        E = num_edges(*g_ptr);
        make_node_shuffle_map(gen, min_vocab, max_vocab, shuffle_nodes);
        make_edge_shuffle_map(gen, shuffle_edges);

        auto edge_list = get_edge_list(g_ptr, edge_shuffle_map);
        graph_tokenizer = make_unique<GraphTokenizer>(edge_list, concat_edges, duplicate_edges,
                                                      include_nodes_in_graph_tokenization,
                                                      use_edges_invariance, use_node_invariance,
                                                      use_graph_invariance, use_graph_structure);
        graph_tokenizer->tokenize(dictionary, node_shuffle_map, pos_dictionary, gen);

        if (include_nodes_in_graph_tokenization) {
            graph_tokenizer->get_distances(g_ptr);
            graph_tokenizer->get_node_ground_truths(is_direct_ranking);
        } else {
            // this also gets the distances due to legacy code
            graph_tokenizer->get_edge_ground_truths(g_ptr, is_causal);
        }

        if (task_type == "shortest_path") {
            auto short_path = make_unique<ShortestPathTask>(gen, graph_tokenizer->distances_ptr,
                                                            max_path_length, min_path_length,
                                                            start, end, task_sample_dist, use_query_invariance);
            start = short_path->path.front();
            end = short_path->path.back();
            task = std::move(short_path);
            if (scratchpad_type == "bfs" || scratchpad_type == "BFS") {
                scratch_pad = make_unique<BFSScratchPad>(start, end, g_ptr, sort_adjacency_lists,
                                                         use_unique_depth_markers);
            } else if (scratchpad_type == "dfs" || scratchpad_type == "DFS") {
                // scratch_pad = make_unique<DFSScratchPad>(start, end, g_ptr, sort_adjacency_lists, use_unique_depth_markers);
            }
        } else if (task_type == "center") {
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
    }

    void tokenize(
        const map<std::string, int> &dictionary,
        const map<std::string, int> pos_dictionary,
        bool query_at_end,
        int num_thinking_tokens,
        bool is_flat_model) {
        int num_tokens = static_cast<int>(graph_tokenizer->tokenized_inputs.shape()[0]) + 2;
        // +2 for start and end tokens
        if (task) {
            num_tokens += static_cast<int>(task->tokenized_query_inputs.shape()[0]);
        }
        if (task and is_flat_model) {
            num_tokens += static_cast<int>(task->tokenized_task_inputs.shape()[0]);
        }
        if (scratch_pad and is_flat_model) {
            num_tokens += static_cast<int>(scratch_pad->tokenized_inputs.shape()[0]);
        }

        auto concat_edges = graph_tokenizer->concat_edges;
        auto use_graph_structure = graph_tokenizer->use_graph_structure;


        if (concat_edges) {
            tokenized_inputs = Matrix<int>(num_tokens, 2, dictionary.at("<pad>"));
            use_graph_structure = false; // cannot have both concat edges and graph structure
        } else {
            tokenized_inputs = Matrix<int>(num_tokens, 1, dictionary.at("<pad>"));
        }

        if (use_graph_structure) {
            tokenized_positions = Matrix<int>(num_tokens, 2, pos_dictionary.at("pad"));
        } else {
            tokenized_positions = Matrix<int>(num_tokens, 1, pos_dictionary.at("pad"));
        }

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
            task_start_idx = cur;
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
            task_start_idx = cur;
            for (size_t i = 0; i < task->tokenized_query_inputs.shape()[0]; i++, cur++) {
                tokenized_inputs(cur, 0) = task->tokenized_query_inputs(i);
                if (concat_edges) {
                    tokenized_inputs(cur, 1) = tokenized_inputs(cur, 0);
                }
                tokenized_positions(cur, 0) = task->tokenized_query_pos(i);
            }
        }
        if (num_thinking_tokens > 0) {
            // write in thinking tokens

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

        if (is_flat_model and task) {
            // write in task and scratchpad
            auto cur_task_pos = 0;
            auto task_start = pos_dictionary.at("task_start");
            auto task_end = pos_dictionary.at("task_end");
            auto task_size = static_cast<int>(task->tokenized_task_inputs.shape()[0]);

            if (scratch_pad) {
                // write in scratchpad tokens
                auto scratch_pad_size = static_cast<int>(scratch_pad->tokenized_inputs.shape()[0]);
                if (task_size + scratch_pad_size > task_end - task_start + 1) {
                    throw std::invalid_argument("Task size exceeds available position tokens.");
                }
                for (size_t i = 0; i < scratch_pad->tokenized_inputs.shape()[0]; i++, cur++, cur_task_pos++) {
                    tokenized_inputs(cur, 0) = scratch_pad->tokenized_inputs(i);
                    if (concat_edges) {
                        tokenized_inputs(cur, 1) = tokenized_inputs(cur, 0);
                    }
                    tokenized_positions(cur, 0) = task_start + cur_task_pos;
                }
            }
            // write in task
            if (task_size > task_end - task_start + 1) {
                throw std::invalid_argument("Task size exceeds available position tokens.");
            }
            for (size_t i = 0; i < task->tokenized_task_inputs.shape()[0]; i++, cur++, cur_task_pos++) {
                tokenized_inputs(cur, 0) = task->tokenized_task_inputs(i);
                if (concat_edges) {
                    tokenized_inputs(cur, 1) = task->tokenized_task_inputs(i);
                }
                tokenized_positions(cur, 0) = task_start + cur_task_pos;
            }
            // end of sequence
            auto end_marker = dictionary.at("</s>");
            tokenized_inputs(cur, 0) = end_marker;
            if (concat_edges) {
                tokenized_inputs(cur, 1) = end_marker;
            }
            tokenized_positions(cur, 0) = task_start + cur_task_pos;
            if (use_graph_structure) {
                tokenized_positions(cur, 1) = task_start + cur_task_pos;
            }
            cur++;
            cur_task_pos++;
        } // not implemented for non-flat models yet
    }

    void pprint() const {
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

        string s = "Tokenized Inputs and Positions:\n";
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
        py::dict d;

        for (size_t i = 0; i < instances.size(); i++) {
            instances[i].tokenize(
                dictionary,
                pos_dictionary,
                query_at_end,
                num_thinking_tokens,
                is_flat_model
            );

            instances[i].pprint();
        }

        return d;
    }

    void print(py::dict) {
    }
};


#endif //GRAPHGEN_INSTANCE_H
