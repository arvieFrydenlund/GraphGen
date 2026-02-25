//
// Created by arvie on 23/02/26.
//

#include <iostream>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace py::literals;

using namespace std;

#ifndef GRAPHGEN_ARGS_H
#define GRAPHGEN_ARGS_H


// Helper function to parse and set values from kwargs
template <typename T>
void parse_and_set_arg(const py::kwargs &kwargs, const std::string &name, T &arg, T default_value) {
    if (kwargs.contains(name)) {
        arg = kwargs[name.c_str()].cast<T>();
    } else {
        arg = default_value;
    }
}

class TaskArgs {
public:
    string task_type;
    optional<vector<float>> task_sample_dist;

    // for random trees
    bool start_at_root = true;
    bool end_at_leaf = true;
    optional<vector<float>> probs = nullopt;

    TaskArgs(const string &task_type = "shortest_path", const py::kwargs &kwargs = py::kwargs()) {
        parse_and_set_arg(kwargs, "task_type", this->task_type, task_type);
        // this can only be passed by kwargs
        if (kwargs.contains("task_sample_dist")) {
            if (!kwargs["task_sample_dist"].is_none() and !kwargs["task_sample_dist"].cast<py::list>().empty()) {
                task_sample_dist = kwargs["task_sample_dist"].cast<vector<float> >();
            }
        } else{
            task_sample_dist = nullopt;
        }

        // sample different paths but lose that it is exactly k-hops
        if (kwargs.contains("start_at_root")) {
            start_at_root = kwargs["start_at_root"].cast<bool>();
        }
        if (kwargs.contains("end_at_leaf")) {
            end_at_leaf = kwargs["end_at_leaf"].cast<bool>();
        }
        // override default probabilities for branching in random tree generator
        if (kwargs.contains("probs")) {
            if (!kwargs["probs"].is_none() and !kwargs["probs"].cast<py::list>().empty()) {
                probs = kwargs["probs"].cast<vector<float> >();
            }
        }

    }

    virtual  ~TaskArgs() = default;

    virtual void print() const {
        cout << "TaskArgs: task_type=" << task_type;
        if (task_sample_dist.has_value()) {
            cout << ", task_sample_dist=[";
            for (const auto &val : task_sample_dist.value()) {
                cout << val << " ";
            }
            cout << "]";
        }
        cout << endl;
    }
};

class ShortestPathTaskArgs : public TaskArgs {
public:
    int max_path_length;
    int min_path_length;

    ShortestPathTaskArgs(const py::kwargs &kwargs = py::kwargs(),
                     int max_path_length = 10, int min_path_length = 1) :
            TaskArgs("shortest_path", kwargs), max_path_length(max_path_length),
            min_path_length(min_path_length) {
        parse_and_set_arg(kwargs, "max_path_length", this->max_path_length, max_path_length);
        parse_and_set_arg(kwargs, "min_path_length", this->min_path_length, min_path_length);

        if (min_path_length > 0 and min_path_length > max_path_length) {
            throw std::invalid_argument("Invalid arguments: min_path_length > max_path_length");
        }
    }

    virtual void print() const {
        // print child and parent args
        this->TaskArgs::print();
        cout << "ShortestPathTaskArgs: max_path_length=" << max_path_length
             << ", min_path_length=" << min_path_length << endl;
    }
};

class BFSTaskArgs : public ShortestPathTaskArgs {
public:
    BFSTaskArgs(const py::kwargs &kwargs,
                int max_path_length = 10, int min_path_length = 1) :
            ShortestPathTaskArgs( kwargs, max_path_length, min_path_length) {
        task_type = "bfs";  // override task type
        parse_and_set_arg(kwargs, "max_path_length", this->max_path_length, max_path_length);
        parse_and_set_arg(kwargs, "min_path_length", this->min_path_length, min_path_length);
    }
};


class CenterCentroidTaskArgs : public TaskArgs {
public:
    int max_query_size;
    int min_query_size;
    bool is_center;
    optional<vector<int>> given_query;

    CenterCentroidTaskArgs(const string &task_type, const py::kwargs &kwargs = py::kwargs(),
                       int max_query_size = -1, int min_query_size = 2) :
            TaskArgs(task_type, kwargs), max_query_size(max_query_size), min_query_size(min_query_size) {
        if (task_type != "center" && task_type != "centroid") {
            throw std::invalid_argument("Invalid task_type for CenterCentroidArgs: " + task_type);
        }
        if (task_type == "center") {
            is_center = true;
        } else {
            is_center = false;
        }
        parse_and_set_arg(kwargs, "max_query_size", this->max_query_size, max_query_size);
        parse_and_set_arg(kwargs, "min_query_size", this->min_query_size, min_query_size);

        // check for given query by kwargs
        if (kwargs.contains("given_query")){
            if (!kwargs["given_query"].is_none() and !kwargs["given_query"].cast<py::list>().empty()) {
                given_query = kwargs["given_query"].cast<vector<int> >();
            }
        } else{
            given_query= nullopt;
        }
    }

    virtual void print() const {
        // print child and parent args
        this->TaskArgs::print();
        cout << "CenterCentroidTaskArgs: max_query_size=" << max_query_size
             << ", min_query_size=" << min_query_size
             << ", is_center=" << is_center;
        if (given_query.has_value()) {
            cout << ", given_query=[";
            for (const auto &val: given_query.value()) {
                cout << val << " ";
            }
            cout << "]";
        }
        cout << endl;
    }
};

class KhopsArgs : public TaskArgs {
public:
    int max_khops;
    int min_khops;
    int min_prefix_length;
    int max_prefix_length;
    bool right_side_connect;
    bool permutation_version;
    bool mask_to_vocab_size;
    string partition_method;

    KhopsArgs(const py::kwargs &kwargs,
              int max_khops = 5, int min_khops = 2, int min_prefix_length = 3, int max_prefix_length = 20,
              bool right_side_connect = true, bool permutation_version = false, bool mask_to_vocab_size = false,
              const string &partition_method = "uniform") :
            TaskArgs("khops", kwargs), max_khops(max_khops), min_khops(min_khops),
            min_prefix_length(min_prefix_length), max_prefix_length(max_prefix_length),
            right_side_connect(right_side_connect), permutation_version(permutation_version),
            mask_to_vocab_size(mask_to_vocab_size), partition_method(partition_method) {
        parse_and_set_arg(kwargs, "max_khops", this->max_khops, max_khops);
        parse_and_set_arg(kwargs, "min_khops", this->min_khops, min_khops);
        parse_and_set_arg(kwargs, "min_prefix_length", this->min_prefix_length, min_prefix_length);
        parse_and_set_arg(kwargs, "max_prefix_length", this->max_prefix_length, max_prefix_length);
        parse_and_set_arg(kwargs, "right_side_connect", this->right_side_connect, right_side_connect);
        parse_and_set_arg(kwargs, "permutation_version", this->permutation_version, permutation_version);
        parse_and_set_arg(kwargs, "mask_to_vocab_size", this->mask_to_vocab_size, mask_to_vocab_size);
        parse_and_set_arg(kwargs, "partition_method", this->partition_method, partition_method);
    }

    virtual void print() const {
        // print child and parent args
        this->TaskArgs::print();
        cout << "KhopsArgs: max_khops=" << max_khops
            << ", min_khops=" << min_khops
            << ", min_prefix_length=" << min_prefix_length
            << ", max_prefix_length=" << max_prefix_length
            << ", right_side_connect=" << right_side_connect
            << ", permutation_version=" << permutation_version
            << ", mask_to_vocab_size=" << mask_to_vocab_size
            << ", partition_method=" << partition_method
            << endl;
    }
};


class ScratchpadArgs {
public:
    string scratchpad_type;
    ScratchpadArgs(const string &scratchpad_type = "none", const py::kwargs &kwargs = py::kwargs()) {
        parse_and_set_arg(kwargs, "scratchpad_type", this->scratchpad_type, scratchpad_type);
    }
    virtual  ~ScratchpadArgs() = default;

    virtual void print() const {
        cout << "ScratchpadArgs: scratchpad_type=" << scratchpad_type << endl;
    }
};

class BFSScratchpadArgs : public ScratchpadArgs {
public:
    bool sort_adjacency_lists;  // doesn't do anything anymore
    bool use_unique_depth_markers;
    bool stop_once_found;

    BFSScratchpadArgs(const py::kwargs &kwargs,
                      bool sort_adjacency_lists=true, bool use_unique_depth_markers=true, bool stop_once_found=true) :
            ScratchpadArgs("bfs", kwargs), sort_adjacency_lists(sort_adjacency_lists),
            use_unique_depth_markers(use_unique_depth_markers), stop_once_found(stop_once_found) {
        parse_and_set_arg(kwargs, "sort_adjacency_lists", this->sort_adjacency_lists, sort_adjacency_lists);
        parse_and_set_arg(kwargs, "use_unique_depth_markers", this->use_unique_depth_markers, use_unique_depth_markers);
        parse_and_set_arg(kwargs, "stop_once_found", this->stop_once_found, stop_once_found);
    }

    virtual void print() const {
        // print child and parent args
        this->ScratchpadArgs::print();
        cout << "BFSScratchpadArgs: "
             << ", sort_adjacency_lists=" << sort_adjacency_lists
             << ", use_unique_depth_markers=" << use_unique_depth_markers
             << ", stop_once_found=" << stop_once_found
             << endl;
    }
};

class TokenizationArgs {
public:
    bool is_causal;
    bool is_direct_ranking;
    bool query_at_end;
    bool no_graph;
    bool concat_edges;
    bool duplicate_edges;
    bool include_nodes_in_graph_tokenization;
    int num_thinking_tokens;
    bool scratchpad_as_prefix;
    bool is_flat_model;
    bool align_prefix_front_pad;

    TokenizationArgs(bool is_causal = false, bool is_direct_ranking = false,
                     bool query_at_end = true, bool no_graph = false,
                     bool concat_edges = true, bool duplicate_edges = false,
                     bool include_nodes_in_graph_tokenization = false,
                     int num_thinking_tokens = 0, bool scratchpad_as_prefix = false,
                     bool is_flat_model = true, bool align_prefix_front_pad = false,
                     const py::kwargs &kwargs = py::kwargs()) {
        parse_and_set_arg(kwargs, "is_causal", this->is_causal, is_causal);
        parse_and_set_arg(kwargs, "is_direct_ranking", this->is_direct_ranking, is_direct_ranking);
        parse_and_set_arg(kwargs, "query_at_end", this->query_at_end, query_at_end);
        parse_and_set_arg(kwargs, "no_graph", this->no_graph, no_graph);
        parse_and_set_arg(kwargs, "concat_edges", this->concat_edges, concat_edges);
        parse_and_set_arg(kwargs, "duplicate_edges", this->duplicate_edges, duplicate_edges);
        parse_and_set_arg(kwargs, "include_nodes_in_graph_tokenization", this->include_nodes_in_graph_tokenization, include_nodes_in_graph_tokenization);
        parse_and_set_arg(kwargs, "num_thinking_tokens", this->num_thinking_tokens, num_thinking_tokens);
        parse_and_set_arg(kwargs, "scratchpad_as_prefix", this->scratchpad_as_prefix, scratchpad_as_prefix);
        parse_and_set_arg(kwargs, "is_flat_model", this->is_flat_model, is_flat_model);
        parse_and_set_arg(kwargs, "align_prefix_front_pad", this->align_prefix_front_pad, align_prefix_front_pad);

        if (num_thinking_tokens < 0) {
            throw std::invalid_argument("Invalid arguments: num_thinking_tokens < 0");
        }
    }

    virtual ~TokenizationArgs() = default;

    virtual void print() const {
        cout << "TokenizationArgs: is_causal=" << is_causal
             << ", is_direct_ranking=" << is_direct_ranking
             << ", query_at_end=" << query_at_end
             << ", no_graph=" << no_graph
             << ", concat_edges=" << concat_edges
             << ", duplicate_edges=" << duplicate_edges
             << ", include_nodes_in_graph_tokenization=" << include_nodes_in_graph_tokenization
             << ", num_thinking_tokens=" << num_thinking_tokens
             << ", scratchpad_as_prefix=" << scratchpad_as_prefix
             << ", is_flat_model=" << is_flat_model
             << ", align_prefix_front_pad=" << align_prefix_front_pad
             << endl;
    }
};


class PosArgs {
public:
    bool return_pos_ids;
    bool use_edges_invariance;
    bool use_node_invariance;
    bool use_graph_invariance;
    bool use_query_invariance;
    bool use_graph_structure;
    bool use_full_structure;

    PosArgs(const py::kwargs &kwargs = py::kwargs(),
            bool return_pos_ids = true, // otherwise model will just make normal range pos ids
            bool use_edges_invariance = false,  // same pos token across edges
            bool use_node_invariance = false, // same pos token across nodes
            bool use_graph_invariance = false, // same pos token across whole graph (overrides edge and node invariance)
            bool use_query_invariance = false, // same pos token across whole query
            bool use_graph_structure = false, // adds sub ids for 'u v |'  i.e. none concat edges
            bool use_full_structure = false  // adds invariant sub ids for all [Q E N S T]
            ){
        parse_and_set_arg(kwargs, "return_pos_ids", this->return_pos_ids, return_pos_ids);
        parse_and_set_arg(kwargs, "use_edges_invariance", this->use_edges_invariance, use_edges_invariance);
        parse_and_set_arg(kwargs, "use_node_invariance", this->use_node_invariance, use_node_invariance);
        parse_and_set_arg(kwargs, "use_graph_invariance", this->use_graph_invariance, use_graph_invariance);
        parse_and_set_arg(kwargs, "use_query_invariance", this->use_query_invariance, use_query_invariance);
        parse_and_set_arg(kwargs, "use_graph_structure", this->use_graph_structure, use_graph_structure);
        parse_and_set_arg(kwargs, "use_full_structure", this->use_full_structure, use_full_structure);

    }

    virtual ~PosArgs() = default;

    virtual void print() const {
        cout << "PosArgs: return_pos_ids=" << return_pos_ids
             << ", use_edges_invariance=" << use_edges_invariance
             << ", use_node_invariance=" << use_node_invariance
             << ", use_graph_invariance=" << use_graph_invariance
             << ", use_query_invariance=" << use_query_invariance
             << ", use_graph_structure=" << use_graph_structure
             << ", use_full_structure=" << use_full_structure
             << endl;
    }
};


class Args{
    // argument that we want to test in c++ should be passed in, otherwise use kwargs
    // do not pass in specific task or scratchpad args
public:
    TaskArgs *task = nullptr;
    ScratchpadArgs *sp = nullptr;
    TokenizationArgs *tok = nullptr;
    PosArgs *pos = nullptr;

    const string &graph_type;
    const int min_vocab;
    const int max_vocab;

    Args(const string &task_type="shortest_path",
         const string &scratchpad_type="none",
         // misc since I don't want to make more classes
         const string &graph_type="none",
         const int min_vocab = -1, const int max_vocab = -1,
         // tokenization
         bool is_causal=false,
         bool is_direct_ranking=false,
         bool query_at_end=true,
         bool no_graph=false,
         bool concat_edges=true,
         bool duplicate_edges=false,
         bool include_nodes_in_graph_tokenization=false,
         int num_thinking_token=0,
         bool scratchpad_as_prefix=false,
         bool is_flat_model=true,
         bool align_prefix_front_pad=false,
         const py::kwargs &kwargs = py::kwargs(),
         bool const print_args = true
    ): graph_type(graph_type), min_vocab(min_vocab), max_vocab(max_vocab) {
        if (task_type == "shortest_path"){
            task = new ShortestPathTaskArgs(kwargs);
        } else if (task_type == "bfs"){
            task = new BFSTaskArgs(kwargs);
            sp = new BFSScratchpadArgs(kwargs);
        } else if (task_type == "center" || task_type == "centroid"){
            task = new CenterCentroidTaskArgs(task_type, kwargs);
        } else if (task_type == "khops"){
            task = new KhopsArgs(kwargs);
        } else if (task_type == "None"){
            task = new TaskArgs("none", kwargs);
        } else{
            throw std::invalid_argument("Invalid task_type: " + task_type);
        }

        if (sp == nullptr) {
            if (scratchpad_type == "bfs") {
                sp = new BFSScratchpadArgs(kwargs);
            } else if (scratchpad_type == "none") {
                sp = new ScratchpadArgs("none", kwargs);
            } else {
                throw std::invalid_argument("Invalid scratchpad_type: " + scratchpad_type);
            }
        }

        tok = new TokenizationArgs(is_causal, is_direct_ranking, query_at_end, no_graph,
                                            concat_edges, duplicate_edges, include_nodes_in_graph_tokenization,
                                            num_thinking_token, scratchpad_as_prefix, is_flat_model, align_prefix_front_pad, kwargs);
        pos = new PosArgs(kwargs);

        if (print_args) {
            this->print();
        }
    }

    ~Args() {
        delete task;
        delete sp;
        delete tok;
        delete pos;
    }

    void print(){
        cout << "Args: graph_type=" << graph_type
             << ", min_vocab=" << min_vocab
             << ", max_vocab=" << max_vocab
             << endl;

        task->print();
        sp->print();
        tok->print();
        pos->print();
        cout << "-----------------------------" << endl << endl;
    }


};

#endif //GRAPHGEN_ARGS_H
