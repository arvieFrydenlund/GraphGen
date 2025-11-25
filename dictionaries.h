//
// Created by arvie on 15/11/25.
//

#ifndef GRAPHGEN_DICTIONARIES_H
#define GRAPHGEN_DICTIONARIES_H

#include <iostream>
#include <map>

#include <Python.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

using namespace std;
namespace py = pybind11;
using namespace py::literals;

/* ************************************************
 *  Dictionary for mapping vocabulary to ids
 *  ***********************************************/

extern map<std::string, int> dictionary; // token to idx map
extern int dictionary_num_special;  // special tokens are the first N tokens in the dictionary
extern int dictionary_num_extra;  //extra tokens
extern int dictionary_max_vocab;  //then the rest are vocab tokens up to max_vocab
extern string dictionary_extra_after_symbol;  // then rest are special extras of indeterminate number of D1, D2, ...

inline void set_dictionary(py::dict &py_dictionary, const bool verbose = false,
                           const int max_num_nodes = -1, const int extra_after=-1, const string extra_after_symbol="D") {
    if (verbose) {
        cout << "Setting dictionary" << endl;
    }
    for (std::pair<py::handle, py::handle> item: py_dictionary) {
        auto key = item.first.cast<std::string>();
        auto value = item.second.cast<int>();
        if (verbose) {
            cout << "\tkey: " << key << ", value=" << value << endl;
        }
        dictionary[key] = value;
    }
    // get counts, assume specials are not int values and extras are symbol + int
    dictionary_extra_after_symbol = extra_after_symbol;
    int num_node_vocab = 0;
    for (const auto &item : dictionary) {
        const auto &key = item.first;
        // if is an int add to num_node_vocab
        try {
            stoi(key);
            num_node_vocab++;
        } catch (std::invalid_argument &) {
            // not an int
            // if starts with extra symbol do not count
            if (key.rfind(dictionary_extra_after_symbol) == 0 and key.size() > dictionary_extra_after_symbol.size()) {
                dictionary_num_extra ++;
            } else{ // else is special
                dictionary_num_special ++;
            }
        }
    }
    dictionary_max_vocab = dictionary_num_special + num_node_vocab;

    if (max_num_nodes > 0 && num_node_vocab != max_num_nodes) {
        const string err_msg = "Dictionary vocab size (" + std::to_string(num_node_vocab) +
                               ") does not match max_num_nodes (" + std::to_string(max_num_nodes) + ")";
        throw std::invalid_argument(err_msg);
    }
    if (extra_after > 0 && dictionary_num_extra != extra_after) {
        const string err_msg = "Dictionary extra tokens (" + std::to_string(dictionary_num_extra) +
                               ") do not match extra_after (" + std::to_string(extra_after) + ")";
        throw std::invalid_argument(err_msg);
    }
}

inline void set_default_dictionary(const int max_num_nodes = 50,
                                   const int extra_after=0, const string extra_after_symbol="D") {
    dictionary_extra_after_symbol = extra_after_symbol;
    // Sets a default dictionary
    dictionary.clear();
    dictionary = {  // should really make this as a list to index for safety, but this is easier to reference
            {"<s>", 0},
            {"<pad>", 1},
            {"</s>", 2},
            {"<unk>", 3},
            {"|", 4},  // edge marker
            {"!", 5},  // thinking token
            {"=", 6},  // task start
            {".", 7},  // task end
            {"t1", 8}, // potentially mark task type for muliti-task learning
            {"t2", 9},
            {"t3", 10},
            {"t4", 11},
            {"t5", 12},
            {"/", 13}, // query start
            {"?", 14}, // query end
            {"@", 15},
            {"#", 16}, // scratchpad start
            {"[", 17}, // bfs adjacency start
            {"]", 18}, // bfs adjacency end
            {"{", 19},
            {"}", 20},
            {"$", 21},
            {dictionary_extra_after_symbol, 22},  // D
    };
    // safety check,  make sure the max value is the same as size of dictionary
    auto max_idx = 0;
    for (const auto &item : dictionary) {
        if (item.second > max_idx) {
            max_idx = item.second;
        }
    }
    if (max_idx + 1 != static_cast<int>(dictionary.size())) {
        throw std::invalid_argument("Default dictionary indices are not contiguous");
    }
    dictionary_num_special = static_cast<int>(dictionary.size());
    if (max_num_nodes > 0) {
        dictionary_max_vocab = dictionary_num_special + max_num_nodes;
        for (int i = dictionary_num_special; i < max_num_nodes + dictionary_num_special; i++) {
            dictionary[std::to_string(i - dictionary_num_special)] = i;
        }
    }
    if (extra_after > 0) {
        dictionary_num_extra = extra_after;
        auto current_size = static_cast<int>(dictionary.size());
        for (int i = 0; i < extra_after; i++) {
            dictionary[dictionary_extra_after_symbol + std::to_string(i)] = current_size + i;
        }
    }
}

inline map<std::string, int> get_dictionary() {
    return dictionary;
}

inline pair<int, int> get_dictionary_vocab_limits() {
    return {dictionary_num_special, dictionary_max_vocab};
}

inline string get_dictionary_extra_after_symbol() {
    return dictionary_extra_after_symbol;
}

inline bool is_valid_extra_dictionary_symbol(const int cur_extra){
    return cur_extra < dictionary_num_extra;
}
inline bool is_valid_extra_dictionary_symbol(const string &extra){
    return dictionary.find(extra) != dictionary.end();
}

/* ************************************************
 *  Dictionary for mapping positions to ids
 *  ***********************************************/

extern map<std::string, int> pos_dictionary; // token to idx map

inline void set_pos_dictionary(py::dict &py_dictionary, const bool verbose = false) {
    if (verbose) {
        cout << "Setting pos dictionary" << endl;
    }
    for (std::pair<py::handle, py::handle> item: py_dictionary) {
        auto key = item.first.cast<std::string>();
        auto value = item.second.cast<int>();
        if (verbose) {
            cout << "\tkey: " << key << ", value=" << value << endl;
        }
        pos_dictionary[key] = value;
    }
    // verify that all keys are present
    vector<string> required_keys = {
            "pad",
            "misc_start",
            "misc_end",
            "query_start",
            "query_end",
            "graph_start",
            "graph_end",
            "graph_sub_start",
            "graph_sub_end",
            "task_start",
            "task_end",
    };
    for (const auto &key : required_keys) {
        if (pos_dictionary.find(key) == pos_dictionary.end()) {
            throw std::invalid_argument("Key " + key + " not found in pos_dictionary");
        }
    }
}

inline void set_default_pos_dictionary() {
    /* Sets a default pos dictionary
    */
    pos_dictionary = {
            {"pad", 0},  // needed for embedding look-up
            {"query_invariance", 1},
            {"edge_invariance", 2},
            {"node_invariance", 3},
            {"graph_invariance", 4},
            {"misc_start", 11},
            {"misc_end", 99},
            {"query_start", 100},
            {"query_end", 199},
            {"graph_start", 200},
            {"graph_end", 499},
            {"graph_sub_start", 500},
            {"graph_sub_end", 599},  // only really need 3
            {"thinking_start", 600},
            {"thinking_end", 699},
            {"task_start", 700},
            {"task_end", 999},
    };
}

map<std::string, int> get_pos_dictionary() {
    return pos_dictionary;
}


#endif //GRAPHGEN_DICTIONARIES_H
