//
// Created by arvie on 05/11/25.
//

#ifndef GRAPH_TOKENIZER_H
#define GRAPH_TOKENIZER_H

#include <iostream>
#include <random>
#include <queue>
#include <map>
#include "matrix.h"
#include "undirected_graphs.h"
#include "directed_graphs.h"
#include "args.h"

using namespace std;

class GraphTokenizer {
public:
    Matrix<int> tokenized_inputs; // possibly 2D inputs when using concat_edges
    Matrix<int> tokenized_pos;  // possibly 2D positions when using use_graph_structure or task_tokens

    TokenizationArgs *tok_args;
    PosArgs *pos_args;
    const vector<pair<int, int> > &edge_list;  // already shuffled
    const vector<int> &node_list;

    GraphTokenizer(TokenizationArgs *tok_args, PosArgs *pos_args,
                   const vector<pair<int, int> > &edge_list, const vector<int> &node_list) :
                   tok_args(tok_args), pos_args(pos_args), edge_list(edge_list), node_list(node_list) {
    }

    void tokenize(const map<std::string, int> &dictionary,
                  const vector<int> &node_shuffle_map,
                  const map<std::string, int> pos_dictionary,
                  std::mt19937 &gen) {
        // node order by edge appearance
        auto num_tokens = static_cast<int>(edge_list.size());
        vector<pair<int, int> > new_edge_list;
        // remap edges according to node shuffle map
        if (tok_args->duplicate_edges) {
            num_tokens *= 2;
            // make copy of edge list with reversed edges
            std::copy(edge_list.begin(), edge_list.end(), std::back_inserter<vector<pair<int, int> > >(new_edge_list));
            for (size_t i = 0; i < edge_list.size(); i++) {  // TODO option to flip being separate from duplicate
                new_edge_list.push_back(make_pair(edge_list[i].second,edge_list[i].first));
            }
        } else {
            new_edge_list = this->edge_list;
        }
        if (!tok_args->concat_edges) {
            num_tokens *= 3;
        }
        if (tok_args->include_nodes_in_graph_tokenization) {
            num_tokens += static_cast<int>(node_list.size());
        }

        auto edge_invariance_marker = pos_dictionary.at("edge_invariance");
        auto node_invariance_marker = pos_dictionary.at("node_invariance");
        auto graph_invariance_marker = pos_dictionary.at("graph_invariance");
        auto graph_start = pos_dictionary.at("graph_start");
        auto graph_end = pos_dictionary.at("graph_end");
        auto graph_sub_start = pos_dictionary.at("graph_sub_start");
        // auto graph_sub_end = pos_dictionary.at("graph_sub_end");

        if (num_tokens > graph_end - graph_start + 1) { // not a correct check but easy enough to just do other values in dictionary
            throw runtime_error("Graph tokenization length exceeds allocated position ids.  Please increase graph position id range in the pos dictionary.");
        }

        auto cur = 0;
        // padding below is for sanity, there should be no pads left after writing in graph (at least in zero dim)
        if (tok_args->concat_edges) {
            tokenized_inputs.resize(num_tokens, 2, dictionary.at("<pad>"));
        } else {
            tokenized_inputs.resize(num_tokens, 1, dictionary.at("<pad>"));
        }
        if (pos_args->use_graph_structure) {
            tokenized_pos.resize(num_tokens, 2, pos_dictionary.at("pad"));
        } else {
            tokenized_pos.resize(num_tokens, 1, pos_dictionary.at("pad"));
        }

        // write in new edge list
        if (!tok_args->concat_edges) {
            for (size_t i = 0; i < new_edge_list.size(); i++, cur += 3) {
                tokenized_inputs(i * 3, 0) = node_shuffle_map.at(new_edge_list[i].first);
                tokenized_inputs(i * 3 + 1, 0) = node_shuffle_map.at(new_edge_list[i].second);
                tokenized_inputs(i * 3 + 2, 0) = dictionary.at("|");
                if (pos_args->use_graph_structure){
                    auto edge_pos = graph_start + static_cast<int>(i);
                    tokenized_pos(i * 3, 0) = edge_pos;
                    tokenized_pos(i * 3 + 1, 0) = edge_pos;
                    tokenized_pos(i * 3 + 2, 0) = edge_pos;
                    tokenized_pos(i * 3, 1) = graph_sub_start;
                    tokenized_pos(i * 3 + 1, 1) = graph_sub_start + 1;
                    tokenized_pos(i * 3 + 2, 1) = graph_sub_start + 2;
                } else {
                    tokenized_pos(i * 3, 0) = graph_start + cur;
                    tokenized_pos(i * 3 + 1, 0) = graph_start + cur + 1;
                    tokenized_pos(i * 3 + 2, 0) = graph_start + cur + 2;
                }
            }
        } else {
            for (size_t i = 0; i < new_edge_list.size(); i++, cur++) {
                tokenized_inputs(i, 0) = node_shuffle_map.at(new_edge_list[i].first);
                tokenized_inputs(i, 1) = node_shuffle_map.at(new_edge_list[i].second);
                if (pos_args->use_edges_invariance) {
                    tokenized_pos(i, 0) = edge_invariance_marker;
                } else if (pos_args->use_graph_invariance){
                    tokenized_pos(i, 0) = graph_invariance_marker;
                } else {
                    tokenized_pos(i, 0) = graph_start + cur;
                }
            }
        }
        if (tok_args->include_nodes_in_graph_tokenization) { // write in node list at end
            for (size_t i = 0; i < node_list.size(); i++, cur++) {
                tokenized_inputs(cur, 0) = node_shuffle_map.at(node_list[i]);
                if (tok_args->concat_edges) {  // duplicate node if concat edges in both dims
                    tokenized_inputs(cur, 1) = tokenized_inputs(cur, 0);
                }
                if (pos_args->use_node_invariance) {
                    tokenized_pos(cur, 0) = node_invariance_marker;
                } else if (pos_args->use_graph_invariance){
                    tokenized_pos(cur, 0) = graph_invariance_marker;
                } else {
                    tokenized_pos(cur, 0) = graph_start + cur;
                }
            }
        }
    }


};

#endif //GRAPH_TOKENIZER_H
