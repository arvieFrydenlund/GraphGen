//
// Created by arvie on 05/11/25.
//

#ifndef GRAPH_TOKENIZER_H
#define GRAPH_TOKENIZER_H

#include <iostream>
#include <random>
#include <queue>
#include <map>
#include "undirected_graphs.h"
#include "directed_graphs.h"

using namespace std;

class GraphTokenizer {
public:
    vector<vector<int> > tokenized_inputs;  // 2D inputs when using concatentated edges
    vector<int> node_list;
    vector<pair<int, int> >  edge_list;
    bool concat_edges = true;
    bool duplicate_edges = false;
    bool include_nodes_in_graph_tokenization;

    GraphTokenizer(vector<int> node_list, vector<pair<int, int> > edge_list,
            const bool concat_edges = true, const bool duplicate_edges = false, const bool include_nodes_in_graph_tokenization = false){
        this->node_list = node_list;
        this->edge_list = edge_list;
        this->concat_edges = concat_edges;
        this->duplicate_edges = duplicate_edges;  // needs to be false for directed graphs
        this->include_nodes_in_graph_tokenization = include_nodes_in_graph_tokenization;
    }

    void tokenize(const map<std::string, int> &dictionary,
                          const unique_ptr<vector<int> > &node_shuffle_map,
                          std::mt19937 &gen) {
        // node order by edge appearance
        auto num_tokens = static_cast<int>(edge_list.size());
        vector<pair<int, int> > new_edge_list;
         // remap edges according to node shuffle map
        if (duplicate_edges) {
            num_tokens *= 2;
            // make copy of edge list with reversed edges
            std::copy(edge_list.begin(), edge_list.end(), std::back_inserter<vector<pair<int, int> > >(new_edge_list));
            for (size_t i = 0; i < edge_list.size(); i++) {
                new_edge_list.push_back(make_pair(
                        node_shuffle_map->at(edge_list[i].second),
                        node_shuffle_map->at(edge_list[i].first)
                                        )
                );
            }
        } else {
            new_edge_list = this->edge_list;
        }
        if ( !concat_edges) {
            num_tokens *= 3;
        }
        if (include_nodes_in_graph_tokenization) {
            num_tokens += static_cast<int>(node_list.size());
        }
        auto cur = 0;
        tokenized_inputs = vector<vector<int> >(num_tokens, vector<int>(2, dictionary.at("<pad>")));
        // write in new edge list
        if ( !concat_edges) {
            for (size_t i = 0; i < new_edge_list.size(); i++, cur += 3) {
                tokenized_inputs[i * 3][0] = node_shuffle_map->at(new_edge_list[i].first);
                tokenized_inputs[i * 3 + 1][0] = node_shuffle_map->at(new_edge_list[i].second);
                tokenized_inputs[i * 3 + 2][0] = dictionary.at("|");
            }
        } else {
            for (size_t i = 0; i < new_edge_list.size(); i++, cur++) {
                tokenized_inputs[i][0] = node_shuffle_map->at(new_edge_list[i].first);
                tokenized_inputs[i][1] = node_shuffle_map->at(new_edge_list[i].second);
            }
        }
        if (include_nodes_in_graph_tokenization) {
            // write in node list at end
            for (size_t i = 0; i < node_list.size(); i++) {
                tokenized_inputs[cur + i][0] = node_shuffle_map->at(node_list[i]);
                if (concat_edges){
                    tokenized_inputs[cur + i][1] = tokenized_inputs[cur + i][0];  // duplicate it
                }
            }
            cur += static_cast<int>(node_list.size());
        }
    };
};

#endif //GRAPH_TOKENIZER_H
