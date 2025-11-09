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
    vector<int> node_list;  // ordered in the constructor, PLEASE NOTE this is not the original order nore the shuffled order but order of nodes seen in shuffled edges
    vector<pair<int, int> >  edge_list; // already shuffled
    unique_ptr<vector<vector<int>>> distances_ptr;
    unique_ptr<vector<vector<int>>> edge_ground_truths_ptr;  // -1 for unreachable
    unique_ptr<vector<vector<int>>> node_ground_truths_ptr;   // -1 for unreachable
    bool concat_edges = true;
    bool duplicate_edges = false;
    bool include_nodes_in_graph_tokenization;

    GraphTokenizer(vector<pair<int, int> > edge_list,
            const bool concat_edges = true, const bool duplicate_edges = false, const bool include_nodes_in_graph_tokenization = false){
        this->edge_list = edge_list;
        this->concat_edges = concat_edges;
        this->duplicate_edges = duplicate_edges;  // needs to be false for directed graphs
        this->include_nodes_in_graph_tokenization = include_nodes_in_graph_tokenization;

        // build node list by edge list order, this is not always needed
        node_list = vector<int>();
        map<int, bool> node_seen;
        for (size_t i = 0; i < edge_list.size(); i++) {
            if (node_seen.find(edge_list[i].first) == node_seen.end()) {
                node_list.push_back(edge_list[i].first);
                node_seen[edge_list[i].first] = true;
            }
            if (node_seen.find(edge_list[i].second) == node_seen.end()) {
                node_list.push_back(edge_list[i].second);
                node_seen[edge_list[i].second] = true;
            }
        }
    }

    void tokenize(const map<std::string, int> &dictionary,
                          const vector<int> &node_shuffle_map,
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
                        node_shuffle_map.at(edge_list[i].second),
                        node_shuffle_map.at(edge_list[i].first)
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
                tokenized_inputs[i * 3][0] = node_shuffle_map.at(new_edge_list[i].first);
                tokenized_inputs[i * 3 + 1][0] = node_shuffle_map.at(new_edge_list[i].second);
                tokenized_inputs[i * 3 + 2][0] = dictionary.at("|");
            }
        } else {
            for (size_t i = 0; i < new_edge_list.size(); i++, cur++) {
                tokenized_inputs[i][0] = node_shuffle_map.at(new_edge_list[i].first);
                tokenized_inputs[i][1] = node_shuffle_map.at(new_edge_list[i].second);
            }
        }
        if (include_nodes_in_graph_tokenization) {
            // write in node list at end
            for (size_t i = 0; i < node_list.size(); i++) {
                tokenized_inputs[cur + i][0] = node_shuffle_map.at(node_list[i]);
                if (concat_edges){
                    tokenized_inputs[cur + i][1] = tokenized_inputs[cur + i][0];  // duplicate it
                }
            }
            // cur += static_cast<int>(node_list.size());  // incase we need cur later
        }
    }

    // helper functions for getting gather_ids  TODO


    /*
     * note none of these are in the node shuffle map order,
     * this is because we will convert everything to gather_ids for the model
     * and that will handle the map and shift to vocab ids
     */
    template<class D>
    void get_distances(unique_ptr<Graph<D>> &g_ptr){
        /*
         * Just the distance matrix [N X N] in the original graph node order
         */
        auto N = num_vertices(*g_ptr);
        unique_ptr<DistanceMatrix<D> > boost_distances_ptr;
        johnson<D>(g_ptr, boost_distances_ptr, false);
        // convert boost to c++ matrix
        distances_ptr = make_unique<vector<vector<int>>>(N, vector<int>(N, inf));
        for (int i = 0; i < static_cast<int>(N); i++) {
            for (int j = 0; j < static_cast<int>(N); j++) {
                (*distances_ptr)[i][j] = (*boost_distances_ptr)[i][j];
            }
        }
    }

    template<class D>
    void get_edge_ground_truths(unique_ptr<Graph<D>> &g_ptr,
                                const bool is_causal){
        /*
         * Just the distance matrix but in edge_list order  [E X N], this is possibly causally constrained
         * Note these only work for the projected ranking loss so N is in vocab order
         */
        if (is_causal) {  // this will calculate the distances as well
            if (distances_ptr) {
                throw runtime_error("Causal ground truths should not be calculated with precomputed distances");
            }
            floyd_warshall_frydenlund(g_ptr, distances_ptr, edge_ground_truths_ptr, edge_list, false);

            // convert both to node shuffle map order
            auto N = distances_ptr->size();
            unique_ptr<vector<vector<int>>> distances_ptr;
            unique_ptr<vector<vector<int>>> ground_truths_ptr;

        } else {
            if (!distances_ptr) {
                get_distances<D>(g_ptr);  // already node_shuffle_map, so edges will be too
            }
            auto N = num_vertices(*g_ptr);
            auto E = edge_list.size();
            // Makes a [E, N] matrix of ground truths where each row is the distance from the edge.first to all other nodes
            edge_ground_truths_ptr = make_unique<vector<vector<int> > >(E, vector<int>(N, -1));
            for (int t = 0; t < static_cast<int>(E); t++) {
                for (int i = 0; i < static_cast<int>(N); i++) {
                    (*edge_ground_truths_ptr)[t][i] = (*distances_ptr)[edge_list[t].first][i];
                }
            }

        }
    }

    void get_node_ground_truths(const bool is_direct_ranking){
        /*
         * Just the distance matrix but in node_list order in the first dimension.
         *
         * If it is direct ranking then the second dimension is the node_list order as well
         * i.e [N in node order by N in node order]
         * So for nodes u,v,w,x,y in node_list order [u,v,w,x,y], the ground_truths_ptr would be:
         * [[d(u,u), d(u,v), d(u,w), d(u,x), d(u,y)],
         *  [d(v,u), d(v,v), d(v,w), d(v,x), d(v,y)],
         *  [d(w,u), d(w,v), d(w,w), d(w,x), d(w,y)],
         *  [d(x,u), d(x,v), d(x,w), d(x,x), d(x,y)],
         *  [d(y,u), d(y,v), d(y,w), d(y,x), d(y,y)] ]
         *
         * However if it is projected_ranking then the second dimension is in vocab order
         * i.e. [N in node order by N in vocab order]
         * So for nodes u,v,w,x,y in node_list order [u,v,w,x,y], and vocab order [w,u,y,v,x], the ground_truths_ptr would be:
         * [[d(u,w), d(u,u), d(u,y), d(u,v), d(u,x)],
         *  [d(v,w), d(v,u), d(v,y), d(v,v), d(v,x)],
         *  [d(w,w), d(w,u), d(w,y), d(w,v), d(w,x)],
         *  [d(x,w), d(x,u), d(x,y), d(x,v), d(x,x)],
         *  [d(y,w), d(y,u), d(y,y), d(y,v), d(y,x)] ]
         */
        auto N = node_list.size();
        edge_ground_truths_ptr = make_unique<vector<vector<int> > >(N, vector<int>(N, -1));
        for (int i = 0; i < static_cast<int>(N); i++) {
            for (int j = 0; j < static_cast<int>(N); j++) {
                if (is_direct_ranking) {
                    (*node_ground_truths_ptr)[i][j] = (*distances_ptr)[node_list[i]][node_list[j]];
                } else {
                    (*node_ground_truths_ptr)[i][j] = (*distances_ptr)[node_list[i]][j];
                }
            }
        }
    }


};

#endif //GRAPH_TOKENIZER_H
