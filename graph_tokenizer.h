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

using namespace std;

class GraphTokenizer {
public:
    Matrix<int> tokenized_inputs; // possibly 2D inputs when using concat_edges
    Matrix<int> tokenized_pos;  // possibly 2D positions when using use_graph_structure

    vector<int> node_list;
    // ordered in the constructor, PLEASE NOTE this is not the original order nore the shuffled order but order of nodes seen in shuffled edges
    vector<pair<int, int> > edge_list; // already shuffled
    unique_ptr<vector<vector<int> > > distances_ptr;
    unique_ptr<vector<vector<int> > > edge_ground_truths_ptr; // -1 for unreachable
    unique_ptr<vector<vector<int> > > node_ground_truths_ptr; // -1 for unreachable
    bool concat_edges = true;
    bool duplicate_edges = false;
    bool include_nodes_in_graph_tokenization;
    bool use_edges_invariance;
    bool use_node_invariance;
    bool use_graph_invariance;
    bool use_graph_structure;

    GraphTokenizer(vector<pair<int, int> > edge_list,
                   const bool concat_edges = true, const bool duplicate_edges = false,
                   const bool include_nodes_in_graph_tokenization = false,
                   const bool use_edges_invariance = false,  // for concated edges this allows true permutation invariance
                   const bool use_node_invariance = false,
                   const bool use_graph_invariance = false,  // divide positions by task structure
                   const bool use_graph_structure = false) {  // 2d positions by graph structure
        this->edge_list = edge_list;
        this->concat_edges = concat_edges;
        this->duplicate_edges = duplicate_edges; // needs to be false for directed graphs
        this->include_nodes_in_graph_tokenization = include_nodes_in_graph_tokenization;
        this->use_edges_invariance = use_edges_invariance;
        this->use_node_invariance = use_node_invariance;
        this->use_graph_invariance = use_graph_invariance;
        this->use_graph_structure = use_graph_structure;

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
                  const map<std::string, int> pos_dictionary,
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
        if (!concat_edges) {
            num_tokens *= 3;
        }
        if (include_nodes_in_graph_tokenization) {
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
        if (concat_edges) {
            tokenized_inputs.resize(num_tokens, 2, dictionary.at("<pad>"));
        } else {
            tokenized_inputs.resize(num_tokens, 1, dictionary.at("<pad>"));
        }
        if (use_graph_structure) {
            tokenized_pos.resize(num_tokens, 2, pos_dictionary.at("pad"));
        } else {
            tokenized_pos.resize(num_tokens, 1, pos_dictionary.at("pad"));
        }

        // write in new edge list
        if (!concat_edges) {
            for (size_t i = 0; i < new_edge_list.size(); i++, cur += 3) {
                tokenized_inputs(i * 3, 0) = node_shuffle_map.at(new_edge_list[i].first);
                tokenized_inputs(i * 3 + 1, 0) = node_shuffle_map.at(new_edge_list[i].second);
                tokenized_inputs(i * 3 + 2, 0) = dictionary.at("|");
                if (use_graph_structure){
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
                if (use_edges_invariance) {
                    tokenized_pos(i, 0) = edge_invariance_marker;
                } else if (use_graph_invariance){
                    tokenized_pos(i, 0) = graph_invariance_marker;
                } else {
                    tokenized_pos(i, 0) = graph_start + cur;
                }
            }
        }
        if (include_nodes_in_graph_tokenization) { // write in node list at end
            for (size_t i = 0; i < node_list.size(); i++, cur++) {
                tokenized_inputs(cur, 0) = node_shuffle_map.at(node_list[i]);
                if (concat_edges) {  // duplicate node if concat edges in both dims
                    tokenized_inputs(cur, 1) = tokenized_inputs(cur, 0);
                }
                if (use_node_invariance) {
                    tokenized_pos(cur, 0) = node_invariance_marker;
                } else if (use_graph_invariance){
                    tokenized_pos(cur, 0) = graph_invariance_marker;
                } else {
                    tokenized_pos(cur, 0) = graph_start + cur;
                }
            }
        }
    }


    // helper functions for getting gather_ids of edges and nodes TODO


    /*
     * note none of these are in the node shuffle map order,
     * this is because we will convert everything to gather_ids for the model
     * and that will handle the map and shift to vocab ids
     */
    template<class D>
    void get_distances(unique_ptr<Graph<D> > &g_ptr) {
        /*
         * Just the distance matrix [N X N] in the original graph node order
         */
        auto N = num_vertices(*g_ptr);
        unique_ptr<DistanceMatrix<D> > boost_distances_ptr;
        johnson<D>(g_ptr, boost_distances_ptr, false);
        // convert boost to c++ matrix
        distances_ptr = make_unique<vector<vector<int> > >(N, vector<int>(N, inf));
        for (int i = 0; i < static_cast<int>(N); i++) {
            for (int j = 0; j < static_cast<int>(N); j++) {
                (*distances_ptr)[i][j] = (*boost_distances_ptr)[i][j];
            }
        }
    }

    template<class D>
    void get_edge_ground_truths(unique_ptr<Graph<D> > &g_ptr,
                                const bool is_causal) {
        /*
         * Just the distance matrix but in edge_list order  [E X N], this is possibly causally constrained
         * Note these only work for the projected ranking loss so N is in vocab order
         */
        if (is_causal) {
            // this will calculate the distances as well
            if (distances_ptr) {
                throw runtime_error("Causal ground truths should not be calculated with precomputed distances");
            }
            floyd_warshall_frydenlund(g_ptr, distances_ptr, edge_ground_truths_ptr, edge_list, false);

            // convert both to node shuffle map order TODO
            // auto N = distances_ptr->size();
            unique_ptr<vector<vector<int> > > distances_ptr;
            unique_ptr<vector<vector<int> > > ground_truths_ptr;
        } else {
            if (!distances_ptr) {
                get_distances<D>(g_ptr); // already node_shuffle_map, so edges will be too
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

    void get_node_ground_truths(const bool is_direct_ranking) {
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
