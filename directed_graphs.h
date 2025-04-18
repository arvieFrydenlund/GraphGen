//
// Created by arvie on 3/26/25.
//

#ifndef BALANCED_H
#define BALANCED_H

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/graph/erdos_renyi_generator.hpp>
#include <boost/random/linear_congruential.hpp>
#include <boost/graph/directed_graph.hpp>
#include <boost/graph/exterior_property.hpp>
#include <boost/graph/random.hpp>
#include <random>
#include <iostream>
#include <string>
#include <vector>

#include "undirected_graphs.h"


using namespace std;

typedef pair<int, int> start_end_pair;


inline vector<unordered_set<int>> get_children(const unique_ptr<Graph<boost::directedS>> &g_ptr, const int N) {
  /*
    * Get the children of each node in the graph. The children are stored in a vector of unordered sets.
	* The index of the vector does not correspond to the node id in the original graph.
   */
    vector<unordered_set<int>> children(N, unordered_set<int>());
    for (int i = 0; i < num_vertices(*g_ptr); i++) {
      	auto children_at_i = children[i];
    	// convert node at idx to graph id
        for (auto e : boost::make_iterator_range(boost::out_edges(i, *g_ptr))) {
         	auto child_of_i = boost::target(e, *g_ptr);
            children_at_i .insert(child_of_i);
        }
    }
    return children;
}


inline vector<unordered_set<int>> get_parents(const vector<unordered_set<int>> &children){
    vector<unordered_set<int>> parents(children.size(), unordered_set<int>());
    for (int i = 0; i < children.size(); i++) {
        for (int j : children[i]) {
            parents[j].insert(i);
        }
    }
    return parents;
}


inline unordered_set<int> get_descendents(const int node, const unique_ptr<Graph<boost::directedS>> &g_ptr){
    unordered_set<int> queue;
    queue.insert(node);
    unordered_set<int> visited;
    unordered_set<int> descendants;
    while (queue.size() != 0) {
        int current = *queue.begin();
        queue.erase(queue.begin());
        visited.insert(current);
        if ( boost::out_degree(current, *g_ptr) == 0 ) {
            continue;
        }
        for (auto e : boost::make_iterator_range(boost::out_edges(current, *g_ptr))) {
            int child = boost::target(e, *g_ptr);
            if (descendants.find(child) == descendants.end()) {
                descendants.insert(child);
            }
            if (visited.find(child) == visited.end()) {
                queue.insert(child);
            }
        }
    }
    return descendants;
}


inline start_end_pair path_star_generator(unique_ptr<Graph<boost::directedS>> &g_ptr,
    const int min_num_arms, const int max_num_arms,
    const int min_arm_length, const int max_arm_length,
    std::mt19937 &gen, const bool verbose = false) {

    g_ptr = make_unique<Graph<boost::directedS>>();

    int num_arms = min_num_arms;
    if ( min_num_arms < max_num_arms) {
        num_arms = std::uniform_int_distribution<int>(min_num_arms, max_num_arms)(gen);
    }
    int arm_length = max_arm_length;
    if ( min_arm_length < max_arm_length) {
        arm_length = std::uniform_int_distribution<int>(min_arm_length, max_arm_length)(gen);
    }

    const int num_nodes = num_arms * (arm_length - 1) + 1;

    // add start node
    int start = 0;
    int end = arm_length - 1;  // -1 since it is the index
    int cur = 1;  // 0 is start
    for (int i = 0; i < num_arms; i++) {
        int prev_node = start;
        for (int j = 0; j < arm_length - 1; j++) {
            boost::add_edge(prev_node, cur, *g_ptr);
            prev_node = cur;
            cur++;
        }
    }
    make_edge_weights(*g_ptr, verbose);

    if (verbose) {
        cout << "Graph has " << num_vertices(*g_ptr) << " vertices and " << num_edges(*g_ptr) << " edges" << endl;
    }
    return pair<int, int>(start, end);
}


bool balanced_graph_size_check(const int num_nodes, int lookahead, const int min_noise_reserve = 0){
      // need two arms of length lookahead (+1) and min_noise_reserve
      // expose this to python
      return num_nodes > 0  &&  num_nodes > lookahead * 2 + min_noise_reserve + 1;
 }


inline start_end_pair balanced_generator(unique_ptr<Graph<boost::directedS>> &g_ptr,
                              const int num_nodes, std::mt19937 &gen, int lookahead, const int min_noise_reserve = 0,
                              const int max_num_parents = 4, int max_noise = -1, const bool verbose = false) {

    assert ( balanced_graph_size_check(num_nodes, lookahead, min_noise_reserve) );

    g_ptr = make_unique<Graph<boost::directedS>>();

    int max_num_paths = (num_nodes - min_noise_reserve - 1 - lookahead) / lookahead;  // integer division
    int num_paths = uniform_int_distribution<int>(1, max_num_paths)(gen);

    int start = 0;
    int end = lookahead - 1;
    int cur = 1;
    // make ground-truth path
    int prev_node = start;
    for (int _ = 0; _ < lookahead; _++) {
        boost::add_edge(prev_node, cur, *g_ptr);
        prev_node = cur;
        cur++;
    }

    // make other paths connected to start
    int available = num_nodes - min_noise_reserve - 1 - lookahead;
    for ( int _ = 0; _ < num_paths; _++) {
        prev_node = start;
        int extra_length = uniform_int_distribution<int>(0, 1)(gen);
        int other_branch_length = lookahead + extra_length;
         if (other_branch_length > available) {
            other_branch_length = available;
        }
        for (int _ = 0; _ < other_branch_length; _++) {
            boost::add_edge(prev_node, cur, *g_ptr);
            prev_node = cur;
            cur++;
        }
        available -= other_branch_length;
    }

    //  build in-arm (of max length lookahead)
    if (cur < num_nodes - 1 ) {
    	int num_prefix_vertices = uniform_int_distribution<int>(0, min(num_nodes - cur - 1, lookahead))(gen);
    	int next_node = start;
    	for (int _ = 0; _ < num_prefix_vertices; _++) {
        	boost::add_edge(cur, next_node, *g_ptr);
        	next_node = cur;
        	cur++;
   		}
	}

    // sample some parent/ancestor vertices
    // initialize sizes
    auto children = get_children(g_ptr, num_nodes); // these are in indices not node ids
    auto parents = get_parents(children); // these are in indices not node ids
    vector<float> in_degrees(num_nodes);
    vector<float> out_degrees(num_nodes);
    for (int i = 0; i < num_nodes; i++) {
        constexpr float alpha = 0.5;
        in_degrees[i] = alpha + static_cast<float>(children[i].size());
        out_degrees[i] = alpha + static_cast<float>(parents[i].size());
    }

    int num_noise = 0;
    if ( max_noise == -1 ) {
        max_noise = num_nodes - cur - 1;  // max noise is the rest of the nodes
    }
    while ( cur < num_nodes && num_noise < max_noise ) {
        const int num_children = uniform_int_distribution<int>(0, max_num_parents)(gen);
        int num_parents = 0;
        if ( num_children != 0 ) {
            num_parents = uniform_int_distribution<int>(0, max_num_parents)(gen);
        } else {
            num_parents = uniform_int_distribution<int>(1, max_num_parents)(gen);
        }
        vector<float> probabilities(cur);
        for (int j = 0; j < cur; j++) {
            probabilities[j] = in_degrees[j];  // only looking at nodes before i
        }
        for (int j = 0; j < num_children; j++) {
            int child_id = std::discrete_distribution<>(probabilities.begin(), probabilities.end())(gen);
            children[cur].insert(child_id);
            parents[child_id].insert(cur);
            in_degrees[child_id] += 1;
            boost::add_edge(cur, child_id, *g_ptr);
        }

        unordered_set<int> descendants;
        if ( num_children > 0 ) {  // since we are using edge list graphs we can not insert nodes
            // this causes an issue with boost if we are looking for an edge for a node that is not in the graph
            descendants = get_descendents(cur, g_ptr);
        } else { // empty
            descendants = unordered_set<int>();
        }
        for (const int descendant : descendants) {
            probabilities[descendant] = 0;  // do not sample descendants of the node
        }
        for (int j = 0; j < num_parents; j++) {
            if (accumulate(probabilities.begin(), probabilities.end(), 0.0) > 0.0) {
                int parent_id = std::discrete_distribution<int>(probabilities.begin(), probabilities.end())(gen);
                children[parent_id].insert(cur);
                parents[cur].insert(parent_id);
                out_degrees[parent_id] += 1;
                boost::add_edge(parent_id, cur, *g_ptr);
                probabilities[parent_id] = 0;  // zero out so fresh ones are sampled
            }
        }
        cur += 1;
    }
    return pair<int, int>(start, end);
}



inline int convert_undirected_to_directed(unique_ptr<Graph<boost::undirectedS>> &ug_ptr,
                                          unique_ptr<Graph<boost::directedS>> &dg_ptr){

  return 0;
}


#endif //BALANCED_H
