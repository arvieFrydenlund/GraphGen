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
#include <boost/graph/floyd_warshall_shortest.hpp>
#include <boost/graph/random.hpp>
#include <random>
#include <iostream>
#include <string>
#include <vector>

#include "graph.h"


using namespace std;


inline vector<unordered_set<int>> get_children(const unique_ptr<Graph<boost::directedS>> &g_ptr,
                                               const vector<int> &node_ids, const vector<int> &reverse_node_ids) {
  /*
    * Get the children of each node in the graph. The children are stored in a vector of unordered sets.
	* The index of the vector does not correspond to the node id in the original graph.
   */



    vector<unordered_set<int>> children(num_vertices(*g_ptr), unordered_set<int>());
    for (int i = 0; i < num_vertices(*g_ptr); i++) {
      	auto children_at_i = children[i];
    	// convert node at idx to graph id
        for (auto e : boost::make_iterator_range(boost::out_edges(node_ids[i], *g_ptr))) {
         	auto child_of_i = boost::target(e, *g_ptr);
            children_at_i .insert(reverse_node_ids[child_of_i]);  // reverse back to the index
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


inline unordered_set<int> get_decendants(const int node, const unique_ptr<Graph<boost::directedS>> &g_ptr){
    unordered_set<int> queue;
    queue.insert(node);
    unordered_set<int> visited;
    unordered_set<int> decendants;
    while (queue.size() != 0) {
        int current = *queue.begin();
        queue.erase(queue.begin());
        visited.insert(current);
        for (auto e : boost::make_iterator_range(boost::out_edges(current, *g_ptr))) {
            int child = boost::target(e, *g_ptr);
            if (decendants.find(child) == decendants.end()) {
                decendants.insert(child);
            }
            if (visited.find(child) != visited.end()) {
                continue;
            }
            queue.insert(child);
        }
    }
    return decendants;
}


inline vector<int> make_node_ids(const int num_nodes, unique_ptr<Graph<boost::directedS>> &g_ptr, std::mt19937 &gen){
    // permute graph before we make it
    auto node_ids = vector<int>(num_nodes);
    for (int i = 0; i < num_nodes; i++) {
        node_ids[i] = i;
        boost::add_vertex(*g_ptr);  // note we are adding vertices here
    }
    std::shuffle(node_ids.begin(), node_ids.end(), gen);
    return node_ids;
 }


inline int path_star_generator(unique_ptr<Graph<boost::directedS>> &g_ptr, unique_ptr<DistanceMatrix<boost::directedS>> &distances_ptr,
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
    auto node_ids = make_node_ids(num_nodes, g_ptr, gen);

    // add start node
    int cur = 0;
    int start = node_ids[cur];
    cur++;
    for (int i = 0; i < num_arms; i++) {
        int prev_node = start;
        for (int j = 0; j < arm_length - 1; j++) {
            boost::add_edge(prev_node, node_ids[cur], *g_ptr);
            prev_node = node_ids[cur];
            cur++;
        }
    }
    make_edge_weights(*g_ptr, verbose);

    if (verbose) {
        cout << "Graph has " << num_vertices(*g_ptr) << " vertices and " << num_edges(*g_ptr) << " edges" << endl;
        for (int node : node_ids) {
            auto decendants = get_decendants(node, g_ptr);
            cout << "Decendants of " << node << " are: ";
            for (auto d : decendants) {
                cout << d << " ";
            }
            cout << endl;
        }
    }

    distances_ptr = make_unique<DistanceMatrix<boost::directedS>>(num_vertices(*g_ptr));
    auto distances = *distances_ptr.get();
    const int r = floyd_warshall(*g_ptr, distances, verbose);
    return r;
}


inline int balanced_generator(unique_ptr<Graph<boost::directedS>> &g_ptr,
                              unique_ptr<DistanceMatrix<boost::directedS>> &distances_ptr,
                              const int num_nodes, std::mt19937 &gen, int lookahead, const int min_noise_reserve = 0,
                              const int max_num_parents = 4, const bool verbose = false) {

    g_ptr = make_unique<Graph<boost::directedS>>();
    auto node_ids = make_node_ids(num_nodes, g_ptr, gen); // this adds the max to the graph, not all may be used

    vector<int> reverse_node_ids(node_ids.size());
    for (int i = 0; i < node_ids.size(); i++) {
        reverse_node_ids[node_ids[i]] = i;
    }

    const int max_num_path_nodes = num_nodes - min_noise_reserve - 1 - lookahead;
    if ( max_num_path_nodes  < 0 ) {
          return -1;
     }
    int max_num_paths = max_num_path_nodes / lookahead;  // integer division
    int num_paths = uniform_int_distribution<int>(1, max_num_paths)(gen);
    // cout << "num_paths " << num_paths << " max_num_paths " << max_num_paths << endl;
    // cout << "max_num_path_nodes " << max_num_path_nodes << " lookahead " << lookahead << endl;

    int start = node_ids[0];
    int end = node_ids[lookahead];

    int cur = 1;
    // make ground-truth path
    int prev_node = start;
    for (int _ = 0; _ < lookahead; _++) {
        boost::add_edge(prev_node, node_ids[cur], *g_ptr);
        prev_node = node_ids[cur];
        cur++;
    }
    // make other paths connected to start
    int available = max_num_path_nodes;
    for ( int _ = 0; _ < num_paths; _++) {
        prev_node = start;
        int extra_length = uniform_int_distribution<int>(0, 1)(gen);
        int other_branch_length = lookahead + extra_length;
         if (other_branch_length > available) {
            other_branch_length = available;
        }
        for (int _ = 0; _ < other_branch_length; _++) {
            boost::add_edge(prev_node, node_ids[cur], *g_ptr);
            prev_node = node_ids[cur];
            cur++;
        }
        available -= other_branch_length;
    }
    //  build in-arm
    if ( cur < num_nodes - 1 ) {
    	int num_prefix_vertices = uniform_int_distribution<int>(0, min(num_nodes - cur - 1, lookahead))(gen);
    	int next_node = start;
    	for (int _ = 0; _ < num_prefix_vertices; _++) {
        	boost::add_edge(node_ids[cur], next_node, *g_ptr);
        	next_node = node_ids[cur];
        	cur++;
   		}
	}

    // sample some parent/ancestor vertices
    float alpha = 0.5;
	// initialize sizes
    auto children = get_children(g_ptr, node_ids, reverse_node_ids); // these are in idicies not node ids
    auto parents = get_parents(children); // these are in idicies not node ids
    vector<float> in_degrees(num_nodes);
    vector<float> out_degrees(num_nodes);
    for (int i = 0; i < num_nodes; i++) {
        in_degrees[i] = alpha + children[i].size();
        out_degrees[i] = alpha + parents[i].size();
    }
    for (int i = cur; i < num_nodes - 1; i++) {
        int num_children = uniform_int_distribution<int>(0, max_num_parents)(gen);
        int num_parents = 0;
        if ( num_children != 0 ) {
            num_parents = uniform_int_distribution<int>(0, max_num_parents)(gen);
        } else {
            num_parents = uniform_int_distribution<int>(1, max_num_parents)(gen);
        }

       	cout << " num children and num parents " << num_children << " " << num_parents << endl;
       	cout << " num nodes and i " << num_nodes << " " << i << " " << num_nodes - i << endl;
        vector<float> probabilities(i);
        for (int j = 0; j < i; j++) {
            probabilities[j] = in_degrees[j];
        }
        for (int j = 0; j < i; j++) {
            cout << probabilities[j] << " ";
        }
        cout << endl;
        for (int j = 0; j < num_children; j++) {
            int child_id = std::discrete_distribution<>(probabilities.begin(), probabilities.end())(gen);
            children[i].insert(child_id);
            parents[child_id].insert(i);
            in_degrees[child_id] += 1;
            boost::add_edge(node_ids[cur], node_ids[child_id], *g_ptr);
        }

        // do not sample descendants of the node
        auto descendants = get_decendants(node_ids[i], g_ptr);
        for (int descendant : descendants) {
            probabilities[reverse_node_ids[descendant]] = 0;
        }
        if (accumulate(probabilities.begin(), probabilities.end(), 0.0) > 0.0) {
            for (int j = 0; j < num_parents; j++) {
                int parent_id = std::discrete_distribution<int>(probabilities.begin(), probabilities.end())(gen);
                children[parent_id].insert(i);
                parents[i].insert(parent_id);
                out_degrees[parent_id] += 1;
                boost::add_edge(node_ids[parent_id], node_ids[cur], *g_ptr);
            }
        }
    }

    /*
    children = get_children(g_ptr);
    parents = get_parents(children);
    for (int i = 0; i < num_vertices(*g_ptr); i++) {
        cout << "Node " << i;
        for (int c : children[i]) {
            cout << " child " << c;
        }
        for (int p : parents[i]) {
            cout << " parent " << p;
        }
        cout << endl;
    }
     */


    return 0;
}



inline int convert_undirected_to_directed(unique_ptr<Graph<boost::undirectedS>> &ug_ptr,
                                          unique_ptr<Graph<boost::directedS>> &dg_ptr){

  return 0;
}


#endif //BALANCED_H
