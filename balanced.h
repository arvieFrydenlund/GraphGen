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


inline vector<unordered_set<int>> get_children(const unique_ptr<Graph<boost::directedS>> &g_ptr){
    vector<unordered_set<int>> children;
    for (int i = 0; i < num_vertices(*g_ptr); i++) {
        unordered_set<int> c;
        for (auto e : boost::make_iterator_range(boost::out_edges(i, *g_ptr))) {
            c.insert(boost::target(e, *g_ptr));
        }
        children.push_back(c);
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

    auto children = get_children(g_ptr);
    auto parents = get_parents(children);
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

    distances_ptr = make_unique<DistanceMatrix<boost::directedS>>(num_vertices(*g_ptr));
    auto distances = *distances_ptr.get();
    const int r = floyd_warshall(*g_ptr, distances, verbose);
    return r;
}


inline int balanced_generator(unique_ptr<Graph<boost::directedS>> &g_ptr, unique_ptr<DistanceMatrix<boost::directedS>> &distances_ptr,
    const int num_nodes, std::mt19937 &gen, int lookahead, const int max_num_parents = 4, int max_prefix_vertices = -1, const bool verbose = false) {

      /*
        # sample some parent/ancestor vertices
        alpha = 0.5
        in_degrees = np.array([alpha + len(vertex.parents) for vertex in vertices[:num_vertices]])
        out_degrees = np.array([alpha + len(vertex.children) for vertex in vertices[:num_vertices]])
        for i in range(index, num_vertices):
            # sample the number of child and parent vertices
            num_children = randrange(0, self.max_num_parents)
            num_parents = randrange(0 if num_children != 0 else 1, self.max_num_parents)
            num_children = min(num_children, i)
            num_parents = min(num_parents, i)

            # sample the children of this new node
            probabilities = in_degrees[:index].copy()
            probabilities /= np.sum(probabilities)
            for child_id in np.random.choice(index, num_children, replace=False, p=probabilities):
                vertices[index].children.append(vertices[child_id])
                vertices[child_id].parents.append(vertices[index])
                in_degrees[child_id] += 1

            # to avoid creating a cycle, we have to remove any descendants from the possible parents
            descendants = self.get_descendants(vertices[index])
            probabilities = out_degrees[:index].copy()

            for descendant in descendants:
                probabilities[descendant.id] = 0
            total_probability = np.sum(probabilities)
            if total_probability != 0.0:
                probabilities /= total_probability
                num_parents = min(num_parents, index - len(descendants))

                # sample the parents of this new node
                for parent_id in np.random.choice(index, num_parents, replace=False, p=probabilities):
                    vertices[parent_id].children.append(vertices[i])
                    vertices[i].parents.append(vertices[parent_id])
                    out_degrees[parent_id] += 1
            index += 1
       */

      if ( max_prefix_vertices  < 0 ) {
          max_prefix_vertices = num_nodes;
      }

    int max_num_paths = (num_nodes - 1) / lookahead;  // integer division
    int num_paths = uniform_int_distribution<int>(2, max_num_paths + 1)(gen);  // B in paper
    // u = randrange(0, 6) in paper
    int num_verts = min(lookahead * num_paths + 1 + uniform_int_distribution<int>(0, 6)(gen), num_nodes);
    num_verts = max(max(2, num_verts), 1 + num_paths * lookahead);  // num_nodes is really max number of vertices

    // pint num_verts
    cout << "num_verts: " << num_verts << endl;

    auto g = make_unique<Graph<boost::directedS>>();
    // auto node_ids = make_node_ids(num_verts, g_ptr, gen);

    int cur = 0;
    // int start = node_ids[cur];
    //int end = node_ids[lookahead];
    // make ground-truth path
    /*
    for (int i = 0; i < lookahead; i++) {
        int prev_node = start;
        boost::add_edge(prev_node, node_ids[cur], *g_ptr);
        prev_node = node_ids[cur];
        cur++;
    }
    */
    // make other paths
    /*
    for ( int j = 0; j < num_paths - 1; j++) {
        int prev_node = start;
        int other_branch_length = lookahead + min(2, num_verts - cur - (num_paths - j - 1) * lookahead + 2);
        for (int _ = 0; _ < other_branch_length; _++) {
            boost::add_edge(prev_node, node_ids[cur], *g_ptr);
            prev_node = node_ids[cur];
            cur++;
        }
    } */
    //  build in-arm
    /*
    int num_prefix_vertices = uniform_int_distribution<int>(0, min(max_prefix_vertices + 1, num_verts - cur + 1))(gen);
    int next_node = start;
    for (int _ = 0; _ < num_prefix_vertices; _++) {
        boost::add_edge(node_ids[cur], next_node, *g_ptr);
        next_node = node_ids[cur];
        cur++;
    } */
    // sample some parent/ancestor vertices
    float alpha = 0.5;

    /*
    auto children = get_children(g_ptr);
    auto parents = get_parents(children);
    vector<int> in_degrees(num_verts);
    vector<int> out_degrees(num_verts);
    for (int i = 0; i < num_verts; i++) {
        in_degrees[i] = alpha + children[i].size();
        out_degrees[i] = alpha + parents[i].size();
    }
     */

     /*
    for (int i = 0; i < num_verts; i++) {
        int num_children = uniform_int_distribution<int>(0, max_num_parents)(gen);
        int num_parents = 0;
        if ( num_children != 0 ) {
            num_parents = uniform_int_distribution<int>(0, max_num_parents)(gen);
        }
        num_children = min(num_children, i);
        num_parents = min(num_parents, i);
        vector<float> probabilities(num_verts);
        for (int j = 0; j < i; j++) {
            probabilities[j] = in_degrees[j];
        }
        float total_probability = accumulate(probabilities.begin(), probabilities.end(), 0.0);
        for (int j = 0; j < num_children; j++) {
            int child_id = std::discrete_distribution<int>(probabilities.begin(), probabilities.end())(gen);
            children[i].insert(child_id);
            parents[child_id].insert(i);
            in_degrees[child_id] += 1;
        }
        auto descendants = get_decendants(i, g_ptr);
        for (int descendant : descendants) {
            probabilities[descendant] = 0;
        }
        total_probability = accumulate(probabilities.begin(), probabilities.end(), 0.0);
        if (total_probability != 0.0) {
            for (int j = 0; j < num_parents; j++) {
                int parent_id = std::discrete_distribution<int>(probabilities.begin(), probabilities.end())(gen);
                children[parent_id].insert(i);
                parents[i].insert(parent_id);
                out_degrees[parent_id] += 1;
            }
        }
    }
    */

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
    }*/


    return 0;
}




#endif //BALANCED_H
