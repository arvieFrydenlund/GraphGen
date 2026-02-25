//
// Created by arvie on 3/24/25.
//

#ifndef GRAPH_H
#define GRAPH_H

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/graph/erdos_renyi_generator.hpp>
#include <boost/random/linear_congruential.hpp>
#include <boost/graph/undirected_graph.hpp>
#include <boost/graph/exterior_property.hpp>
#include <boost/graph/floyd_warshall_shortest.hpp>
#include <boost/graph/johnson_all_pairs_shortest.hpp>
#include <boost/graph/random.hpp>
#include <random>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <queue>
#include <cmath>

/*
 * Undirected graphs and utils, read before directed_graphs.h and py_bindings.h
 */

using namespace std;

typedef pair<int, int> start_end_pair;

typedef int t_weight;
typedef boost::property<boost::edge_weight_t, t_weight> EdgeWeightProperty;

const int inf = std::numeric_limits<int>::max() / 8;  // careful because we add max + 1 + max -> overflow

template <typename D>  // undirectedS  or directedS
using Graph = boost::adjacency_list<
                                    boost::setS            // edge set!  otherwise this allows for edge duplicates
                                  , boost::vecS            // vertex list
                                  , D    // directedness
                                  , boost::no_property     // property associated with vertices, float or int etc.
                                  , EdgeWeightProperty     // property associated with edges
                                  >;

using ERGen = boost::sorted_erdos_renyi_iterator<std::mt19937, Graph<boost::undirectedS>>;
using ERGen2 = boost::sorted_erdos_renyi_iterator<boost::minstd_rand, Graph<boost::undirectedS>>;

template <typename D>
using Cluster = boost::adjacency_list<boost::listS, boost::listS, D>;

template <typename D>
using  WeightMap = typename boost::property_map<Graph<D>, boost::edge_weight_t>::type;

template <typename D>
using DistanceProperty = boost::exterior_vertex_property<Graph<D>, t_weight>;

template <typename D>
using DistanceMatrix = typename DistanceProperty<D>::matrix_type;
template <typename D>
using DistanceMatrixMap = typename DistanceProperty<D>::matrix_map_type;


template<typename T>
void print_matrix(T &matrix_ptr, const int N, const int M,
    bool full, const int cutoff = 100000, const string max_value = "inf") {
    // cant figure out the damn shape of the distance matrix, so just pass in the size, sigh
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = (full) ? 0 : i; j < M; ++j) {  // triangular matrix only if not full
            if(cutoff > 0 && (*matrix_ptr)[i][j] >= cutoff)
                std::cout << max_value << " ";
            else
                std::cout << (*matrix_ptr)[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

template<typename T>
vector<T> list_to_vector(list<T> &l) {  // convert list to vector
    vector<T> v;
    v.reserve(l.size());
    std::copy(std::begin(l), std::end(l), std::back_inserter(v));
    return v;
}


inline map<int, list<int>> get_connected_components_map(Graph<boost::undirectedS> &g, bool verbose = false) {
    std::vector<int> component (num_vertices (g));
    size_t num_components = connected_components(g, &component[0]);
    if (verbose) {
        cout << "Number of connected components: " << num_components << " num edges " << num_edges(g) << endl;
    }
    // make map of connected components
    map<int, list<int>> component_map;
    for (size_t i = 0; i < num_vertices(g); i++) {
        component_map[component[i]].push_back(i);
    }
    return component_map;
}


inline int sample_num_connected(std::mt19937 &gen, const int num_nodes, const int c_min = 75, const int c_max = 125) {
    if ( num_nodes < c_min ) {
        return 1;
    }
    if ( num_nodes > c_max ) {
        return 2;
    }
    std::uniform_int_distribution dist(1, 2);
    return dist(gen);
}


template <typename D>
 WeightMap<D> make_edge_weights(Graph<D> &g, bool verbose = false) {

    // make all edge weights 1
    WeightMap<D> weight_map = boost::get(boost::edge_weight, g);
    typename boost::graph_traits<Graph<D>>::edge_iterator ei, ei_end;
    for (boost::tie(ei, ei_end) = boost::edges(g); ei != ei_end; ++ei) {
        weight_map[*ei] = 1;
        if (verbose) {
            cout << *ei << " " << weight_map[*ei] << endl;
        }
    }
    return weight_map;
}


template <typename D>
int floyd_warshall(unique_ptr<Graph<D>> &g_ptr, unique_ptr<DistanceMatrix<D>> &distances_ptr, bool verbose = false) {
    // https://stackoverflow.com/questions/26855184/floyd-warshall-all-pairs-shortest-paths-on-weighted-undirected-graph-boost-g
    distances_ptr = make_unique<DistanceMatrix<D>>(num_vertices(*g_ptr));
    auto weight_pmap = make_edge_weights(*g_ptr, false);
    DistanceMatrixMap<D> dm(*distances_ptr, *g_ptr);
    bool valid = floyd_warshall_all_pairs_shortest_paths(*g_ptr, dm, boost::weight_map(weight_pmap));

    if (!valid && verbose) {
        std::cerr << "Error - Negative cycle in matrix" << std::endl;
        return -1;
    }
    return valid;
}

template <typename D>
int johnson(unique_ptr<Graph<D>> &g_ptr, unique_ptr<DistanceMatrix<D>> &distances_ptr, bool verbose = false) {
    //  much faster than floyd_warshall for sparse graphs
	// https://stackoverflow.com/questions/47757973/obtain-predecessors-with-boost-bgl-for-an-all-pair-shortest-path-search
    distances_ptr = make_unique<DistanceMatrix<D>>(num_vertices(*g_ptr));
    // auto weight_pmap = make_edge_weights(*g_ptr, false);
    make_edge_weights(*g_ptr, false);
    DistanceMatrixMap<D> dm(*distances_ptr, *g_ptr);
    johnson_all_pairs_shortest_paths(*g_ptr, *distances_ptr);
	return 0;
}

template <typename D>
inline int floyd_warshall_frydenlund(unique_ptr<Graph<D>> &g_ptr,
                                     unique_ptr<vector<vector<int>>> &distances_ptr,
                                     unique_ptr<vector<vector<int>>> &ground_truths_ptr,
                                     vector<pair<int, int>> &edge_list, bool verbose) {
    // assume edge weights of 1, no need to make them in the graph (unlike boost)
  	auto N = num_vertices(*g_ptr);
    auto E = num_edges(*g_ptr);
    distances_ptr = make_unique<vector<vector<int>>>(N, vector<int>(N, inf));
    ground_truths_ptr = make_unique<vector<vector<int>>>(E, vector<int>(N, -1));

    auto connected_components = vector(N, shared_ptr<set<int>>());
    for (int i = 0; i < static_cast<int>(N); i++) {  // initialize distance matrix and connected components
        (*distances_ptr)[i][i] = 0;
        connected_components[i] = make_shared<set<int>>(initializer_list<int>{i});
    }
    for (int t = 0; t < static_cast<int>(E); t++) {  // stream edges as pivot instead of nodes
    	auto node_i = edge_list[t].first;
        auto node_j = edge_list[t].second;
        if (node_i == node_j) {
            continue;  // skip self loops
        }
        (*distances_ptr)[node_i][node_j] = 1;  // add new edge
        if (std::is_same<D, boost::undirectedS>::value) {
            (*distances_ptr)[node_j][node_i] = 1;  // add new edge for undirected graph
        }
        // update distances
        for (int ki : *(connected_components[node_i])) {  //
        // for (int ki = 0; ki < N; ki++) {  // slower
            for (int kj : *(connected_components[node_j])) {  //for (int kj = 0; kj < N; kj++) {  // slower
            // for (int kj = 0; kj < N; kj++) {  // slower
            	auto d = (*distances_ptr)[ki][node_i] + 1 + (*distances_ptr)[node_j][kj];
            	if ( (*distances_ptr)[ki][kj] > d ){
                	(*distances_ptr)[ki][kj] = d;
                    if (std::is_same<D, boost::undirectedS>::value) {
                		(*distances_ptr)[kj][ki] = d;  // update distance for undirected graph
                    }
            	}
            }
        }
        // merge connected components by making all node_j's connected components node_i's connected components
        // note this is for undirected and could be made faster for directed
        (connected_components[node_i])->insert((connected_components[node_j])->begin(), (connected_components[node_j])->end());
        for (auto it = (connected_components[node_j])->begin(), end = (connected_components[node_j])->end(); it != end; ++it) {
            if (*it != node_i && *it != node_j) {
                connected_components[*it] = connected_components[node_i];  // add new connected component as same set
            }
        }
        connected_components[node_j] = connected_components[node_i];  // make node_j's the same as node_i's
        // copy current distances to ground truths for node i in edge t after observing <= t edges
        for (int i = 0; i < static_cast<int>(N); i++) {
            if ((*distances_ptr)[node_i][i] >= inf) {
                (*ground_truths_ptr)[t][i] = -1;
            } else {
                (*ground_truths_ptr)[t][i] = (*distances_ptr)[node_i][i];  // edge_i, makes more sense for directed
            }
        }
    }
    return 1;
}


inline int erdos_renyi_generator(unique_ptr<Graph<boost::undirectedS>> &g_ptr,  const int num_nodes, std::mt19937 &gen,
    float p = -1.0, const int c_min = 75, const int c_max = 125, const bool verbose = false) {

    if ( p < 0 ) {
        p = 1.0 / static_cast<float>(num_nodes);
    }

    g_ptr = make_unique<Graph<boost::undirectedS>>(ERGen(gen, num_nodes, p), ERGen(), num_nodes);


    if (verbose) {
        cout << "Graph has " << num_vertices(*g_ptr) << " vertices and " << num_edges(*g_ptr) << " edges " << p << " p" << endl;
    }

    // connect components
    auto component_map = get_connected_components_map(*g_ptr , verbose);
    while ( component_map.size() > 1 ) {
        for (size_t i = 0; i < component_map.size(); i++) {
            for (int _ = 0; _ < sample_num_connected(gen, num_nodes, c_min, c_max); _++) {
                uniform_int_distribution<int> dc(i, component_map.size() - 1);
                int j = dc(gen);
                // sample a node from each component
                vector<int> c1 = list_to_vector(component_map[i]);
                uniform_int_distribution<int> d1(0, c1.size() - 1);
                int v1 = c1[d1(gen)];
                vector<int> c2 = list_to_vector(component_map[j]);
                uniform_int_distribution<int> d2(0, c2.size() - 1);
                int v2 = c2[d2(gen)];
                if (v1 == v2) {
                    continue;  // skip self loops
                }
                boost::add_edge(v1, v2, *g_ptr);
            }
        }
        //cout << "Number of connected components: " << component_map.size() << endl;
        component_map = get_connected_components_map(*g_ptr, verbose);  // remake connected components
    }
    // component_map = get_connected_components_map(*g_ptr, verbose);
    //cout << "Number of connected components: " << component_map.size() << endl;
    return 0;
}


inline int euclidean_generator(unique_ptr<Graph<boost::undirectedS>> &g_ptr,
    unique_ptr<vector<vector<float>>> &positions_ptr, const int num_nodes, std::mt19937 &gen,
    const int dim = 2, float radius = -1.0, const int c_min = 75, const int c_max = 125, const bool verbose = true) {
    /*  These are the only graphs which have a true property which is the position a vector of length dim
     *  Because of this, we just keep this as a num_nodes x dim matrix
     */

    if ( radius <= 0 ) {
        radius = 1.0 / sqrt(static_cast<float>(num_nodes));
    }
    g_ptr = make_unique<Graph<boost::undirectedS>>(num_nodes);

    // uniformly generates num_nodes points in dim
    std::uniform_real_distribution<float> distr(0, 1);
    positions_ptr = make_unique<vector<vector<float>>>(num_nodes, vector<float>(dim));
    for (int i = 0; i < num_nodes; i++) {
        for (int j = 0; j < dim; j++) {
            (*positions_ptr)[i][j] = distr(gen);
        }
    }

    // add edges if the distance between two nodes is less than radius
    for (int i = 0; i < num_nodes; i++) {
        for (int j = i+1; j < num_nodes; j++) {
            float dist = 0;
            for (int k = 0; k < dim; k++) {
                dist += pow((*positions_ptr)[i][k] - (*positions_ptr)[j][k], 2);
            }
            dist = sqrt(dist);
            if ( dist < radius ) {
                boost::add_edge(i, j, *g_ptr);
            }
        }
    }

    if ( verbose ) {
        for (int i = 0; i < num_nodes; i++) {
            cout << "Node " << i << " : ";
            for (int j = 0; j < dim; j++) {
                cout << (*positions_ptr)[i][j] << " ";
            }
            // edges
            cout << "Edges: ";
            boost::graph_traits<Graph<boost::undirectedS>>::adjacency_iterator ai, ai_end;
            for (boost::tie(ai, ai_end) = boost::adjacent_vertices(i, *g_ptr); ai != ai_end; ++ai) {
                cout << *ai << " ";
            }
            cout << endl;
        }
    }

    // connect components
    if ( true ) {
    auto component_map = get_connected_components_map(*g_ptr, verbose);
    while ( component_map.size() > 1 ) {
        for (size_t i = 0; i < component_map.size(); i++) {
            vector<tuple<int, int, float>> closest;
            for (size_t j = 0; j < component_map.size(); j++) {
                if ( i == j ) {
                    continue;
                }
                tuple<int, int, float> closest_in_component = make_tuple(-1, -1, 10000.0);
                for (int u : component_map[i]) {
                    for (int v : component_map[j]) {
                        float distance = 0;
                        for (int k = 0; k < dim; k++) {
                            distance += pow((*positions_ptr)[u][k] - (*positions_ptr)[v][k], 2);
                        }
                        distance = sqrt(distance);
                        if ( distance < get<2>(closest_in_component) ) {
                            closest_in_component = make_tuple(u, v, distance);
                        }
                    }
                }
                if ( get<0>(closest_in_component) == -1 ) {
                    continue;
                }
                closest.push_back(closest_in_component);
            }
            sort(closest.begin(), closest.end(), [](auto &left, auto &right) {
                return get<2>(left) < get<2>(right);
            });
            auto num_connected = sample_num_connected(gen, num_nodes, c_min, c_max);
            for (size_t k = 0; k < static_cast<size_t>(num_connected) && k < closest.size(); k++) {
                // cout << "Adding edge " << k << " " << num_connected << " " << closest.size() << endl;
                int u = get<0>(closest[k]);
                int v = get<1>(closest[k]);
                boost::add_edge(u, v, *g_ptr);
            }
        }
        component_map = get_connected_components_map(*g_ptr , verbose);  // remake connected components
    }
    }
    return 0;
}

inline bool node_check(const int cur_node, const int num_nodes) {
    if (num_nodes > 0 && cur_node >= num_nodes) {
        return false;
    }
    return true;
}

inline bool depth_check(const int cur_depth, const int max_depth) {
    if (max_depth > 0 && cur_depth >= max_depth) {
        return false;
    }
    return true;
}


inline start_end_pair random_tree_generator(unique_ptr<Graph<boost::undirectedS>> &g_ptr,
                                const int num_nodes, std::mt19937 &gen,
                                const int d,
                                int sample_depth,
                                const int max_depth,  // if -1 then go until num_nodes otherwise stop at max_depth whichever comes first
                                const float bernoulli_p=0., optional<vector<float>> probs=nullopt,
                                const bool verbose = false) {
    // Either full d-ary trees (bernoulli_p=0.) or random binomial trees with at most d children per node depending on bernoulli_p
    // binomial random tree, modeled as a Galton-Watson branching process with a binomial offspring distribution
    // E[children per node] = d * bernoulli_p
    // E[total children at gen k] = (d * bernoulli_p)^k
    // E[total nodes] = sum i = 0 to k (d * bernoulli_p)^i = (1 - (d * bernoulli_p)^(k + 1)) / (1 - d * bernoulli_p) if d * bernoulli_p != 1 else k + 1

    // do not do this check
    //if ( num_nodes < pow(d, max_depth)) {
    //    throw invalid_argument("num_nodes must be at least d^max_depth");
    //}

    g_ptr = make_unique<Graph<boost::undirectedS>>();

    if (sample_depth <= 0) {
        assert (max_depth > 0);
        sample_depth = max_depth;
    }
    int start = 0;
    int end = -1;  // return a leaf node
    int cur_depth = 0;
    int cur_node = start;
    // make expansion queue of leaf nodes

    queue<int> expansion_q{};
    queue<int> expansion_q_next{};
    expansion_q.push(cur_node);
    while ( depth_check(cur_depth, max_depth) && !expansion_q.empty() && node_check(cur_node, num_nodes) ) {
        while ( !expansion_q.empty() && node_check(cur_node, num_nodes) ) {
            auto to_expand = expansion_q.front();
            expansion_q.pop();

            int num_children;
            if (probs.has_value() && !probs.value().empty()) {  // old version
                std::discrete_distribution<int> dist(probs->begin(), probs->end());
                num_children = dist(gen) + 1;  // at least one child
            } else {
                num_children = d;
                if (bernoulli_p > 0.) {
                    std::binomial_distribution<> dist(d, bernoulli_p);
                    num_children = dist(gen);
                }
            }
            if ( num_children == 0 && cur_depth < 3 ) {  // safety to avoid empty graphs
                num_children = 1;  // ensure at least one child if we have not reached the max number of nodes
            }

            for (int i = 0; i < num_children && node_check(cur_node + 1, num_nodes); i++) {
                cur_node++;
                boost::add_edge(to_expand, cur_node, *g_ptr);
                expansion_q_next.push(cur_node);
            }
        }
        cur_depth++;

        if (cur_depth <= sample_depth and !expansion_q_next.empty()) {
            // choose either front or end randomly
            // important because end-side may be biased by weight due to num nodes limit
            uniform_int_distribution<int> dist(0, 1);
            if (dist(gen) == 0) {
                end = expansion_q_next.front();
            } else {
                end = expansion_q_next.back();
            }
        }
        swap(expansion_q, expansion_q_next);
    }


    return make_pair(start, end);
}

#endif //GRAPH_H
