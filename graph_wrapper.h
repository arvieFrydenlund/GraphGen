//
// Created by arvie on 23/02/26.
//

#include <iostream>
#include <random>
#include <map>

#include "matrix.h"
#include "args.h"
#include "undirected_graphs.h"
#include "directed_graphs.h"

using namespace std;

#ifndef GRAPHGEN_GRAPH_WARPPER_H
#define GRAPHGEN_GRAPH_WARPPER_H


template<typename D>
class GraphWrapper {
public:
    int min_num_nodes;  // global, not unique to this.  This is me mixing in graph args with specific values
    int max_num_nodes;  // global
    int min_vocab;
    int max_vocab;
    bool shuffle_edges;
    bool shuffle_nodes;
    int c_min;
    int c_max;

    unique_ptr<Graph<D> > g_ptr;
    unique_ptr<vector<vector<float> > > positions_ptr;

    unique_ptr<vector<vector<int> > > distances_ptr;
    unique_ptr<vector<vector<int> > > graph_ground_truths_ptr; // -1 for unreachable, either [E * N] or [N * N]

    int start{};
    int end{};

    int N{};
    int E{};

    vector<int> node_shuffle_map;
    vector<int> edge_shuffle_map;

    vector<int> node_list; // ordered in the constructor, PLEASE NOTE this is not the original order nore the shuffled order but order of nodes seen in shuffled edges
    vector<pair<int, int> > edge_list;

    GraphWrapper(int min_num_nodes, int max_num_nodes,
                 int min_vocab, int max_vocab,
                 bool shuffle_edges, bool shuffle_nodes,
                 const int c_min = 75, const int c_max = 125) {

        this->min_num_nodes = min_num_nodes;
        this->max_num_nodes = max_num_nodes;
        this->min_vocab = min_vocab;
        this->max_vocab = max_vocab;
        this->shuffle_edges = shuffle_edges;
        this->shuffle_nodes = shuffle_nodes;

        this->c_min = c_min;
        this->c_max = c_max;

        if (c_min > 0 and c_min > c_max) { throw std::invalid_argument("Invalid arguments: c_min > c_max"); }
    }

    inline int attempt_check(const int max_edges, const int attempts, const int max_attempts) {
        if (E > max_edges) {
            if (attempts > max_attempts) {
                cout << "Failed to generate graph after " << attempts << " attempts" << endl;
            }
            return 1;
        }
        return 0;
    }

    int sample_num_nodes(std::mt19937 &gen) {
        if (max_num_nodes == min_num_nodes) {
            return max_num_nodes;
        }
        uniform_int_distribution<int> d(min_num_nodes, max_num_nodes);
        return d(gen);
    };

    void make_erdos_renyi(std::mt19937 &gen, float p=-1.0){
        auto num_nodes = sample_num_nodes(gen);
        // auto graph_t = time_before();
        erdos_renyi_generator(g_ptr, num_nodes, gen, p, c_min, c_max, false);
        // time_after(graph_t, "graph gen");
        finish(gen);
    }

    void make_euclidean(std::mt19937 &gen, int dim = 2, float radius = -1.0){
        auto num_nodes = sample_num_nodes(gen);
        // auto graph_t = time_before();
        euclidean_generator(g_ptr, positions_ptr, num_nodes, gen, dim, radius, c_min, c_max, false);
        // time_after(graph_t, "graph gen");
        finish(gen);
    }

    void make_random_tree(std::mt19937 &gen, const int max_degree, int sample_depth, const int max_depth, const float bernoulli_p,
                          optional<vector<float>> probs = nullopt, const bool start_at_root=true, const bool end_at_leaf=true) {
        auto num_nodes = sample_num_nodes(gen);
        // auto graph_t = time_before();
        auto start_end = random_tree_generator(g_ptr, num_nodes, gen, max_degree, sample_depth, max_depth, bernoulli_p, probs);
        start = start_end.first, end = start_end.second;
        if (!start_at_root) {
            start = -1;
        }
        if (!end_at_leaf) {
            end = -1;
        }
        // time_after(graph_t, "graph gen");
        finish(gen);
    }

    void make_path_star(std::mt19937 &gen, const int min_num_arms, const int max_num_arms,
                        const int min_arm_length, const int max_arm_length) {
        // auto graph_t = time_before();
        auto start_end = path_star_generator(g_ptr, min_num_arms, max_num_arms, min_arm_length, max_arm_length, gen);
        start = start_end.first, end = start_end.second;
        // time_after(graph_t, "graph gen");
        finish(gen);
    }

    void make_balanced(std::mt19937 &gen, const int lookahead, const int min_noise_reserve, const int max_num_parents,
                       const int max_noise_sample) {
        auto num_nodes = sample_num_nodes(gen);
        // auto graph_t = time_before();
        auto start_end = balanced_generator(g_ptr, num_nodes, gen, lookahead, min_noise_reserve, max_num_parents,
                                            max_noise_sample);
        start = start_end.first, end = start_end.second;
        // time_after(graph_t, "graph gen");
        finish(gen);
    }


    void finish(std::mt19937 &gen){
        N = num_vertices(*g_ptr);
        E = num_edges(*g_ptr);
        make_node_shuffle_map(gen);
        make_edge_shuffle_map(gen);
        make_edge_list();
        make_node_list();
    }

    void make_node_shuffle_map(std::mt19937 &gen) {
        // Shuffle nodes and map to the new range [min_vocab, max_vocab)
        if (max_vocab > 0) {
            // asserts do not work on python side, use throws
            if (max_vocab - min_vocab < N) {
                throw std::invalid_argument("max_vocab - min_vocab < N with " + std::to_string(max_vocab
                )
                                            + " - " + std::to_string(min_vocab) + " < " + std::to_string(N));
            }
            if (min_vocab < 0) { throw std::invalid_argument("min_vocab < 0 with " + std::to_string(min_vocab)); }
            if (max_vocab - min_vocab <= 0) {
                throw std::invalid_argument("max_vocab - min_vocab <= 0 with " + std::to_string(max_vocab - min_vocab));
            }
            if (max_vocab - min_vocab < N) { throw std::invalid_argument("max_vocab - min_vocab < N"); }
        } else {
            if (min_vocab != 0) { throw std::invalid_argument("min_vocab != 0 with " + std::to_string(min_vocab)); }
            max_vocab = N;
        }
        auto m = std::vector<int>(max_vocab - min_vocab);
        std::iota(m.begin(), m.end(), min_vocab);
        if (shuffle_nodes) {
            std::shuffle(m.begin(), m.end(), gen);
        }
        node_shuffle_map = std::vector<int>(m.begin(), m.begin() + N);
    }

    void make_edge_shuffle_map(std::mt19937 &gen) {
        // shuffle the edges around, this will be the shuffled order given to the model
        edge_shuffle_map = std::vector<int>(E);
        std::iota(edge_shuffle_map.begin(), edge_shuffle_map.end(), 0);
        if (shuffle_edges) {
            std::shuffle(edge_shuffle_map.begin(), edge_shuffle_map.end(), gen);
        }
    }

    void make_edge_list() {
        // Get the edge list of the graph in the shuffled order
        edge_list = vector<pair<int, int> >(num_edges(*g_ptr), make_pair(-1, -1));
        typename boost::graph_traits<Graph<D> >::edge_iterator ei, ei_end;
        int cur = 0;
        for (boost::tie(ei, ei_end) = boost::edges(*g_ptr); ei != ei_end; ++ei) {
            edge_list[edge_shuffle_map[cur]] = make_pair(source(*ei, *g_ptr), target(*ei, *g_ptr));
            cur += 1;
        }
    }

    void make_node_list(){
        // build node list by edge list order, this is not always needed
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


    /*
     * Note THESE ARE NOT IN NODE SHUFFLE MAP ORDER!!!
     * this is because we will convert everything to gather_ids for the model
     * and that will handle the map and shift to vocab ids, except for the final distance matrix
     */
    void get_distances() {
        /*
         * Just the distance matrix [N X N] in the original graph node order
         */
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

    void get_edge_ground_truths(const bool is_causal) {
        /*
         * Just the distance matrix but in edge_list order  [E X N], this is possibly causally constrained
         * Note these only work for the projected ranking loss so N is in vocab order
         */
        if (is_causal) {
            // this will calculate the distances as well
            if (distances_ptr) {
                throw runtime_error("Causal ground truths should not be calculated with precomputed distances");
            }
            floyd_warshall_frydenlund(g_ptr, distances_ptr, graph_ground_truths_ptr, edge_list, false);
        } else {
            if (!distances_ptr) {
                get_distances(); // already node_shuffle_map, so edges will be too
            }
            // Makes a [E, N] matrix of ground truths where each row is the distance from the edge.first to all other nodes
            graph_ground_truths_ptr = make_unique<vector<vector<int> > >(E, vector<int>(N, -1));
            for (int t = 0; t < static_cast<int>(E); t++) {
                for (int i = 0; i < static_cast<int>(N); i++) {  // permute first dim by edge order, other stays in vocab order
                    (*graph_ground_truths_ptr)[t][i] = (*distances_ptr)[edge_list[t].first][i];
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
        graph_ground_truths_ptr = make_unique<vector<vector<int> > >(N, vector<int>(N, -1));
        for (int i = 0; i < static_cast<int>(N); i++) {
            for (int j = 0; j < static_cast<int>(N); j++) {
                if (is_direct_ranking) {  // permute both dims by tokenization order
                    (*graph_ground_truths_ptr)[i][j] = (*distances_ptr)[node_list[i]][node_list[j]];
                } else {  // only permute first dim by tokenization order, other stays in vocab order
                    (*graph_ground_truths_ptr)[i][j] = (*distances_ptr)[node_list[i]][j];
                }
            }
        }
    }


};

#endif //GRAPHGEN_GRAPH_WARPPER_H
