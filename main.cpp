#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/graph/erdos_renyi_generator.hpp>
#include <boost/random/linear_congruential.hpp>
#include <boost/graph/undirected_graph.hpp>
#include <boost/graph/exterior_property.hpp>
#include <boost/graph/floyd_warshall_shortest.hpp>
#include <boost/graph/random.hpp>
#include <random>
#include <iostream>
#include <string>
#include <vector>

#include "graph.h"


using namespace std;

int erdos_renyi_generator(const int num_nodes, const float p, boost::minstd_rand gen) {

    UnDirGraph g(ERGen(gen, num_nodes, p), ERGen(), num_nodes);
    cout << num_edges(g)<<endl;

    // make weights all 1
    WeightMap weight_map = boost::get(boost::edge_weight, g);
    boost::graph_traits<UnDirGraph>::edge_iterator ei, ei_end;
    for (boost::tie(ei, ei_end) = boost::edges(g); ei != ei_end; ++ei) {
        weight_map[*ei] = 1;
        //    cout << *ei << " " << weight_map[*ei] << endl;
    }

    DistanceMatrix distances(num_vertices(g));
    floyd_warshall(g, distances, true);

    return 0;
}


int main(){
    unsigned int seed = std::random_device{}();
    cout << "Seed: " << seed << endl;
    boost::minstd_rand gen(seed);

    int num_nodes = 100;
    float p = 1 / static_cast<float>(num_nodes) * 2;


    erdos_renyi_generator(num_nodes, p, gen);


    return 0;
};

// TIP See CLion help at <a
// href="https://www.jetbrains.com/help/clion/">jetbrains.com/help/clion/</a>.
//  Also, you can try interactive lessons for CLion by selecting
//  'Help | Learn IDE Features' from the main menu.