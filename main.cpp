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

// https://stackoverflow.com/questions/26855184/floyd-warshall-all-pairs-shortest-paths-on-weighted-undirected-graph-boost-g

using namespace std;

typedef int t_weight;
typedef boost::property<boost::edge_weight_t, t_weight> EdgeWeightProperty;

typedef
  boost::adjacency_list<
    boost::vecS            // edge list
  , boost::vecS            // vertex list
  , boost::undirectedS     // directedness
  , boost::no_property     // property associated with vertices, float or int etc.
  , EdgeWeightProperty     // property associated with edges
  >
Graph;

typedef boost::sorted_erdos_renyi_iterator<boost::minstd_rand, Graph> ERGen;

typedef boost::property_map<Graph, boost::edge_weight_t>::type WeightMap;
typedef boost::exterior_vertex_property<Graph, t_weight> DistanceProperty;
typedef DistanceProperty::matrix_type DistanceMatrix;
typedef DistanceProperty::matrix_map_type DistanceMatrixMap;

int erdos_renyi_generator(const int num_nodes, const float p, boost::minstd_rand gen) {

    Graph g(ERGen(gen, num_nodes, p), ERGen(), num_nodes);
    cout << num_edges(g)<<endl;

    // make weights all 1
    WeightMap weight_map = boost::get(boost::edge_weight, g);
    boost::graph_traits<Graph>::edge_iterator ei, ei_end;
    for (boost::tie(ei, ei_end) = boost::edges(g); ei != ei_end; ++ei) {
        weight_map[*ei] = 1;
        //    cout << *ei << " " << weight_map[*ei] << endl;
    }


    vector<int> component (boost::num_vertices (g));
    size_t num_components = boost::connected_components (g, &component[0]);
    cout << "Number of components: " << num_components << endl;

    std::cout << "Vertices in the first component:" << std::endl;
    for (size_t i = 0; i < boost::num_vertices (g); ++i)
        if (component[i] == 0)
            std::cout << i << " ";
    std::cout << std::endl;

    WeightMap weight_pmap = boost::get(boost::edge_weight, g);
    DistanceMatrix distances(num_vertices(g));


    bool valid = floyd_warshall_all_pairs_shortest_paths(g, dm, boost::weight_map(weight_pmap));

    if (!valid) {
        std::cerr << "Error - Negative cycle in matrix" << std::endl;
        return -1;
    }

    std::cout << "Distance matrix: " << std::endl;
    for (std::size_t i = 0; i < num_vertices(g); ++i) {
        for (std::size_t j = i; j < num_vertices(g); ++j) {
            std::cout << "From vertex " << i+1 << " to " << j+1 << " : ";
            if(distances[i][j] > 100000)
                std::cout << "inf" << std::endl;
            else
                std::cout << distances[i][j] << std::endl;
        }
        std::cout << std::endl;
    }

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