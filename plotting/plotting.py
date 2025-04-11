import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

np.finfo(np.dtype("float32"))
np.finfo(np.dtype("float64"))


class GraphPlotter(object):
    def __init__(self, G=None, d=None):
        self.G = G
        if d is not None:
            pos = d['positions'] if 'positions' in d else None
            self.G = self.make_graph(d['edge_list'], pos=pos)

    def make_graph(self, edge_list, pos=None):
        """
        Make a graph from an edge list.
        :param edge_list: list of edges
        :param pos: node positions
        :return: graph
        """
        self.G = nx.Graph()
        self.G.add_edges_from(edge_list)
        if pos is not None:
            pos = pos.tolist() if isinstance(pos, np.ndarray) else pos
            pos_d = {}
            for i in range(len(pos)):
                pos_d[i] = (pos[i][0], pos[i][1])
            nx.set_node_attributes(self.G, pos_d, 'pos')
        return self.G

    def floyd_warshall_nx(self):
        d = nx.floyd_warshall(self.G)
        dist = np.zeros((len(self.G), len(self.G)))
        for i, j in d.items():
            for k, v in j.items():
                dist[i][k] = v
        return dist.astype(np.int32)

    def floyd_warshall_frydenlund(self, return_ground_truths: bool = False, is_undirected=True):
        """
        Did I name this after myself?  Yes.  Come at me bro.
        Or call it Edge-Streaming Floyd-Warshall.

        Returns: [num_nodes, num_nodes] distance matrix, [num_edges, num_nodes] ground truth matrix
        """
        vertices = list(self.G.nodes)
        n = len(vertices)
        dist = np.zeros((n, n)) + float('inf')
        connected = [{v} for v in vertices]  # keep track of connected components
        gts = None
        if return_ground_truths:
            gts = np.zeros((len(self.G.edges), n)) + float('inf')
        for i in range(n):  # initialize distance matrix
            dist[i, i] = 0
        for t, (i, j) in enumerate(list(self.G.edges)):  # pivot edge, stream of edges
            dist[i][j] = 1
            if is_undirected:
                dist[j][i] = 1

            for ki in connected[i]:
                for kj in connected[j]:
                    if dist[ki][kj] > dist[ki][i] + dist[i][j] + dist[j][kj]:
                        dist[ki][kj] = dist[ki][i] + dist[i][j] + dist[j][kj]
                        if is_undirected:
                            dist[kj][ki] = dist[ki][i] + dist[i][j] + dist[j][kj]
            new_connected = connected[i].union(connected[j])  # merge connected components
            for c in new_connected:  # point nodes to the new connected component set
                connected[c] = new_connected  # note this is for undirected and could be made faster for directed

            if return_ground_truths:
                gts[t, :] = dist[i, :]  # distances to node i after observing <=t edges

        dist = np.where(dist == np.inf, -1, dist).astype(np.int32)
        if return_ground_truths:
            return dist, gts.astype(np.int32)
        return dist

    def make_plot_positions_for_layout(self, spring_k=1.5, spring_scale=1.5, verbose=False, **kwargs):
        if nx.get_node_attributes(self.G, 'pos'):
            print('Using pre-defined node positions')
            return nx.get_node_attributes(self.G, 'pos')
        try:
            return nx.nx_agraph.graphviz_layout(self.G, prog="neato")
        except Exception as e:
            if verbose:
                print('Using spring layout due to error:', e)
            if spring_k is None:
                spring_k = 1 / np.sqrt(len(self.G.nodes))
            else:
                spring_k = 1 / np.sqrt(len(self.G.nodes)) * spring_k
            return nx.spring_layout(self.G, k=spring_k, scale=spring_scale)

    def plot_graph(self, node_size=12,  with_labels=True, **kwargs):
        pos = self.make_plot_positions_for_layout(**kwargs)
        nx.draw(self.G, with_labels=with_labels, pos=pos, node_size=node_size)
        plt.show()



def _try_networkx_plot():
    np.set_printoptions(threshold=np.inf, edgeitems=10, linewidth=np.inf, precision=2, suppress=True)
    G = nx.random_geometric_graph(100, 0.125)
    plotter = GraphPlotter(G)
    plotter.plot_graph()

def try_cpp_plot(graph_type='erdos_renyi'):
    # Assuming you have a C++ function that generates a graph and returns an edge list
    # For example, let's say you have a function `generate_graph` in your C++ module
    import generator  # Assuming this is your C++ module
    if graph_type == 'erdos_renyi':
        d = generator.erdos_renyi(15, -1.0, 75, 125, False, False, shuffle_edges=False)
    elif graph_type == 'euclidean':
        d = generator.euclidian(15, 2, -1, False, False, shuffle_edges=False)
    else:
        raise ValueError("Unsupported graph type. Use 'erdos_renyi' or 'euclidean'.")
    plotter = GraphPlotter(d=d)
    plotter.plot_graph()



if __name__ == "__main__":
    # _try_networkx_plot()

    try_cpp_plot()
