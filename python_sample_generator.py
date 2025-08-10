import networkx as nx
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict
from random import sample, randrange, choice, shuffle


class PyGraph(object):
    gtype = None

    def __init__(self, num_nodes: int | None, path_length: int | None,
                 c_min: int = 75,  c_max: int = 125,
                 rng: np.random.Generator = np.random.default_rng(), needs_visual=False):
        self.num_nodes = num_nodes
        self.path_length = path_length

        self.c_min = c_min
        self.c_max = c_max

        self.rng = rng
        self.needs_visual = needs_visual

        self.G = None
        self.start = None
        self.end = None
        self.arms = None
        self.colour_map = None
        self.pos = None

    def __str__(self):
        return (f'Graph with {len(self.G)} nodes and {len(self.G.edges)} edges.')

    def __len__(self):
        return len(self.G)

    def __hash__(self):
        return hash(self.G)

    def c_rule(self) -> int:
        # random interpolation
        if self.num_nodes < self.c_min:
            return 1
        elif self.num_nodes > self.c_max:
            return 2
        else:
            return self.rng.integers(1, 3)

    def random_directed(self):
        # idea sample paths and turn directed until all edges are used
        raise NotImplementedError

    def connect_c_components(self, verbose: bool = False) -> None:
        """
        Makes graph only have a single connected component by connecting c components for each component
        Args:
            c: number of components to connect
            verbose:
        Returns: None
        """
        raise NotImplementedError

    def get_histogram_path_lengths(self) -> tuple[Dict[int, int], List[Tuple[int, Dict[int, int]]]]:
        """
        Get the histogram of path lengths in the graph
        Returns: dictionary of path lengths  to their counts, list of path lengths for each node
        """
        path_lengths = nx.all_pairs_shortest_path_length(self.G)
        lengths = defaultdict(int)
        for node, paths in path_lengths:
            for target, length in paths.items():
                lengths[length] += 1
        return lengths, path_lengths

    def floyd_warshall_nx(self):
        """
        Returns:
        """
        d = nx.floyd_warshall(self.G)
        # convert to numpy array
        dist = np.zeros((len(self.G), len(self.G)))
        for i, j in d.items():
            for k, v in j.items():
                dist[i][k] = v
        return dist.astype(np.int32)

    def floyd_warshall_undirected(self):
        """
        https://en.wikipedia.org/wiki/Floyd%E2%80%93Warshall_algorithm
        Floyd-Warshall which streams in pivot nodes
        Returns:
        """
        vertices = list(self.G.nodes)
        n = len(vertices)
        dist = np.zeros((n, n)) + float('inf')
        for i in range(n):
            dist[i, i] = 0
        for edge in self.G.edges:
            dist[edge[0]][edge[1]], dist[edge[1]][edge[0]] = 1, 1  # undirected
        for i in range(n):  # i pivot node, steam of pivots
            for j in range(n):
                for k in range(n):
                    if dist[j][k] > dist[j][i] + dist[i][k]:
                        dist[j][k] = dist[j][i] + dist[i][k]
        return dist.astype(np.int32)

    def floyd_warshall_frydenlund_undirected(self, return_ground_truths: bool = False):
        """
        Simple test version of the algorithm, dont use.
        Returns: [num_nodes, num_nodes] distance matrix, [num_edges, num_nodes] ground truth matrix
        """
        vertices = list(self.G.nodes)
        n = len(vertices)
        dist = np.zeros((n, n)) + float('inf')
        gts = None
        if return_ground_truths:
            gts = np.zeros((len(self.G.edges), n)) + float('inf')
        for i in range(n):  # initialize distance matrix
            dist[i, i] = 0
        for t, (i, j) in enumerate(list(self.G.edges)):  # pivot edge, stream of edges
            dist[i][j], dist[j][i] = 1, 1  # add new edge
            for ki in range(n):
                for kj in range(n):
                    if dist[ki][kj] > dist[ki][i] + dist[i][j] + dist[j][kj]:
                        dist[ki][kj] = dist[ki][i] + dist[i][j] + dist[j][kj]
                        dist[kj][ki] = dist[ki][i] + dist[i][j] + dist[j][kj]
            if return_ground_truths:
                gts[t, :] = dist[i, :]  # distances to node i after observing <t edges

        dist = np.where(dist == np.inf, -1, dist).astype(np.int32)
        if return_ground_truths:
            return dist, gts.astype(np.int32)
        return dist

    def floyd_warshall_frydenlund(self, return_ground_truths: bool = False, is_undirected=True):
        """
        Did I name this after myself?  Yes.  Come at me bro.
        Or call it Edge-Streaming Floyd-Warshall.

        Because of the way streaming edges works, given a sparse graph and a random edge list, we will expect a lot of
        unconnected components. This allows for a more optimal update of the distance matrix by only consider 1 or 2
        connected components per new edge.

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

    def get_center_and_centroid(self, query=None, dist=None, max_query_length=-1, min_query_length=0):
        # TODO RASP OF THESE
        #  ASSUME N Embeddings which contain the distances

        # https://en.wikipedia.org/wiki/1-center_problem
        # https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.distance_measures.center.html

        if dist is None:
            dist = self.floyd_warshall_nx()
        N = len(self)
        if query is None:
            if max_query_length == -1:
                max_query_length = N
            num_queries = self.rng.integers(min_query_length, max_query_length)
            query = self.rng.choice(range(N), num_queries, replace=False)
        # get the center and centroid
        centers = []
        centroids = []
        center_values = np.zeros(N)
        centroid_values = np.zeros(N)
        for v in range(N):
            d = dist[v, query]  # [N, num_queries]
            center_values[v] = np.max(d)
            centroid_values[v] = np.sum(d)
        min_center = np.min(center_values)
        min_centroid = np.min(centroid_values)
        for v in range(N):
            if center_values[v] == min_center:
                centers.append(v)
            if centroid_values[v] == min_centroid:
                centroids.append(v)
        return centers, centroids, query


    def make_plot_positions_for_layout(self, spring_k=1.5, spring_scale=1.5, verbose=False, **kwargs):
        try:
            pos = nx.nx_agraph.graphviz_layout(self.G, prog="neato")
        except Exception as e:
            if verbose:
                print('Using spring layout due to error:', e)
            if spring_k is None:
                spring_k = 1 / np.sqrt(len(self.G.nodes))
            else:
                spring_k = 1 / np.sqrt(len(self.G.nodes)) * spring_k
            pos = nx.spring_layout(self.G, k=spring_k, scale=spring_scale)
        self.pos = pos

    def make_plot_colours(self, default=False, **kwargs):
        if default:
            self.colour_map = '#1f78b4'
        else:
            import matplotlib.pyplot as plt
            colour_map = []
            connected_components = list(nx.connected_components(self.G))
            # print(f'num components: {len(connected_components)}, num edge {len(self.G.edges)}')
            colours = plt.cm.rainbow(np.linspace(0, 1, len(connected_components)))
            self.rng.shuffle(colours)
            self.rng.shuffle(connected_components)
            for node in self.G.nodes:
                for i, component in enumerate(connected_components):
                    if node in component:
                        colour_map.append(colours[i])
                        break
            self.colour_map = colour_map

    def make_from_ground_truths(self):
        # for plotting attention on edges  # visualize search frontier
        # number edges by position in edge list [connected to start/end]
        raise NotImplementedError


class EuclidianGraph(PyGraph):
    """
    An undirected graph generated in a Euclidian space without self-loops
    https://networkx.org/documentation/stable/reference/generated/networkx.generators.geometric.random_geometric_graph.html
    """
    gtype = 'Euclidian'

    def __init__(self, num_nodes: int, path_length: int | None,
                 c_min: int = 75, c_max: int = 125,
                 rng: np.random.Generator = np.random.default_rng(), needs_visual=False,
                 dim: int = 2, radius: float = -1, spring_k=1.5, spring_scale=1.5):
        super().__init__(num_nodes, path_length, c_min, c_max, rng, needs_visual)
        self.dim = dim
        self.radius = radius if radius != -1 else 1 / np.sqrt(self.num_nodes)
        G = nx.random_geometric_graph(self.num_nodes, self.radius, dim=self.dim, pos=None, seed=self.rng)
        self.G = G
        if needs_visual:
            self.make_plot_colours()
            self.make_plot_positions_for_layout(spring_k=spring_k, spring_scale=spring_scale)
        self.connect_c_components()

    def __str__(self):
        return super().__str__() + f'\nEuclidian Graph with dim={self.dim}, radius={self.radius}'

    def connect_c_components(self, verbose: bool = False) -> None:
        """
        Makes graph only have a single connected component by connecting the c closest components for each component
        Returns: None
        """
        components = list(nx.connected_components(self.G))
        num_round = 0
        while len(components) > 1:
            for i in range(len(components)):
                closest = []
                for j in range(len(components)):
                    if i == j:
                        continue
                    closest_in_component = (None, None, float('inf'))
                    for u in components[i]:
                        for v in components[j]:
                            distance = np.linalg.norm(
                                np.array(self.G.nodes[u]['pos']) - np.array(self.G.nodes[v]['pos']))
                            if distance < closest_in_component[2]:
                                closest_in_component = (u, v, distance)
                    closest.append(closest_in_component)
                closest.sort(key=lambda x: x[2])
                for u, v, _ in closest[:self.c_rule()]:
                    self.G.add_edge(u, v)
            num_round += 1
            components = list(nx.connected_components(self.G))

    def make_plot_positions_for_layout(self, **kwargs):
        self.pos = nx.get_node_attributes(self.G, 'pos')


class ErdosRenyiGraph(PyGraph):
    """
    Erdős-Rényi graph
    https://networkx.org/documentation/stable/reference/generated/networkx.generators.random_graphs.fast_gnp_random_graph.html#networkx.generators.random_graphs.fast_gnp_random_graph
    """
    gtype = 'Erdos-Renyi'

    def __init__(self, num_nodes: int, path_length: int | None,
                 c_min: int = 75, c_max: int = 125,
                 rng: np.random.Generator = np.random.default_rng(), needs_visual=False,
                 p: float = -1):
        super().__init__(num_nodes, path_length, c_min, c_max, rng, needs_visual)
        self.p = p if p != -1 else 1./num_nodes

        self.G = nx.fast_gnp_random_graph(self.num_nodes, self.p, directed=False, seed=self.rng)
        if needs_visual:
            self.make_plot_colours()
        self.connect_c_components()
        if needs_visual:
            self.make_plot_positions_for_layout()

    def connect_c_components(self, verbose: bool = False) -> None:
        """
        Makes graph only have a single connected component by connecting the c closest components for each component

        Note close doesn't make sense for Erdos-Renyi graphs so we randomly connect c components

        Args:
            c: number of components to connect
            verbose:
        Returns: None
        """
        components = list(nx.connected_components(self.G))
        # print(f'num components: {len(components)}, num edge {len(self.G.edges)}')
        num_round = 0
        while len(components) > 1:
            for i in range(len(components)-1):
                for _ in range(self.c_rule()):
                    j = self.rng.choice(range(i + 1, len(components)))
                    u = self.rng.choice(list(components[i]))
                    v = self.rng.choice(list(components[j]))
                    self.G.add_edge(u, v)
            num_round += 1
            # print(f'Number of components: {len(components)}, number of edges: {len(self.G.edges)}')
            components = list(nx.connected_components(self.G))


class PathStarGraph(PyGraph):
    gtype = 'PathStar'

    def __init__(self, num_nodes: int | None, path_length: int,
                 c_min: int = 75,  c_max: int = 125,
                 rng: np.random.Generator = np.random.default_rng(), needs_visual=False,
                 num_arms: int = 2, **kwargs):
        super().__init__(num_nodes, path_length, c_min, c_max, rng, needs_visual)
        self.num_arms = num_arms
        self.arms = []

        G = nx.DiGraph()
        self.start, self.end = 0, path_length
        G.add_node(self.start)
        cur = 1
        for a in range(num_arms):
            arm = []
            for i in range(1, path_length + 1):
                G.add_node(cur)
                if i == 1:
                    G.add_edge(self.start, cur)
                else:
                    G.add_edge(cur - 1, cur)
                arm.append(cur)
                cur += 1
            self.arms.append(arm)
        self.G = G

        if needs_visual:
            self.make_plot_colours()
            self.make_plot_positions_for_layout()

    def connect_c_components(self, verbose: bool = False) -> None:
        raise ValueError('Balanced graphs do not have unconnected components by construction')

    def make_plot_positions_for_layout(self, **kwargs):
        # https://stackoverflow.com/questions/57512155/how-to-draw-a-tree-more-beautifully-in-networkx
        try:
            self.pos = nx.nx_agraph.graphviz_layout(self.G, prog="twopi")
        except ImportError:
            super().make_plot_positions_for_layout()

    def make_plot_colours(self, **kwargs):
        import matplotlib.pyplot as plt
        # https://networkx.org/documentation/stable/auto_examples/drawing/plot_labels_and_colours.html
        colour_map = []
        colours = plt.cm.rainbow(np.linspace(0, 1, len(self.arms)))
        node_to_arm_id = {}
        for i, arm in enumerate(self.arms):
            for node in arm:
                node_to_arm_id[node] = i
        for v in self.G.nodes:
            if v not in node_to_arm_id:
                colour_map.append('black')
            else:
                colour_map.append(colours[node_to_arm_id[v]])
            if v == self.start:
                colour_map[-1] = 'green'
            elif v == self.end:
                colour_map[-1] = 'red'
        self.colour_map = colour_map


class BalancedGraph(PathStarGraph):
    gtype = 'Balanced'

    def __init__(self, num_nodes: int, path_length: int,
                 c_min: int = 75,  c_max: int = 125,
                 rng: np.random.Generator = np.random.default_rng(), needs_visual=False,
                 max_num_parents: int = 4, max_prefix_vertices: int = None, old_version=False, **kwargs):
        super().__init__(num_nodes, path_length, c_min, c_max, rng, needs_visual)
        self.max_num_parents = max_num_parents
        self.max_prefix_vertices = max_prefix_vertices

        if old_version:
            self.G, self.start, self.end = self.generate_graph_with_lookahead_original(num_nodes, lookahead=path_length)
        else:
            self.G, self.start, self.end = self.generate_graph_with_lookahead(num_nodes, lookahead=path_length)
        if needs_visual:
            self.make_plot_colours()
            self.make_plot_positions_for_layout()

    def generate_graph_with_lookahead(self, max_num_vertices: int, lookahead: int, min_noise_reserve: int = 0):
        """
        Args:
            self:
            max_num_vertices: v_max in paper -- note this can sample above this??
            lookahead: L in paper
            min_noise_reserve:  NEW parameter to reserve some noise vertices from max_num_vertices, this effecitively
            replaces 'u' in the paper
        Returns: nx.Graph, start, end
        """

        if lookahead == 0:
            raise ValueError('Lookahead must be > 0')
        if (max_num_vertices - min_noise_reserve - 1 - lookahead) // lookahead < 1:
            raise ValueError(f'Not enough vertices to create a graph with lookahead {lookahead} '
                             f'from {max_num_vertices - min_noise_reserve}')

        class Node(object):
            def __init__(self, id):
                self.id = id
                self.children = []
                self.parents = []

            def __eq__(self, other):
                return self.id == other.id

        # sample number of paths
        max_num_paths = (max_num_vertices - min_noise_reserve - 1 - lookahead) // lookahead
        num_paths = self.rng.integers(1, max_num_paths + 1)

        # sample path lengths
        path_lengths = []
        path_noise = self.rng.integers(0, 1, num_paths)
        n = max_num_vertices - min_noise_reserve - 1 - lookahead
        for p in range(num_paths):
            l = lookahead + path_noise[p]
            if l > n:
                l = n
            path_lengths.append(l)
            n -= l

        vertices = []
        for i in range(max_num_vertices):
            vertices.append(Node(i))

        start = vertices[0]
        end = vertices[lookahead]

        # create desired path
        cur = 1
        for i in range(lookahead):
            vertices[cur].parents.append(vertices[cur - 1])
            vertices[cur -1].children.append(vertices[cur])
            cur += 1
        self.arms.append([v.id for v in vertices[1:lookahead]])

        # create other paths
        for path_length in path_lengths:
            arm = []
            vertices[cur].parents.append(vertices[0])
            vertices[0].children.append(vertices[cur])
            arm.append(vertices[cur].id)
            cur += 1
            for _ in range(1, path_length):
                vertices[cur].parents.append(vertices[cur - 1])
                vertices[cur - 1].children.append(vertices[cur])
                arm.append(vertices[cur].id)
                cur += 1
            self.arms.append(arm)

        # create in path
        if cur < max_num_vertices - 1:
            num_prefix_vertices = self.rng.integers(0, lookahead + 1)
            num_prefix_vertices = min(num_prefix_vertices, max_num_vertices - cur - 1)
            prev_vertex = vertices[0]
            arm = []
            for _ in range(num_prefix_vertices):  # build in arm
                vertices[cur].children.append(prev_vertex)
                prev_vertex.parents.append(vertices[cur])
                prev_vertex = vertices[cur]
                arm.append(prev_vertex.id)
                cur += 1
            self.arms.append(arm)

        # sample some parent/ancestor vertices
        alpha = 0.5
        in_degrees = np.array([alpha + len(vertex.parents) for vertex in vertices[:max_num_vertices]])
        out_degrees = np.array([alpha + len(vertex.children) for vertex in vertices[:max_num_vertices]])
        for i in range(cur, max_num_vertices):
            # sample the number of child and parent vertices
            num_children = self.rng.integers(0, self.max_num_parents)
            num_parents = self.rng.integers(0 if num_children != 0 else 1, self.max_num_parents)
            num_children = min(num_children, i)
            num_parents = min(num_parents, i)

            # sample the children of this new node
            probabilities = in_degrees[:cur].copy()
            probabilities /= np.sum(probabilities)
            for child_id in np.random.choice(cur, num_children, replace=False, p=probabilities):
                vertices[cur].children.append(vertices[child_id])
                vertices[child_id].parents.append(vertices[cur])
                in_degrees[child_id] += 1

            # to avoid creating a cycle, we have to remove any descendants from the possible parents
            descendants = self.get_descendants(vertices[cur])
            probabilities = out_degrees[:cur].copy()

            for descendant in descendants:
                probabilities[descendant.id] = 0
            total_probability = np.sum(probabilities)
            if total_probability != 0.0:
                probabilities /= total_probability
                num_parents = min(num_parents, cur - len(descendants))

                # sample the parents of this new node
                for parent_id in np.random.choice(cur, num_parents, replace=False, p=probabilities):
                    vertices[parent_id].children.append(vertices[i])
                    vertices[i].parents.append(vertices[parent_id])
                    out_degrees[parent_id] += 1
            cur += 1

        # convert to nx
        G = nx.DiGraph()
        for vertex in vertices:
            G.add_node(vertex.id)
        for vertex in vertices:
            for child in vertex.children:
                G.add_edge(vertex.id, child.id)

        # print(len(G.nodes) > max_num_vertices, len(G.nodes), max_num_vertices, lookahead)
        return G, start.id, end.id

    def generate_graph_with_lookahead_original(self, max_num_vertices: int, lookahead: int):
        """
        Args:
            self:
            max_num_vertices: v_max in paper
            lookahead: L in paper
            max_prefix_vertices:  ??? in paper
        Returns: nx.Graph, start, end
        """

        class Node(object):
            def __init__(self, id):
                self.id = id
                self.children = []
                self.parents = []

            def __eq__(self, other):
                return self.id == other.id

        max_prefix_vertices = self.max_prefix_vertices
        if self.max_prefix_vertices is None:
            max_prefix_vertices = max_num_vertices

        if lookahead == 0:
            num_paths = randrange(1, 3)
        else:
            max_num_paths = (max_num_vertices - 1) // lookahead
            num_paths = randrange(2, max_num_paths + 1)  # B in paper

        #  u = randrange(0, 6) in paper
        num_vertices = min(lookahead * num_paths + 1 + randrange(0, 6), max_num_vertices)
        num_vertices = max(2, num_vertices, 1 + num_paths * lookahead)

        vertices = []
        for i in range(num_vertices):
            vertices.append(Node(i))

        vertices[1].parents.append(vertices[0])
        vertices[0].children.append(vertices[1])
        for i in range(1, lookahead):
            vertices[1 + i].parents.append(vertices[i])
            vertices[i].children.append(vertices[1 + i])
        self.arms.append([v.id for v in vertices[1:lookahead]])
        if lookahead == 0:
            index = 2
        else:
            index = 1 + lookahead
            for j in range(num_paths - 1):
                arm = []
                vertices[index].parents.append(vertices[0])
                vertices[0].children.append(vertices[index])
                arm.append(vertices[index].id)
                index += 1
                other_branch_length = lookahead + randrange(
                    min(2, num_vertices - index - (num_paths - j - 1) * lookahead + 2))
                for _ in range(1, other_branch_length):
                    vertices[index].parents.append(vertices[index - 1])
                    vertices[index - 1].children.append(vertices[index])
                    arm.append(vertices[index].id)
                    index += 1
                self.arms.append(arm)

        num_prefix_vertices = randrange(min(max_prefix_vertices + 1, num_vertices - index + 1))
        prev_vertex = vertices[0]
        arm = []
        for _ in range(num_prefix_vertices):  # build in arm
            vertices[index].children.append(prev_vertex)
            prev_vertex.parents.append(vertices[index])
            prev_vertex = vertices[index]
            arm.append(prev_vertex.id)
            index += 1
        self.arms.append(arm)

        start = vertices[0]
        end = vertices[max(1, lookahead)]

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

        # convert to nx
        G = nx.DiGraph()
        for vertex in vertices:
            G.add_node(vertex.id)
        for vertex in vertices:
            for child in vertex.children:
                G.add_edge(vertex.id, child.id)
        return G, start.id, end.id

    @staticmethod
    def get_descendants(node):
        queue = [node]
        visited = []
        descendants = []
        while len(queue) != 0:
            current = queue.pop()
            visited.append(current)
            for child in current.children:
                if child not in descendants:
                    descendants.append(child)
                if child in visited:
                    continue
                queue.append(child)
        return descendants


class PyGraphGen(object):
    """
    Generates random graphs
    """
    def __init__(self, graph_type,
                 max_nodes: int | None, max_path_length: int | None,
                 min_nodes: int | None = -1, min_path_length: int | None = -1,
                 c_min: int = 75,  c_max: int = 125,
                 rng: np.random.Generator = np.random.default_rng(),
                 **kwargs):
        self.graph_type = graph_type
        self.graph_type_str = graph_type.gtype

        self.max_nodes = max_nodes
        self.min_nodes = min_nodes if min_nodes != -1 else max_nodes
        self.max_path_length = max_path_length
        self.min_path_length = min_path_length if min_path_length != -1 else max_path_length

        self.c_min = c_min
        self.c_max = c_max

        self.max_num_arms = kwargs.get('max_num_arms', None)
        self.min_num_arms = kwargs.get('min_num_arms', None)
        self.rng = rng

        self.kwargs = kwargs

    def get_random_graph_size(self) -> int:
        if self.min_nodes == self.max_nodes:
            return self.min_nodes
        return self.rng.integers(self.min_nodes, self.max_nodes + 1)

    def get_random_path_length(self, lengths: List[int] = None) -> int:
        if lengths:  # uniform over lengths
            return self.rng.choice(lengths)
        else:
            if self.min_path_length == self.max_path_length:
                return self.min_path_length
            return self.rng.integers(self.min_path_length, self.max_path_length + 1)

    def get_random_num_arms(self) -> int:
        if self.min_num_arms == self.max_num_arms:
            return self.min_num_arms
        return self.rng.integers(self.min_num_arms, self.max_num_arms + 1)

    def generate(self, needs_visual=False) -> PyGraph:
        if self.graph_type == EuclidianGraph:
            return EuclidianGraph(self.get_random_graph_size(), None, rng=self.rng,
                                  needs_visual=needs_visual, **self.kwargs)
        elif self.graph_type == ErdosRenyiGraph:
            return ErdosRenyiGraph(self.get_random_graph_size(), None, rng=self.rng,
                                   needs_visual=needs_visual, **self.kwargs)
        elif self.graph_type == PathStarGraph:
            num_arms = self.get_random_num_arms()
            return PathStarGraph(self.get_random_graph_size(), self.get_random_path_length(), rng=self.rng,
                                 needs_visual=needs_visual, num_arms=num_arms, **self.kwargs)
        elif self.graph_type == BalancedGraph:
            return BalancedGraph(self.get_random_graph_size(), self.get_random_path_length(), rng=self.rng,
                                 needs_visual=needs_visual, **self.kwargs)
        else:
            raise ValueError(f'Unknown graph type {self.graph_type}')

    def histogram_of_shortest_path_lengths(self, num_graphs, average=True, **kwargs):
        lengths = defaultdict(list)
        for i in range(num_graphs):
            graph = self.generate(**kwargs)
            lengths_i, _ = graph.get_histogram_path_lengths()
            for length, count in lengths_i.items():
                lengths[length].append(count)
        if average:
            for length, counts in lengths.items():
                lengths[length] = np.mean(counts)
        return lengths


