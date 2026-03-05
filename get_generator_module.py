import os
import sys
import time
import pydoc
import numpy as np
import torch

try:
    import networkx as nx
except ImportError as e:
    print(f"NetworkX is not installed or broken. {e}")
    nx = None

from sympy.polys.polyconfig import query

"""
All code belonging to the generator is in here.  This is the python interface to the C++ code.

TThere is also code for recreating the graphs in python for plotting and statistics.
"""


def get_args_parser():
    """
    :return: the parser i.e. not the parsed arguments i.e. parser.parse_args()
    """
    import argparse

    parser = argparse.ArgumentParser(description='Test generator functions')

    # graph settings
    parser.add_argument('--graph_type', type=str, default='erdos_renyi')  # 'erdos_renyi' # 'euclidean'  # 'path_star'  # 'balanced'
    parser.add_argument('--min_num_nodes', type=int, default=15,
                        help="Minimum number of nodes for generated graphs.  "
                             "We strongly recommend using shuffle_nodes and a vocab range map via min_vocab and max_vocab.")
    parser.add_argument('--max_num_nodes', type=int, default=25,
                        help='If -1 use max=min only i.e. only sample a single size')
    parser.add_argument('--c_min', type=int, default=75,
                        help='Min number of sampled edges to form a single connected component')  # for graphs with multiple connected components
    parser.add_argument('--c_max', type=int, default=125,
                        help='Max number of sampled edges to form a single connected component')
    parser.add_argument('--shuffle_edges', action='store_true', default=True,
                        help='Whether to shuffle edge list when generating the graph tokenization.')
    parser.add_argument('--dont_shuffle_edges', action='store_false', dest='shuffle_edges')
    parser.add_argument('--shuffle_nodes', action='store_true', default=True,
                        help='Whether to shuffle node ids when generating the graph tokenization.'
                             'This randomly maps nodes across the whole spectrum of available vocab ids.')
    parser.add_argument('--dont_shuffle_nodes', action='store_false', dest='shuffle_nodes')

    parser.add_argument('--min_vocab', type=int, default=-1,
                        help='Minimum vocab id to use when shuffling nodes.'
                             '-1 and -1 max_vocab means uses set dictionary values')
    parser.add_argument('--max_vocab', type=int, default=-1,
                        help='Maximum vocab id to use when shuffling nodes.'
                             '-1 with a set minimum, will use the number of nodes.')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--max_edges', type=int, default=512)
    parser.add_argument('--max_attempts', type=int, default=1000)

    # erdos_renyi graphs settings
    parser.add_argument('--p', type=float, default=-1.0,
                        help="Probability for edge creation for erdos_renyi graphs, -1.0 means random")
    # euclidean graphs settings
    parser.add_argument('--dims', type=int, default=2,
                        help="Number of dimensions for euclidean graphs")
    parser.add_argument('--radius', type=float, default=-1.0,
                        help="Radius for euclidean graphs, -1.0 means random")
    # random tree graphs settings
    parser.add_argument('--max_degree', type=int, default=3)
    parser.add_argument('--max_depth', type=int, default=7)
    parser.add_argument('--bernoulli_p', type=float, default=0.5)
    parser.add_argument('--probs', type=float, nargs='+',
                        help="Probabilities for random tree graphs, should sum to 1.0")
    parser.add_argument('--start_at_root', action='store_true', default=True,)
    parser.add_argument('--end_at_leaf', action='store_true', default=True,)

    # path star graphs settings
    parser.add_argument('--min_num_arms', type=int, default=2,
                        help="Minimum number of arms for star graphs")
    parser.add_argument('--max_num_arms', type=int, default=5,
                        help="Maximum number of arms for star graphs")
    parser.add_argument('--min_arm_length', type=int, default=5,
                        help="Minimum length of each arm in star graphs")
    parser.add_argument('--max_arm_length', type=int, default=8,
                        help="Maximum length of each arm in star graphs")

    # balanced graphs settings
    parser.add_argument('--min_lookahead', type=int, default=3,
                        help="Minimum lookahead for balanced graphs")
    parser.add_argument('--max_lookahead', type=int, default=8,
                        help="Maximum lookahead for balanced graphs")
    parser.add_argument('--min_noise_reserve', type=int, default=0,
                        help="Minimum noise reserve for balanced graphs")
    parser.add_argument('--max_num_parents', type=int, default=4,
                        help="Maximum number of parents for balanced graphs")

    # task settings
    parser.add_argument('--task_type', type=str, default='shortest_path')
    parser.add_argument('--scratchpad_type', type=str, default='none') #'none' 'BFS'  # 'DFS'

    parser.add_argument('--min_path_length', type=int, default=3,
                        help='Minimum path length for shortest path tasks (inclusive)')
    parser.add_argument('--max_path_length', type=int, default=12,
                        help='Maximum path length for shortest path tasks (inclusive)')
    parser.add_argument('--sort_adjacency_lists', action='store_true', default=False,
                        help='Whether to sort adjacency lists when generating BFS/DFS scratchpad.')
    parser.add_argument('--use_unique_depth_markers', action='store_true', default=False,
                        help='Whether to use unique depth markers when generating BFS/DFS scratchpad.')
    parser.add_argument('--stop_once_found', action='store_true', default=True, help='')
    parser.add_argument('--task_sample_dist', nargs='*', default=None,
                        help='Optional sampling distribution for different task types.'
                             ' E.g. for shortest path tasks, [shortest_path, center, centroid].'
                             ' Should sum to 1.0.')

    parser.add_argument('--min_query_size', type=int, default=2,
                        help='Minimum query size for center/centroid tasks (inclusive)')
    parser.add_argument('--max_query_size', type=int, default=10,
                        help='Maximum query size for center/centroid tasks (inclusive)')

    parser.add_argument('--min_khops', type=int, default=1,
                        help='Minimum hops for khops path tasks (inclusive)')
    parser.add_argument('--max_khops', type=int, default=7)
    parser.add_argument('--min_prefix_length', type=int, default=70)
    parser.add_argument('--max_prefix_length', type=int, default=100)
    parser.add_argument('--right_side_connect', action='store_true', default=False,
                        help='The usual khops version, but not how I have BFS set up.')
    parser.add_argument('--khops_no_repeats', action='store_true', default=False,
                        help='for khops gen and MTP')
    parser.add_argument('--permutation_version', action='store_true', default=False,)
    parser.add_argument('--mask_to_vocab_size', action='store_true', default=False,)
    parser.add_argument('--partition_method', type=str, default='uniform')

    # tokenization settings
    parser.add_argument('--is_causal', action='store_true', default=False,
                        help='Whether to use causal tokenization of distances'
                             ' (i.e. decoder-only style) or non-causal (i.e. encoder-only style).')
    parser.add_argument('--is_direct_ranking', action='store_true', default=False,
                        help='Whether to use direct ranking for graph distance ranking loss on node-list.'
                             'This assumes include_nodes_in_graph_tokenization'
                        )
    parser.add_argument('--query_at_end', action='store_true', default=False,
                        help='Whether to place the query at the end of the graph or before.')
    parser.add_argument('--no_graph', action='store_true', default=False,
                        help='Used for scratchpad -> path without graph experiments')
    parser.add_argument('--concat_edges', action='store_true', default=True,
                        help='Whether to concatenate edge pairs into a single token or use separate tokens.'
                             'This happens in the model but produces a 2d tensor for src_tokens [seq_len, 2]')
    parser.add_argument('--dont_concat_edges', action='store_false', dest='concat_edges')
    parser.add_argument('--duplicate_edges', action='store_true', default=False,
                        help='Whether to allow duplicate edges when generating the graph tokenization.'
                             'This repeats the edge list twice, thus bypassing the causal constraint.'
                             'Only makes sense for undirected graphs since we also swap (u, v) to (v, u).')
    parser.add_argument('--include_nodes_in_graph_tokenization', action='store_true', default=False,
                        help='Whether to include node tokens in the graph tokenization, after the edges, '
                             'i.e. edge list and then node list.')
    parser.add_argument('--num_thinking_tokens', type=int, default=0)
    parser.add_argument('--is_flat_model', action='store_true', default=True,
                        help='Whether the model is flat (i.e. single input tensor) or uses separate encoder/decoder inputs.')
    parser.add_argument('--align_prefix_front_pad', action='store_true', default=False,
                        help='Whether to align the prefix (up to target seq) by front padding the input. '
                             'Only makes sense for flat models.')

    # positional encoding settings
    parser.add_argument('--use_edges_invariance', action='store_true', default=False,)
    parser.add_argument('--use_node_invariance', action='store_true', default=False,)
    parser.add_argument('--use_graph_invariance', action='store_true', default=False,)
    parser.add_argument('--use_query_invariance', action='store_true', default=False,)
    parser.add_argument('--use_task_structure', action='store_true', default=False,)
    parser.add_argument('--use_graph_structure', action='store_true', default=False,)
    parser.add_argument('--use_full_structure', action='store_true', default=False,)

    return parser


#############################
# Extra Generator Functions #
#############################

class ReconstructedGraph(object):
    def __init__(self, graph_type, task_type,
                 edge_list, query, task_input, task_targets, pos=None,
                 spring_k=1.5, spring_scale=1.5, verbose=False, **kwargs):
        """

        :param graph_type:
        :param edge_list: [num_edges, 2]
        :param pos:
        """

        self.graph_type = graph_type
        self.task_type = task_type
        self.edge_list = edge_list
        self.query = query
        self.task_input = task_input
        self.task_targets = task_targets

        self.default_colour = '#1f78b4'  # matplotlib tab:blue
        self.query_colour = 'purple'  # matplotlib tab:purple
        self.target_colour = 'green'  # matplotlib tab:green

        if self.is_directed():
            self.G = nx.DiGraph()  # or nx.Graph() for undirected graphs
        else:
            self.G = nx.Graph()
        self.pos, self.colour_map, self.node_edge_colour_map = None, None, None
        if pos is not None:
            self.pos = {}
            for node, node_pos in pos.items():
                self.G.add_node(node, pos=node_pos)
        self.G.add_edges_from(edge_list)
        self.process_task()
        print('verbose is', verbose)
        self.set_plot_positions_for_layout(spring_k, spring_scale, verbose, **kwargs)


    def is_directed(self):
        return self.graph_type in ('path_star', 'balanced')

    def get_node_list(self):
        nodes = []
        for e in self.edge_list:
            nodes.append(e[0])
            nodes.append(e[1])
        nodes = sorted(list(set(nodes)))
        return nodes

    # RECONSTRUCTION METHODS
    def process_task(self):
        # print(self.query)
        # print(self.task_input)
        # print(self.task_targets)
        # print(self.get_node_list())
        # print()
        if self.task_input is None or self.task_targets is None or self.task_type in (None, 'none', 'None'):
            self.colour_map = self.default_colour
        elif self.task_type in ('shortest_path', 'path'):
            self.colour_map = []
            self.node_edge_colour_map = []
            target_nodes = []
            target_node_to_rank = {}
            for t in self.task_targets[1:-1]:  # cut special tokens
                for i, node in enumerate(t):
                    target_nodes.append(node)
                    target_node_to_rank[node] = i
            query = self.query[1:-1]  # cut special tokens
            for node in self.G:

                if node in query:
                    self.colour_map.append(self.query_colour)
                    self.node_edge_colour_map.append(self.query_colour)
                elif node in target_nodes:
                    self.colour_map.append(self.target_colour)
                    # colour alternative paths with optional red boundary
                    self.node_edge_colour_map.append(self.target_colour if target_node_to_rank[node] == 0 else 'red')
                else:
                    self.colour_map.append(self.default_colour)
                    self.node_edge_colour_map.append(self.default_colour)
        elif self.task_type in ('center', 'centroid'):
            self.colour_map = []
            self.node_edge_colour_map= []
            target_nodes = []
            for t in self.task_targets[1:-1]:  # cut special tokens
                for node in t:
                    target_nodes.append(node)
            query = self.query[1:-1]  # cut special tokens
            for node in self.G:
                if node in target_nodes and node in query:
                    self.colour_map.append(self.target_colour)
                    self.node_edge_colour_map.append(self.query_colour)
                elif node in target_nodes:
                    self.colour_map.append(self.target_colour)
                    self.node_edge_colour_map.append(self.target_colour)
                elif node in query:
                    self.colour_map.append(self.query_colour)
                    self.node_edge_colour_map.append(self.query_colour)
                else:
                    self.colour_map.append(self.default_colour)
                    self.node_edge_colour_map.append(self.default_colour)
        else:
            raise ValueError(f"Unexpected task_type: {self.task_type}")

    # PLOTTING METHODS
    def plot(self, save_path=None, save_name=None, node_size=200,  with_labels=True, **kwargs):
        assert nx is not None, "NetworkX is required for plotting. Please install it with 'pip install networkx'."
        import matplotlib.pyplot as plt

        # the edgecolors keyword argument (for setting the outline of nodes)
        # is different from the edge_color keyword argument (for setting the colour of lines)
        print('pos', self.pos)

        nx.draw(self.G, with_labels=with_labels, pos=self.pos, node_size=node_size, linewidths=2,
                node_color=self.colour_map,
                edgecolors=self.node_edge_colour_map)

        plt.show()
        if save_path is not None:
            #if graph.type not in save_path:
            #    save_path = os.path.join(save_path, graph.type)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            if save_name is not None:
                save_path = os.path.join(save_path, save_name)
            else:
                save_path = os.path.join(save_path, 'graph.png')
            plt.savefig(save_path)


    def set_plot_positions_for_layout(self, spring_k=1.5, spring_scale=1.5, verbose=False, **kwargs):
        if self.pos is None:
            if verbose:
                print('No pos provided, using layout to compute positions.')

            try:
                if self.graph_type in ('path_star', 'balanced', 'random_tree'):
                    # pos = nx.nx_agraph.graphviz_layout(self.G, prog='dot', args='-Grankdir=LR')
                    pos = nx.nx_agraph.graphviz_layout(self.G, prog="twopi")
                    print(pos)
                else:
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

    def pprint(self):
        pass





def create_reconstruct_graphs(b_n, token_dict, ids=None, verbose=False, **kwargs):
    """
    Take the c++ output and reconstruct the graphs for plotting and sanity checking.

    :param b_n: c++ output dictionary
    :param token_dict: mapping from symbols to tokenized ids
    :param ids: batch ids to reconstruct, if None, reconstruct all
    :return:
    """

    reconstructions = []

    id_to_symbol_ = {v: k for k, v in token_dict.items()}
    def id_to_symbol(id):
        symbol = id_to_symbol_[id]
        if symbol.isdigit():
            return int(symbol)
        return symbol

    pad = token_dict.get('<pad>', -1)

    src_tokens = b_n['src_tokens']
    prev_output_tokens = b_n['prev_output_tokens']  # task targets
    num_nodes = b_n['num_nodes']
    num_edges = b_n['num_edges']
    true_task_length = b_n['true_task_lengths']
    query_lengths = b_n['query_lengths']

    graph_edge_gather_indices = edge_gather_ids(b_n, pad_value=-1)
    graph_node_gather_indices = node_gather_ids(b_n, pad_value=-1)
    task_targets = b_n['prev_output_tokens']
    true_task_gather_indices = task_gather_ids(b_n, is_true_task=True, pad_value=-1)
    query_gather_indices = query_gather_ids(b_n, pad_value=-1)
    node_positions = None
    if 'node_positions' in b_n and b_n['node_positions'] is not None:
        node_positions = b_n['node_positions']

    concat_edges = b_n['concat_edges']
    graph_type = b_n['graph_type']
    task_type = b_n['task_type']

    is_directed = graph_type in ('path_star', 'balanced', 'khops', 'khops_gen')

    if ids is None:
        ids = list(range(src_tokens.shape[0]))

    def gather(tensor, gather_indices, id):
        gather_indices, mask = gather_indices
        if tensor is None or gather_indices is None:
            raise ValueError("Tensor and gather_indices cannot be None")
        if tensor.ndim == 2:
            t = tensor[id, gather_indices[id], 0]
        elif tensor.ndim == 3:
            t = tensor[id, gather_indices[id], :]
        else:
            raise ValueError(f"Unexpected tensor shape: {tensor.shape}")
        # cut down to actual lengths
        if mask is not None:
            valid_length = mask[id].sum()
            if tensor.ndim == 2:
                t = t[:valid_length]
            elif tensor.ndim == 3:
                t = t[:valid_length, :]
        return t

    for id in ids:
        if b_n["graph_type"] in ('khops', 'khops_gen'):  # these have no edge list
            raise NotImplementedError
        else:
            assert not b_n['no_graph'], "No graph to reconstruct"
            if concat_edges:
                edges = gather(src_tokens, graph_edge_gather_indices, id)
                edge_list = edges.reshape(-1, 2).tolist()
            else:  # here we actually need to parse the src
                graph_start = b_n['graph_edge_start_indices'][id]
                assert src_tokens.shape[-1] == 1, 'src can not have structure if concat_edges is False'
                edge_list = []
                for i in range(num_edges[id]):
                    edge_tokens = src_tokens[id, graph_start + (i * 3): graph_start + (i * 3) + 2, 1]
                    edge_list.append(edge_tokens[:2].tolist())  # cut off edge marker if it exists
            for i in range(len(edge_list)):
                edge_list[i] = [id_to_symbol(e) for e in edge_list[i]]

            edge_list = [sorted(e) if not is_directed else e for e in edge_list]
            edge_list = sorted(edge_list, key=lambda x: (x[0], x[1]))

        query, task_input, task_targets = None, None, None
        if true_task_gather_indices is not None:
            query = gather(src_tokens, query_gather_indices, id)[:, 0].tolist()
            query = [id_to_symbol(q) for q in query]
            task_input = gather(src_tokens, true_task_gather_indices, id)[:, 0].tolist()
            task_input = [id_to_symbol(t) for t in task_input]
            task_targets_t = prev_output_tokens[id, :true_task_length[id], :]
            task_targets= []
            for i in range(true_task_length[id]):
                targets_at_i = []
                for j in range(task_targets_t.shape[-1]):
                    if task_targets_t[i, j] != pad:
                        targets_at_i.append(id_to_symbol(task_targets_t[i, j]))
                task_targets.append(targets_at_i)

            print(edge_list)
            print(query)
            print(task_targets)

            if nx is not None:
                r = ReconstructedGraph(graph_type, task_type, edge_list, query, task_input, task_targets, pos=None, verbose=verbose)
                reconstructions.append(r)

    return reconstructions

def pprint_distance(distances, min_node=0, max_node=500, idxs=(0,1,2), use_node_ids=True):
    """
    d in a n * n matrix, we add a column and a row for the range ids, then cut off at the min and max num nodes
    """
    if distances.ndim == 3:
        if isinstance(idxs, int):
            if idxs > 0:
                idxs_ = list(range(idxs))
            else:
                idxs_ = list(range(distances.shape[0]))
        elif len(idxs) == 0:
            idxs_ = list(range(distances.shape[0]))
        else:
            idxs_ = [b for b in idxs if b < distances.shape[0]]
    else:
        distances = np.expand_dims(distances, 0)
        idxs_ = [0]

    for b in idxs_:
        print(f'BATCH INDEX: {b}\n')
        d_out = distances[b, ...].copy()
        d_out = d_out[min_node:max_node + 1, min_node:max_node + 1]
        n = d_out.shape[0]
        if use_node_ids:
            a1 = np.arange(min_node, max_node + 1)[:n]
            a2 = np.arange(min_node, max_node + 2)[:n + 1]
        else:
            a1 = np.arange(n)
            a2 = np.arange(n + 1)
        a2[-1] = -1
        d_out = np.concatenate([d_out, a1[None, :]], axis=0)
        d_out = np.concatenate([d_out, a2[:, None]], axis=1)
        print(d_out)

def pprint_batched_dict(b_n, token_dict, pos_dict, title='', print_distances=False, print_graph_gts=False, idxs=(0,1,2),
                        print_dist=False, print_shapes=True):
    """
    :param b_n: batched dict
    :param title:
    :param print_distances:
    :param print_graph_gts:
    :param idxs:
    :return: None
    """

    rev_token_dict = {v: k for k, v in token_dict.items()}
    # rev_pos_dict = {v: k for k, v in pos_dict.items()}  # just print the idxs because they are readable
    src_tokens = b_n['src_tokens']
    graph_edge_gather_indices = edge_gather_ids(b_n, pad_value=-1)[0]
    graph_node_gather_indices = node_gather_ids(b_n, pad_value=-1)[0]
    task_targets = b_n['prev_output_tokens']
    true_task_gather_indices = task_gather_ids(b_n, is_true_task=True, pad_value=-1)[0]
    scratch_pad_gather_indices = scratchpad_gather_ids(b_n, pad_value=-1)[0]
    positions = b_n['positions']
    if positions is None:
        positions = np.arange(src_tokens.shape[1])[None, :].repeat(src_tokens.shape[0], axis=0)

    if print_shapes:
        print('src_tokens shape:', src_tokens.shape)
        if task_targets is not None:
            print('task_targets shape:', task_targets.shape)
        if positions is not None:
            print('positions shape:', positions.shape)

    if isinstance(idxs, int):
        if idxs > 0:
            idxs_ = list(range(idxs))
        else:
            idxs_ = list(range(src_tokens.shape[0]))
    elif len(idxs) == 0:
        idxs_ = list(range(src_tokens.shape[0]))
    else:
        idxs_ = [b for b in idxs if b < src_tokens.shape[0]]

    max_num_chars = 0
    def update_max(b_, tensor, dict_, max_num_chars_):
        if tensor.ndim < 3:
            tensor = np.expand_dims(tensor, -1)
        for i in range(tensor.shape[1]):
            for j in range(tensor.shape[2]):
                token_id = tensor[b_, i, j] if tensor.ndim > 2 else tensor[b_, i]
                if dict_:
                    token_str = dict_.get(token_id, str(token_id))
                else:
                    token_str = str(token_id)
                if len(token_str) > max_num_chars_:
                    max_num_chars_ = len(token_str)
        return max_num_chars_

    for b in idxs_:
        max_num_chars = update_max(b, src_tokens, rev_token_dict, max_num_chars)
        if task_targets is not None:
            max_num_chars = update_max(b, task_targets, None, max_num_chars)
        if positions is not None:
            max_num_chars =update_max(b, positions, pos_dict, max_num_chars)

    def pprint_tensor(b_, tensor, dict_, pad, offset1=0, offset2=len('Src:   '), skip=0):
        s = ''
        if tensor.ndim < 3:
            tensor = np.expand_dims(tensor, -1)
        max_j_dim = tensor.shape[2]
        m = (tensor == pad).all(axis=1)
        for j in range(max_j_dim):
            if m[b_, j]:
                max_j_dim = j
                break
        for j in range(max_j_dim):
            s += ' ' * (max_num_chars + 1) * offset1
            for i in range(tensor.shape[1]):
                token_id = tensor[b_, i, j] if tensor.ndim > 2 else tensor[b_, i]
                if dict_:
                    token_str = dict_.get(token_id, str(token_id))
                else:
                    token_str = str(token_id)
                if token_id == pad:
                    token_str = ' '
                s += token_str.ljust(max_num_chars + 1)
                if skip > 0:
                    s += ' '.ljust(max_num_chars + 1) * skip
            if j < max_j_dim - 1:
                s += '\n' + ' ' * offset2
            else:
                s += '\n'
        return s


    if title:
        print(f'{title}')

    pad = token_dict.get('<pad>', -1)
    pos_pad = pos_dict.get('pad', -1)
    for b in idxs_:
        # print so all tokens line up
        s = f'BATCH INDEX: {b}\n'
        target_start_idx = 0
        if task_targets is not None:
            target_start_idx = b_n['task_start_indices'][b]
        if positions is not None:
            s += 'Pos:   '
            s += pprint_tensor(b, positions, None, pos_pad)
        s += 'Src:   '
        s += pprint_tensor(b, src_tokens, rev_token_dict, pad)
        if task_targets is not None:
            s += 'Tgt:   '
            s += pprint_tensor(b, task_targets, rev_token_dict, pad, offset1=target_start_idx)
            s += f'TgtIdx:'
            s += pprint_tensor(b, np.expand_dims(true_task_gather_indices, -1), None, pad=-1, offset1=b_n['true_task_start_indices'][b])
            if scratch_pad_gather_indices is not None:
                s += f'SP Idx:'
                s += pprint_tensor(b, np.expand_dims(scratch_pad_gather_indices, -1), None, pad=-1, offset1=b_n['scratch_pad_start_indices'][b])
        if graph_edge_gather_indices is not None:
            s += f'EdgIdx:'
            edge_offset = b_n['graph_edge_start_indices'][b]
            if not b_n['concat_edges']:
                edge_offset += 2
            s += pprint_tensor(b, np.expand_dims(graph_edge_gather_indices, -1), None, pad=-1, offset1=edge_offset,
                               skip = 0 if b_n['concat_edges'] else 2)
        if graph_node_gather_indices is not None:
            s += f'NodIdx:'
            s += pprint_tensor(b, np.expand_dims(graph_node_gather_indices, -1), None, pad=-1, offset1=b_n['graph_node_start_indices'][b])
        s += 'Idx:   '
        a = np.expand_dims(np.expand_dims(np.arange(b_n['src_lengths'][b]), 1), 0)
        s += pprint_tensor(0, a, None, pad=-1,)
        print(s)

        if print_dist and b_n["distances"] is not None:
            pprint_distance(b_n["distances"], min_node=b_n["min_vocab"], max_node=b_n["max_vocab"], idxs=[b], use_node_ids=False)

    if b_n['align_prefix_front_pad']:
        print('Showing that align_prefix_front_pad works as targets are aligned to the right (either at scratchpad = 16 or task = 6):')
        print(src_tokens[:3, :, 0])

    # print out the shape of distance matrix, and ground_truths_gather_distances
    if b_n["distances"] is not None:
        distances = b_n["distances"]
        print(f'Distances shape: {distances.shape}')
        ground_truths_gather_distances = b_n["ground_truths_gather_distances"]
        print(f'Ground truths gather distances shape: {ground_truths_gather_distances.shape}')


def _gather_ids(starts, lengths, stride=1, offset=0, pad_value=0):
    bs = starts.shape[0]
    if isinstance(starts, np.ndarray):
        max_len = lengths.max()
        gather_indices = np.arange(max_len)[None, :].repeat(bs, axis=0)  # [bs, max_len]
        gather_indices = gather_indices * stride + offset  # apply stride and offset
        gather_indices = gather_indices + starts[:, None]  # [bs, max_len]
        mask = gather_indices < (starts + lengths)[:, None]  # [bs, max_len]
        gather_indices = gather_indices * mask + pad_value * (~mask)  # [bs, max_len]
    elif isinstance(starts, torch.Tensor):
        max_len = lengths.max().item()
        gather_indices = torch.arange(max_len)[None, :].expand(bs, -1)  # [bs, max_len]
        gather_indices = gather_indices * stride + offset  # apply stride and offset
        gather_indices = gather_indices + starts[:, None]  # [bs, max_len]
        mask = gather_indices < (starts + lengths)[:, None]  # [bs, max_len]
        gather_indices = gather_indices * mask + pad_value * (~mask)  # [bs, max_len]
    else:
        raise ValueError(f"Unexpected type for starts: {type(starts)}")
    return gather_indices, mask

# these parse the output tensor to generate indicies for gathering the relevant tokens for each component of the input (graph, query, task, scratchpad)
def _gather_ids_helper(b_n, start_n, lengths_n, stride=1, offset=0, pad_value=0):
    if start_n not in b_n or lengths_n not in b_n or b_n[start_n] is None or b_n[lengths_n] is None:
        return None, None
    starts = b_n[start_n]
    lengths = b_n[lengths_n]
    return _gather_ids(starts, lengths, stride=stride, offset=offset, pad_value=pad_value)


def task_gather_ids(b_n, is_true_task, pad_value=0, **kwargs):
    if is_true_task:
        if 'true_task_gather_ids' in b_n and b_n['true_task_gather_ids'] is not None:
            return b_n['true_task_gather_ids']
        out = _gather_ids_helper(b_n, 'true_task_start_indices', 'true_task_lengths', pad_value=pad_value, **kwargs)
        b_n['true_task_gather_ids'] = out
    else:
        if 'task_gather_ids' in b_n and b_n['task_gather_ids'] is not None:
            return b_n['task_gather_ids']
        out = _gather_ids_helper(b_n, 'task_start_indices', 'task_lengths', pad_value=pad_value, **kwargs)
        b_n['true_task_gather_ids'] = out
    return out


def scratchpad_gather_ids(b_n, pad_value=0, **kwargs):
    if 'scratchpad_gather_ids' in b_n and b_n['scratchpad_gather_ids'] is not None:
        return b_n['scratchpad_gather_ids']
    out = _gather_ids_helper(b_n, 'scratch_pad_start_indices', 'scratch_pad_lengths', pad_value=pad_value, **kwargs)
    b_n['scratchpad_gather_ids'] = out
    return out


def edge_gather_ids(b_n, pad_value=0, *kwargs):
    if 'graph_edge_gather_ids' in b_n and b_n['graph_edge_gather_ids'] is not None:
        return b_n['graph_edge_gather_ids']

    if 'graph_edge_start_indices' not in b_n or 'graph_edge_lengths' not in b_n or b_n['graph_edge_start_indices'] is None or b_n['graph_edge_lengths'] is None or b_n['no_graph']:
        return None, None

    starts = b_n['graph_edge_start_indices']
    lengths = b_n['graph_edge_lengths']
    if 'graph_node_lengths' in b_n and b_n['graph_node_lengths'] is not None:
        node_lengths = b_n['graph_node_lengths']
        lengths -= node_lengths  # adjust edge lengths to account for node tokens if they are included in the graph tokenization

    if b_n["concat_edges"]:
        out = _gather_ids(starts, lengths, pad_value=pad_value)
    else:  # gather at edge markers
        out = _gather_ids(starts, lengths, stride=3, offset=2, pad_value=pad_value)
    b_n['graph_edge_gather_ids'] = out
    return out


def node_gather_ids(b_n, pad_value=0, **kwargs):
    if 'graph_node_gather_ids' in b_n and b_n['graph_node_gather_ids'] is not None:
        return b_n['graph_node_gather_ids']
    out = _gather_ids_helper(b_n, 'graph_node_start_indices', 'graph_node_lengths', pad_value=pad_value, **kwargs)
    b_n['graph_node_gather_ids'] = out
    return out


def query_gather_ids(b_n, pad_value=0, **kwargs):
    if 'query_gather_ids' in b_n and b_n['query_gather_ids'] is not None:
        return b_n['query_gather_ids']
    out = _gather_ids_helper(b_n, 'query_start_indices', 'query_lengths', pad_value=pad_value, **kwargs)
    b_n['query_gather_ids'] = out
    return out


def create_task_pos_for_inference():
    pass  # TODO


def get_generator_module(cpp_files=('undirected_graphs.h', 'directed_graphs.h', 'utils.h', 'dictionaries.h', 'matrix.h',
                                    'args.h', 'graph_wrapper.h', 'graph_tokenizer.h', 'tasks.h', 'scratch_pads.h',
                                    'instance.h', 'generator.cpp'),
                         cpp_path='',
                         boost_path='/usr/include/boost/graph/',):
    """
    This will import the C++ module `generator` and may compile it from source if it is not found or is out-of-date.
    """

    def build_module(name, cpp_files, cpp_path, boost_path):
        from os import system  #  -Ofast, don't use this cause numpy warnings
        import sysconfig
        sys_config = sysconfig.get_config_var("EXT_SUFFIX")
        includes = " ".join([f"{cpp_path}{f}" for f in cpp_files]) + " "
        if system(f"g++ --std=c++20 -DNDEBUG -fno-stack-protector -Wall -Wpedantic -shared "
                  f"-Wno-sign-compare -Wunused-variable "  # I should fix these instead of suppressing the warnings
                  f"-fPIC $(python3 -m pybind11 --includes) "
                  f"-I{boost_path} "
                  f"-I. {includes} "
                  f"-o generator{sys_config}") != 0:
            print(f"ERROR: Unable to compile `{name}.cpp`.")
            import sys
            sys.exit(1)

    def check_fresh(cpp_files, cpp_path):
        for f in cpp_files:
            if getmtime(generator_module.origin) < os.path.getmtime(f'{cpp_path}{f}'):
                print(f"ERROR: `{f}` is newer than `generator object`. Please recompile.")
                return False
        return True

    try:
        from os.path import getmtime
        import importlib.machinery
        from importlib.util import find_spec

        generator_module = find_spec("generator")
        if generator_module is None:
            raise ModuleNotFoundError
        elif not check_fresh(cpp_files, cpp_path):
            print("C++ module `generator` is out-of-date. Compiling from source...")
            build_module("generator", cpp_files, cpp_path, boost_path)
        import generator
    except ModuleNotFoundError:
        print("C++ module `generator` not found. Compiling from source...")
        build_module("generator", cpp_files, cpp_path, boost_path)
        print(find_spec('generator'))
        import generator

    setattr(generator, "get_args_parser", get_args_parser)


    def get_graph(args, graph_type=None, batch_size=None, task_sample_dist=None):
        if not isinstance(args, dict):
            args = vars(args)
        if graph_type is None:
            graph_type = args['graph_type']
        if batch_size is not None:
            args['batch_size'] = batch_size
        if task_sample_dist is not None:
            args['task_sample_dist'] = task_sample_dist
        else:
            if args.get('task_sample_dist', None) is None:
                args['task_sample_dist'] = []
        if graph_type == 'erdos_renyi':
            b_n = generator.erdos_renyi_n(**args)
        elif graph_type == 'euclidian':
            b_n = generator.euclidian_n(**args)
        elif graph_type == 'random_tree':
            b_n = generator.random_tree_n(**args)
        elif graph_type == 'path_star':
            b_n = generator.path_star_n(**args)
        elif graph_type == 'balanced':
            b_n = generator.balanced_n(**args)
        elif graph_type == 'khops_gen':  # kinda a graph if you squint
            b_n = generator.khops_gen_n(**args)
        elif graph_type == 'khops':  # kinda a graph if you squint
            b_n = generator.khops_n(**args)
        else:
            raise ValueError(f"Unknown graph type: {graph_type}")
        return b_n


    setattr(generator, "get_graph", get_graph)

    setattr(generator, "task_gather_ids", task_gather_ids)
    setattr(generator, "scratchpad_gather_ids", scratchpad_gather_ids)
    setattr(generator, "edge_gather_ids", edge_gather_ids)
    setattr(generator, "node_gather_ids", node_gather_ids)

    setattr(generator, "pprint_distance", pprint_distance)
    setattr(generator, "pprint_batched_dict", pprint_batched_dict)
    setattr(generator, 'create_reconstruct_graphs', create_reconstruct_graphs)

    def help_str():  # displays docstrings from cpp files with print(generator.help_str())
        # note this only works for the cpp functions not the added python functions above
        return pydoc.render_doc(generator, "\nDocstring for %s:")

    setattr(generator, "help_str", help_str)

    return generator


if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf, edgeitems=10, linewidth=np.inf, precision=2, suppress=True, )

    print(f'Running from CWD: {os.getcwd()}')
    generator = get_generator_module()
    print("C++ module `generator` loaded.")

    print(f'Random seed is {generator.get_seed()}')
    generator.set_seed(42)
    # generator.set_seed(3172477368)
    print(f'Random seed is {generator.get_seed()} after setting to 42')

    print(generator.help_str())

    # d = generator.erdos_renyi(15, -1.0, 75, 125, False, False, shuffle_edges=False)
    d = generator.euclidean(15, 2, -1, False, False, shuffle_edges=False)

    for k, v in d.items():
        print(f'{k}: {type(v)}')
        if isinstance(v, np.ndarray):
            print(f'\t {k}: {v.shape}, {v.dtype}')
            print(v)
        print()


