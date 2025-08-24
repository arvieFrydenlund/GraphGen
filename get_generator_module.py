import os
import sys
import time
import pydoc
import numpy as np

try:
    import networkx as nx
except ImportError:
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
    parser.add_argument('--min_num_nodes', type=int, default=20)
    parser.add_argument('--max_num_nodes', type=int, default=25)
    parser.add_argument('--c_min', type=int, default=75)  # for graphs with multiple connected components
    parser.add_argument('--c_max', type=int, default=125)

    parser.add_argument('--p', type=float, default=-1.0,
                        help="Probability for edge creation for erdos_renyi graphs, -1.0 means random")

    parser.add_argument('--dims', type=int, default=2,
                        help="Number of dimensions for euclidean graphs")
    parser.add_argument('--radius', type=float, default=-1.0,
                        help="Radius for euclidean graphs, -1.0 means random")

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
    parser.add_argument('--task_type', type=str, default='shortest_path') #'shortest_path')  # 'center'  # 'centroid'
    parser.add_argument('--max_path_length', type=int, default=12)
    parser.add_argument('--min_path_length', type=int, default=3)
    parser.add_argument('--max_query_size', type=int, default=10)
    parser.add_argument('--min_query_size', type=int, default=2)

    # tokenization settings
    parser.add_argument('--is_causal', action='store_true', default=False)
    parser.add_argument('--shuffle_edges', action='store_false', default=True)
    parser.add_argument('--shuffle_nodes', action='store_false', default=True)
    parser.add_argument('--dont_shuffle_edges', action='store_false', dest='shuffle_edges')
    parser.add_argument('--dont_shuffle_nodes', action='store_false', dest='shuffle_nodes')
    parser.add_argument('--min_vocab', type=int, default=22)
    parser.add_argument('--max_vocab', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--max_edges', type=int, default=512)
    parser.add_argument('--max_attempts', type=int, default=1000)
    parser.add_argument('--concat_edges', action='store_true', default=True)
    parser.add_argument('--dont_concat_edges', action='store_false', dest='concat_edges')
    parser.add_argument('--query_at_end', action='store_true', default=False)
    parser.add_argument('--num_thinking_tokens', type=int, default=0)
    parser.add_argument('--is_decoder_model', action='store_true', dest='is_flat_model', default=True)
    parser.add_argument('--is_encoder_model', action='store_false', dest='is_flat_model')
    parser.add_argument('--for_plotting', action='store_true', default=False)

    return parser



# Extra generator functions

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
        if self.pos is not None:
            try:
                if self.graph_type in ('path_star', 'balanced'):
                    pos = nx.nx_agraph.graphviz_layout(self.G, prog="twopi")
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


def create_reconstruct_graphs(batched_dict, symbol_to_id, for_plotting=False, ids=None):
    """
    Take the c++ output and reconstruct the graphs for plotting and sanity checking.

    :param batched_dict: c++ output dictionary
    :param symbol_to_id: mapping from symbols to tokenized ids
    :param for_plotting: bool
    :param ids: batch ids to reconstruct, if None, reconstruct all
    :return:
    """

    reconstructions = []

    id_to_symbol_ = {v: k for k, v in symbol_to_id.items()}
    def id_to_symbol(id):
        symbol = id_to_symbol_[id]
        if symbol.isdigit():
            return int(symbol)
        return symbol

    pad = symbol_to_id.get('<pad>', -1)

    edge_list, query, task = None, None, None
    if for_plotting:
        raise NotImplementedError
    else:
        src_tokens = batched_dict['src_tokens']
        src_lengths = batched_dict['src_lengths']
        query_start_indices = batched_dict['query_start_indices']
        query_lengths = batched_dict['query_lengths']
        graph_start_indices = batched_dict['graph_start_indices']
        graph_lengths = batched_dict['graph_lengths']
        graph_gather_indices = batched_dict['graph_gather_indices']
        task = batched_dict['prev_output_tokens']
        task_start_indices = batched_dict['task_start_indices']
        task_lengths = batched_dict['task_lengths']
        task_gather_indices = batched_dict['task_gather_indices']

        print('is_flat_model', batched_dict['is_flat_model'])

        if ids is None:
            ids = list(range(src_tokens.shape[0]))

        if src_tokens.shape[-1] == 1:  # not concat_edges
            concat_edges = False
        elif src_tokens.shape[-1] == 2:  # concat_edges
            concat_edges = True
        else:
            raise ValueError(f"Unexpected src_tokens shape: {src_tokens.shape[-1]}")

        if batched_dict['task_type'] == 'shortest_path':
            pass
        elif batched_dict['task_type'] in ('center', 'centroid'):
            pass

        pos = None
        if batched_dict['graph_type'] == 'euclidean':
            pos = batched_dict['positions']

        for id in ids:
            print('\n\nID: ', id)
            print('src_tokens[id]', src_tokens[id, :, 0] if src_tokens.ndim > 2 else src_tokens[id, :])
            print('range         ', np.arange(src_tokens.shape[1]))
            print('query_start_indices[id]',  query_start_indices[id], 'length ', query_lengths[id])
            print('graph_start_indices[id]', graph_start_indices[id], 'length ', graph_lengths[id])
            print('graph_gather_indices[id]', graph_gather_indices[id])
            print('task_start_indices[id]', task_start_indices[id], 'length ', task_lengths[id])
            print('task_gather_indices[id]', task_gather_indices[id], 'src_size', src_tokens.shape[1])

            query = src_tokens[id, query_start_indices[id]: query_start_indices[id] + query_lengths[id], 0]
            edge_list = src_tokens[id, graph_start_indices[id]: graph_start_indices[id] + graph_lengths[id], :]
            if batched_dict['is_flat_model']:
                task_input = src_tokens[id, task_start_indices[id]: task_start_indices[id] + task_lengths[id], 0]
            else:
                task_input = task[id, :task_lengths[id], 0]
            prev_output_tokens = task[id, :task_lengths[id], :]
            if not concat_edges:
                edge_list = edge_list.reshape(-1, 3)[:, :2]  # remove the edge marker
            edge_list = edge_list.tolist()
            for i in range(len(edge_list)):
                edge_list[i] = [id_to_symbol(e) for e in edge_list[i]]
            query = query.tolist()  # remove the tokenized ids
            print('query', query)
            query = [id_to_symbol(q) for q in query]
            task_input = task_input.tolist()
            task_input = [id_to_symbol(t) for t in task_input]
            task_targets = [[id_to_symbol(j) for j in t if j != pad] for t in prev_output_tokens]
            pos_i = {id_to_symbol(int(pos[id, i, 0])): (pos[id, i, 1], pos[id, i, 2])
                     for i in range(pos.shape[1]) if pos[id, i, 0] > 0} if pos is not None else None
            if pos_i is not None:
                pos_i = dict(sorted(pos_i.items(), key=lambda item: item[0]))

            print('edge_list', edge_list)
            r = ReconstructedGraph(batched_dict['graph_type'], batched_dict['task_type'],
                                   edge_list, query, task_input, task_targets, pos=pos_i)
            reconstructions.append(r)


    return reconstructions

def get_generator_module(cpp_files=('undirected_graphs.h', 'directed_graphs.h', 'utils.h', 'generator.cpp'),
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


