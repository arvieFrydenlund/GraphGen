import os
import sys
import time
import pydoc
import numpy as np

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
    parser.add_argument('--graph_type', type=str, default='euclidean')
    parser.add_argument('--min_num_nodes', type=int, default=10)
    parser.add_argument('--max_num_nodes', type=int, default=50)
    parser.add_argument('--p', type=float, default=-1.0,
                        help="Probability for edge creation for erdos_renyi graphs, -1.0 means random")
    parser.add_argument('--dims', type=int, default=2,
                        help="Number of dimensions for euclidean graphs")
    parser.add_argument('--radius', type=float, default=-1.0,
                        help="Radius for euclidean graphs, -1.0 means random")
    parser.add_argument('--c_min', type=int, default=75)
    parser.add_argument('--c_max', type=int, default=125)
    parser.add_argument('--min_num_arms', type=int, default=1,
                        help="Minimum number of arms for star graphs")
    parser.add_argument('--max_num_arms', type=int, default=5,
                        help="Maximum number of arms for star graphs")
    parser.add_argument('--min_arm_length', type=int, default=1,
                        help="Minimum length of each arm in star graphs")
    parser.add_argument('--max_arm_length', type=int, default=5,
                        help="Maximum length of each arm in star graphs")
    parser.add_argument('--min_lookahead', type=int, default=1,
                        help="Minimum lookahead for balanced graphs")
    parser.add_argument('--max_lookahead', type=int, default=3,
                        help="Maximum lookahead for balanced graphs")
    parser.add_argument('--min_noise_reserve', type=int, default=0,
                        help="Minimum noise reserve for balanced graphs")
    parser.add_argument('--max_num_parents', type=int, default=4,
                        help="Maximum number of parents for balanced graphs")

    # task settings
    parser.add_argument('--max_length', type=int, default=10)
    parser.add_argument('--min_length', type=int, default=1)
    parser.add_argument('--sample_target_paths', action='store_true', default=True)
    parser.add_argument('--max_query_length', type=int, default=-1)
    parser.add_argument('--min_query_length', type=int, default=2)
    parser.add_argument('--sample_center', action='store_true', default=False)
    parser.add_argument('--sample_centroid', action='store_true', default=False)

    # tokenization settings
    parser.add_argument('--is_causal', action='store_true', default=False)
    parser.add_argument('--shuffle_edges', action='store_true', default=False)
    parser.add_argument('--shuffle_nodes', action='store_true', default=False)
    parser.add_argument('--min_vocab', type=int, default=0)
    parser.add_argument('--max_vocab', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--max_edges', type=int, default=512)
    parser.add_argument('--max_attempts', type=int, default=1000)
    parser.add_argument('--concat_edges', action='store_true', default=True)
    parser.add_argument('--query_at_end', action='store_true', default=True)
    parser.add_argument('--num_thinking_tokens', type=int, default=0)
    parser.add_argument('--is_flat_model', action='store_true', default=True)
    parser.add_argument('--for_plotting', action='store_true', default=False)

    return parser


def determine_task(args):
    if args.sample_target_paths:
        task = 'min_path'


# Extra generator functions

class ReconstructedGraph(object):
    def __init__(self, graph_type, edge_list, pos=None, is_undirected=True):

        self.G = None

    def plot(self):
        pass

    def print(self):
        pass


def create_reconstruct_graphs(graph_type, batched_dict, for_plotting=False, ids=None):
    """
    Take the c++ output and reconstruct the graphs for plotting or further processing.

    :param batched_dict: c++ output dictionary
    :param for_plotting: bool
    :param ids: batch ids to reconstruct, if None, reconstruct all
    :return:
    """
    if for_plotting:
        raise NotImplementedError
    else:
        if d['is_flat_model']:

            src_tokens = d['src_tokens']
            src_lengths = d['src_lengths']
            graph_start_indices = d['graph_start_indices']
            graph_lengths = d['graph_lengths']
            task_start_indices = d['task_start_indices']

            if ids is None:
                ids = list(range(src_tokens[0]))

            edge_lists = []

            if src_tokens.shape[-1] == 1:  # not concat_edges
                pass
            elif src_tokens.shape[-1] == 2:  # concat_edges
                pass
            else:
                raise ValueError(f"Unexpected src_tokens shape: {src_tokens.shape[-1]}")





        else:
            raise NotImplementedError


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
    d = generator.euclidian(15, 2, -1, False, False, shuffle_edges=False)

    for k, v in d.items():
        print(f'{k}: {type(v)}')
        if isinstance(v, np.ndarray):
            print(f'\t {k}: {v.shape}, {v.dtype}')
            print(v)
        print()


