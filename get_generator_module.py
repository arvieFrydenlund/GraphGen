import os
import sys
import time
import pydoc
import numpy as np


# Extra generator functions

class ReconstructedGraph(object):
    def __init__(self, edge_list, pos=None, is_undirected=True):

        self.G = None

    def plot(self):
        pass

    def print(self):
        pass


def create_reconstruct_graphs(batched_dict, for_plotting=False, ids=None, generator_=None):
    """
    Take the c++ output and reconstruct the graphs for plotting or further processing.

    :param batched_dict: c++ output dictionary
    :param for_plotting: bool
    :param ids: batch ids to reconstruct, if None, reconstruct all
    :param generator_:
    :return:
    """
    if for_plotting:
        pass
    else:
        if d['is_flat_model']:
            """num_attempts: <class 'int'> 0
            vocab_min_size: <class 'int'> 25
            vocab_max_size: <class 'int'> 55
            success: <class 'bool'> True
            concat_edges: <class 'bool'> True
            query_at_end: <class 'bool'> False
            num_thinking_tokens: <class 'int'> 0
            is_flat_model: <class 'bool'> True
            for_plotting: <class 'bool'> False
            src_tokens: <class 'numpy.ndarray'> 	src_tokens: (3, 62, 2), int32
            src_lengths: <class 'numpy.ndarray'> 	src_lengths: (3,), int32
            prev_output_tokens: <class 'numpy.ndarray'> 	prev_output_tokens: (3, 11, 2), int32
            task_lengths: <class 'numpy.ndarray'> 	task_lengths: (3,), int32
            query_start_indices: <class 'numpy.ndarray'> 	query_start_indices: (3,), int32
            query_lengths: <class 'numpy.ndarray'> 	query_lengths: (3,), int32
            graph_start_indices: <class 'numpy.ndarray'> 	graph_start_indices: (3,), int32
            graph_lengths: <class 'numpy.ndarray'> 	graph_lengths: (3,), int32
            task_start_indices: <class 'numpy.ndarray'> 	task_start_indices: (3,), int32
            distances: <class 'numpy.ndarray'> 	distances: (3, 55, 55), int32
            hashes: <class 'numpy.ndarray'> 	hashes: (3,), uint64
            ground_truths: <class 'numpy.ndarray'> 	ground_truths: (3, 45, 55), int32
            positions:"""



            src_tokens = d['src_tokens']
            graph_start_indices = d['graph_start_indices']
            graph_lengths = d['graph_lengths']

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
            pass


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

    def help_string():  # displays docstrings from cpp files with print(generator.help_str())
        return pydoc.render_doc(generator, "\nDocstring for %s:")

    setattr(generator, "help_str", help_string)

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


