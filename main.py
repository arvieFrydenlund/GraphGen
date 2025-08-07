import time
import pydoc
import numpy as np

# np.finfo(np.dtype("float32"))  # gets rid of warnings, hope they aint important
# np.finfo(np.dtype("float64"))
np.set_printoptions(threshold=np.inf, edgeitems=10, linewidth=np.inf, precision=2, suppress=True,)

from graph_datasets import GeneratorParser, batch_pprint

from get_generator_module import get_generator_module
get_generator_module()
import generator


"""
Testing generation functions and pybind compile. 
"""

def _t_single_graph():
    # d = generator.erdos_renyi(15, -1.0, 75, 125, False, False, shuffle_edges=False)
    d = generator.euclidian(15, 2, -1, False, False, shuffle_edges=False)

    for k, v in d.items():
        print(f'{k}: {type(v)}')
        if isinstance(v, np.ndarray):
            print(f'\t {k}: {v.shape}, {v.dtype}')
            print(v)
        print()
    print('\n\nTesting N generator')
    print(f'Test size is {generator.get_test_size()}')

def _t_batched_verify_paths():
    # todo
    pass

def _t_batched_graphs_for_plotting_and_hashes():

    d_n = generator.euclidian_n(50, 56, 2, -1, is_causal=True,
                                shuffle_nodes=True, min_vocab=4, max_vocab=60, shuffle_edges=True, batch_size=8,
                                for_plotting=True)

    batch_pprint(d_n)

    print(f'Settings hashes')
    hashes = d_n['hashes']
    generator.set_test_hashes(hashes)
    print(f'Test size is {generator.get_test_size()}')
    print(f'Is in same: {generator.is_in_test(hashes)}')
    hashes[1] = 0  # fake a different hash
    print(f'Is in with 2nd faked: {generator.is_in_test(hashes)}')

    d_n = generator.euclidian_n(50, 55, 2, -1, is_causal=True, min_vocab=4, max_vocab=59, shuffle_edges=True, batch_size=8,
                                for_plotting=True)
    print(f"Is in with new hashes: {generator.is_in_test(d_n['hashes'])}")

    print('\n\nSetting dictionary')
    fake_dict = {str(i): i for i in range(12)}
    generator.set_dictionary(fake_dict, verbose=True)


def _t_batched_graphs_flat_model():

    print('\n\nSetting default dictionary')
    generator.set_default_dictionary()

    min_num_nodes = 25
    max_num_nodes = 30
    num_special_tokens = 25

    d_n = generator.euclidian_n(min_num_nodes, max_num_nodes,
                                is_causal=True, shuffle_nodes=True,
                                min_vocab=num_special_tokens, max_vocab=max_num_nodes+num_special_tokens,
                                shuffle_edges=True,
                                batch_size=3,
                                is_flat_model=True, concat_edges=True, query_at_end=False)
    batch_pprint(d_n, title='Flat model with concat edges and query at start')


    d_n = generator.euclidian_n(min_num_nodes, max_num_nodes,
                                is_causal=True, shuffle_nodes=True,
                                min_vocab=num_special_tokens, max_vocab=max_num_nodes+num_special_tokens,
                                shuffle_edges=True,
                                batch_size=3,
                                is_flat_model=True, concat_edges=False, query_at_end=False)
    batch_pprint(d_n, title='Flat model without concat edges and query at start')


    d_n = generator.euclidian_n(min_num_nodes, max_num_nodes,
                                is_causal=True, shuffle_nodes=True,
                                min_vocab=num_special_tokens, max_vocab=max_num_nodes+num_special_tokens,
                                shuffle_edges=True,
                                batch_size=3,
                                is_flat_model=True, concat_edges=True, query_at_end=True)
    batch_pprint(d_n, title='Flat model with concat edges and query at end')



def _t_reconstruct(args):
    pass




if __name__ == '__main__':

    import argparse


    h = pydoc.render_doc(generator, "Help on %s")
    print(h + '\n\n\n')
    print('\n\nStarting Testing')

    print(f'Random seed is {generator.get_seed()}')
    generator.set_seed(42)
    # generator.set_seed(3172477368)
    print(f'Random seed is {generator.get_seed()} after setting to 42')

    #_t_batched_graphs_for_plotting_and_hashes()
    # _t_batched_graphs_flat_model()

    """
    py::arg("min_num_nodes"),
          py::arg("max_num_nodes"),
          py::arg("p") = -1.0,
          py::arg("c_min") = 75,
          py::arg("c_max") = 125,
          py::arg("max_length") = 10,
          py::arg("min_length") = 1,
          py::arg("sample_target_paths") = true,
          py::arg("max_query_length") = -1,
          py::arg("min_query_length") = 2,
          py::arg("sample_center") = false,
          py::arg("sample_centroid") = false,
          py::arg("is_causal") = false,
          py::arg("shuffle_edges") = false,
          py::arg("shuffle_nodes") = false,
          py::arg("min_vocab") = 0,
          py::arg("max_vocab") = -1,
          py::arg("batch_size") = 256,
          py::arg("max_edges") = 512,
          py::arg("max_attempts") = 1000,
          py::arg("concat_edges") = true,
          py::arg("query_at_end") = true,
          py::arg("num_thinking_tokens") = 0,
          py::arg("is_flat_model") = true,
          py::arg("for_plotting") = false);
    """

    parser = argparse.ArgumentParser(description='Test generator functions')
    parser.add_argument('--graph_type', type=str, default='euclidean')
    parser.add_argument('--max_num_nodes', type=int, default=50)
    parser.add_argument('--p', type=float, default=-1.0)
    parser.add_argument('--c_min', type=int, default=75)
    parser.add_argument('--c_max', type=int, default=125)
    parser.add_argument('--max_length', type=int, default=10)
    parser.add_argument('--min_length', type=int, default=1)
    parser.add_argument('--sample_target_paths', action='store_true', default=True)
    parser.add_argument('--max_query_length', type=int, default=-1)
    parser.add_argument('--min_query_length', type=int, default=2)





    _t_reconstruct()



    print('\n\nDone Testing')