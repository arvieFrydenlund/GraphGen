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
    d = generator.euclidean(15, 2, -1, False, False, shuffle_edges=False)

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

    d_n = generator.euclidean_n(50, 56, 2, -1, is_causal=True,
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

    d_n = generator.euclidean_n(50, 55, 2, -1, is_causal=True, min_vocab=4, max_vocab=59, shuffle_edges=True, batch_size=8,
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

    d_n = generator.euclidean_n(min_num_nodes, max_num_nodes,
                                is_causal=True, shuffle_nodes=True,
                                min_vocab=num_special_tokens, max_vocab=max_num_nodes+num_special_tokens,
                                shuffle_edges=True,
                                batch_size=3,
                                is_flat_model=True, concat_edges=True, query_at_end=False)
    batch_pprint(d_n, title='Flat model with concat edges and query at start')


    d_n = generator.euclidean_n(min_num_nodes, max_num_nodes,
                                is_causal=True, shuffle_nodes=True,
                                min_vocab=num_special_tokens, max_vocab=max_num_nodes+num_special_tokens,
                                shuffle_edges=True,
                                batch_size=3,
                                is_flat_model=True, concat_edges=False, query_at_end=False)
    batch_pprint(d_n, title='Flat model without concat edges and query at start')


    d_n = generator.euclidean_n(min_num_nodes, max_num_nodes,
                                is_causal=True, shuffle_nodes=True,
                                min_vocab=num_special_tokens, max_vocab=max_num_nodes+num_special_tokens,
                                shuffle_edges=True,
                                batch_size=3,
                                is_flat_model=True, concat_edges=True, query_at_end=True)
    batch_pprint(d_n, title='Flat model with concat edges and query at end')


def get_batch(args, batch_size=20):
    if batch_size is not None:
        args.batch_size = batch_size

    if args.graph_type == 'erdos_renyi':
        d_n = generator.erdos_renyi_n(**vars(args))
    elif args.graph_type == 'euclidean':
        d_n = generator.euclidean_n(**vars(args))
    elif args.graph_type == 'random_tree':
        d_n = generator.random_tree_n(**vars(args))
    elif args.graph_type == 'path_star':
        d_n = generator.path_star_n(**vars(args))
    elif args.graph_type == 'balanced':
        d_n = generator.balanced_n(**vars(args))
    else:
        raise NotImplementedError
    return d_n

def _t_reconstruct(args, d, batch_size=20, plot=True):

    np.set_printoptions(threshold=np.inf)

    args.task_type = 'shortest_path'
    d_n = get_batch(args, batch_size)

    for k, v in d_n.items():
        if isinstance(v, np.ndarray):
            print(f'{k}: {v.shape}, {v.dtype}')
        else:
            print(f'{k}: {v}, {type(v)}')

    reconstructions = generator.create_reconstruct_graphs(d_n, d)
    for r in reconstructions:
        if plot:
            r.plot()

def _t_verify_paths(args, d, batch_size=20):
    args.task_type = 'shortest_path'
    # given distance matrices, make fake paths and verify them
    d_n = get_batch(args, batch_size)
    distances = d_n['distances']
    print(distances.shape)

    src_tokens = d_n['src_tokens']
    task_start_indices = d_n['task_start_indices'] + 1
    task_lengths = d_n['task_lengths'] - 2

    query_start_indices = d_n['query_start_indices'] + 1

    gt_paths = np.ones((src_tokens.shape[0], max(task_lengths)), dtype=np.int32)
    queries = np.ones((src_tokens.shape[0], 2), dtype=np.int32)
    for b in range(src_tokens.shape[0]):
        gt_paths[b, :task_lengths[b]] = src_tokens[b, task_start_indices[b]:task_start_indices[b] + task_lengths[b], 0]
        queries[b, 0] = src_tokens[b, query_start_indices[b], 0]
        queries[b, 1] = src_tokens[b, query_start_indices[b] + 1, 0]

    print(gt_paths, 'should be all ones')
    # print(distances[0])
    verify = generator.verify_paths(distances, queries, gt_paths, task_lengths)
    print(verify)




    # generate incorrect paths

def _concat(first, second):
    a = np.arange(first.shape[0])
    if first.ndim == 1:
        return np.stack([first, second, a], axis=-1).transpose(1, 0)
    else:
        return np.concatenate([first, np.stack([second, a], axis=-1)], axis=-1).transpose(1, 0)

def _t_positions(args, d, batch_size=7):

    args.align_prefix_front_pad = True
    args.concat_edges = False # True # False
    d_n = get_batch(args, batch_size)
    generator.set_default_pos_dictionary()
    src_tokens = d_n['src_tokens']
    graph_start_indices = d_n['graph_start_indices']
    graph_lengths = d_n['graph_lengths']
    query_start_indices = d_n['query_start_indices']
    pos_ids, task_start_pos = generator.get_position_ids(**d_n, mask_edges=False, use_task_structure=False, use_graph_structure=False)
    print('Regular position ids:')
    for b in range(pos_ids.shape[0]):
        print(f'Batch {b} position ids:')
        print(src_tokens[b, :, 0])
        print(pos_ids[b])

    print('Masking edges')
    pos_ids, task_start_pos = generator.get_position_ids(**d_n, mask_edges=True, use_task_structure=False, use_graph_structure=False)
    for b in range(pos_ids.shape[0]):
        print(f'Batch {b} position ids:')
        print(_concat(src_tokens[b, :, 0], pos_ids[b]))
        print(graph_start_indices[b], graph_lengths[b], query_start_indices[b])


    print('Using task structure')
    pos_ids, task_start_pos = generator.get_position_ids(**d_n, mask_edges=False, use_task_structure=True, use_graph_structure=False)
    for b in range(pos_ids.shape[0]):
        print(f'Batch {b} position ids:')
        print(_concat(src_tokens[b, :, 0], pos_ids[b]))
        print(graph_start_indices[b], graph_lengths[b], query_start_indices[b])


    print('Using graph structure')
    pos_ids, task_start_pos = generator.get_position_ids(**d_n, mask_edges=False, use_task_structure=True, use_graph_structure=True)
    for b in range(pos_ids.shape[0]):
        print(f'Batch {b} position ids:')
        print(src_tokens[b, :, 0]),
        print(pos_ids[b].transpose(1, 0))
        print(graph_start_indices[b], graph_lengths[b], query_start_indices[b])
        print(task_start_pos[b])


def _t_positions2(args, d, batch_size=7):
    args.align_prefix_front_pad = True
    args.concat_edges = True # True # False
    args.include_nodes_in_graph_tokenization = True

    d_n = get_batch(args, batch_size)
    generator.set_default_pos_dictionary()
    src_tokens = d_n['src_tokens']
    graph_start_indices = d_n['graph_start_indices']
    graph_lengths = d_n['graph_lengths']
    query_start_indices = d_n['query_start_indices']
    pos_ids, task_start_pos = generator.get_position_ids(**d_n, use_edges_invariance=True, use_node_invariance=True, use_graph_invariance=True,
                                                         use_task_structure=True, use_graph_structure=False)
    for b in range(pos_ids.shape[0]):
        print(f'Batch {b} position ids:')
        print(_concat(src_tokens[b], pos_ids[b]))



if __name__ == '__main__':

    parser = generator.get_args_parser()
    args = parser.parse_args()

    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')


    # h = pydoc.render_doc(generator, "Help on %s")
    # print(generator.help_str() + '\n\n\n')
    print('\n\nStarting Testing')

    print(f'Random seed is {generator.get_seed()}')
    generator.set_seed(42)
    # generator.set_seed(3172477368)
    print(f'Random seed is {generator.get_seed()} after setting to 42')

    print("Setting dictionary")
    generator.set_default_dictionary(args.max_vocab)
    d = generator.get_dictionary()
    # sort by value
    d = dict(sorted(d.items(), key=lambda item: item[1]))
    print('Dictionary ...')
    for k, v in d.items():
        print(f'{k}: {v}')
    print()

    #_t_batched_graphs_for_plotting_and_hashes()
    # _t_batched_graphs_flat_model()

    # args.graph_type = 'random_tree'
    # args.start_at_root = True
    # args.align_prefix_front_pad = True
    # _t_reconstruct(args, d)
    # _t_verify_paths(args, d)

    _t_positions2(args, d, batch_size=20)

    print('\n\nDone Testing')