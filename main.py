import time
import pydoc
import numpy as np
import copy

from PyQt5.pyrcc_main import verbose

# np.finfo(np.dtype("float32"))  # gets rid of warnings, hope they aint important
# np.finfo(np.dtype("float64"))
np.set_printoptions(threshold=np.inf, edgeitems=10, linewidth=np.inf, precision=2, suppress=True, )

from graph_datasets import GeneratorParser, batch_pprint

from get_generator_module import get_generator_module

get_generator_module()
import generator  # ignore warnings, get_generator_module sets this up and it is needed for the rest of the code to work

"""
Testing generation functions and pybind compile. 
"""


def _graph_print(args, token_dict, pos_dict,
                 graph_type='path_star', # 'random_tree',  #'erdos_renyi',  #'path_star',
                 task_type ='shortest_path', concat_edges=False, duplicate_edges=False,
                 include_nodes_in_graph_tokenization=True, query_at_end=False, num_thinking_tokens=0,
                 scratchpad_type='bfs', use_unique_depth_markers=True,
                 scratchpad_as_prefix=False, no_graph=True,
                 align_prefix_front_pad=False, use_graph_invariance=False, use_task_structure=False,
                 use_graph_structure=True, use_full_structure=True,
                 batch_size=3):
    args.min_num_nodes = 25
    args.max_num_nodes = 25
    args.min_path_length = 3
    args.max_path_length = 5
    args.graph_type = graph_type
    args.task_type = task_type
    args.concat_edges = concat_edges
    args.duplicate_edges = duplicate_edges
    args.include_nodes_in_graph_tokenization = include_nodes_in_graph_tokenization
    args.query_at_end = query_at_end
    args.num_thinking_tokens = num_thinking_tokens
    args.scratchpad_type = scratchpad_type
    args.scratchpad_as_prefix = scratchpad_as_prefix
    args.no_graph = no_graph
    args.use_unique_depth_markers = use_unique_depth_markers
    args.align_prefix_front_pad = align_prefix_front_pad
    args.use_graph_invariance = use_graph_invariance
    args.use_task_structure = use_task_structure
    args.use_graph_structure = use_graph_structure
    args.use_full_structure = use_full_structure

    b_n = generator.get_graph(args, batch_size=batch_size)
    generator.pprint_batched_dict(b_n, token_dict, pos_dict, idxs=-1, print_dist=True)


def _graph_plot(args, token_dict, pos_dict,
                 graph_type='path_star', # 'random_tree',  #'erdos_renyi',  #'path_star',
                 task_type ='shortest_path', concat_edges=True, duplicate_edges=False,
                 include_nodes_in_graph_tokenization=True, query_at_end=False, num_thinking_tokens=0,
                 scratchpad_type='none', use_unique_depth_markers=True,
                 scratchpad_as_prefix=False,
                 align_prefix_front_pad=False, use_graph_invariance=False, use_task_structure=False,
                 use_graph_structure=True, use_full_structure=True,
                 batch_size=1):
    args.min_num_nodes = 190
    args.max_num_nodes = 190
    args.min_path_length = 10
    args.max_path_length = 15
    args.graph_type = graph_type
    args.task_type = task_type
    args.concat_edges = concat_edges
    args.duplicate_edges = duplicate_edges
    args.include_nodes_in_graph_tokenization = include_nodes_in_graph_tokenization
    args.query_at_end = query_at_end
    args.num_thinking_tokens = num_thinking_tokens
    args.scratchpad_type = scratchpad_type
    args.scratchpad_as_prefix = scratchpad_as_prefix
    args.no_graph = False
    args.use_unique_depth_markers = use_unique_depth_markers
    args.align_prefix_front_pad = align_prefix_front_pad
    args.use_graph_invariance = use_graph_invariance
    args.use_task_structure = use_task_structure
    args.use_graph_structure = use_graph_structure
    args.use_full_structure = use_full_structure

    b_n = generator.get_graph(args, batch_size=batch_size)
    generator.pprint_batched_dict(b_n, token_dict, pos_dict, idxs=-1, print_dist=False)

    print('Creating reconstructed graph plots')
    trees_to_left = False
    reconstructions = generator.create_reconstruct_graphs(b_n, token_dict, for_plotting=False, ids=None,
                                                          trees_to_left=trees_to_left, verbose=True)
    for i, recon in enumerate(reconstructions):
        recon.plot(verbose=True)


# KHOPS
def _t_khops(args, token_dict, pos_dict, right_side_connect=True, permutation_version=False, mask_to_vocab_size=False, batch_size=3):
    args.task_type = "khops"
    args.right_side_connect = right_side_connect
    args.permutation_version = permutation_version
    args.mask_to_vocab_size = mask_to_vocab_size
    args.batch_size = batch_size

    b_n = generator.khops_n(**vars(args))

    print('KHOPS batch:')
    generator.pprint_batched_dict(b_n, token_dict, pos_dict, idxs=-1, print_dist=False)


def _t_int_partition(Q=200, N=9, num=10000):
    print('Testing integer partitioning')
    for _i in range(num):
        partition = generator.uniform_random_int_partition(Q, N, shuffle=True)
        print(f'Partition of {Q} into {N} parts: {partition}, sum: {sum(partition)}')


def _t_khops_gen(args, token_dict, pos_dict, right_side_connect=True, khops_no_repeats=True, batch_size=20):
    args.task_type = "khops_gen"
    args.right_side_connect = right_side_connect
    args.khops_no_repeats = khops_no_repeats
    args.batch_size = batch_size

    b_n = generator.khops_gen_n(**vars(args))
    generator.pprint_batched_dict(b_n, token_dict, pos_dict, idxs=-1, print_dist=False)

    # verify
    src = b_n['src_tokens'].squeeze(-1)
    task_start_indices = b_n['task_start_indices']
    task_lengths = b_n['task_lengths']
    max_prefix = np.max(task_start_indices)
    prefixes = np.ones((src.shape[0], max_prefix), dtype=src.dtype)
    for b in range(src.shape[0]):
        prefixes[b, :task_start_indices[b]] = src[b, :task_start_indices[b]]
    ground_truths = np.ones((src.shape[0], task_lengths.max() - 2), dtype=src.dtype)
    for b in range(src.shape[0]):
        ground_truths[b, :task_lengths[b]-2] = src[b, task_start_indices[b]+1:task_start_indices[b]+task_lengths[b]-1]

    print('Verifying khops_gen outputs')
    verify = generator.verify_khop_gens(prefixes, task_start_indices, ground_truths, task_lengths-2, right_side_connect)
    print(verify)


# Scratchpads
def _t_bfs_task(args, token_dict, pos_dict, use_unique_depth_markers=True, batch_size=20):
    args.batch_size = batch_size
    args.task_type = 'bfs'
    args.scratchpad_type = 'none'
    args.use_unique_depth_markers = use_unique_depth_markers
    b_n = generator.get_graph(args, batch_size=batch_size)
    generator.pprint_batched_dict(b_n, token_dict, pos_dict, idxs=-1, print_dist=False)


def _t_scratchpad_validation(args, token_dict, pos_dict, use_unique_depth_markers=True, batch_size=20, scratchpad_type='bfs'):
    args.batch_size = batch_size
    args.task_type = 'shortest_path'
    args.scratchpad_type = scratchpad_type
    args.use_unique_depth_markers = use_unique_depth_markers
    b_n = generator.get_graph(args, batch_size=batch_size)
    # generator.pprint_batched_dict(b_n, token_dict, pos_dict, idxs=-1, print_dist=False)

    # construct inputs to verify_bfs_gens
    distances = b_n['distances']
    queries = np.zeros((batch_size, 2), dtype=np.int32)
    lengths = b_n['scratch_pad_lengths']  # add noise?
    gens = np.zeros((batch_size, lengths.max()), dtype=np.int32)
    for b in range(batch_size):
        query_start = b_n['query_start_indices'][b] + 1
        queries[b, 0] = b_n['src_tokens'][b, query_start, 0]
        queries[b, 1] = b_n['src_tokens'][b, query_start + 1, 0]

        scratchpad_start = b_n['scratch_pad_start_indices'][b] + 1
        scratchpad_length = b_n['scratch_pad_lengths'][b] - 1 # exclude start of scatchpad
        gens[b, :scratchpad_length] = b_n['src_tokens'][b,
                                        scratchpad_start:scratchpad_start + scratchpad_length, 0]
        if b >= batch_size // 2:
            # add some errors
            r = np.random.randint(0, scratchpad_length)
            gens[b, r] = gens[b, r] + 1  # wrong node id

    generator.pprint_batched_dict(b_n, token_dict, pos_dict, idxs=-1, print_shapes=True)
    if scratchpad_type in ('bfs', ):
        out = generator.verify_bfs_gens(distances, queries, gens, lengths, check_special_tokens=True)
        print(f'BFS verify output: {out[:batch_size//2]} should be all 1s and {out[batch_size//2:]} should be < 1s')
    elif scratchpad_type in ('dfs',):
        #out = generator.verify_dfs_gens(distances, queries, gens, lengths, use_unique_depth_markers=use_unique_depth_markers)
        # print(f'DFS verify output: {out[:batch_size//2]} should be all 1s and {out[batch_size//2:]} should be < 1s')
        pass


def _t_random_trees(args, token_dict, pos_dict, batch_size=20):
    args.batch_size = batch_size
    args.min_num_nodes = 100
    args.max_num_nodes = 100
    args.task_type = 'shortest_path'
    args.graph_type = 'random_tree'

    print('With bernoulli_p')
    b_n = generator.get_graph(args, batch_size=batch_size)
    generator.pprint_batched_dict(b_n, token_dict, pos_dict, idxs=-1, print_dist=False)

    args.probs = [0.1, 0.4, 0.5]
    print('With probs')
    b_n = generator.get_graph(args, batch_size=batch_size)
    generator.pprint_batched_dict(b_n, token_dict, pos_dict, idxs=-1, print_dist=False)

def _distance_scores(args, token_dict, pos_dict,
                 graph_type='random_tree', # 'random_tree',  #'erdos_renyi',  #'path_star',
                 task_type ='shortest_path', concat_edges=False, duplicate_edges=False,
                 include_nodes_in_graph_tokenization=True, query_at_end=False, num_thinking_tokens=0,
                 scratchpad_type='none', use_unique_depth_markers=True,
                 scratchpad_as_prefix=False, no_graph=False,
                 align_prefix_front_pad=False, use_graph_invariance=False, use_task_structure=False,
                 use_graph_structure=True, use_full_structure=True,
                 batch_size=3):
    """
    The outcome of this is that the only valid score is dot product with a one-hot vector
    This literally selects the true distance element and doesn't really test anything about the geometry of the space.
    """

    args.min_num_nodes = 25
    args.max_num_nodes = 25
    args.min_path_length = 3
    args.max_path_length = 5
    args.graph_type = graph_type
    args.task_type = task_type
    args.concat_edges = concat_edges
    args.duplicate_edges = duplicate_edges
    args.include_nodes_in_graph_tokenization = include_nodes_in_graph_tokenization
    args.query_at_end = query_at_end
    args.num_thinking_tokens = num_thinking_tokens
    args.scratchpad_type = scratchpad_type
    args.scratchpad_as_prefix = scratchpad_as_prefix
    args.no_graph = no_graph
    args.use_unique_depth_markers = use_unique_depth_markers
    args.align_prefix_front_pad = align_prefix_front_pad
    args.use_graph_invariance = use_graph_invariance
    args.use_task_structure = use_task_structure
    args.use_graph_structure = use_graph_structure
    args.use_full_structure = use_full_structure

    b_n = generator.get_graph(args, batch_size=batch_size)
    generator.pprint_batched_dict(b_n, token_dict, pos_dict, idxs=-1, print_dist=True)

    dist = b_n['distances'][0, ...]
    mask = dist < 0
    dist = np.where(mask, 0, dist)
    print(dist)

    start_node = b_n['src_tokens'][0, b_n['query_start_indices'][0]+1, 0]
    print(f'Start node: {start_node}')

    # order nodes by distance from start node
    start_column = dist[start_node, :]
    print(f'Start column: {start_column}')
    order = np.argsort(start_column, axis=-1)
    print(order)

    def euclidean_dist(u, v):
        return -np.sqrt(np.sum((u - v) ** 2))

    def manhattan_dist(u, v):
        return -np.sum(np.abs(u - v))

    def cosine_sim(u, v):
        return 1-np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

    def dot_product(u, v):
        return -np.dot(u, v)

    invert_map = {v: k for k, v in token_dict.items()}

    distance_orders = {'euclidean': [], 'manhattan': [], 'dot_product': [], 'cosine': []}

    for other in order:
        if not mask[start_node, other]:
            s = invert_map[start_node]
            o = invert_map[other]
            su = start_column
            if True:  # one hot, this just selects the true distance, which is actually correct
                su = np.zeros_like(start_column)
                su[start_node] = 1
            du = dist[other, :]

            print(f'Node {o} is at distance {start_column[other]} from start node {s}')
            print(f'Node {o} is at euclidean distance {euclidean_dist(su, du):.2f} in embedding space from start node {s}')
            print(f'Node {o} is at manhattan distance {manhattan_dist(su, du):.2f} in embedding space from start node {s}')
            print(f'Node {o} is at dot product {dot_product(su, du):.2f} in embedding space from start node {s}')
            print(f'Node {o} is at cosine similarity {cosine_sim(su, du):.2f} in embedding space from start node {s}')
            print()

            distance_orders['euclidean'].append((start_column[other], euclidean_dist(su, du)))
            distance_orders['manhattan'].append((start_column[other], manhattan_dist(su, du)))
            distance_orders['dot_product'].append((start_column[other], dot_product(su, du)))
            distance_orders['cosine'].append((start_column[other], cosine_sim(su, du)))

    for key in distance_orders:
        # make sure the order respects the true distance, that is if dist(u, v) < dist(u, w) then sim(u, v) > sim(u, w)
        print(f'Checking order for {key}')
        pairs = distance_orders[key]
        for i in range(len(pairs)):
            for j in range(i+1, len(pairs)):
                d_i, s_i = pairs[i]
                d_j, s_j = pairs[j]
                if d_i < d_j and not s_i > s_j:
                    print(s_i > s_j, f'Order violation for {key}: node with true distance {d_i} has similarity {s_i} which is not greater than similarity {s_j} of node with true distance {d_j}')

        print()

    reconstructions = generator.create_reconstruct_graphs(b_n, token_dict, for_plotting=False, ids=None,
                                                          trees_to_left=False, verbose=True)
    reconstructions[0].plot(verbose=True)


def main(max_vocab_size=100):

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
    generator.set_default_dictionary(max_vocab_size, 20, 'D')
    token_dict = generator.get_dictionary()
    # sort by value
    d = dict(sorted(token_dict.items(), key=lambda item: item[1]))
    s = 'Dictionary: '
    for k, v in d.items():
        s += f'{k}: {v} '
    print(s)
    generator.set_default_pos_dictionary()
    pos_dict = generator.get_pos_dictionary()

    # _graph_print(args, token_dict, pos_dict, batch_size=3)

    # _t_khops(args, token_dict, pos_dict)
    # _t_int_partition()
    # _t_khops_gen(args, token_dict, pos_dict)

    # _t_bfs_task(args, token_dict, pos_dict)
    # _t_scratchpad_validation(args, token_dict, pos_dict)

    # _t_random_trees(args, token_dict, pos_dict)

    _graph_plot(args, token_dict, pos_dict)

    # _distance_scores(args, token_dict, pos_dict, batch_size=1)




    print('\n\nDone Testing')


if __name__ == '__main__':
    main()