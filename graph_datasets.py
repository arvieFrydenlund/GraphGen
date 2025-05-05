"""
Online Datastreams for graph generation via pybind
"""


import time
import numpy as np
import torch
from ctypes import c_bool
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader

# from fairseq_plugins.data3.get_generator_module import get_generator_module
from get_generator_module import get_generator_module
get_generator_module()  # compiles the generator module if needed
import generator


class GeneratorParser(object):
    """
    Used to verify the correctness of the c++ code and to convert graphs to networkx for plotting.
    """
    def __init__(self):
        pass


def batch_pprint(batch, print_tensors=False, title='', level=1):
    s = ''
    if title:
        print(title)
        s += '\t' * level
    for k, v in batch.items():
        print(f'{s}{k}: {type(v)}', end=' ')
        if isinstance(v, np.ndarray) or isinstance(v, torch.Tensor):
            t = "\t" *  level
            print(f'{t}{k}: {v.shape}, {v.dtype}')
            if print_tensors:
                print(v)
        elif isinstance(v, dict):
            print()
            batch_pprint(v, print_tensors, title='', level=level + 1)
        else:
            print(v)
    print()

def example_from_batch_pprint(batch, idx, title=''):
    pass


def batch_check(batch, check_gts=False):
    if 'success' in batch and not batch['success']:
        return False
    if check_gts:
        if 'ground_truths' not in batch:
            return False
        if batch['ground_truths'] is None:
            return False
        if batch['ground_truths'].shape[0] == 0:
            return False
    return True


def to_tensors(batch, exclude=('hashes',)):
    for k, v in batch.items():
        if exclude is not None and k in exclude:
            continue
        if isinstance(v, np.ndarray):
            batch[k] = torch.from_numpy(v)
        elif isinstance(v, list):
            batch[k] = torch.tensor(v)
    return batch


def to_device(batch, device, include=None):
    for k, v in batch.items():
        if include is not None and k not in include:
            continue
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device, non_blocking=True)
    return batch


class StaticDataset(Dataset):
    """Does this need to be in its own tread?  No"""
    def __init__(self, c_min=75, c_max=125,
                 max_path_length=10, min_path_length=3, sample_target_paths=True,
                 is_causal=False, shuffle_edges=False,
                 shuffle_nodes=False, min_vocab=0, max_vocab=-1,
                 batch_size=256,
                 max_edges=512, max_attempts=1000,
                 partition='train', base_seed=42,
                 dataset_size=10000, add_to_hashes=True, *args, **kwargs):

        self.c_min = c_min
        self.c_max = c_max
        self.max_path_length = max_path_length
        self.min_path_length = min_path_length
        self.sample_target_paths = sample_target_paths
        self.is_causal = is_causal
        self.shuffle_edges = shuffle_edges
        self.shuffle_nodes = shuffle_nodes
        self.min_vocab = min_vocab
        self.max_vocab = max_vocab
        self.batch_size = batch_size
        self.max_edges = max_edges
        self.max_attempts = max_attempts

        self.dataset_size = dataset_size  # number in examples, will go over by max batch_size
        self.add_to_hashes = add_to_hashes

        self.partition = partition
        self.base_seed = base_seed
        self.batches = []

    def create_batches(self):

        cur = 0
        while cur < self.dataset_size:
            batch = self.generator_call()
            if not batch_check(batch):
                print(f"Batch check failed.  This is often because you have bad graph generation parameters.  "
                      f"Like trying to sample length 10 paths from a fully connected Erdos Reny graph due to p=.9 etc.")
                continue
            self.batches.append(batch)
            if self.add_to_hashes:
                if self.partition == 'valid':
                    generator.set_validation_hashes(batch['hashes'])
                elif self.partition == 'test':
                    generator.set_test_hashes(batch['hashes'])
            cur += self.batch_size

    def set_seed(self):
        generator.set_seed(self.base_seed)

    @staticmethod
    def batch_check(batch):
        return batch_check(batch)

    @staticmethod
    def batch_pprint(batch, print_tensors, title=''):
        batch_pprint(batch, print_tensors, title)

    @staticmethod
    def to_tensors(batch, exclude=('hashes',)):
        return to_tensors(batch, exclude)

    def generator_call(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return self.batches[idx]


class ErdosRenyiStaticDataset(StaticDataset):
    def __init__(self,
                 min_num_nodes, max_num_nodes,
                 p=-1.0, c_min=75, c_max=125,
                 max_path_length=10, min_path_length=3, sample_target_paths=True,
                 is_causal=False, shuffle_edges=False,
                 shuffle_nodes=False, min_vocab=0, max_vocab=-1,
                 batch_size=256,
                 max_edges=512, max_attempts=1000,
                 partition='train', base_seed=42,
                 dataset_size=10000, add_to_hashes=True, *args, **kwargs):
        super(ErdosRenyiStaticDataset, self).__init__(
            c_min, c_max, max_path_length, min_path_length, sample_target_paths,
            is_causal, shuffle_edges, shuffle_nodes, min_vocab, max_vocab,
            batch_size, max_edges, max_attempts,
            partition, base_seed, dataset_size, add_to_hashes, *args, **kwargs)
        self.min_num_nodes = min_num_nodes
        self.max_num_nodes = max_num_nodes
        self.p = p

        self.create_batches()

    def generator_call(self):
        batch = generator.erdos_renyi_n(
            self.min_num_nodes, self.max_num_nodes,
            self.p, self.c_min, self.c_max,
            self.max_path_length, self.min_path_length, self.sample_target_paths,
            self.is_causal, self.shuffle_edges,
            self.shuffle_nodes, self.min_vocab, self.max_vocab,
            self.batch_size, self.max_edges, self.max_attempts
        )
        return batch


class EuclidianStaticDataset(StaticDataset):
    def __init__(self,
                 min_num_nodes, max_num_nodes,
                 dims=2, radius=-1.0, c_min=75, c_max=125,
                 max_path_length=10, min_path_length=1, sample_target_paths=True,
                 is_causal=False, shuffle_edges=False,
                 shuffle_nodes=False, min_vocab=0, max_vocab=-1,
                 batch_size=256,
                 max_edges=512, max_attempts=1000,
                 partition='train', base_seed=42,
                 dataset_size=10000, add_to_hashes=True, *args, **kwargs):
        super(EuclidianStaticDataset, self).__init__(
            c_min, c_max, max_path_length, min_path_length, sample_target_paths,
            is_causal, shuffle_edges, shuffle_nodes, min_vocab, max_vocab,
            batch_size, max_edges, max_attempts,
            partition, base_seed, dataset_size, add_to_hashes, *args, **kwargs)
        self.min_num_nodes = min_num_nodes
        self.max_num_nodes = max_num_nodes
        self.dims = dims
        self.radius = radius

        self.create_batches()

    def generator_call(self):
        batch = generator.euclidian_n(
            self.min_num_nodes, self.max_num_nodes,
            self.dims, self.radius, self.c_min, self.c_max,
            self.max_path_length, self.min_path_length, self.sample_target_paths,
            self.is_causal, self.shuffle_edges,
            self.shuffle_nodes, self.min_vocab, self.max_vocab,
            self.batch_size, self.max_edges, self.max_attempts
        )
        return batch


class PathStarStaticDataset(StaticDataset):
    def __init__(self,
                 min_num_arms, max_num_arms, min_arm_length, max_arm_length,
                 sample_target_paths=True,
                 is_causal=False, shuffle_edges=False,
                 shuffle_nodes=False, min_vocab=0, max_vocab=-1,
                 batch_size=256,
                 max_edges=512, max_attempts=1000,
                 partition='train', base_seed=42,
                 dataset_size=10000, add_to_hashes=True, *args, **kwargs):
        super(PathStarStaticDataset, self).__init__(
            -1, -1, -1, -1,
            sample_target_paths,
            is_causal, shuffle_edges, shuffle_nodes, min_vocab, max_vocab,
            batch_size, max_edges, max_attempts,
            partition, base_seed, dataset_size, add_to_hashes, *args, **kwargs)
        self.min_num_arms = min_num_arms
        self.max_num_arms = max_num_arms
        self.min_arm_length = min_arm_length
        self.max_arm_length = max_arm_length

        self.create_batches()

    def generator_call(self):
        batch = generator.path_star_n(
            self.min_num_arms, self.max_num_arms, self.min_arm_length, self.max_arm_length,
            self.sample_target_paths,
            self.is_causal, self.shuffle_edges,
            self.shuffle_nodes, self.min_vocab, self.max_vocab,
            self.batch_size, self.max_edges, self.max_attempts
        )
        return batch


class BalancedStaticDataset(StaticDataset):
    def __init__(self,
                 min_num_nodes, max_num_nodes,
                 min_lookahead, max_lookahead,
                 min_noise_reserve=0, max_num_parents=4, max_noise=-1,
                 sample_target_paths=True,
                 is_causal=False, shuffle_edges=False,
                 shuffle_nodes=False, min_vocab=0, max_vocab=-1,
                 batch_size=256,
                 max_edges=512, max_attempts=1000,
                 partition='train', base_seed=42,
                 dataset_size=10000, add_to_hashes=True, *args, **kwargs):
        super(BalancedStaticDataset, self).__init__(
            -1, -1, -1, -1,
            sample_target_paths,
            is_causal, shuffle_edges, shuffle_nodes, min_vocab, max_vocab,
            batch_size, max_edges, max_attempts,
            partition, base_seed, dataset_size, add_to_hashes, *args, **kwargs)
        self.min_num_nodes = min_num_nodes
        self.max_num_nodes = max_num_nodes
        self.min_lookahead = min_lookahead
        self.max_lookahead = max_lookahead
        self.min_noise_reserve = min_noise_reserve
        self.max_num_parents = max_num_parents
        self.max_noise = max_noise

        self.create_batches()

    def generator_call(self):
        batch = generator.balanced_n(
            self.min_num_nodes, self.max_num_nodes,
            self.min_lookahead, self.max_lookahead,
            self.min_noise_reserve, self.max_num_parents, self.max_noise,
            self.sample_target_paths,
            self.is_causal, self.shuffle_edges,
            self.shuffle_nodes, self.min_vocab, self.max_vocab,
            self.batch_size, self.max_edges, self.max_attempts
        )
        return batch


class StreamingDataset(torch.utils.data.IterableDataset):
    """
    """
    def __init__(self, c_min=75, c_max=125,
                 max_path_length=10, min_path_length=3, sample_target_paths=True,
                 is_causal=False, shuffle_edges=False,
                 shuffle_nodes=False, min_vocab=0, max_vocab=-1,
                 batch_size=256,
                 max_edges=512, max_attempts=1000,
                 partition='train', base_seed=42, cur_epoch=0, is_running=None, break_me=False, *args, **kwargs):
        super(StreamingDataset, self).__init__()

        self.c_min = c_min
        self.c_max = c_max
        self.max_path_length = max_path_length
        self.min_path_length = min_path_length
        self.sample_target_paths = sample_target_paths
        self.is_causal = is_causal
        self.shuffle_edges = shuffle_edges
        self.shuffle_nodes = shuffle_nodes
        self.min_vocab = min_vocab
        self.max_vocab = max_vocab
        self.batch_size = batch_size
        self.max_edges = max_edges
        self.max_attempts = max_attempts

        self.partition = partition
        self.base_seed = base_seed
        self.cur_epoch = cur_epoch
        self.is_running = is_running
        self.break_me = break_me
        self.my_seed = None

    def set_seed(self, num_workers, worker_id):
        """
        seed needs to be seed spread out by the current epoch is so that on restart we do not start at beginning again.
        The other option is to shut down the workers and restart them at every epoch.
        This makes the training reproducible, but it is not efficient.
        """
        if self.break_me:  # all workers will have the same seed, used for testing
            seed = self.base_seed
        else:
            seed = self.base_seed + self.cur_epoch * num_workers + worker_id
        generator.set_seed(seed)
        return seed

    @staticmethod
    def batch_check(batch):
        return batch_check(batch)

    @staticmethod
    def batch_pprint(batch, print_tensors, title=''):
        batch_pprint(batch, print_tensors, title)

    @staticmethod
    def to_tensors(batch, exclude=('hashes',)):
        return to_tensors(batch, exclude)

    def get_hash_mask(self, batch):
        if self.partition == 'train':  # make mask for training examples, not this should almost never happen if ever.
            is_valid = generator.is_invalid_example(batch['hashes'])
            batch['is-valid'] = is_valid

    def stop(self):
        self.is_running.value = False

    def generator_call(self):
        raise NotImplementedError

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers
        worker_id = worker_info.id

        self.my_seed = self.set_seed(num_workers, worker_id)

        my_cur = 0
        while self.is_running.value:
            t = time.time()
            batch = self.generator_call()
            t = time.time() - t  # as seconds
            if not self.batch_check(batch):
                # raise ValueError(f"Batch check failed: {d}")
                print(f"Batch check failed.  This is often because you have bad graph generation parameters.  "
                      f"Like trying to sample length 10 paths from a fully connected Erdos Reny graph due to p=.9 etc.")
                yield {'bad_batch': True, 'worker_id': worker_id, 'worker_cur': my_cur}
            self.get_hash_mask(batch)
            # self.to_tensors(batch)  # done in collate_fn
            batch['batch_hash'] = batch['hashes'][0]  # for visual confirmation bach is new across threads
            batch['batch_time'] = t
            batch['worker_id'] = worker_id
            batch['worker_cur'] = my_cur
            if self.partition == 'train':  # visual confirmation safety check that threads do not matter
                batch['test_size'] = generator.get_test_size()
                batch['valid_size'] = generator.get_validation_size()
            yield batch
            my_cur += 1


class ErdosRenyiDataset(StreamingDataset):
    def __init__(self,
                 min_num_nodes, max_num_nodes,
                 p=-1.0, c_min=75, c_max=125,
                 max_path_length=10, min_path_length=3, sample_target_paths=True,
                 is_causal=False, shuffle_edges=False,
                 shuffle_nodes=False, min_vocab=0, max_vocab=-1,
                 batch_size=256,
                 max_edges=512, max_attempts=1000,
                 partition='train', base_seed=42, cur_epoch=0, is_running=None, break_me=False, *args, **kwargs):
        super(ErdosRenyiDataset, self).__init__(
            c_min, c_max, max_path_length, min_path_length, sample_target_paths,
            is_causal, shuffle_edges, shuffle_nodes, min_vocab, max_vocab,
            batch_size, max_edges, max_attempts,
            partition, base_seed, cur_epoch, is_running, break_me, *args, **kwargs)
        self.min_num_nodes = min_num_nodes
        self.max_num_nodes = max_num_nodes
        self.p = p

    def generator_call(self):
        batch = generator.erdos_renyi_n(
            self.min_num_nodes, self.max_num_nodes,
            self.p, self.c_min, self.c_max,
            self.max_path_length, self.min_path_length, self.sample_target_paths,
            self.is_causal, self.shuffle_edges,
            self.shuffle_nodes, self.min_vocab, self.max_vocab,
            self.batch_size, self.max_edges, self.max_attempts
        )
        return batch


class EuclidianDataset(StreamingDataset):
    def __init__(self,
                 min_num_nodes, max_num_nodes,
                 dims=2, radius=-1.0, c_min=75, c_max=125,
                 max_path_length=10, min_path_length=1, sample_target_paths=True,
                 is_causal=False, shuffle_edges=False,
                 shuffle_nodes=False, min_vocab=0, max_vocab=-1,
                 batch_size=256,
                 max_edges=512, max_attempts=1000,
                 partition='train', base_seed=42, cur_epoch=0, is_running=None, break_me=False, *args, **kwargs):
        super(EuclidianDataset, self).__init__(
            c_min, c_max, max_path_length, min_path_length, sample_target_paths,
            is_causal, shuffle_edges, shuffle_nodes, min_vocab, max_vocab,
            batch_size, max_edges, max_attempts,
            partition, base_seed, cur_epoch, is_running, break_me, *args, **kwargs)
        self.min_num_nodes = min_num_nodes
        self.max_num_nodes = max_num_nodes
        self.dims = dims
        self.radius = radius

    def generator_call(self):
        batch = generator.euclidian_n(
            self.min_num_nodes, self.max_num_nodes,
            self.dims, self.radius, self.c_min, self.c_max,
            self.max_path_length, self.min_path_length, self.sample_target_paths,
            self.is_causal, self.shuffle_edges, 
            self.shuffle_nodes, self.min_vocab, self.max_vocab,
            self.batch_size, self.max_edges, self.max_attempts
        )
        return batch


class PathStarDataset(StreamingDataset):
    def __init__(self,
                 min_num_arms, max_num_arms, min_arm_length, max_arm_length,
                 sample_target_paths=True,
                 is_causal=False, shuffle_edges=False,
                 shuffle_nodes=False, min_vocab=0, max_vocab=-1,
                 batch_size=256,
                 max_edges=512, max_attempts=1000,
                 partition='train', base_seed=42, cur_epoch=0, is_running=None, break_me=False, *args, **kwargs):
        super(PathStarDataset, self).__init__(
            -1, -1, -1, -1,
            sample_target_paths,
            is_causal, shuffle_edges, shuffle_nodes, min_vocab, max_vocab,
            batch_size, max_edges, max_attempts,
            partition, base_seed, cur_epoch, is_running, break_me, *args, **kwargs)
        self.min_num_arms = min_num_arms
        self.max_num_arms = max_num_arms
        self.min_arm_length = min_arm_length
        self.max_arm_length = max_arm_length

    def generator_call(self):
        batch = generator.path_star_n(
            self.min_num_arms, self.max_num_arms, self.min_arm_length, self.max_arm_length,
            self.sample_target_paths,
            self.is_causal, self.shuffle_edges,
            self.shuffle_nodes, self.min_vocab, self.max_vocab,
            self.batch_size, self.max_edges, self.max_attempts
        )
        return batch


class BalancedDataset(StreamingDataset):
    def __init__(self,
                 min_num_nodes, max_num_nodes,
                 min_lookahead, max_lookahead,
                 min_noise_reserve=0, max_num_parents=4, max_noise=-1,
                 sample_target_paths=True,
                 is_causal=False, shuffle_edges=False,
                 shuffle_nodes=False, min_vocab=0, max_vocab=-1,
                 batch_size=256,
                 max_edges=512, max_attempts=1000,
                 partition='train', base_seed=42, cur_epoch=0, is_running=None, break_me=False, *args, **kwargs):
        super(BalancedDataset, self).__init__(
            -1, -1, -1, -1,
            sample_target_paths,
            is_causal, shuffle_edges, shuffle_nodes, min_vocab, max_vocab,
            batch_size, max_edges, max_attempts,
            partition, base_seed, cur_epoch, is_running, break_me, *args, **kwargs)
        self.min_num_nodes = min_num_nodes
        self.max_num_nodes = max_num_nodes
        self.min_lookahead = min_lookahead
        self.max_lookahead = max_lookahead
        self.min_noise_reserve = min_noise_reserve
        self.max_num_parents = max_num_parents
        self.max_noise = max_noise

    def generator_call(self):
        batch = generator.balanced_n(
            self.min_num_nodes, self.max_num_nodes,
            self.min_lookahead, self.max_lookahead,
            self.min_noise_reserve, self.max_num_parents, self.max_noise,
            self.sample_target_paths,
            self.is_causal, self.shuffle_edges,
            self.shuffle_nodes, self.min_vocab, self.max_vocab,
            self.batch_size, self.max_edges, self.max_attempts
        )
        return batch


class GraphEpochIterator(object):
    """To fake having epochs in the streaming dataset, we need to create a new iterator over the generator."""
    def __init__(self, dataloader, batch_size, dataset_size, *args, **kwargs):
        self.dataloader = dataloader
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        self.iterations_in_epoch = -1
        self.num_iters = dataset_size // batch_size + 1

        self._dataloader_iter = iter(self.dataloader)  # this starts up the workers, do not delete to make infinite

    def __iter__(self):
        return self

    def __len__(self):
        return self.num_iters

    def __next__(self):
        self.iterations_in_epoch += 1
        if self.iterations_in_epoch < self.num_iters:
            batch = next(self._dataloader_iter)
            return batch
        if not isinstance(self.dataloader.dataset, StreamingDataset):
            self._dataloader_iter = iter(self.dataloader)
        self.iterations_in_epoch = -1
        raise StopIteration

    def end_of_epoch(self) -> bool:
        """Returns whether the most recent epoch iterator has been exhausted"""
        return self.iterations_in_epoch >= self.num_iters - 1


class GraphEpochBatchIterator(object):
    """This is Fairseq wrapper stuff."""
    def __init__(self, dataset, epoch, batch_size, dataset_size,
                 num_workers=0, pin_memory=None, timeout=0, prefetch_factor=8,
                 persistent_workers=False, pin_memory_device='', in_order=True, **kwargs,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        self._cur_epoch_itr = None
        self.epoch = epoch

        self.data_loader = DataLoader(dataset, batch_size=None, batch_sampler=None,
                                      collate_fn=dataset.to_tensors,
                                      num_workers=num_workers, pin_memory=num_workers > 0 if pin_memory is None else pin_memory,
                                      prefetch_factor=prefetch_factor, timeout=timeout)

        self._cur_epoch_itr = GraphEpochIterator(self.data_loader, batch_size, dataset_size)

    def __len__(self):
        return len(self._cur_epoch_itr) if self._cur_epoch_itr is not None else 0

    @property
    def next_epoch_idx(self):
        """Return the epoch index after *next_epoch_itr* is called."""
        if self._cur_epoch_itr is not None and self.end_of_epoch():
            return self.epoch + 1
        else:
            return self.epoch

    def next_epoch_itr(
        self, shuffle=True, fix_batches_to_gpus=False, set_dataset_epoch=True
    ):
        """Return a new iterator over the dataset.

        Args:
            shuffle (bool, optional): shuffle batches before returning the
                iterator (default: True).
            fix_batches_to_gpus (bool, optional): ensure that batches are always
                allocated to the same shards across epochs. Requires
                that :attr:`dataset` supports prefetching (default: False).
            set_dataset_epoch (bool, optional): update the wrapped Dataset with
                the new epoch number (default: True).
        """
        self.epoch = self.next_epoch_idx
        return self._cur_epoch_itr

    def end_of_epoch(self) -> bool:
        """Returns whether the most recent epoch iterator has been exhausted"""
        return self._cur_epoch_itr.end_of_epoch()

    @property
    def iterations_in_epoch(self):
        """The number of consumed batches in the current epoch."""
        if self._cur_epoch_itr is not None:
            return self._cur_epoch_itr.iterations_in_epoch
        return 0

    @property
    def n(self):
        return self.iterations_in_epoch

    def state_dict(self):
        """Returns a dictionary containing a whole state of the iterator."""
        if self.end_of_epoch():
            epoch = self.epoch + 1
            iter_in_epoch = 0
        else:
            epoch = self.epoch
            iter_in_epoch = self.iterations_in_epoch
        return {
            "version": 2,
            "epoch": epoch,
            "iterations_in_epoch": iter_in_epoch,
            "shuffle": False,
        }

    def load_state_dict(self, state_dict):
        """Copies the state of the iterator from the given *state_dict*."""
        self.epoch = state_dict["epoch"]
        itr_pos = state_dict.get("iterations_in_epoch", 0)
        version = state_dict.get("version", 1)
        # if itr_pos > 0:  # this check doesn't work because it saves before resetting the iterator
        #    raise ValueError('Should not have saved mid epoch state, itr_pos: {}'.format(itr_pos))

        # self.dataset.update_epoch(epoch=self.epoch)
        # self._cur_epoch_itr = PathStarIterator(self.dataset)

        raise NotImplementedError

    def ordered_batches(self, epoch, fix_batches_to_gpus, shuffle):
        raise NotImplementedError


def t_datastream(graph_type='erdos-renyi', break_me=False):

    base_seed = 42
    num_workers = 4
    is_running = mp.Value(c_bool, True, lock=False)

    if graph_type == 'erdos-renyi':
        static_dataset = ErdosRenyiStaticDataset(50, 55,
                                             p=-1, c_min=75, c_max=125,
                                             max_path_length=10, min_path_length=3, sample_target_paths=True,
                                             is_causal=False, shuffle_edges=False,
                                             shuffle_nodes=True, min_vocab=5, max_vocab=100,
                                             batch_size=7,
                                             max_edges=512, max_attempts=1000,
                                             partition='test', base_seed=base_seed,
                                             dataset_size=19, add_to_hashes=True)


        iterable_dataset = ErdosRenyiDataset(50, 55,
                                             p=-1, c_min=75, c_max=125,
                                             max_path_length=10, min_path_length=3, sample_target_paths=True,
                                             is_causal=False, shuffle_edges=False,
                                             shuffle_nodes=True, min_vocab=5, max_vocab=100,
                                             batch_size=7,
                                             max_edges=512, max_attempts=1000,
                                             partition='train', base_seed=base_seed, cur_epoch=0, is_running=is_running,
                                             break_me=break_me)
    elif graph_type == 'euclidian':
        static_dataset = EuclidianStaticDataset(50, 55,
                                            dims=2, radius=-1.0, c_min=75, c_max=125,
                                                max_path_length=10, min_path_length=3, sample_target_paths=True,
                                                is_causal=False, shuffle_edges=False,
                                                shuffle_nodes=True, min_vocab=5, max_vocab=100,
                                                batch_size=7,
                                                max_edges=512, max_attempts=1000,
                                                partition='test', base_seed=base_seed,
                                                dataset_size=19, add_to_hashes=True)



        iterable_dataset = EuclidianDataset(50, 55,
                                            dims=2, radius=-1.0, c_min=75, c_max=125,
                                            max_path_length=10, min_path_length=3, sample_target_paths=True,
                                            is_causal=False, shuffle_edges=False,
                                            shuffle_nodes=True, min_vocab=5, max_vocab=100,
                                            batch_size=7,
                                            max_edges=512, max_attempts=1000,
                                            partition='train', base_seed=base_seed, cur_epoch=0, is_running=is_running,
                                            break_me=break_me)
    elif graph_type == 'path-star':
        static_dataset = PathStarStaticDataset(2, 5, 5, 10,
                                               sample_target_paths=True,
                                               is_causal=False, shuffle_edges=False,
                                               shuffle_nodes=True, min_vocab=5, max_vocab=100,
                                               batch_size=7,
                                               max_edges=512, max_attempts=1000,
                                               partition='test', base_seed=base_seed,
                                               dataset_size=19, add_to_hashes=True)


        iterable_dataset = PathStarDataset(min_num_arms=2, max_num_arms=5, min_arm_length=5, max_arm_length=10,
                                           sample_target_paths=True,
                                           is_causal=False, shuffle_edges=False,
                                           shuffle_nodes=True, min_vocab=5, max_vocab=100,
                                           batch_size=7,
                                           max_edges=512, max_attempts=1000,
                                           partition='train', base_seed=base_seed, cur_epoch=0, is_running=is_running,
                                           break_me=break_me)
    elif graph_type == 'balanced':
        static_dataset = BalancedStaticDataset(min_num_nodes=50, max_num_nodes=55,
                                               min_lookahead=5, max_lookahead=10,
                                               min_noise_reserve=0, max_num_parents=4, max_noise=-1,
                                               sample_target_paths=True,
                                               is_causal=False, shuffle_edges=False,
                                               shuffle_nodes=True, min_vocab=5, max_vocab=100,
                                               batch_size=7,
                                               max_edges=512, max_attempts=1000,
                                               partition='test', base_seed=base_seed,
                                               dataset_size=19, add_to_hashes=True)

        iterable_dataset = BalancedDataset(min_num_nodes=50, max_num_nodes=55,
                                           min_lookahead=5, max_lookahead=10,
                                           min_noise_reserve=0, max_num_parents=4, max_noise=-1,
                                           sample_target_paths=True,
                                           is_causal=False, shuffle_edges=False,
                                           shuffle_nodes=True, min_vocab=5, max_vocab=100,
                                           batch_size=7,
                                           max_edges=512, max_attempts=1000,
                                           partition='train', base_seed=base_seed, cur_epoch=0, is_running=is_running,
                                           break_me=break_me)
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")

    test_loader = DataLoader(static_dataset, batch_size=None, num_workers=0, collate_fn=static_dataset.to_tensors)
    test_epoch_iter = GraphEpochIterator(test_loader, static_dataset.batch_size, static_dataset.dataset_size)

    for cur, batch in enumerate(test_epoch_iter):
        static_dataset.batch_pprint(batch, print_tensors=False, title=f'Test Batch: {cur}')

    print(f"Test set size: {len(static_dataset)}")
    print(f"And from generator: {generator.get_test_size()}")

    train_loader = DataLoader(iterable_dataset, batch_size=None, batch_sampler=None,
                              collate_fn=iterable_dataset.to_tensors,
                              num_workers=num_workers, pin_memory=True,
                              prefetch_factor=8, timeout=60)
    train_epoch_iter = GraphEpochIterator(train_loader,
                                          batch_size=iterable_dataset.batch_size,
                                          dataset_size=iterable_dataset.batch_size * 5 + 2)

    for cur, batch in enumerate(train_epoch_iter):
        iterable_dataset.batch_pprint(batch, print_tensors=False, title=f'Train Batch: {cur} in epoch')
    for cur, batch in enumerate(train_epoch_iter):
        iterable_dataset.batch_pprint(batch, print_tensors=False, title=f'Train Batch: {cur} in epoch')

        #  Example of how to iterate over the raw train_loader
        # if cur > 5 * num_workers + 1:
        #    iterable_dataset.stop()
        #    break  # need this to stop iterating over remaining prefetched
    iterable_dataset.stop()


if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf, edgeitems=10, linewidth=np.inf, precision=2, suppress=True, )
    torch.set_printoptions(threshold=np.inf, edgeitems=10, linewidth=np.inf, precision=2, sci_mode=False)

    break_me_ = False
    print(generator.help_str())
    print('\n\n\n')

    graph_type_ = 'erdos-renyi'
    # graph_type_ = 'euclidian'
    # graph_type_ = 'path-star'
    # graph_type_ = 'balanced'

    t_datastream(graph_type_, break_me_)


