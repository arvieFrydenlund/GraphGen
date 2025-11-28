import numpy as np

"""
Prototype code for generating k-hop prefixes for use in language modeling tasks.
"""

class SampleIntPartition(object):
    def __init__(self, seed=None, suggested_cache_size=1000000000, max_cache_size=10):
        """
        cashes for partition functions.
        suggested_cache clears on first call, max_cache clears when size exceeds max_cache_size * suggested_cache_size.
        :param seed:
        """
        self.QK_cache = {}
        self.QN_cache = {}
        self.QNK_cache = {}
        self.qk_needs_emptying = False  #  self.QK_cache.clear()
        self.qn_needs_emptying = False
        self.qnk_needs_emptying = False

        if seed is None:
            self.rng = np.random.default_rng()
        self.rng = np.random.default_rng(seed)

        self.suggested_cache_size = suggested_cache_size
        self.max_cache_size = suggested_cache_size * max_cache_size

    def print(self):
        s = ''
        if len(self.QK_cache):
            s += f"QK cache size: {len(self.QK_cache)},\t"
        s += f"QN cache size: {len(self.QN_cache)},\t"
        s += f"QNK cache size: {len(self.QNK_cache)}"
        print(s)

    def qk_cache(self, Q, K, result):
        self.QK_cache[(Q, K)] = result
        if len(self.QK_cache) > self.max_cache_size:
            self.QK_cache.clear()

    def qn_cache(self, Q, N, result):
        self.QN_cache[(Q, N)] = result
        if len(self.QN_cache) > self.max_cache_size:
            self.QN_cache.clear()

    def qnk_cache(self, Q, N, K, result):
        self.QNK_cache[(Q, N, K)] = result
        if len(self.QNK_cache) > self.max_cache_size:
            self.QNK_cache.clear()

    def _partition_QK(self, Q, K):
        """
        Counts the number of partitions of Q where no part exceeds K.
        """
        if (Q, K) in self.QK_cache:
            return self.QK_cache[(Q, K)]
        if Q == 0:
            return 1
        if Q < 0 or K == 0:
            return 0
        result = self.partition_QN(Q - K, K) + self.partition_QN(Q, K - 1)
        self.qk_cache(Q, K, result)
        return result

    def partition_QK(self, Q, K):
        if len(self.QK_cache) < self.suggested_cache_size:
            return self._partition_QK(Q, K)
        return self._partition_QK(Q, K)

    def _partition_QN(self, Q, N):
        """
        Counts the number of partitions of Q into N parts.
        """
        if (Q, N) in self.QN_cache:
            return self.QN_cache[(Q, N)]
        if N == 0:
            return 1 if Q == 0 else 0
        if N > Q or Q <= 0:
            return 0
        if N == Q:
            return 1
        result = self.partition_QN(Q - 1, N - 1) + self.partition_QN(Q - N, N)
        self.qn_cache(Q, N, result)
        return result

    def partition_QN(self, Q, N):
        if len(self.QN_cache) < self.suggested_cache_size:
            return self._partition_QN(Q, N)
        return self._partition_QN(Q, N)

    def _partition_QNK(self, Q, N, K):
        """
        Counts the number of partitions of Q into N parts where no part exceeds K.
        """
        if (Q, N, K) in self.QNK_cache:
            return self.QNK_cache[(Q, N, K)]
        if Q > N * K or K <= 0:
            return 0
        if Q == N * K:
            return 1
        result = sum(self.partition_QNK(Q - i * K, N - i, K - 1) for i in range(N))
        self.qnk_cache(Q, N, K, result)
        return result

    def partition_QNK(self, Q, N, K):
        if len(self.QNK_cache) < self.suggested_cache_size:
            return self._partition_QNK(Q, N, K)
        return self._partition_QNK(Q, N, K)

    def _min_max(self, n, s):
        min_max = int(np.floor(float(n) / float(s)))
        if int(n % s) > 0:
            min_max += 1
        return min_max

    def uniform_random_partition(self, Q, N, shuffle=True):
        """
        file:///h/arvie/Downloads/Partitioning_paper1.pdf
        @article{loceyrandom,
        title={Random integer partitions with restricted numbers of parts},
        author={Locey, Kenneth J}
        }
        :return:
        """
        segment_lengths = []
        _min = self._min_max(Q, N)
        _max = Q - N + 1
        total = self.partition_QN(Q, N)
        which = self.rng.integers(1, total)  # change from random.randrange
        while Q:
            for K in range(_min, _max + 1):
                count = self.partition_QNK(Q, N, K)
                if count >= which:
                    count = self.partition_QNK(Q, N, K - 1)
                    break
            segment_lengths.append(K)
            Q -= K
            if Q == 0:
                break
            N -= 1
            which -= count
            _min = self._min_max(Q, N)
            _max = K
        if shuffle:
            self.rng.shuffle(segment_lengths)
        return segment_lengths

    def non_uniform_random_partition(self, Q, N, shuffle=True):
        """
        A faster  non-uniform version
        :return:
        """
        # each segment has at least length 1 but then otherwise it is randomly distributed
        remaining_length = Q - N
        segment_lengths = [1] * N
        for i in range(segment_lengths - 1):
            if remaining_length <= 0:
                break
            r = self.rng.integers(0, remaining_length + 1)
            segment_lengths[i] += r
            remaining_length -= r
        segment_lengths[-1] += remaining_length
        if shuffle:
            self.rng.shuffle(segment_lengths)
        else:
            segment_lengths.sort()
        return segment_lengths


class KHopsGen(object):
    def __init__(self, min_k, max_k, min_value, max_value, min_prefix_length, max_prefix_length, dictionary=None,
                 right_side_connect=True, partition_method='uniform', partition_func=None, seed=None):
        self.min_k, self.max_k = min_k, max_k
        self.min_value, self.max_value = min_value, max_value
        self.min_prefix_length, self.max_prefix_length = min_prefix_length, max_prefix_length

        assert max_prefix_length - min_prefix_length >= 3 * max_k,\
            "prefix_length must be long enough to accommodate k hops"

        self.vocabulary = np.ones([max_value - min_value + 1])  # use this to sample from
        self.dictionary = dictionary

        self.right_side_connect = right_side_connect

        self.partition_method = partition_method
        self.partition_func = partition_func
        if self.partition_func is None:
            self.partition_func = SampleIntPartition(seed=seed)
        if seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(seed)

    def get_segment_lengths(self):
        """
        Notice that segments sum to prefix_length - k, since we will add one token to each segment later
        :return: list[int] length of k, int, int
        """
        k = self.rng.integers(self.min_k, self.max_k + 1)
        prefix_length = self.rng.integers(self.min_prefix_length, self.max_prefix_length + 1)
        if self.partition_method == 'uniform':
            segment_lengths = self.partition_func.uniform_random_partition(prefix_length - k, k)
            if len(segment_lengths) < k:  # safety that should never happen
                segment_lengths = self.partition_func.non_uniform_random_partition(prefix_length - k, k)
        else:
            segment_lengths = self.partition_func.non_uniform_random_partition(prefix_length - k, k)
        return segment_lengths, k, prefix_length

    def get_segment(self, cur_value, segment_len, is_last=False):
        """
        return a list of segement_size elements from vocabulary where cur_value is the last or second last element
        and no other element is equal to cur_value
        :param cur_value:
        :param segment_size:
        :param vocabulary:
        :return:
        """
        cur_vocab = self.vocabulary.copy() * 1./(len(self.vocabulary) - 1)

        cur_vocab[cur_value - self.min_value] = 0.  # get id of cur_value in vocabulary
        segment = np.random.choice(np.arange(self.min_value, self.max_value + 1),
                                   size=segment_len, replace=True, p=cur_vocab)
        segment = segment.tolist()
        if is_last or self.right_side_connect:  # then the new cur_value is at the last position
            segment.append(cur_value)
        else:  # the new cur_value is the second last position
            segment.insert(-1, cur_value)
        return segment

    def generate(self, flat=True):
        segment_lengths, k, prefix_length = self.get_segment_lengths()
        segments = []
        ground_truths = []
        cur_value = self.rng.integers(self.min_value, self.max_value + 1)
        for i in range(len(segment_lengths)):
            ground_truths.append(cur_value)
            segment = self.get_segment(cur_value, segment_lengths[i], is_last = (i == len(segment_lengths) - 1))
            if self.right_side_connect:
                cur_value = segment[-2]
            else:
                cur_value = segment[-1]
            segments.append(segment)  # easier to verify but we can just extend, or better make vector of correct size
        if flat:
            prefix = [item for sublist in segments for item in sublist]  # flatten
        else:
            prefix = segments
        return prefix, ground_truths, k, prefix_length




def _t_khops_gen(seed=42):
    # np.random.seed(46)

    pf = SampleIntPartition()

    #khops = KHopsGen(k=10, min_value=0, max_value=4, min_prefix_length=50, max_prefix_length=250,
    #                 right_side_connect=True, partition_method='uniform', partition_func=pf)

    #segments, prefix_length = khops.get_segment_lengths_uniform()
    #print(segments, sum(segments), prefix_length)

    segments = pf.uniform_random_partition(247, 15)
    print(segments, sum(segments), len(segments))
    pf.print()

    segments = pf.uniform_random_partition(300, 10)
    print(segments, sum(segments), len(segments))
    pf.print()


    segments = pf.uniform_random_partition(250, 11)
    print(segments, sum(segments), len(segments))
    pf.print()

    khops = KHopsGen(min_k=5, max_k=7, min_value=0, max_value=10, min_prefix_length=22, max_prefix_length=50,
                     right_side_connect=True, partition_method='uniform', partition_func=pf, seed=seed)

    for i in range(100):
         prefix, ground_truths, k, prefix_length = khops.generate(flat=False)
         print(prefix, ground_truths)

if __name__ == '__main__':
    _t_khops_gen()