

class KHopsGen(object):
    def __init__(self, k, min_value, max_value, prefix_length, dictionary=None,
                 right_side_connect=True):
        self.k = k
        self.min_value = min_value
        self.max_value = max_value
        self.prefix_length = prefix_length

        self.segments = []  # k + 1 segments

    def get_k(self):
        return self.k

    def get_segement(self, cur_value):
        pass




if __name__ == '__main__':
    pass