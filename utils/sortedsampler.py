import random
import torch.utils.data as datautils

class SortedSampler(datautils.sampler.Sampler):
    r"""
    """
    def __init__(self, src, batch_size):
        it_end = len(src) - batch_size + 1
        # TODO: sort by descending length of file
        self.batches = [range(i,i+batch_size) for i in range(0, it_end, batch_size)]
        self.src = src

    def __iter__(self):
        r"""
        """
        return (i for b in self.batches for i in b)

    def __len__(self):
        r"""
        """
        return len(self.src)

