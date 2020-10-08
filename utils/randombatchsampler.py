import random
import torch.utils.data as datautils

class RandomBatchSampler(datautils.sampler.Sampler):
    r"""Sampler which returns a batch, randomly chosen from the data
    given, of entries a forming a contiguous range in the source provided.
    """
    def __init__(self, src, batch_size):
        it_end = len(src) - batch_size + 1
        self.batches = [range(i,i+batch_size) for i in range(0, it_end, batch_size)]
        self.src = src
        self.flag = 0

    def __iter__(self):
        r"""
        """
        batches_out = (i for b in self.batches for i in b)
        #print(self.batches)
        if(self.flag == 0):
            random.shuffle(self.batches)
            self.flag = 1
#        print([self.src.data[i]['length'] for i in batches_out])
        return batches_out
            

    def __len__(self):
        r"""
        """
        return len(self.src)

