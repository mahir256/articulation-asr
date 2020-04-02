import numpy as np
import random
import codecs
import librosa
import soundfile
import os

import torch
import torchaudio
import torch.autograd as autograd
import torch.utils.data as datautils

class RandomBatchSampler(datautils.sampler.Sampler):
    def __init__(self, src, batch_size):
        it_end = len(src) - batch_size + 1
        self.batches = [range(i,i+batch_size) for i in range(0, it_end, batch_size)]
        self.src = src

    def __iter__(self):
        random.shuffle(self.batches)
        return (i for b in self.batches for i in b)

    def __len__(self):
        return len(self.src)

