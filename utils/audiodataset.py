import numpy as np
import random
import codecs
import soundfile
import os

import torch
import torchaudio
import torch.autograd as autograd
import torch.utils.data as datautils

from scipy.signal import spectrogram

from utils.mappings import *
from utils.preprocessor import file_log_spectrogram
from utils.text_io import read_tsv

class AudioDataset(datautils.Dataset):
    r"""A dataset encapsulating a particular set of audio files.
    """

    # TODO: reduce argument count by moving stuff into dict, or perhaps using *(*kw)args
    def __init__(self, lang_in, tsv_filename, preprocessor_in, batch_size):
        self.lang = lang_in
        self.preprocessor = preprocessor_in
        data = read_tsv(os.path.join(corpora_path, speech_langs[self.lang], tsv_filename))
        data.sort(key=lambda x: x['length'])
        
        # TODO: add argument to turn off buckets
        #bucket_diff = 4
        #max_text_length = max(len(x['text']) for x in data)
        #bucket_num = max_text_length // bucket_diff
        #buckets = [[] for _ in range(bucket_num)]
        #for x in data:
        #    bid = min(len(x['text']) // bucket_diff, bucket_num - 1)
        #    buckets[bid].append(x)
        #sort_fn = lambda x : (round(x['length'], 1), len(x['text']))
        #for bucket in buckets:
        #    bucket.sort(key=sort_fn)
        #self.data = [d for b in buckets for d in b]
        self.data = data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        utterance = self.data[index]
        soundfile = os.path.join(corpora_path,speech_langs[self.lang],
                                    'data',utterance['sound'][:2],utterance['sound']+'.flac')
        return self.preprocessor.preprocess(soundfile,utterance['text'])

