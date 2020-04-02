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

from scipy.signal import spectrogram

from utils.mappings import *
from utils.preprocessor import file_log_spectrogram

class AudioDataset(datautils.Dataset):
    def __init__(self, lang_in, tsv_filename, preprocessor_in, batch_size):
        self.lang = lang_in
        self.preprocessor = preprocessor_in
        data = read_tsv(os.path.join(corpora_path, speech_langs[self.lang], tsv_filename))
        bucket_diff = 4
        max_text_length = max(len(x['text']) for x in data)
        bucket_num = max_text_length // bucket_diff
        buckets = [[] for _ in range(bucket_num)]
        for x in data:
            bid = min(len(x['text']) // bucket_diff, bucket_num - 1)
            buckets[bid].append(x)
        sort_fn = lambda x : (round(x['length'], 1), len(x['text']))
        for bucket in buckets:
            bucket.sort(key=sort_fn)
        self.data = [d for b in buckets for d in b]
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        utterance = self.data[index]
        soundfile = os.path.join(corpora_path,speech_langs[self.lang],
                                    'data',utterance['sound'][:2],utterance['sound']+'.flac')
        return self.preprocessor.preprocess(soundfile,utterance['text'])

def read_tsv(file):
    utterances = []
    with codecs.open(os.path.join('corpora-googleasr',file),'r','utf-8') as f:
        for line in f.read().splitlines():
            [soundfile, speaker, text, length] = line.split('\t')
            utterances.append({'sound': soundfile,
                               'speaker': speaker,
                               'text': text,
                               'length': float(length)})
    return utterances

