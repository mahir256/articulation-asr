import codecs
import os
from utils.mappings import *

def read_tsv(input_file):
    r"""Reads in a tab-separated-value file containing utterances, their speakers, and their lengths.
    """
    utterances = []
    with codecs.open(input_file,'r','utf-8') as f:
        for line in f.read().splitlines():
            [soundfile, speaker, text, length] = line.split('\t')
            utterances.append({'sound': soundfile,
                               'speaker': speaker,
                               'text': text,
                               'length': float(length)})
    return utterances

