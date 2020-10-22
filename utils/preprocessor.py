import random
import os

import torch
import torchaudio
import torchaudio.functional as F
import unicodedata
import h5py
from utils.mappings import *
from utils.audio_io import mean_stddev, file_log_spectrogram
from utils.text_io import read_tsv

class Preprocessor():
    r"""
    """

    def __init__(self, lang_in, tsv_filename, feature_classes=["phones"], max_samples=100, min_audio_length=3):
        self.lang = lang_in
        self.feature_classes = feature_classes
        transcript_path = os.path.join(corpora_path,speech_langs[self.lang],tsv_filename)

        self.phone_mapping = festvox_to_phones[self.lang]
        self.lexicon = self.get_lexicon(self.lang)

        utterance_data = read_tsv(transcript_path)
        random.shuffle(utterance_data)
        sample_list = []
        sample_num = 0
        while(sample_num < max_samples):
            sample = utterance_data[sample_num]
            if(sample['length'] >= min_audio_length):
                sample_list.append(sample)
                if(len(sample_list) >= max_samples):
                    break
            sample_num += 1
        soundfiles = []
        for x in sample_list:
            # TODO: move this filepath scheme someplace else!
            soundfiles.append(os.path.join(corpora_path,speech_langs[self.lang],
                                    'data',x['sound'][:3],x['sound']+'_1.wav'))
        self.mean, self.std = mean_stddev(soundfiles)
        self._input_dim = self.mean.shape[0]
    
    def get_lexicon(self, lang):
        r"""Processes a tab-separated-value file containing a lexicon for a given language.

        """
        lexicon = {}
        lexicon_path = os.path.join(language_resources_path,self.lang,lexicon_langs[self.lang])
        with open(lexicon_path) as f:
            for line in f.read().splitlines():
                if(line == ""):
                    continue
                if(line[0] == '#'):
                    continue
                line_tabsplit = line.split('\t')
                word = line_tabsplit[0].lower()
                phones = line_tabsplit[1].split(' ')
                lexicon[word] = list(filter(lambda a : a != ".", phones))
        return lexicon

    def encode(self, text, feature_classes=["phones"]):
        r"""Renders an utterance in terms of an integral indexing of the common phone set and its features.
        """
        words = text.split(' ')
        encoded_all = {}
        original_phones = []
        # TODO: handle out-of-vocabulary words better
        for word in words:
            original_phones.extend(self.lexicon[unicodedata.normalize('NFC',word).lower()])
            original_phones.append(" ")
        original_phones.pop()
        equivalent_phones = []
        for phone in original_phones:
            equivalent_phones.extend(festvox_to_phones[self.lang][phone])
        for feature_class in feature_classes:
            encoded = []
            if(feature_class == "phones"):
                encoded.extend(equivalent_phones)
            else:
                for phone in equivalent_phones:
                    if (phone == ' '):
                        encoded.append(' ')
                    elif (feature_class in common_phone_set[phone].keys()):
                        encoded.append(common_phone_set[phone][feature_class])
                    else:
                        encoded.append(common_phone_set[phone]["_type"])
            encoded_all[feature_class] = [char_to_int[feature_class][t] for t in encoded]
        return encoded_all
    
    def decode(self, sequence, feature_class):
        r"""Converts from an integral representation to the specified feature class.
        """
        return [int_to_char[feature_class][s] for s in sequence]

    def preprocess(self, sound, text):
        r"""Normalizes the log spectogram of ``sound`` and encodes ``text`` using
        the feature classes passed into the preprocessor.
        """
        cur_input = torch.squeeze(file_log_spectrogram(sound))
        preprocessed_sound = ((cur_input - self.mean) / self.std)
        return preprocessed_sound, self.encode(text, self.feature_classes)

    @property
    def input_dim(self):
        r"""Returns the dimensionality of utterances.
        """
        return self._input_dim

    def phone_size(self, feature_class="phones"):
        r"""Returns the number of possible values for a given feature class.
        """
        return len(int_to_char[feature_class])
