import numpy as np
import random
import codecs
import librosa
import soundfile
import os
import json

import torch, torchaudio
import torchaudio.functional as F
import torch.autograd as autograd
import torch.utils.data as datautils
from mappings import *

from scipy.signal import spectrogram

class Preprocessor():
    def __init__(self, lang_in, tsv_filename, max_samples=100, min_audio_length=5):
        self.lang = lang_in
        transcript_path = os.path.join(corpora_path,speech_langs[self.lang],tsv_filename)

        self.phonology = self.get_phonology(self.lang)
        self.lexicon = self.get_lexicon(self.lang)
        data = read_tsv(transcript_path)
        soundfiles = [os.path.join(corpora_path,speech_langs[self.lang],'data',x['sound'][:2],x['sound']+'.flac')
                        for x in data]
        random.shuffle(soundfiles)
        sample_list = []
        sample_num = 0
        while(sample_num < max_samples):
            sample = data[sample_num]
            if(sample['length'] >= min_audio_length):
                sample_list.append(sample['sound'])
                if(len(sample_list) >= max_samples):
                    break
            sample_num += 1
        self.mean, self.std = mean_stddev(soundfiles[:max_samples])
        self._input_dim = self.mean.shape[0]
        self.int_to_char = {}
        self.int_to_char[" "] = dict(enumerate(self.phonology.keys()))
        for feature_class in feature_class_values.keys():
            self.int_to_char[feature_class] = dict(enumerate(feature_class_values[feature_class]))
            default_representation = "C" if feature_classes_consonant[feature_class] else "V"
            self.int_to_char[feature_class][len(self.int_to_char[feature_class])] = default_representation
        self.char_to_int = {}
        for feature_class in self.int_to_char.keys():
            self.char_to_int[feature_class] = {v: k for k, v in self.int_to_char[feature_class].items()}
    
    def get_phonology(self, lang):
        # TODO: prepare festvox phonologies for languages lacking them
        phonology_out = {}
        harmonize_mapping = feature_harmonize[self.lang]
        phonology_path = os.path.join(language_resources_path,self.lang,phonology_langs[self.lang])
        with open(phonology_path) as f:
            phonology = json.load(f)
            vowel_classes = phonology["feature_types"][0]
            consonant_classes = phonology["feature_types"][1]
            for phone in phonology["phones"]:
                if(phone[1] == "silence"):
                    continue
                phonology_out[phone[0]] = {}
                if(phone[1] == "vowel"): # TODO: merge this and the next block somehow?
                    phonology_out[phone[0]]["_"] = "V"
                    for i in range(len(vowel_classes)-1):
                        cur_vc = vowel_classes[i+1]
                        if(cur_vc not in feature_harmonize["bn"].keys()):
                            continue
                        harmonize_class = harmonize_mapping[cur_vc]["_equivalent"]
                        harmonize_value = harmonize_mapping[cur_vc][phone[i+2]]
                        phonology_out[phone[0]][harmonize_class] = harmonize_value
                elif(phone[1] == "consonant"):
                    phonology_out[phone[0]]["_"] = "C"
                    for i in range(len(consonant_classes)-1):
                        cur_cc = consonant_classes[i+1]
                        if(cur_cc not in feature_harmonize["bn"].keys()):
                            continue
                        harmonize_class = harmonize_mapping[cur_cc]["_equivalent"]
                        harmonize_value = harmonize_mapping[cur_cc][phone[i+2]]
                        phonology_out[phone[0]][harmonize_class] = harmonize_value
        return phonology_out
    
    def get_lexicon(self, lang):
        lexicon = {}
        lexicon_path = os.path.join(language_resources_path,self.lang,lexicon_langs[self.lang])
        with open(lexicon_path) as f:
            for line in f.read().splitlines():
                if(line == ""):
                    continue
                if(line[0] == '#'):
                    continue
                line_tabsplit = line.split('\t')
                word = line_tabsplit[0]
                phones = line_tabsplit[1].split(' ')
                lexicon[word] = filter(lambda a : a != ".", phones)
        return lexicon

    def encode(self, text, feature_class=None):
        words = text.split(' ')
        encoded = []
        for word in words:
            # TODO: handle out-of-vocabulary words
            cur_word = self.lexicon[word]
            if(feature_class is None):
                encoded.extend(cur_word.split(' '))
            else:
                for phone in cur_word:
                    if (feature_class in self.phonology[phone].keys()):
                        encoded.append(self.phonology[phone][feature_class])
                    else:
                        encoded.append(self.phonology[phone]["_"])
            encoded.append(' ')
        return [self.char_to_int[feature_class][t] for t in encoded]
    
    def decode(self, sequence, feature_class=None):
        if(feature_class is None):
            return [self.int_to_char[" "][s] for s in sequence]
        return [self.int_to_char[feature_class][s] for s in sequence]

    def preprocess(self, sound, text):
        cur_input = file_log_spectrogram(sound)
        return ((cur_input - self.mean) / self.std), self.encode(text)

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def phone_size(self):
        return len(self.int_to_char[0])

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

def mean_stddev(sounds):
    samples = [file_log_spectrogram(soundfile)[:,:,:150] for soundfile in sounds]
    samples = torch.cat(samples)
    mean = torch.mean(samples, axis=0)
    std = torch.std(samples, axis=0)
    return mean, std

def file_log_spectrogram(sound):
    waveform, fs = torchaudio.load(sound)
    nperseg = int(20 * fs / 1000) # TODO: do not hardcode these
    noverlap = int(10 * fs / 1000)
    cur_input = torch.log(F.spectrogram(waveform,0,
                                    torch.hann_window(nperseg),
                                    nperseg,
                                    nperseg-noverlap,
                                    nperseg,
                                    2,0))
    return cur_input
