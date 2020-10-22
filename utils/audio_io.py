import torch
import torchaudio
import torchaudio.functional as F

def mean_stddev(sounds):
    r"""Computes the mean and standard deviation of a list of sound files.
    """
    samples = [file_log_spectrogram(soundfile) for soundfile in sounds]
    samples = torch.cat(samples)
    mean = torch.mean(samples, axis=0)
    std = torch.std(samples, axis=0)
    return mean, std

def file_log_spectrogram(sound,segment_time=20,overlap_time=10):
    r"""Generates a spectrogram of a given sound file.
    """
    waveform, fs = torchaudio.load(sound)
    nperseg = int(segment_time * fs / 1000) # TODO: do not hardcode these
    noverlap = int(overlap_time * fs / 1000)
    cur_input = torch.log(F.spectrogram(waveform,0,
                                    torch.hann_window(nperseg),
                                    nperseg,
                                    nperseg-noverlap,
                                    nperseg,
                                    2,0) + 1e-10)
    return torch.squeeze(torch.transpose(cur_input,1,2))
