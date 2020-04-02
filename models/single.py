import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import collections

class BasicNetwork(nn.Module):

    def __init__(self, freq_dim, output_dim, config):
        super(BasicNetwork, self).__init__()
        self.dim = freq_dim
        encoder_cfg = config["encoder"]
        convolution_cfg = encoder_cfg["conv"]
        self.dropout = config["dropout"]

        conv_layers = []
        in_channels = 1
        for out_channels, height, width, stride in convolution_cfg:
            conv = nn.Conv2d(in_channels, out_channels, (height, width),
                stride=(stride, stride), padding=0)
            conv_layers.extend([conv, nn.ReLU()])
            if(self.dropout != 0):
                conv_layers.append(nn.Dropout(p=self.dropout))
            in_channels = out_channels
        self.conv = nn.Sequential(*conv_layers)
        conv_out = out_channels * self.conv_out_size(self.dim, 1)
        assert conv_out > 0, "Non-positive convolutional output frequency dimension."

        rnn_cfg = encoder_cfg["rnn"]
        rnns = nn.ModuleList()
        self.rnn_bidirectional = rnn_cfg["bidirectional"]
        num_directions = 2 if self.rnn_bidirectional else 1
        for i in range(rnn_cfg["layers"]):
            rnns.append(nn.GRU(input_size=(conv_out if i==0 else rnn_cfg["dim"]*num_directions),
                            hidden_size=rnn_cfg["dim"],
                            batch_first=True,
                            bidirectional=self.rnn_bidirectional))
        self.rnns = rnns
        self._encoder_dim = rnn_cfg["dim"]
        self.volatile = False
        self.blank = output_dim
        self.fc = LinearND(self._encoder_dim, output_dim+1)

    def conv_out_size(self, n, dim):
        for c in self.conv.children():
            if type(c) == nn.Conv2d:
                kernel_size = c.kernel_size[dim]
                stride = c.stride[dim]
                n = int(math.ceil((n-kernel_size+1)/stride))
        return n
    
    def encode(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)

        x = torch.transpose(x, 1, 2).contiguous()
        batch, time, freq, channels = x.size()
        x = x.view((batch, time, freq*channels))

        for layernum, layer in enumerate(self.rnns):
            if(layernum != 0):
                if(self.dropout != 0):
                    x = F.dropout(x,p=self.dropout)
            x, h = layer(x)
        if self.rnn_bidirectional:
            half = x.size()[-1] // 2
            x = x[:, :, :half] + x[:, :, half:]
        return x
    
    def forward(self, batch):
        x, y, x_lens, y_lens = self.collate(*batch)
        return self.forward_impl(x)
    
    def forward_impl(self, x, softmax=False):
        if self.is_cuda:
            x = x.cuda()
        x = self.encode(x)
        x = self.fc(x)
        if softmax:
            return torch.nn.functional.softmax(x, dim=2)
        return x

    def infer(self, inputs, labels):
        x, y, x_lens, y_lens = self.collate(inputs, labels)
        probs = self.forward_impl(x, softmax=True)
        probs = probs.data.cpu().numpy()
        return [decode(p, beam_size=1, blank=self.blank)[0] for p in probs]
    
    def loss(self, inputs, labels):
        x, y, x_lens, y_lens = self.collate(inputs, labels)
        out = self.forward_impl(x)
        loss_function = torch.nn.CTCLoss(reduction='none')
        return loss_function(torch.transpose(out,0,1), y, x_lens, y_lens)
    
    def collate(self, inputs, labels):
        max_t = self.conv_out_size(max(i.shape[0] for i in inputs), 0)
        x_lens = torch.IntTensor([max_t] * len(inputs))
        y_lens = torch.IntTensor([len(l) for l in labels])
        x = torch.FloatTensor(zero_pad_concat(inputs))
        y = torch.IntTensor([l for label in labels for l in label])
        batch = [x, y, x_lens, y_lens]
        return batch

    @staticmethod
    def max_decode(pred, blank):
        prev = pred[0]
        seq = [prev] if prev != blank else []
        for p in pred[1:]:
            if p != blank and p != prev:
                seq.append(p)
            prev = p
        return seq
    # These are just simple settings.
    def set_eval(self):
        self.eval()
        self.volatile = True

    def set_train(self):
        self.train()
        self.volatile = False

    @property
    def is_cuda(self):
        return list(self.parameters())[0].is_cuda

    @property
    def encoder_dim(self):
        return self._encoder_dim

NEG_INF = -float("inf")

class LinearND(nn.Module):
    def __init__(self, *args):
        super(LinearND, self).__init__()
        self.fc = nn.Linear(*args)
    
    def forward(self, x):
        size = x.size()
        n = int(np.prod(size[:-1]))
        out = x.contiguous().view(n, size[-1])
        out = self.fc(out)
        size = list(size)
        size[-1] = out.size()[-1]
        return out.view(size)

def zero_pad_concat(inputs):
    max_t = max(inp.shape[0] for inp in inputs)
    shape = (len(inputs), max_t, inputs[0].shape[1])
    input_mat = np.zeros(shape, dtype=np.float32)
    for e, inp in enumerate(inputs):
        input_mat[e, :inp.shape[0], :] = inp
    return input_mat

def make_new_beam():
    fn = lambda : (NEG_INF, NEG_INF)
    return collections.defaultdict(fn)

def logsumexp(*args):
    """
    Stable log sum exp.
    """
    if all(a == NEG_INF for a in args):
        return NEG_INF
    a_max = max(args)
    lsp = math.log(sum(math.exp(a - a_max)
                       for a in args))
    return a_max + lsp

def decode(probs, beam_size=10, blank=0):
    """
    Performs inference for the given output probabilities.

    Arguments:
      probs: The output probabilities (e.g. post-softmax) for each
        time step. Should be an array of shape (time x output dim).
      beam_size (int): Size of the beam to use during inference.
      blank (int): Index of the CTC blank label.

    Returns the output label sequence and the corresponding negative
    log-likelihood estimated by the decoder.
    """
    T, S = probs.shape
    probs = np.log(probs)

    # Elements in the beam are (prefix, (p_blank, p_no_blank))
    # Initialize the beam with the empty sequence, a probability of
    # 1 for ending in blank and zero for ending in non-blank
    # (in log space).
    beam = [(tuple(), (0.0, NEG_INF))]

    for t in range(T): # Loop over time

        # A default dictionary to store the next step candidates.
        next_beam = make_new_beam()

        for s in range(S): # Loop over vocab
            p = probs[t, s]

            # The variables p_b and p_nb are respectively the
            # probabilities for the prefix given that it ends in a
            # blank and does not end in a blank at this time step.
            for prefix, (p_b, p_nb) in beam: # Loop over beam

                # If we propose a blank the prefix doesn't change.
                # Only the probability of ending in blank gets updated.
                if s == blank:
                  n_p_b, n_p_nb = next_beam[prefix]
                  n_p_b = logsumexp(n_p_b, p_b + p, p_nb + p)
                  next_beam[prefix] = (n_p_b, n_p_nb)
                  continue

                # Extend the prefix by the new character s and add it to
                # the beam. Only the probability of not ending in blank
                # gets updated.
                end_t = prefix[-1] if prefix else None
                n_prefix = prefix + (s,)
                n_p_b, n_p_nb = next_beam[n_prefix]
                if s != end_t:
                  n_p_nb = logsumexp(n_p_nb, p_b + p, p_nb + p)
                else:
                  # We don't include the previous probability of not ending
                  # in blank (p_nb) if s is repeated at the end. The CTC
                  # algorithm merges characters not separated by a blank.
                  n_p_nb = logsumexp(n_p_nb, p_b + p)

                # *NB* this would be a good place to include an LM score.
                next_beam[n_prefix] = (n_p_b, n_p_nb)

                # If s is repeated at the end we also update the unchanged
                # prefix. This is the merging case.
                if s == end_t:
                  n_p_b, n_p_nb = next_beam[prefix]
                  n_p_nb = logsumexp(n_p_nb, p_nb + p)
                  next_beam[prefix] = (n_p_b, n_p_nb)

        # Sort and trim the beam before moving on to the
        # next time-step.
        beam = sorted(next_beam.items(),
                key=lambda x : logsumexp(*x[1]),
                reverse=True)
        beam = beam[:beam_size]

    best = beam[0]
    return best[0], -logsumexp(*best[1])
