import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.linearnd import LinearND
from utils.ctc_tools import zero_pad_concat, decode

def conv_out_size(convs, n, dim):
    r"""Returns the size of the output of a sequence of convolutional layers.
    """
    for c in convs.children():
        if type(c) == nn.Conv2d:
            kernel_size = c.kernel_size[dim]
            stride = c.stride[dim]
            n = int(math.ceil((n-kernel_size+1)/stride))
    return n

class BasicNetwork(nn.Module):

    def __init__(self, freq_dim, output_features, config):
        super(BasicNetwork, self).__init__()
        self.dim = freq_dim
        encoder_cfg = config["encoder"]
        convolution_cfg = encoder_cfg["conv"]
        recurrent_cfg = encoder_cfg["rnn"]
        self.dropout = config["dropout"]

        self.conv, out_channels = self.build_convolution_layers(convolution_cfg, self.dropout)
        conv_out = out_channels * conv_out_size(self.conv, self.dim, 1)
        assert conv_out > 0, "Non-positive convolutional output frequency dimension"

        # TODO: can I unpack fewer values?
        self.rnns, self._encoder_dim, self.rnn_bidirectional = self.build_recurrent_layers(recurrent_cfg, conv_out)

        self.blanks, self.fcs = self.build_fc_layers(output_features, self._encoder_dim)

    def build_convolution_layers(self, convolution_cfg, dropout):
        conv_layers = nn.ModuleList()
        in_channels = 1
        for out_channels, height, width, stride in convolution_cfg:
            conv = nn.Conv2d(in_channels, out_channels, (height, width),
                stride=(stride, stride), padding=0)
            conv_layers.extend([conv, nn.ReLU()])
            if(dropout != 0):
                conv_layers.append(nn.Dropout(p=dropout))
            in_channels = out_channels
        return nn.Sequential(*conv_layers), out_channels
    
    def build_recurrent_layers(self, recurrent_cfg, n):
        rnns = nn.ModuleList()
        rnn_bidirectional = recurrent_cfg["bidirectional"]
        num_directions = 2 if rnn_bidirectional else 1
        for i in range(recurrent_cfg["layers"]):
            rnns.append(nn.GRU(input_size=(n if i==0 else recurrent_cfg["dim"]*num_directions),
                            hidden_size=recurrent_cfg["dim"],
                            batch_first=True,
                            bidirectional=rnn_bidirectional))
        return rnns, recurrent_cfg["dim"], rnn_bidirectional

    def build_fc_layers(self, output_features, in_dim):
        blanks = {}
        fcs = nn.ModuleDict()
        for output_feature, output_dim in output_features:
            blanks[output_feature] = output_dim
            fcs[output_feature] = LinearND(in_dim, output_dim+1)
        return blanks, fcs

    def encode(self, x):
        r"""
        """
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
    
    def forward_batch(self, batch):
        r"""
        """
        x, y, x_lens, y_lens = self.collate(*batch)
        return self.forward(x)
    
    def forward(self, x, softmax=False):
        r"""
        """
        if self.is_cuda:
            x = x.cuda()
        x = self.encode(x)
        x_fcs = {}
        for feature_class, fc in self.fcs.items():
            current_fc_out = fc(x)
            if softmax:
                x_fcs[feature_class] = torch.nn.functional.softmax(current_fc_out, dim=2)
            else:
                x_fcs[feature_class] = current_fc_out
        return x_fcs

    def infer(self, inputs, labels):
        r"""
        """
        x, _, _, _ = self.collate(inputs, labels)
        probs = self.forward(x, softmax=True)
        for feature_class, prob in probs.items():
            prob = prob.data.cpu().numpy()
            # TODO: is there a way to speed this up for phones?
            prob = [decode(p, beam_size=1, blank=self.blanks[feature_class])[0] for p in prob]
            probs[feature_class] = prob
        return probs
    
    def loss(self, inputs, labels):
        r"""
        """
        x, y, x_lens, y_lens = self.collate(inputs, labels)
        outs = self.forward(x)

        loss_function = torch.nn.CTCLoss(reduction='none')
        losses = {}
        for feature_class, out in outs.items():
            losses[feature_class] = loss_function(torch.transpose(out,0,1), y, x_lens, y_lens)
        return losses
    
    def collate(self, inputs, labels):
        r"""
        """
        max_t = conv_out_size(self.conv, max(i.shape[0] for i in inputs), 0)
        x_lens = torch.IntTensor([max_t] * len(inputs))
        y_lens = torch.IntTensor([len(l) for l in labels])
        x = torch.FloatTensor(zero_pad_concat(inputs))
        y = torch.IntTensor([l for label in labels for l in label])
        batch = [x, y, x_lens, y_lens]
        return batch

    @staticmethod
    def max_decode(pred, blank):
        r"""
        """
        prev = pred[0]
        seq = [prev] if prev != blank else []
        for p in pred[1:]:
            if p != blank and p != prev:
                seq.append(p)
            prev = p
        return seq

    def set_eval(self):
        r"""
        """
        self.eval()

    def set_train(self):
        r"""
        """
        self.train()

    @property
    def is_cuda(self):
        r"""
        """
        return list(self.parameters())[0].is_cuda

    @property
    def encoder_dim(self):
        r"""
        """
        return self._encoder_dim

