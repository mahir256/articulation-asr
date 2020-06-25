import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.linearnd import LinearND
from utils.ctc_tools import zero_pad_concat, decode

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
        r"""
        """
        for c in self.conv.children():
            if type(c) == nn.Conv2d:
                kernel_size = c.kernel_size[dim]
                stride = c.stride[dim]
                n = int(math.ceil((n-kernel_size+1)/stride))
        return n
    
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
    
    def forward(self, batch):
        r"""
        """
        x, y, x_lens, y_lens = self.collate(*batch)
        return self.forward_impl(x)
    
    def forward_impl(self, x, softmax=False):
        r"""
        """
        if self.is_cuda:
            x = x.cuda()
        x = self.encode(x)
        x = self.fc(x)
        if softmax:
            return torch.nn.functional.softmax(x, dim=2)
        return x

    def infer(self, inputs, labels):
        r"""
        """
        x, y, x_lens, y_lens = self.collate(inputs, labels)
        probs = self.forward_impl(x, softmax=True)
        probs = probs.data.cpu().numpy()
        return [decode(p, beam_size=1, blank=self.blank)[0] for p in probs]
    
    def loss(self, inputs, labels):
        r"""
        """
        x, y, x_lens, y_lens = self.collate(inputs, labels)
        out = self.forward_impl(x)
        loss_function = torch.nn.CTCLoss(reduction='none')
        return loss_function(torch.transpose(out,0,1), y, x_lens, y_lens)
    
    def collate(self, inputs, labels):
        r"""
        """
        max_t = self.conv_out_size(max(i.shape[0] for i in inputs), 0)
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
        self.volatile = True

    def set_train(self):
        r"""
        """
        self.train()
        self.volatile = False

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

