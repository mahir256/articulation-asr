import math
import numpy as np

from models.linearnd import LinearND
from utils.ctc_tools import zero_pad_concat, decode

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.nn_manipulation import graves_const

class ProgressiveNetwork(nn.Module):

    def __init__(self, freq_dim, output_dim, config, priornets, currentnet):
        super(ProgressiveNetwork, self).__init__()
        self.priornets = priornets
        self.currentnet = currentnet
        for priornet in priornets:
            for parameter in priornet.parameters():
                parameter.requires_grad = False

        # TODO: adjust for flexibility with alternate nets
        encoder_cfg = config["encoder"]
        convolution_cfg = encoder_cfg["conv"]
        rnn_cfg = encoder_cfg["rnn"]

        num_directions = 2 if self.currentnet.rnn_bidirectional else 1
        self.gates = []
        for rnn in self.currentnet.rnns:
            gate = nn.Linear(rnn_cfg["dim"]*num_directions,rnn_cfg["dim"]*num_directions)
            gate.apply(graves_const)
            gate.cuda() if currentnet.is_cuda else gate.cpu()
            self.gates.append(gate)

        self.dropout = config["dropout"]
        self._encoder_dim = rnn_cfg["dim"]
        self.volatile = False
        self.blank = output_dim

    def encode(self, x):
        r"""
        """
        x = x.unsqueeze(1)
        priornet_hidden_layers = [0 for net in self.priornets]
        for j in range(len(self.priornets)):
            z = self.priornets[j].conv(x)
            z = torch.transpose(z, 1, 2).contiguous()
            batch, time, freq, channels = z.size()
            z = z.view((batch, time, freq*channels))
            priornet_hidden_layers[j] = z
        x = self.currentnet.conv(x)
        x = torch.transpose(x, 1, 2).contiguous()
        batch, time, freq, channels = x.size()
        x = x.view((batch, time, freq*channels))
        cur_layer = x
        for i in range(len(self.currentnet.rnns)):
            hidden_layer_sum = 0.
            for j in range(len(self.priornets)):
                priornet_hidden_layers[j], _ = self.priornets[j].rnns[i](priornet_hidden_layers[j])
                if(i != len(self.currentnet.rnns)-1 and self.dropout != 0):
                    priornet_hidden_layers[j] = F.dropout(priornet_hidden_layers[j],p=self.dropout)
                hidden_layer_sum += priornet_hidden_layers[j]
            cur_layer, _ = self.currentnet.rnns[i](cur_layer)
            if(i != len(self.currentnet.rnns)-1 and self.dropout != 0):
                cur_layer = F.dropout(cur_layer,p=self.dropout)
            input_to_relu = self.gates[i](cur_layer + hidden_layer_sum)
            cur_layer = F.relu(input_to_relu)
        if self.currentnet.rnn_bidirectional:
            half = cur_layer.size()[-1] // 2
            cur_layer = cur_layer[:, :, :half] + cur_layer[:, :, half:]
        return cur_layer

    def forward_impl(self, x, softmax=False):
        r"""
        """
        if self.is_cuda:
            x = x.cuda()
        x = self.encode(x)
        x = self.currentnet.fc(x)
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
        max_t = self.currentnet.conv_out_size(max(i.shape[0] for i in inputs), 0)
        x_lens = torch.IntTensor([max_t] * len(inputs))
        y_lens = torch.IntTensor([len(l) for l in labels])
        x = torch.FloatTensor(zero_pad_concat(inputs))
        y = torch.IntTensor([l for label in labels for l in label])
        batch = [x, y, x_lens, y_lens]
        return batch

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