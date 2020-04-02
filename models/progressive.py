import math
import numpy as np
from models.single import LinearND

import torch
import torch.nn as nn
import torch.nn.functional as F

class ProgressiveNetwork(nn.Module):

    def __init__(self, priornets, freq_dim, output_dim, config):
        super(ProgressiveNetwork, self).__init__()
        self.priornets = priornets
        for priornet in priornets:
            for parameter in priornet.parameters():
                parameter.requires_grad = False

        # TODO: adjust for flexibility with alternate nets
        encoder_cfg = config["encoder"]
        convolution_cfg = encoder_cfg["conv"]
        self.dropout = config["dropout"]

        conv_layers = []
        in_channels = 1
        for out_channels, height, width, stride in convolution_cfg:
            conv = nn.Conv2d(in_channels, out_channels, (height, width), stride=(stride, stride), padding=0)
            conv_layers.extend([conv, nn.ReLU()])
            if(config["dropout"] != 0):
                conv_layers.append(nn.Dropout(p=self.dropout))
            in_channels = out_channels
        self.conv = nn.Sequential(*conv_layers)
        conv_out = out_channels * self.conv_out_size(freq_dim, 1)
        assert conv_out > 0, "Non-positive convolutional output frequency dimension."

        connections = nn.ModuleList()
        encoder_cfg = config["encoder"]
        rnn_cfg = encoder_cfg["rnn"]
        for k in rnn_cfg["layers"]:
            rnn = nn.GRU(input_size=(conv_out if k==0 else rnn_cfg["dim"]),
                        hidden_size=rnn_cfg["dim"],
                        batch_first=True,
                        bidirectional=rnn_cfg["bidirectional"])
            connections.append(rnn)
            connections.append(nn.Linear(rnn_cfg["dim"],rnn_cfg["dim"]))
        self.connections = connections
        self._encoder_dim = rnn_cfg["dim"]
        self.fc = LinearND(self._encoder_dim, output_dim+1)

    def forward(self, batch):
        x, y, x_lens, y_lens = self.collate(*batch)
        cur_layer = x
        if self.is_cuda:
            x = x.cuda()
        x = x.unsqueeze(1)
        x = self.conv(x)
        priornet_hidden_layers = [0 for net in self.priornets]
        for j in len(self.priornets):
            priornet_hidden_layers[j] = self.prionets[j].conv(x)
        for i in range(len(self.connections)//2):
            for j in len(self.priornets):
                priornet_hidden_layers[j], _ = self.priornets[j].rnns[i](priornet_hidden_layers[j])
                hidden_layer_sum += F.dropout(priornet_hidden_layers[j],p=self.dropout)
            x, h = self.connections[2*i](x)
            cur_layer = F.relu(hidden_layer_sum+F.dropout(x,p=self.dropout))
        cur_layer = self.fc(cur_layer)
        return cur_layer

    @property
    def is_cuda(self):
        return list(self.parameters())[0].is_cuda

    @property
    def encoder_dim(self):
        return self._encoder_dim

    # These are just simple settings.
    def set_eval(self):
        self.eval()
        self.volatile = True

    def set_train(self):
        self.train()
        self.volatile = False