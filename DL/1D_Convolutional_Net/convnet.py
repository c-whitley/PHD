import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from sklearn.metrics import roc_auc_score, accuracy_score


def calc_lout(lin, pad, dil, k, strd):

    try:
        return ((lin + 2*pad[0]-dil[0]*(k[0]-1)-1)/strd[0]) + 1
    except:
        return ((lin + 2*pad-dil*(k-1)-1)/strd) + 1


class SpecConvNet(nn.Module):

    def __init__(self):

        # Run init method from nn.Module
        super().__init__()


    def build(self):

        self.stack = nn.Sequential(
                        nn.Conv1d(1, 3, 5),
                        nn.ReLU(),
                        nn.Dropout(0.25),
                        nn.MaxPool1d(7),
                        nn.Conv1d(3, 1, 1),
                        nn.ReLU(),
                        nn.Dropout(0.25),
                        nn.MaxPool1d(3),
                        nn.Linear(7, 10),
                        nn.ReLU(),
                        nn.Linear(10,2),
                        nn.Softmax(2)
                        )

    def Optuna_build(self, h_params: dict):

        stack = []

        for n_conv_layer, pool_size in zip(range(1, h_params['n_conv_layers']), h_params['pool_list']):

            layer = [
                nn.Conv1d(h_params['chan_list'][n_conv_layer-1], h_params['chan_list'][n_conv_layer], kernel_size=(3,)),
                nn.ReLU(),
                nn.MaxPool1d(pool_size, stride=1)
            ]

            stack = stack + layer

        stack.append(nn.Dropout(h_params['dropout']))
        stack.append(nn.Flatten())

        lin = 169

        # Calculate size of input for linear layer

        for l in stack:

            if isinstance(l, (nn.Conv1d, nn.MaxPool1d)):

                #print(vars(l))

                lout = calc_lout(lin
                            , vars(l).get('padding', (0,))
                            , vars(l).get('dilation', (0,))
                            , vars(l).get('kernel_size', (0,))
                            , vars(l).get('stride', (0,)))

                lin = lout

            else:

                pass

        stack.append(nn.Linear(int(lout*h_params['chan_list'][-2]), h_params['fc_neurons']))
        stack.append(nn.ReLU())
        stack.append(nn.Linear(h_params['fc_neurons'],1))
        stack.append(nn.Softmax(0))

        self.stack = nn.Sequential(*stack)

    def forward(self, x):

        seq = self.stack(x)

        return seq