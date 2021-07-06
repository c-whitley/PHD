import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class SpecConvNet(nn.Module):

    def __init__(self):

        # Run init method from nn.Module
        super().__init__()

        self.stack = nn.Sequential(
                        nn.Conv1d(1, 1, 5),
                        nn.MaxPool1d(3),
                        nn.Conv1d(1, 1, 5),
                        nn.Dropout(0.5),
                        nn.Linear(51, 1),
                        )

    def forward(self, x):

        seq = self.stack(x)

        return seq


class FTIR_Dataset(Dataset):

    def __init__(self, dataframe, y_label, transform=None, target_transform=None):
        self.y = dataframe.reset_index()[y_label].values
        self.X = dataframe.values
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):

        # Make X into tensor compatible with CONV1D layers
        spectra = torch.tensor(self.X[idx,:], dtype=torch.float).unsqueeze(-2)
        
        # Make y compatible with binary cross entropy loss
        label = torch.tensor(self.y[idx], dtype=torch.int).unsqueeze(0).unsqueeze(0)
    

        if self.transform:
            spectra = self.transform(spectra)
        if self.target_transform:
            label = self.target_transform(label)
        return spectra, label


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")