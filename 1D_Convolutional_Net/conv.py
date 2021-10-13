import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from sklearn.metrics import roc_auc_score, accuracy_score

class SpecConvNet(nn.Module):

    def __init__(self):

        # Run init method from nn.Module
        super().__init__()


    def build(self):

        self.stack = nn.Sequential(
                        nn.Conv1d(3, 8, 5),
                        nn.ReLU(),
                        nn.Dropout(0.25),
                        nn.MaxPool1d(5),
                        nn.Conv1d(8, 1, 3),
                        nn.ReLU(),
                        nn.Dropout(0.25),
                        nn.MaxPool1d(3),
                        nn.Linear(10, 8),
                        nn.ReLU(),
                        nn.Linear(8,2),
                        nn.Softmax(2)
                        )

    def forward(self, x):

        seq = self.stack(x)

        return seq

class FTIR_Dataset_C(Dataset):

    def __init__(self, dataframe, y_label, transform=None, target_transform=None):
        self.y = y_label
        self.X = dataframe
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):

        # Make X into tensor compatible with CONV1D layers
        spectra = torch.tensor(self.X[idx,:], dtype=torch.float)
        spectra = torch.rot90(spectra)#, 1, 0)#.unsqueeze(0)
        #spectra = self.X[idx,:]
        
        # Make y compatible with binary cross entropy loss
        label = torch.tensor(self.y[idx], dtype=torch.float).unsqueeze(0)#.unsqueeze(0)
    

        if self.transform:
            spectra = self.transform(spectra)
        if self.target_transform:
            label = self.target_transform(label)
        return spectra, label

class FTIR_Dataset(Dataset):

    def __init__(self, dataframe, y_label, transform=None, target_transform=None):
        self.y = y_label
        self.X = dataframe
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):

        # Make X into tensor compatible with CONV1D layers
        spectra = torch.tensor(self.X[idx,:], dtype=torch.float).unsqueeze(-2)
        #spectra = self.X[idx,:]
        
        # Make y compatible with binary cross entropy loss
        label = torch.tensor(self.y[idx], dtype=torch.float).unsqueeze(0)#.unsqueeze(0)
    

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

            probs = pred.numpy().squeeze()[:,1]

            auc = roc_auc_score(y.numpy().squeeze()[:,1], probs)
            accuracy = accuracy_score(y.numpy().squeeze()[:,1], (probs>0.5))

    test_loss /= num_batches
    correct /= size
    print(f'ROC: {auc}')
    print(f"Test Error: \n Accuracy: {accuracy:>0.1f}%, Avg loss: {test_loss:>8f} \n")