import torch.nn as nn
import torch.nn.functional as F


class SpecConvNet(nn.Module):
    def __init__(self):
        super(SpecConvNet, self).__init__()

        self.stack = nn.Sequential(
        nn.Conv1d(1, 1, 5),
        nn.MaxPool1d(3),
        nn.Conv1d(1, 1, 5),
        nn.Dropout(0.5),
        nn.Linear(5, 2),
        )

    def forward(self, x):

        return self.stack(x)