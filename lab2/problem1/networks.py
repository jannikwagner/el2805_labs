import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch


class Network1(nn.Module):
    def __init__(self, n, m, hidden_size=8, hidden_layers=1, activation=nn.ReLU):
        super(Network1, self).__init__()
        layers = []
        last = n
        for i in range(hidden_layers):
            layers.append(nn.Linear(last, hidden_size))
            layers.append(activation())
            last = hidden_size
        layers.append(nn.Linear(last, m))
        self.sequence = nn.Sequential(*layers)

    def forward(self, x):
        return self.sequence(x)
