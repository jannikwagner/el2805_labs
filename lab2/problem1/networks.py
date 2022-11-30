import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch


class Network1(nn.Module):
    def __init__(self, n, m, hidden_size=8):
        super(Network1, self).__init__()
        self.fc1 = nn.Linear(n, hidden_size)
        self.fc2 = nn.Linear(hidden_size, m)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
