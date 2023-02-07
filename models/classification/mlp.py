import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, dim, num_classes=10, embSize=256):
        super(MLP, self).__init__()
        self.embSize = embSize
        self.dim = dim
        self.lm1 = nn.Sequential(
            nn.Linear(self.dim, embSize*4),
            nn.ReLU(),
            nn.Linear(embSize*4, embSize*2),
            nn.ReLU(),
            nn.Linear(embSize*2, embSize),
            nn.ReLU()
        )
        self.lm2 = nn.Linear(embSize, num_classes)
        self.penultimate_layer = None

    def forward(self, x):
        x = x.view(-1, self.dim)
        emb = self.lm1(x)
        self.penultimate_layer = emb
        out = self.lm2(emb)
        return out

    def get_penultimate_dim(self):
        return self.embSize