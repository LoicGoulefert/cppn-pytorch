import torch
import torch.nn as nn
import numpy as np


class CPPN(nn.Module):
    def __init__(self, z_dim=32, model_size=32):
        super(CPPN, self).__init__()

        self.z_dim = z_dim
        self.model_size = model_size

        self.ln1 = nn.Linear(self.z_dim + 3, self.model_size, bias=False)
        self.ln2 = nn.Linear(self.model_size, self.model_size, bias=False)
        self.ln3 = nn.Linear(self.model_size, self.model_size, bias=False)
        self.ln4 = nn.Linear(self.model_size, 1, bias=False)
        nn.init.uniform_(self.ln1.weight, a=-1, b=1)
        nn.init.uniform_(self.ln2.weight, a=-1, b=1)
        nn.init.uniform_(self.ln3.weight, a=-1, b=1)
        nn.init.uniform_(self.ln4.weight, a=-1, b=1)
        self.tanh1 = nn.Tanh()
        self.tanh2 = nn.Tanh()
        self.tanh3 = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.ln1(x)
        out = self.tanh1(out)
        out = self.ln2(out)
        out = self.tanh2(out)
        out = self.ln3(out)
        out = self.tanh3(out)
        out = self.ln4(out)
        out = self.sigmoid(out)

        return out
