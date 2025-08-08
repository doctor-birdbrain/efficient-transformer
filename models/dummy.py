# A dummy model, for testing

import torch
import torch.nn as nn

class DummyModel(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1) # don't flatten batch dim
        x = self.linear(x)
        x = self.softmax(x)
        return x
