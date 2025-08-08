# A dummy model, for testing

import torch.nn as nn

class DummyModel(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.linear(x)
        x = self.softmax(x)
        return x
