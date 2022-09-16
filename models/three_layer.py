import torch.nn as nn


class ThreeLayerMLP(nn.Module):
    def __init__(self):
        super(ThreeLayerMLP, self).__init__()

        self.dense_1 = nn.Linear(16, 64)
        self.dense_2 = nn.Linear(64, 32)
        self.dense_3 = nn.Linear(32, 32)
        self.dense_4 = nn.Linear(32, 5)

        self.act = nn.ReLU()
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        x = self.act(self.dense_1(x))
        x = self.act(self.dense_2(x))
        x = self.act(self.dense_3(x))
        return self.softmax(self.dense_4(x))


def three_layer_mlp():
    return ThreeLayerMLP()
