import torch 
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


class ThreeLayer_BN(nn.Module):
    def __init__(self):
        super(ThreeLayer_BN, self).__init__()

        self.dense_1 = nn.Linear(16, 64)
        self.bn1 = nn.BatchNorm1d(64)

        self.dense_2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)

        self.dense_3 = nn.Linear(32, 32)
        self.bn3 = nn.BatchNorm1d(32)

        self.dense_4 = nn.Linear(32, 5)

        self.act = nn.ReLU()
        self.softmax = nn.Softmax(1)

        # self.init_weights()

    def init_weights(self):
        for module in self.named_modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_normal_(module.weight, torch.nn.init.calculate_gain("relu"))
                torch.nn.init.xavier_normal_(module.bias, torch.nn.init.calculate_gain("relu"))

    def forward(self, x):
        x = self.act(self.bn1(self.dense_1(x)))
        x = self.act(self.bn2(self.dense_2(x)))
        x = self.act(self.bn3(self.dense_3(x)))
        return self.softmax(self.dense_4(x))


def get_model(args):
    if args.batch_norm:
        return ThreeLayer_BN()
    return ThreeLayerMLP()
