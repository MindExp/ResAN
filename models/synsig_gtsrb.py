import torch.nn as nn
import torch.nn.functional as F
from models.utils.gradient_reverse_layer import grl_hook, grad_reverse


class _netF(nn.Module):
    def __init__(self):
        super(_netF, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(96)
        self.conv2 = nn.Conv2d(96, 144, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(144)
        self.conv3 = nn.Conv2d(144, 256, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(256)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), stride=2, kernel_size=2, padding=0)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), stride=2, kernel_size=2, padding=0)
        x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), stride=2, kernel_size=2, padding=0)
        x = x.view(x.size(0), 6400)
        return x

    def get_parameters(self):
        return self.parameters()


class _netC(nn.Module):
    def __init__(self):
        super(_netC, self).__init__()
        self.fc2 = nn.Linear(6400, 512)
        self.bn2_fc = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 43)
        self.bn_fc3 = nn.BatchNorm1d(43)

    def forward(self, x, reverse=False, grl_coefficient=1.0):
        if reverse:
            # x.register_hook(grl_hook(grl_coefficient))
            x = grad_reverse(x, grl_coefficient)
        x = F.relu(self.bn2_fc(self.fc2(x)))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return x

    def get_parameters(self):
        return self.parameters()


def get_netF(backbone=None):
    netF = _netF()

    return netF


def get_netC(backbone=None):
    netC = _netC()

    return netC