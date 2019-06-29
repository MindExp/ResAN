import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils.gradient_reverse_layer import grl_hook, grad_reverse


class _netF(nn.Module):
    def __init__(self):
        super(_netF, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 48, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm2d(48)

    def forward(self, x):
        x = torch.mean(x, 1).view(x.size()[0], 1, x.size()[2], x.size()[3])
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), stride=2, kernel_size=2, dilation=(1, 1))
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), stride=2, kernel_size=2, dilation=(1, 1))
        # print(x.size())
        x = x.view(x.size(0), 48 * 4 * 4)
        return x


class _netC(nn.Module):
    def __init__(self, prob=0.5):
        super(_netC, self).__init__()
        self.fc1 = nn.Linear(48 * 4 * 4, 100)
        self.bn1_fc = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, 100)
        self.bn2_fc = nn.BatchNorm1d(100)
        self.fc3 = nn.Linear(100, 10)
        self.bn_fc3 = nn.BatchNorm1d(10)
        self.prob = prob

    def forward(self, x, reverse=False, grl_coefficient=1.0):
        if reverse:
            # x.register_hook(grl_hook(grl_coefficient))
            x = grad_reverse(x, grl_coefficient)
        x = F.dropout(x, training=self.training, p=self.prob)
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.dropout(x, training=self.training, p=self.prob)
        x = F.relu(self.bn2_fc(self.fc2(x)))
        x = F.dropout(x, training=self.training, p=self.prob)
        x = self.fc3(x)
        return x


def get_netF(config=None):
    netF = _netF()

    return netF


def get_netC(config):
    netC = _netC()

    return netC