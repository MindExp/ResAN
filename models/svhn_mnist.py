import torch.nn as nn
from models.utils.gradient_reverse_layer import grl_hook, grad_reverse


class _netF(nn.Module):
    def __init__(self):
        super(_netF, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(stride=2, kernel_size=3, padding=1),

            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(stride=2, kernel_size=3, padding=1),

            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.bottleneck = nn.Sequential(
            nn.Linear(8192, 3072),
            nn.BatchNorm1d(3072),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )

    def forward(self, input):
        x = self.feature(input)
        x = self.bottleneck(x.view(x.size(0), 8192))
        return x


class _netC(nn.Module):
    def __init__(self):
        super(_netC, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(3072, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),

            nn.Linear(2048, 10)
        )

    def forward(self, x, reverse=False, grl_coefficient=1.0):
        if reverse:
            # x.register_hook(grl_hook(grl_coefficient))
            x = grad_reverse(x, grl_coefficient)
        x = self.classifier(x)
        return x


def get_netF(backbone=None):
    netF = _netF()

    return netF


def get_netC(backbone=None):
    netC = _netC()

    return netC
