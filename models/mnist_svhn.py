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

    def forward(self, input):
        x = self.feature(input)
        return x


class _netC(nn.Module):
    def __init__(self, prob=0.5):
        super(_netC, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(8192, 3072),
            nn.BatchNorm1d(3072),
            nn.ReLU(inplace=True),
            nn.Dropout(p=prob),

            nn.Linear(3072, 2048),
            nn.BatchNorm1d(2048),

            nn.Linear(2048, 10)
        )

    def forward(self, input, reverse=False, grl_coefficient=1.0):
        x = input.view(input.size(0), 8192)
        if reverse:
            # x.register_hook(grl_hook(grl_coefficient))
            x = grad_reverse(x, grl_coefficient)
        x = self.classifier(x)
        return x


def get_netF(config=None):
    netF = _netF()

    return netF


def get_netC(config):
    netC = _netC()

    return netC
