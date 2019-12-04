from torch.autograd import Function
"""
method 1: implementation of GRL
x.register_hook(grl_hook(grl_coefficient))

method 2: implementation of GRL
x = grad_reverse(x, grl_coefficient)
"""


# TODO 待修复
def grl_hook(coeff):
    def func(grad):
        return -coeff * grad.clone()

    return func


class GradReverse(Function):
    def __init__(self, coeff):
        self.coeff = coeff

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad):
        return -self.coeff * grad


def grad_reverse(x, coeff=1.0):
    grl = GradReverse(coeff)
    return grl(x)
