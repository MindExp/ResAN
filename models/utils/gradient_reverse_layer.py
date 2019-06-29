from torch.autograd import Function


def grl_hook(coeff):
    def func(grad):
        return -coeff * grad.clone()

    return func


class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return -self.lambd * grad_output


def grad_reverse(x, lambd=1.0):
    grl = GradReverse(lambd)
    return grl(x)
