import math
from torch.autograd import Function


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lamda):
        ctx.lamda = lamda
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = (grad_output.neg() * ctx.lamda)
        return output, None


def adjust_alpha(i, epoch, min_len, nepochs):
    p = float(i + epoch * min_len) / nepochs / min_len
    o = -10
    alpha = 2. / (1. + math.exp(o * p)) - 1
    return alpha
