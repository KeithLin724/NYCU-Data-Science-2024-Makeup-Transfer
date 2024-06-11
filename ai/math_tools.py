import torch
from torch.autograd import Variable


def to_var(x: torch.Tensor, requires_grad: bool = True):
    if torch.cuda.is_available():
        x = x.cuda()

    if not requires_grad:
        return Variable(x, requires_grad=requires_grad)

    return Variable(x)


def de_norm(x: torch.Tensor):
    out = (x + 1) / 2
    return out.clamp(0, 1)
