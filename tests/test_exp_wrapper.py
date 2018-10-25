#!/usr/bin/env python3

import torch as th
import torch.nn as nn
from torch import Tensor as T
from torch.autograd import Variable as V

import torch_igt


class Vector(nn.Module):
    def __init__(self, init):
        super(Vector, self).__init__()
        self.vector = nn.Parameter(init)

    def forward(self):
        return self.vector


def dist(x, y):
    return (x - y).pow(2).sum()


def close(x, y):
    return (x - y).pow(2).sum() < 1e-8


if __name__ == '__main__':
    H = V(T([[2, 0], [0, 1]]))
    model = Vector(T([1, 1]).view(2, 1))
    opt = th.optim.SGD(model.parameters(), lr=0.1,
                       momentum=0.0, weight_decay=0.0)
    opt = torch_igt.Exp(model.parameters(), opt=opt, delta=1)

    xs = [
        T([1, 1]).view(2, 1),
        T([0.8, 0.9]).view(2, 1),
        T([0.64, 0.81]).view(2, 1),
        T([0.496, 0.7245]).view(2, 1),
        T([0.3968, 0.65205]).view(2, 1),
        T([0.30752, 0.5832225]).view(2, 1),
    ]
    for i in range(6):
        # Compute one step on the reference
        assert close(xs[i], model.vector.data)
        opt.zero_grad()
        loss = 0.5 * th.mm(model().t(), th.mm(H, model()))
        loss.backward()
        opt.step()
