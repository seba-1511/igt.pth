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


def close(x, y):
    return (x - y).pow(2).sum() < 1e-8


if __name__ == '__main__':
    H = V(T([[2, 0], [0, 1]]))
    model1 = Vector(th.ones(2, 1))
    model2 = Vector(th.ones(2, 1))

    ref = torch_igt.MomentumIGT(model1.parameters(), lr=0.1, momentum=0.5)
    opt = th.optim.SGD(model2.parameters(), lr=0.1, momentum=0.5)
    igt = torch_igt.IGTransporter(model2.parameters(), opt)

    for i in range(100):
        # Compute one step on the reference
        ref.train()
        ref.zero_grad()
        loss1 = 0.5 * th.mm(model1().t(),
                            th.mm(H, model1()))
        loss1.backward()
        ref.step()

        # Compute 1 step on the wrapper
        igt.train()
        igt.zero_grad()
        loss2 = 0.5 * th.mm(model2().t(),
                            th.mm(H, model2()))
        loss2.backward()
        igt.step()

        # Test identical parameters (train and eval)
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert close(p1.data, p2.data)

        ref.eval()
        igt.eval()
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert close(p1.data, p2.data)

        ref.train()
        igt.train()
