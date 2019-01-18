#!/usr/bin/env python3

import torch as th
import torch.nn as nn
import torch.optim as optim
from torch import Tensor as T
from torch.autograd import Variable as V

import torch_igt

"""
The reference values were hand-derived for this toy example,
following the algorithm from the paper.
(C.f. ./derivations/ncigt.pdf)
"""


class Vector(nn.Module):
    def __init__(self, init):
        super(Vector, self).__init__()
        self.vector = nn.Parameter(init)

    def forward(self):
        return self.vector


def close(x, y):
    return (x - y).pow(2).sum() < 1e-8


reference = [

        {}, # In NC-IGT, t is init at 1
        {
            'theta': T([1, 1]).view(2, 1),
            'theta_tilde': T([1, 1]).view(2, 1),
            'g_tilde': T([2, 1]).view(2, 1),
            'g_hat': T([2, 1]).view(2, 1),
        },
        {
            'theta': T([0.8, 0.9]).view(2, 1),
            'theta_tilde': T([0.8, 0.9]).view(2, 1),
            'g_tilde': T([1.6, 0.9]).view(2, 1),
            'g_hat': T([1.6, 0.9]).view(2, 1),
        },
        {
            'theta': T([0.64, 0.81]).view(2, 1),
            'theta_tilde': T([0.56, 0.765]).view(2, 1),
            'g_tilde': T([1.12, 0.765]).view(2, 1),
            'g_hat': T([1.28, 0.81]).view(2, 1),
        },
        {
            'theta': T([0.512, 0.729]).view(2, 1),
            'theta_tilde': T([0.464, 0.693]).view(2, 1),
            'g_tilde': T([0.928, 0.693]).view(2, 1),
            'g_hat': T([1.024, 0.729]).view(2, 1),
        },
        {
            'theta': T([0.4096, 0.6561]).view(2, 1),
            'theta_tilde': T([0.256, 0.54675]).view(2, 1),
            'g_tilde': T([0.512, 0.54675]).view(2, 1),
            'g_hat': T([0.8192, 0.6561]).view(2, 1),
        },
        {
            'theta': T([0.32768, 0.59049]).view(2, 1),
            'theta_tilde': T([0.21504, 0.49572]).view(2, 1),
            'g_tilde': T([0.43008, 0.49572]).view(2, 1),
            'g_hat': T([0.65536, 0.59049]).view(2, 1),
        },
        {
            'theta': T([0.262144, 0.531441]).view(2, 1),
            'theta_tilde': T([0.1905, 0.4531]).view(2, 1),
            'g_tilde': None,
            'g_hat': None,
        },


]


if __name__ == '__main__':
    model = Vector(th.ones(2, 1))
    opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.0)
    opt = torch_igt.NCIGT(model.parameters(), opt=opt)

    H = V(T([[2, 0], [0, 1]]))
    for t in range(1, 1+6):
        opt.train()
        assert close(model.vector.data,
                     reference[t]['theta_tilde'])
        opt.zero_grad()
        loss = 0.5 * th.mm(model().t(),
                           th.mm(H, model()))
        loss.backward()
        assert close(model.vector.data,
                     reference[t]['theta_tilde'])
        assert close(model.vector.grad.data,
                     reference[t]['g_tilde'])

        opt.step()
        assert close(model.vector.grad.data,
                     reference[t]['g_hat'])

        assert close(model.vector.data,
                     reference[t+1]['theta_tilde'])
        opt.eval()
        assert close(model.vector.data,
                     reference[t+1]['theta'])

        opt.train()
        assert close(model.vector.data,
                     reference[t+1]['theta_tilde'])

