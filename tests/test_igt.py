#!/usr/bin/env python3

import torch as th
import torch.nn as nn
from torch import Tensor as T
from torch.autograd import Variable as V

import torch_igt

"""
The reference values were hand-derived for this toy example,
following the algorithm from the paper.
(C.f. ./derivations/igt.svg)
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

        {
            'transported_grad': T([2, 1]).view(2, 1),
            'true_param': T([0.8, 0.9]).view(2, 1),
            'transported_param': T([0.6, 0.8]).view(2, 1),
            'igt_velocity': T([2, 1]).view(2, 1),
        },

        {
            'transported_grad': T([1.2, 0.8]).view(2, 1),
            'true_param': T([0.64, 0.81]).view(2, 1),
            'transported_param': T([0.32, 0.63]).view(2, 1),
            'igt_velocity': T([1.6, 0.9]).view(2, 1),
        },
        {
            'transported_grad': T([0.64, 0.63]).view(2, 1),
            'true_param': T([0.512, 0.729]).view(2, 1),
            'transported_param': T([0.128, 0.486]).view(2, 1),
            'igt_velocity': T([1.28, 0.81]).view(2, 1),
        },

]


if __name__ == '__main__':
    model = Vector(th.ones(2, 1))
    opt = torch_igt.MomentumIGT(model.parameters(), lr=0.1, momentum=0.0)

    H = V(T([[2, 0], [0, 1]]))
    for i in range(3):
        opt.zero_grad()
        loss = 0.5 * th.mm(model().t(),
                           th.mm(H, model()))
        loss.backward()
        assert(close(model.vector.grad.data,
                     reference[i]['transported_grad']))

        opt.step()
        params = opt.state[model.vector]
        assert(close(params['igt_velocity'],
                     reference[i]['igt_velocity']))
        assert(close(model().data,
                     reference[i]['transported_param']))

        opt.eval()
        assert(close(model().data,
                     reference[i]['true_param']))

        opt.train()
        assert(close(model().data,
                     reference[i]['transported_param']))
