#!/usr/bin/env python3

import torch as th
import torch.nn as nn
from torch import Tensor as T
from torch.autograd import Variable as V

import torch_igt

"""
The reference values were hand-derived for this toy example,
following the algorithm from the paper.
(C.f. ./derivations/nesterov_igt.svg)
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
            'true_param': T([0.7, 0.85]).view(2, 1),
            'transported_param': T([0.4, 0.7]).view(2, 1),
            'igt_velocity': T([2, 1]).view(2, 1),
            'momentum_velocity': T([-0.2, -0.1]).view(2, 1),
        },

        {
            'transported_grad': T([0.8, 0.7]).view(2, 1),
            'true_param': T([0.44, 0.6975]).view(2, 1),
            'transported_param': T([-0.08, 0.3925]).view(2, 1),
            'igt_velocity': T([1.4, 0.85]).view(2, 1),
            'momentum_velocity': T([-0.24, -0.135]).view(2, 1),
        },
        {
            'transported_grad': T([-0.16, 0.3925]).view(2, 1),
            'true_param': T([0.248, 0.559125]).view(2, 1),
            'transported_param': T([-0.328, 0.144]).view(2, 1),
            'igt_velocity': T([0.88, 0.6975]).view(2, 1),
            'momentum_velocity': T([-0.208, -0.13725]).view(2, 1),
        },
        {
            'transported_grad': T([-0.656, 0.144]).view(2, 1),
            'true_param': T([0.1216, 0.44094375]).view(2, 1),
            'transported_param': T([-0.384, -0.0318]).view(2, 1),
            'igt_velocity': T([0.496, 0.559125]).view(2, 1),
            'momentum_velocity': T([-0.1536, -0.1245375]).view(2, 1),
        },

]


if __name__ == '__main__':
    model = Vector(th.ones(2, 1))
    opt = torch_igt.MomentumIGT(model.parameters(), lr=0.1,
                                momentum=0.5, nesterov=True)

    H = V(T([[2, 0], [0, 1]]))
    for i in range(4):
        opt.train()
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
        # Adjust for Pytorch's (equivalent) way of computing momentum
        assert(close(params['momentum_velocity'] * -0.1,
                     reference[i]['momentum_velocity']))
        assert(close(model().data,
                     reference[i]['transported_param']))

        opt.eval()
        assert(close(model().data,
                     reference[i]['true_param']))

        opt.train()
        assert(close(model().data,
                     reference[i]['transported_param']))
