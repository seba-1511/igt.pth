#!/usr/bin/env python3

import torch as th
import torch.nn as nn
from torch import Tensor as T
from torch.autograd import Variable as V
import torch.nn.functional as F

import torch_igt


class Convnet(nn.Module):
    def __init__(self):
        super(Convnet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def dist(x, y):
    return (x - y).pow(2).sum()


def close(x, y):
    return dist(x, y) < 1e-8


if __name__ == '__main__':
    th.manual_seed(1234)
    model1 = Convnet().double()
    model2 = Convnet().double()
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        p1.data.copy_(p2.data)

    ref = torch_igt.MomentumIGT(model1.parameters(), lr=0.1, momentum=0.5)
    opt = th.optim.SGD(model2.parameters(), lr=0.1, momentum=0.5)
    igt = torch_igt.IGTransporter(model2.parameters(), opt)

    x = V(th.randn(3, 1, 28, 28).double(), requires_grad=False)

    for i in range(100):
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert close(p1.data, p2.data)

        # Compute reference gradients
        ref.train()
        ref.zero_grad()
        loss1 = model1.forward(x).pow(2).mean()
        loss1.backward()

        # Compute wrapper gradients
        igt.train()
        igt.zero_grad()
        loss2 = model2.forward(x).pow(2).mean()
        loss2.backward()

        assert close(loss1.data, loss2.data)

        # Test identical gradients
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert close(p1.grad.data, p2.grad.data)

        # Take on step
        ref.step()
        igt.step()

        # Test identical parameters (train)
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert close(p1.data, p2.data)

        # Test identical parameters (eval)
        ref.eval()
        igt.eval()
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert close(p1.data, p2.data)

        ref.train()
        igt.train()
