#!/usr/bin/env python3

from torch.optim.Optimizer import Optimizer, required


class MomentumIGT(Optimizer):

    """
    Implementation of Implicit Gradient Transport, and its Heavyball and Nesterov versions.

    Arguments:

    Example:

    """

    def __init__(self, params, lr=required, momentum=required, delta=1.0, nesterov=False):
        """
        Arguments:
        """
        defaults = {
                'lr': lr,
                'momentum': momentum,
                'delta': delta,
                'nesterov': nesterov,
        }
        super(MomentumIGT, self).__init__(params, defaults)
        self.train()

    def __setstate__(self, state):
        super(MomentumIGT, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        return loss

    def train(self):
        """
        Set the parameters of the model to the transported ones. (for gradient computation)
        """
        pass

    def eval(self):
        """
        Set the parameters of the model to the best ones. (for evaluation)
        """
        pass
