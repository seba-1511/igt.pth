#!/usr/bin/env python3

import math
import torch as th
from torch.optim.optimizer import required

from torch_igt.wrapper import IGTransporter


def exp_ata_weight(c, num_steps):
    if num_steps < 2:
        return 0
    gamma = c * num_steps / (1.0 + c * num_steps)
    gamma *= 1.0 - math.sqrt((1.0 - c) / (num_steps * (num_steps + 1))) / c
    return gamma


class ITA(IGTransporter):

    """
    In this implementation, the following variables are defined as

    * igt_estimate: the IGT buffers that get updated with every new
      stochastic gradient.
    * true_p: keeps track of the unshifted parameters when training,
      and the model parameters are set at the shifted position. Conversely,
      when calling `eval()` the parameters of the model are set to the
      unshifted values, and true_p becomes the shifted ones. You can switch
      back using `train()`.
    """

    def __init__(self, params=required, opt=required, delta=1.0, interval=2.0):
        self.opt = opt
        defaults = {
            'delta': delta,
            'c': 1.0 / interval,
            'num_steps': 0.0,
            'train': True,
        }
        super(IGTransporter, self).__init__(params, defaults)

    def step(self, closure=None):
        for group in self.param_groups:
            num_steps = group['num_steps']
            c = group['c']

            # Compute the exponential ATA weighting
            gamma = exp_ata_weight(c, num_steps)
            for p in group['params']:

                if p.grad is None:
                    continue

                d_p = p.grad.data
                param_state = self.state[p]

                # Compute the IGT estimate
                if num_steps == 0:
                    param_state['igt_estimate'] = th.zeros_like(d_p)
                    param_state['true_p'] = th.zeros_like(p.data)
                    param_state['igt_estimate'].add_(d_p)
                    param_state['true_p'].add_(p.data)
                    true_p = param_state['true_p']
                else:
                    igt_estimate = param_state['igt_estimate']
                    true_p = param_state['true_p']
                    igt_estimate.mul_(gamma).add_((1.0 - gamma), d_p)
                    # Sets gradients to the IGT estimate
                    d_p.copy_(igt_estimate)
                    p.data.copy_(true_p)  # Revert to true params

        # Take the step according to opt
        loss = self.opt.step(closure)

        # Transport to the next parameter point
        for group in self.param_groups:
            num_steps = group['num_steps']

            # Compute the next exponential ATA weighting
            c = group['c']
            future_gamma = exp_ata_weight(c, num_steps + 1.0)
            future_transport = future_gamma / (1.0 - future_gamma)
            for p in group['params']:
                true_p = self.state[p]['true_p']
                temp_p = p.data.clone()
                vector_change = p.data.add(-1.0, true_p)
                """
                NOTE: The numerical problem is here.
                Essentially, computing vector change involves a subtraction,
                while computing the update with opt.step() is a multiplication.

                Subtraction is numerically unstable and hence the observed
                differences in algorithms.
                Note: this mainly depends on the loss computation. If it is
                stable, then using the parameter difference doesn't greatly
                diverge.
                """
                p.data.add_(future_transport, vector_change)
                true_p.copy_(temp_p)
            group['num_steps'] += 1.0
        return loss
