#!/usr/bin/env python3

import torch as th
from torch.optim.optimizer import Optimizer, required


class IGTransporter(Optimizer):

    def __init__(self, params=required, opt=required, delta=1.0):
        self.opt = opt
        defaults = {
            'delta': delta,
            'num_steps': 0,
            'train': True,
        }
        super(IGTransporter, self).__init__(params, defaults)

    def step(self, closure=None):
        for group in self.param_groups:
            delta = group['delta']
            num_steps = group['num_steps']
            gamma = (num_steps) / (num_steps + delta)
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
            delta = group['delta']
            num_steps = group['num_steps']
            future_gamma = (num_steps + 1) / (num_steps + 1 + delta)
            future_transport = future_gamma / (1.0 - future_gamma)
            for p in group['params']:
                true_p = self.state[p]['true_p']
                temp_p = p.data.clone()
                vector_change = p.data.add(-1.0, true_p)
                """
                TODO: The numerical problem is here.
                Essentially, computing vector change involves a subtraction,
                while computing the update with opt.step() is a multiplication.

                Subtraction is numerically unstable and hence the observed
                differences in algorithms.
                """
                p.data.add_(future_transport, vector_change)
                true_p.copy_(temp_p)
            group['num_steps'] += 1
        return loss

    def train(self):
        for group in self.param_groups:
            if not group['train']:
                for p in group['params']:
                    param_state = self.state[p]
                    true_p = param_state['true_p']
                    temp_p = p.data.clone()
                    p.data.copy_(true_p)
                    true_p.copy_(temp_p)
                group['train'] = True

    def eval(self):
        for group in self.param_groups:
            if group['train']:
                for p in group['params']:
                    param_state = self.state[p]
                    true_p = param_state['true_p']
                    temp_p = p.data.clone()
                    p.data.copy_(true_p)
                    true_p.copy_(temp_p)
                group['train'] = False
