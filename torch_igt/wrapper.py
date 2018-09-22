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
            future_gamma = (num_steps + 1) / (num_steps + 1 + delta)
            future_transport = future_gamma / (1.0 - future_gamma)
            for p in group['params']:

                d_p = p.grad.data
                param_state = self.state[p]

                if 'igt_estimate' not in param_state:
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
                loss = self.opt.step(closure)
                temp_p = p.data.clone()
                # Transport to the next parameter point
                p.data.add_(future_transport * (p.data - true_p))
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
