#!/usr/bin/env python3

import math
import torch as th
from torch.optim.optimizer import Optimizer, required


class AdamIGT(Optimizer):

    def __init__(self,
                 params=required,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0,
                 delta=1.0):
        defaults = {
            'delta': delta,
            'num_steps': 0,
            'train': True,
            'lr': lr,
            'betas': betas,
            'eps': eps,
            'weight_decay': weight_decay,
        }
        super(AdamIGT, self).__init__(params, defaults)

    def compute_update(self, p, param_state, group):
        exp_avg = param_state['exp_avg']
        exp_avg_sq = param_state['exp_avg_sq']
        beta1, beta2 = group['betas']
        lr = group['lr']
        grad = p.grad.data

        if group['weight_decay'] != 0:
            grad = grad.add(group['weight_decay'], p.data)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(1 - beta1, grad)
        exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

        denom = exp_avg_sq.sqrt().add_(group['eps'])

        # NOTE: the + 1 is because IGT and Adam don't count steps the same way.
        bias_correction1 = 1 - beta1 ** (group['num_steps'] + 1)
        bias_correction2 = 1 - beta2 ** (group['num_steps'] + 1)
        step_size = lr * math.sqrt(bias_correction2) / bias_correction1
        update = -step_size * (exp_avg / denom)
        return update

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            delta = group['delta']
            num_steps = group['num_steps']
            gamma = (num_steps) / (num_steps + delta)
            future_gamma = (num_steps + 1) / (num_steps + 1 + delta)
            future_transport = future_gamma / (1.0 - future_gamma)
            for p in group['params']:

                if p.grad is None:
                    continue

                d_p = p.grad.data
                param_state = self.state[p]

                # Compute the IGT estimate
                if num_steps == 0:
                    param_state['igt_estimate'] = th.zeros_like(d_p)
                    param_state['igt_estimate'].add_(d_p)
                    param_state['true_p'] = th.zeros_like(p.data)
                    param_state['true_p'].add_(p.data)
                    param_state['exp_avg'] = th.zeros_like(p.data)
                    param_state['exp_avg_sq'] = th.zeros_like(p.data)
                    true_p = param_state['true_p']
                else:
                    igt_estimate = param_state['igt_estimate']
                    true_p = param_state['true_p']
                    igt_estimate.mul_(gamma).add_((1.0 - gamma), d_p)
                    # Sets gradients to the IGT estimate
                    d_p.copy_(igt_estimate)
                    p.data.copy_(true_p)  # Revert to true params

                # Take the step according to opt
                update = self.compute_update(p, param_state, group)

                # Transport to the next parameter point
                true_p.copy_(p.data).add_(update)
                p.data.add_(1.0 + future_transport, update)
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
