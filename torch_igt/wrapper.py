#!/usr/bin/env python3

import math
import torch as th
from torch.optim.optimizer import Optimizer, required


class IGTransporter(Optimizer):

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


class ExpIGT(IGTransporter):

    def __init__(self, params=required, opt=required, delta=1.0):
        self.opt = opt
        defaults = {
            'delta': delta,
            'num_steps': 0,
            'train': True,
            'num_resets': 0,
            'exp_power': 2,
        }
        super(IGTransporter, self).__init__(params, defaults)

    def step(self, closure=None):
        result = super(ExpIGT, self).step(closure)
        for group in self.param_groups:
            assert group['train'], 'Called step not in train mode.'

            num_steps = group['num_steps']
            num_resets = group['num_resets']
            exp_power = group['exp_power']
            if exp_power**num_resets == num_steps:
                group['num_resets'] = num_resets + 1
                group['num_steps'] = 0
                # Then we perform a reset
                for p in group['params']:
                    param_state = self.state[p]
                    # First, move the true params to shifted ones
                    true_p = param_state['true_p']
                    true_p.copy_(p.data)
                    # Second, zero-out the IGT buffers
                    param_state['igt_estimate'].mul_(0)
        return result


class SoftExpIGT(ExpIGT):

    def step(self, closure=None):
        result = super(ExpIGT, self).step(closure)
        for group in self.param_groups:
            assert group['train'], 'Called step not in train mode.'

            num_steps = group['num_steps']
            num_resets = group['num_resets']
            exp_power = group['exp_power']
            if exp_power**num_resets == num_steps:
                group['num_resets'] = num_resets + 1
                group['num_steps'] = 1
                # Then we perform a reset
                for p in group['params']:
                    param_state = self.state[p]
                    # Only, move the true params to shifted ones
                    true_p = param_state['true_p']
                    true_p.copy_(p.data)
        return result


class SoftResetExpIGT(ExpIGT):

    def step(self, closure=None):
        result = super(ExpIGT, self).step(closure)
        for group in self.param_groups:
            assert group['train'], 'Called step not in train mode.'

            num_steps = group['num_steps']
            num_resets = group['num_resets']
            exp_power = group['exp_power']
            if exp_power**num_resets == num_steps:
                group['num_resets'] = num_resets + 1
                group['num_steps'] = group['num_resets']
                # Then we perform a reset
                for p in group['params']:
                    param_state = self.state[p]
                    # Only, move the true params to shifted ones
                    true_p = param_state['true_p']
                    true_p.copy_(p.data)
        return result


class ExpIGTCont(IGTransporter):

    def __init__(self, params=required, opt=required, delta=1.0):
        self.opt = opt
        defaults = {
            'delta': delta,
            'num_steps': 0,
            'train': True,
            'num_resets': 0,
            'exp_power': 2,
        }
        super(IGTransporter, self).__init__(params, defaults)

    def step(self, closure=None):
        result = super(ExpIGTCont, self).step(closure)
        for group in self.param_groups:
            assert group['train'], 'Called step not in train mode.'

            num_steps = group['num_steps']
            num_resets = group['num_resets']
            exp_power = group['exp_power']
            if exp_power**num_resets == num_steps:
                group['num_resets'] = num_resets + 1
                group['num_steps'] = 0
                # Then we perform a reset
                for p in group['params']:
                    param_state = self.state[p]
                    # First, move the shifted params to true ones
                    true_p = param_state['true_p']
                    p.data.copy_(true_p)
                    # Second, zero-out the IGT buffers
                    param_state['igt_estimate'].mul_(0)
        return result


class Exp(Optimizer):

    def __init__(self, params=required, opt=required, delta=1):
        self.opt = opt
        defaults = {
            'delta': delta,
            'num_steps': 0,
            'num_resets': 0,
            'exp_power': 2,
        }
        super(Exp, self).__init__(params, defaults)

    def step(self, closure=None):
        # Replace each gradient with exponential average
        for group in self.param_groups:
            num_steps = group['num_steps']
            num_resets = group['num_resets']
            exp_power = group['exp_power']
            delta = group['delta']
            gamma = (num_steps) / (num_steps + delta)
            for p in group['params']:
                param_state = self.state[p]
                if 'exp_average' not in param_state:
                    # Init with stochastic gradient
                    param_state['exp_average'] = p.grad.data.clone()
                else:
                    # Compute new exponential average
                    p.grad.data.mul_(1.0 - gamma).add_(gamma,
                                                       param_state['exp_average'])
                    param_state['exp_average'].copy_(p.grad.data)
            group['num_steps'] += 1

        # Take optimization step
        result = self.opt.step(closure)

        # Reset buffers if necessary
        for group in self.param_groups:
            num_steps = group['num_steps']
            num_resets = group['num_resets']
            exp_power = group['exp_power']
            if exp_power**num_resets == num_steps:
                group['num_resets'] = num_resets + 1
                group['num_steps'] = 0
                # Then we perform a reset
                for p in group['params']:
                    param_state = self.state[p]
                    # Second, zero-out the averaging buffers
                    param_state['exp_average'].mul_(0)
        return result
