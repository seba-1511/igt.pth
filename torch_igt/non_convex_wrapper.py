#!/usr/bin/env python3

import torch as th
from torch.optim.optimizer import Optimizer, required
from torch_igt.wrapper import IGTransporter


class NCIGT(IGTransporter, Optimizer):
    """ Implementation of Non-Convex IGT. """

    def __init__(self, params=required, opt=required, interval=2.0):
        self.opt = opt
        defaults = {
            'num_steps': 1,
            'train': True,
            'interval': interval,
        }
        super(IGTransporter, self).__init__(params, defaults)

    def step(self, closure=None):
        for group in self.param_groups:
            t = group['num_steps']
            c = group['interval']
            for p in group['params']:

                if p.grad is None:
                    continue

                d_p = p.grad.data
                param_state = self.state[p]

                # Compute the NC-IGT estimate
                if t == 1:
                    param_state['true_p'] = th.zeros_like(p.data)
                    param_state['true_p'].add_(p.data)
                    param_state['g_hat_1'] = th.zeros_like(d_p)
                    param_state['g_hat_2'] = th.zeros_like(d_p)
                    param_state['theta_hat_1'] = th.zeros_like(p.data)
                    param_state['theta_hat_1'].add_(p.data)
                    param_state['theta_hat_2'] = th.zeros_like(p.data)
                    param_state['N_t'] = 0
                else:
                    # Compute the gradient estimate
                    true_p = param_state['true_p']
                    g_hat_1 = param_state['g_hat_1']
                    g_hat_2 = param_state['g_hat_2']
                    theta_hat_1 = param_state['theta_hat_1']
                    theta_hat_2 = param_state['theta_hat_2']
                    N_t = param_state['N_t']
                    g_hat_1.mul_((N_t-1)/N_t).add_(1.0/N_t, d_p)
                    d_p.mul_(0).add_(g_hat_2).add_(c*N_t/t, g_hat_1 - g_hat_2)

                    # Perform a reset
                    if N_t >= t / c:
                        g_hat_2.copy_(g_hat_1)
                        theta_hat_2.copy_(theta_hat_1)
                        g_hat_1.mul_(0)
                        param_state['N_t'] = 0

                    # Revert to true params for update
                    p.data.copy_(true_p)

        # Take the optimization step
        result = self.opt.step(closure)

        # Compute and set transported params theta_tilde
        for group in self.param_groups:
            group['num_steps'] += 1
            t = group['num_steps']
            for p in group['params']:

                if p.grad is None:
                    continue

                param_state = self.state[p]
                param_state['N_t'] += 1
                N_t = param_state['N_t']
                true_p = param_state['true_p']
                theta_hat_1 = param_state['theta_hat_1']
                theta_hat_2 = param_state['theta_hat_2']

                # First, copy true params from after the update
                true_p.copy_(p.data)

                # Then compute theta_tilde
                theta_tilde = p.data
                theta_tilde.mul_(0).add_(-(N_t - 1), theta_hat_1)
                theta_hat_1.mul_(0)
                theta_hat_1.add_(1/N_t,
                                 t/c * true_p - (t/c - N_t) * theta_hat_2)
                theta_tilde.add_(N_t, theta_hat_1)

        return result
