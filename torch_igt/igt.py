#!/usr/bin/env python3

import torch as th
from torch.optim.Optimizer import Optimizer, required


class MomentumIGT(Optimizer):

    """
    Implementation of Implicit Gradient Transport,
    and its Heavyball and Nesterov versions.

    Arguments:

    Example:

    Notes:
        
        * This implementation requires 4 copies of the model's parameters.
          I think it's possible to have a version with only 3 copies,
          but it would sacrifice some clarity.
        * Currently dampening is not supported.
        * Heavyball and Nesterov are not implemented as in PyTorch's SGD.
    """

    def __init__(self, params, lr=required, momentum=0.0, dampening=0.0, weight_decay=0.0, nesterov=False, delta=1.0):
        """
        Arguments:
        """
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        if delta <= 0.0:
            raise ValueError("Invalid delta value: {}".format(weight_decay))

        if dampening != 0.0:
            raise ValueError("Dampening is not currently supported.")

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        defaults = {
                'lr': lr,
                'momentum': momentum,
                'dampening': dampening,
                'weight_decay': weight_decay,
                'nesterov': nesterov,
                'delta': delta,
                'num_steps': 0,
        }
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(MomentumIGT, self).__init__(params, defaults)


    def __setstate__(self, state):
        super(MomentumIGT, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)


    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            dampening = group['dampening']
            weight_decay = group['weight_decay']
            nesterov = group['nesterov']
            delta = group['delta']
            num_steps = group['num_steps']

            # gamma = num_steps / (num_steps + delta)
            # transport = gamma / (1.0 - gamma)
            future_gamma = (num_steps + 1) / (num_steps + 1 + delta)
            future_transport = future_gamma / (1.0 - gamma)

            for p in group['params']:

                if p.grad is None:
                    continue

                d_p = p.grad.data  # Transported gradients
                param_state = self.state[p]

                # Apply weight decay
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                # Compute the IGT update
                if num_steps == 0:

                    if 'igt_velocity' not in param_state:
                        param_state['igt_velocity'] = th.zeros_like(p.data)
                    if 'true_params' not in param_state:
                        param_state['true_params'] = th.clone(p.data)
                    if momentum != 0 and 'momentum_velocity' not in param_state:
                        param_state['momentum_velocity'] = th.zeros_like(p.data)

                    igt_velocity = param_state['igt_velocity']
                    true_params = param_state['true_params']

                    # Compute first step and initial values
                    igt_velocity.add_(d_p)
                    if momentum != 0:
                        param_state['momentum_velocity'].add_(-lr, igt_velocity)
                    true_params.add_(-lr, igt_velocity)

                    # Set parameters to transported ones
                    p.data.add_(-lr * (1.0 + future_transport), d_p) 

                else:
                    igt_velocity = param_state['igt_velocity']
                    true_params = param_state['true_params']

                    # Update IGT's velocity
                    igt_velocity.add(gamma, velocity)
                    igt_velocity.add((1.0 - gamma), d_p)

                    # Apply momentum if necessary
                    if momentum == 0:
                        # Update true parameters
                        true_params.add_(-lr, igt_velocity)

                        # Set parameters to transported ones
                        p.data.copy_(true_params)
                        p.data.add_(-lr * (1.0 + future_transport), igt_velocity)
                    else:
                        momentum_velocity = param_state['momentum_velocity']
                        if not nesterov:
                            momentum_velocity.mul_(momentum).add_(lr, igt_velocity)
                            true_params.add_(-1.0, momentum_velocity)

                            # Set parameters to transported ones
                            p.data.copy_(true_params)
                            p.data.add_(-(1.0 + future_transport),
                                        momentum_velocity)
                        else:
                            raise('Not implemented')

            group['num_steps'] = num_steps + 1
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
