import math
from collections import defaultdict

import torch
import torch.optim as optim
from torch.optim.optimizer import Optimizer


def build_optimizer(cfg_optimizer, model):
    weights, biases = [], []
    for name, param in model.named_parameters():
        if 'bias' in name:
            biases += [param]
        else:
            weights += [param]

    parameters = [{'params': biases, 'weight_decay': 0},
                  {'params': weights,
                   'weight_decay': cfg_optimizer['weight_decay']}]

    if cfg_optimizer['type'] == 'sgd':
        optimizer = optim.SGD(
            parameters, lr=cfg_optimizer['lr'], momentum=0.9)
    elif cfg_optimizer['type'] == 'adam':
        optimizer = optim.Adam(parameters, lr=cfg_optimizer['lr'])
    elif cfg_optimizer['type'] == 'adamw':
        optimizer = AdamW(
            parameters, lr=cfg_optimizer['lr'],
            foreach=bool(cfg_optimizer.get('use_foreach_adamw', False)))
    else:
        raise NotImplementedError(
            "%s optimizer is not supported" % cfg_optimizer['type'])

    return optimizer


class AdamW(Optimizer):
    """Historical MonoDGP AdamW with an optional batched execution path."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, foreach=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super().__init__(params, defaults)
        # This is an execution switch, not part of the optimizer state or
        # update equation.
        self.use_foreach = bool(foreach)

    def __setstate__(self, state):
        super().__setstate__(state)
        self.use_foreach = bool(getattr(self, 'use_foreach', False))
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @staticmethod
    def _initialize_state(param, state, amsgrad):
        if state:
            return
        state['step'] = 0
        state['exp_avg'] = torch.zeros_like(param.data)
        state['exp_avg_sq'] = torch.zeros_like(param.data)
        if amsgrad:
            state['max_exp_avg_sq'] = torch.zeros_like(param.data)

    @staticmethod
    def _single_tensor_update(param, grad, state, group):
        exp_avg = state['exp_avg']
        exp_avg_sq = state['exp_avg_sq']
        beta1, beta2 = group['betas']
        state['step'] += 1

        exp_avg.mul_(beta1).add_(1 - beta1, grad)
        exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
        if group['amsgrad']:
            max_exp_avg_sq = state['max_exp_avg_sq']
            torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
            denom = max_exp_avg_sq.sqrt().add_(group['eps'])
        else:
            denom = exp_avg_sq.sqrt().add_(group['eps'])

        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']
        step_size = (group['lr'] * math.sqrt(bias_correction2)
                     / bias_correction1)
        param.data.add_(
            -step_size,
            torch.mul(param.data, group['weight_decay']).addcdiv_(
                1, exp_avg, denom))

    @staticmethod
    def _foreach_bucket_update(entries, group):
        params = [entry[0].data for entry in entries]
        grads = [entry[1] for entry in entries]
        states = [entry[2] for entry in entries]
        exp_avgs = [state['exp_avg'] for state in states]
        exp_avg_sqs = [state['exp_avg_sq'] for state in states]
        beta1, beta2 = group['betas']

        for state in states:
            state['step'] += 1
        step = states[0]['step']
        torch._foreach_mul_(exp_avgs, beta1)
        torch._foreach_add_(exp_avgs, grads, alpha=1 - beta1)
        torch._foreach_mul_(exp_avg_sqs, beta2)
        torch._foreach_addcmul_(
            exp_avg_sqs, grads, grads, value=1 - beta2)
        if group['amsgrad']:
            max_exp_avg_sqs = [
                state['max_exp_avg_sq'] for state in states]
            torch._foreach_maximum_(max_exp_avg_sqs, exp_avg_sqs)
            denom = torch._foreach_sqrt(max_exp_avg_sqs)
        else:
            denom = torch._foreach_sqrt(exp_avg_sqs)
        torch._foreach_add_(denom, group['eps'])

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step
        step_size = (group['lr'] * math.sqrt(bias_correction2)
                     / bias_correction1)
        updates = torch._foreach_mul(params, group['weight_decay'])
        torch._foreach_addcdiv_(updates, exp_avgs, denom, value=1)
        torch._foreach_add_(params, updates, alpha=-step_size)

    def _foreach_group_step(self, group):
        buckets = defaultdict(list)
        for param in group['params']:
            if param.grad is None:
                continue
            grad = param.grad.data
            if grad.is_sparse:
                raise RuntimeError(
                    'Adam does not support sparse gradients, please '
                    'consider SparseAdam instead')
            state = self.state[param]
            self._initialize_state(param, state, group['amsgrad'])
            # Different shapes are allowed in one foreach call. Device, dtype
            # and state step must agree; missing gradients can split the step.
            key = (param.device, param.dtype, state['step'])
            buckets[key].append((param, grad, state))
        for entries in buckets.values():
            self._foreach_bucket_update(entries, group)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            if self.use_foreach:
                self._foreach_group_step(group)
                continue
            for param in group['params']:
                if param.grad is None:
                    continue
                grad = param.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'Adam does not support sparse gradients, please '
                        'consider SparseAdam instead')
                state = self.state[param]
                self._initialize_state(param, state, group['amsgrad'])
                self._single_tensor_update(param, grad, state, group)

        return loss
