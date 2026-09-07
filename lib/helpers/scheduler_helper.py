import torch.nn as nn
import torch.optim.lr_scheduler as lr_sched
import math


def build_lr_scheduler(cfg, optimizer, last_epoch):
    scheduler_type = str(cfg.get('type', 'step')).lower()
    if scheduler_type == 'step':
        def lr_lbmd(cur_epoch):
            cur_decay = 1
            for decay_step in cfg['decay_list']:
                if cur_epoch >= decay_step:
                    cur_decay = cur_decay * cfg['decay_rate']
            return cur_decay
        lr_scheduler = lr_sched.LambdaLR(
            optimizer, lr_lbmd, last_epoch=last_epoch)
    elif scheduler_type == 'cos':
        t_max = int(cfg['t_max'])
        eta_min = float(cfg.get('eta_min', 0.0))
        if t_max <= 0:
            raise ValueError('cosine scheduler t_max must be positive')
        if any(float(group['lr']) <= 0.0
               for group in optimizer.param_groups):
            raise ValueError('cosine scheduler requires positive base LR')
        lr_scheduler = lr_sched.CosineAnnealingLR(
            optimizer, T_max=t_max, eta_min=eta_min,
            last_epoch=last_epoch)
    else:
        raise ValueError(
            f"unsupported lr_scheduler type: {scheduler_type}")
    warmup_lr_scheduler = None
    if cfg['warmup']:
        warmup_epochs = int(cfg.get('warmup_epochs', 5))
        warmup_init_lr = float(cfg.get('warmup_init_lr', 0.00001))
        warmup_type = str(cfg.get('warmup_type', 'cosine')).lower()
        if warmup_epochs <= 0:
            raise ValueError('warmup_epochs must be positive')
        if warmup_init_lr < 0.0:
            raise ValueError('warmup_init_lr must be non-negative')
        if warmup_type == 'cosine':
            warmup_class = CosineWarmupLR
        elif warmup_type == 'linear':
            warmup_class = LinearWarmupLR
        else:
            raise ValueError(
                f'unsupported warmup_type: {warmup_type}')
        warmup_lr_scheduler = warmup_class(
            optimizer, num_epoch=warmup_epochs,
            init_lr=warmup_init_lr)
    return lr_scheduler, warmup_lr_scheduler


def build_bnm_scheduler(cfg, model, last_epoch):
    if not cfg['enabled']:
        return None

    def bnm_lmbd(cur_epoch):
        cur_decay = 1
        for decay_step in cfg['decay_list']:
            if cur_epoch >= decay_step:
                cur_decay = cur_decay * cfg['decay_rate']
        return max(cfg['momentum']*cur_decay, cfg['clip'])

    bnm_scheduler = BNMomentumScheduler(model, bnm_lmbd, last_epoch=last_epoch)
    return bnm_scheduler


def set_bn_momentum_default(bn_momentum):
    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum

    return fn


class BNMomentumScheduler(object):

    def __init__(
            self, model, bn_lambda, last_epoch=-1,
            setter=set_bn_momentum_default
    ):
        if not isinstance(model, nn.Module):
            raise RuntimeError("Class '{}' is not a PyTorch nn Module".format(type(model).__name__))

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))


class CosineWarmupLR(lr_sched._LRScheduler):
    def __init__(self, optimizer, num_epoch, init_lr=0.0, last_epoch=-1):
        self.num_epoch = num_epoch
        self.init_lr = init_lr
        super(CosineWarmupLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.init_lr + (base_lr - self.init_lr) *
                (1 - math.cos(math.pi * self.last_epoch / self.num_epoch)) / 2
                for base_lr in self.base_lrs]


class LinearWarmupLR(lr_sched._LRScheduler):
    def __init__(self, optimizer, num_epoch, init_lr=0.0, last_epoch=-1):
        self.num_epoch = num_epoch
        self.init_lr = init_lr
        super(LinearWarmupLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.init_lr + (base_lr - self.init_lr) * self.last_epoch / self.num_epoch
                for base_lr in self.base_lrs]
