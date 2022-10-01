from __future__ import print_function, absolute_import
import torch
import math


__all__ = [
    'init_scheduler',
    'adjust_learning_rate'
]


def adjust_learning_rate(optimizer, current_epoch, max_epoch, lr_min=0., lr_max=0.1, warmup_epoch=10, warmup=True):
    if warmup:
        warmup_epoch = warmup_epoch
    else:
        warmup_epoch = 0

    if current_epoch < warmup_epoch:
        # 如果当前epoch为0, current_epoch就赋值为0.1
        current_epoch = 0.1 if current_epoch == 0 else current_epoch
        lr = lr_max * current_epoch / warmup_epoch
    else:
        lr = lr_min + (lr_max-lr_min)*(1 + math.cos(math.pi * (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch))) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def init_scheduler(scheduler, optimizer, step_size, gamma, args):
    if scheduler == 'step_lr':
        # 每 step 个epoch之后, lr 衰减为 lr * gamma (gamma一般为0.1)
        return torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                               step_size=step_size,
                                               gamma=gamma)
    elif scheduler == 'cosine_annealing_lr':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                          T_max=args.max_epoch)
    else:
        raise KeyError('Unsupported scheduler: {}'.format(scheduler))
