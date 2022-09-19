from __future__ import print_function, absolute_import
import torch


__all__ = [
    'init_scheduler'
]


def init_scheduler(scheduler, optimizer, step_size, gamma):
    if scheduler == 'steplr':
        # 每 step 个epoch之后, lr 衰减为 lr * gamma (gamma一般为0.1)
        return torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                               step_size=step_size,
                                               gamma=gamma)
    else:
        raise KeyError('Unsupported scheduler: {}'.format(scheduler))
