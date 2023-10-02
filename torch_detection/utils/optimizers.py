from __future__ import print_function, absolute_import
import torch

__all__ = [
    'init_optimizer'
]


def init_optimizer(optim, params, lr, weight_decay):
    if optim == 'adam':
        return torch.optim.Adam(params, lr, weight_decay=weight_decay)
    elif optim == 'sgd':
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise KeyError('Unsupported optim: {}'.format(optim))

