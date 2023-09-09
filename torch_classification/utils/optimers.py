from __future__ import print_function, absolute_import
import torch

__all__ = [
    'init_optimizer'
]


def init_optimizer(optim, params, lr, weight_decay, **kwargs):
    if optim == 'adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optim == 'sgd':
        return torch.optim.SGD(params, lr=lr, momentum=kwargs["momentum"], weight_decay=weight_decay)
    else:
        raise KeyError('Unsupported optim: {}'.format(optim))

