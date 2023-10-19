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


def build_optimizer(opts, model):
    optimizer_name = opts[0]
    optimizer_parameters = opts[1]
    assert optimizer_name in ['SGD', 'AdamW'], 'Unsupported optimizer!'

    lr = optimizer_parameters['lr']
    global_weight_decay = optimizer_parameters['global_weight_decay']
    weight_decay = optimizer_parameters['weight_decay']
    if global_weight_decay:
        pass
    else:
        no_weight_decay_layer_name_list = optimizer_parameters[
            'no_weight_decay_layer_name_list']

    if optimizer_name == 'SGD':
        momentum = optimizer_parameters['momentum']
        nesterov = False if 'nesterov' not in optimizer_parameters.keys(
        ) else optimizer_parameters['nesterov']
        return torch.optim.SGD(
            model_params_weight_decay_list,
            lr=lr,
            momentum=momentum,
            nesterov=nesterov), model_layer_weight_decay_list
