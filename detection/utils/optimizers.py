from __future__ import print_function, absolute_import
import torch

__all__ = [
    'init_optimizer',
    'build_optimizer'
]


def init_optimizer(optim, params, lr, weight_decay):
    if optim == 'adam':
        return torch.optim.Adam(params, lr, weight_decay=weight_decay)
    elif optim == 'sgd':
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise KeyError('Unsupported optim: {}'.format(optim))


"""
opts:
optimizer = (
    'AdamW',
    {
        'lr': 1e-4,
        'global_weight_decay': False,
        # if global_weight_decay = False
        # all bias, bn and other 1d params weight set to 0 weight decay
        'weight_decay': 1e-3,
        'no_weight_decay_layer_name_list': [],
    },
)
"""


def build_optimizer(opts, model):
    optimizer_name = opts[0]
    optimizer_parameters = opts[1]
    assert optimizer_name in ['SGD', 'AdamW'], 'Unsupported optimizer!'

    lr = optimizer_parameters['lr']
    global_weight_decay = optimizer_parameters['global_weight_decay']
    weight_decay = optimizer_parameters['weight_decay']
    if global_weight_decay:
        decay_weight_list, decay_weight_name_list = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            decay_weight_list.append(param)
            decay_weight_name_list.append(name)
            model_params_weight_decay_list = [
                {
                    'params': decay_weight_list,
                    'weight_decay': weight_decay
                },
            ]
            model_layer_weight_decay_list = [
                {
                    'name': decay_weight_name_list,
                    'weight_decay': weight_decay
                },
            ]
    else:
        no_weight_decay_layer_name_list = optimizer_parameters['no_weight_decay_layer_name_list']
        decay_weight_list, no_decay_weight_list = [], []
        decay_weight_name_list, no_decay_weight_name_list = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            if param.ndim == 1 or any(no_weight_decay_layer_name in name
                                      for no_weight_decay_layer_name in
                                      no_weight_decay_layer_name_list):
                no_decay_weight_list.append(param)
                no_decay_weight_name_list.append(name)
            else:
                decay_weight_list.append(param)
                decay_weight_name_list.append(name)

            model_params_weight_decay_list = [
                {
                    'params': no_decay_weight_list,
                    'weight_decay': 0.
                },
                {
                    'params': decay_weight_list,
                    'weight_decay': weight_decay
                },
            ]

            model_layer_weight_decay_list = [
                {
                    'name': no_decay_weight_name_list,
                    'weight_decay': 0.
                },
                {
                    'name': decay_weight_name_list,
                    'weight_decay': weight_decay
                },
            ]

    if optimizer_name == 'SGD':
        momentum = optimizer_parameters['momentum']
        nesterov = False if 'nesterov' not in optimizer_parameters.keys() \
            else optimizer_parameters['nesterov']
        return torch.optim.SGD(model_params_weight_decay_list,
                               lr=lr,
                               momentum=momentum,
                               nesterov=nesterov), model_layer_weight_decay_list
    elif optimizer_name == 'AdamW':
        beta1 = 0.9 if 'beta1' not in optimizer_parameters.keys() else optimizer_parameters['beta1']
        beta2 = 0.999 if 'beta2' not in optimizer_parameters.keys() else optimizer_parameters['beta2']
        return torch.optim.AdamW(model_params_weight_decay_list,
                                 lr=lr,
                                 betas=(beta1,
                                        beta2)), model_layer_weight_decay_list




