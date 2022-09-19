from __future__ import print_function, absolute_import

from .osnet import *
from .resnet import *
from .mobilenet import *

__factory = {
    # image classification models
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
}


def get_names():
    return __factory.keys()


def init_model(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError('Unknown model: {}'.format(name))

    return __factory[name](*args, **kwargs)

