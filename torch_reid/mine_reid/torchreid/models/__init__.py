from __future__ import print_function, absolute_import

from .horizontal_pool import *
from .osnet import *
from .resnet import *
from .mobilenet import *

__factory = {
    # image classification models
    'resnet50': resnet50,
}


def get_names():
    return __factory.keys()


def init_model(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError('Unknown model: {}'.format(name))

    return __factory[name](*args, **kwargs)

