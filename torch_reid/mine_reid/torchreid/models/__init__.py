from __future__ import print_function, absolute_import

from .other_modules import *
from .osnet import *
from .resnet import *
from .mobilenet import *
from .resnet import resnet50
from .osnet import osnet_x1_0, osnet_x0_75


__factory = {
    # image classification models
    'resnet50': resnet50,
    'osnet_x1_0': osnet_x1_0,
    'osnet_x0_75': osnet_x0_75,
}


def get_names():
    return __factory.keys()


def init_model(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError('Unknown model: {}'.format(name))

    return __factory[name](*args, **kwargs)

