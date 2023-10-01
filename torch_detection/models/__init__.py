# from .retinanet import resnet50_retinanet
# from .retinanet import resnet34_retinanet
from .retinanet2 import *


__factory = {
    # image classification models
    'resnet34_retinanet': resnet34_retinanet,
    'resnet50_retinanet': resnet50_retinanet,
}


def get_names():
    return __factory.keys()


def init_model(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError('Unknown model: {}'.format(name))

    return __factory[name](*args, **kwargs)

