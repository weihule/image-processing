from .resnet import *
from .vit_model import *
from .mobilenet import *


models = [
    'resnet18',
    'resnet34half',
    'resnet34',
    'resnet50half',
    'resnet50',
    'resnet101',
    'resnet152',

    'vit_base_patch16_224',
    'vit_base_patch16_224_in21k',
    'vit_base_patch32_224',
    'vit_base_patch32_224_in21k',

    'mobilenetv2_x1_0',
    'mobilenetv2_x1_4'
]


def init_model(backbone_type: str, num_classes: int):
    if backbone_type not in models:
        raise "Unsupported model!"
    model = eval(backbone_type)(
        **{'num_classes': num_classes}
    )

    return model


