from .resnet import *


models = [
    'resnet18',
    'resnet34half',
    'resnet34',
    'resnet50half',
    'resnet50',
    'resnet101',
    'resnet152',
]


def init_model(backbone_type: str, num_classes: int):
    if backbone_type not in models:
        raise "Unsupported model!"
    model = eval(backbone_type)(
        **{'num_classes': num_classes}
    )

    return model


