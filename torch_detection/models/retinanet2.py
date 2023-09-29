import torch
import torch.nn as nn
from . import backbones
from .fpn import RetinaFPN


class RetinaFPN(nn.Module):
    def __init__(self, inplanes, planes, use_p5=False):
        super(RetinaFPN, self).__init__()

    def forward(self, inputs):
        [C3, C4, C5] = inputs


class RetinaNet(nn.Module):
    def __init__(self,
                 backbone_type,
                 backbone_pretrained_path='',
                 planes=256,
                 num_anchors=9,
                 num_classes=80
                 ):
        super(RetinaNet, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes

        self.backbone = backbones.__dict__[backbone_type]()
        # TODO: 考虑下怎么加载backbone的权重信息

    def forward(self, inputs):
        features = self.backbone(inputs)
        del inputs

        features = self.fpn(features)

        cls_heads, reg_heads = [], []
