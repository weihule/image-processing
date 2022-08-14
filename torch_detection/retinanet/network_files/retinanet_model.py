import os
import sys
import torch
import torch.nn as nn
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from torch_detection.retinanet.network_files import resnet
from torch_detection.retinanet.network_files.fpn import FPN
from torch_detection.retinanet.network_files.heads import RetinaClsHead, RetinaRegHead
from torch_detection.retinanet.network_files.anchors import RetinaAnchors

__all__ = [
    'resnet34_retinanet',
    'resnet50_retinanet'
]


class RetinaNet(nn.Module):
    def __init__(self,
                 backbone_type,
                 backbone_pretrained_path='',
                 planes=256,
                 num_anchors=9,
                 num_classes=80):
        super(RetinaNet, self).__init__()
        self.planes = planes
        self.num_anchors = num_anchors
        self.num_classes = num_classes

        self.backbone = resnet.__dict__[backbone_type](
            **{'pretrained_path': backbone_pretrained_path}
        )

        self.fpn = FPN(inplanes=self.backbone.out_channels,
                       planes=self.planes)
        self.cls_head = RetinaClsHead(in_channels=self.planes,
                                      num_anchors=self.num_anchors,
                                      num_classes=self.num_classes)
        self.reg_head = RetinaRegHead(in_channels=self.planes,
                                      num_anchors=self.num_anchors)
        self.feature_sizes = list()

    def forward(self, inputs):
        features = self.backbone(inputs)
        del inputs

        features = self.fpn(features)
        cls_heads, reg_heads = [], []
        for feature in features:
            cls_head = self.cls_head(feature)
            # [B, 9*num_classes, H, W] -> [B, H, W, 9*num_classes] -> [B, H, W, 9, num_classes]
            cls_head = cls_head.permute(0, 2, 3, 1).contiguous()
            cls_head = cls_head.view(cls_head.shape[0], cls_head.shape[1], cls_head.shape[2],
                                     -1, self.num_classes)
            cls_heads.append(cls_head)

            reg_head = self.reg_head(feature)
            # [B, 9*4, H, W] -> [B, H, W, 9*4] -> [B, H, W, 9, 4]
            reg_head = reg_head.permute(0, 2, 3, 1).contiguous()
            reg_head = reg_head.view(reg_head.shape[0], reg_head.shape[1], reg_head.shape[2],
                                     -1, 4)
            reg_heads.append(reg_head)
        del features

        # if input size:[B,3,640,640]
        # features shape:[[B, 256, 80, 80],[B, 256, 40, 40],[B, 256, 20, 20],[B, 256, 10, 10],[B, 256, 5, 5]]
        # cls_heads shape:[[B, 80, 80, 9, 80],[B, 40, 40, 9, 80],[B, 20, 20, 9, 80],[B, 10, 10, 9, 80],[B, 5, 5, 9, 80]]
        # reg_heads shape:[[B, 80, 80, 9, 4],[B, 40, 40, 9, 4],[B, 20, 20, 9, 4],[B, 10, 10, 9, 4],[B, 5, 5, 9, 4]]
        return [cls_heads, reg_heads]


def _retinanet(backbone_type, backbone_pretrained_path, num_classes, **kwargs):
    model = RetinaNet(backbone_type,
                      backbone_pretrained_path=backbone_pretrained_path,
                      num_classes=num_classes,
                      **kwargs)

    return model


def resnet50_retinanet(num_classes, pre_train=''):
    return _retinanet('resnet50_backbone', pre_train, num_classes)


def resnet34_retinanet(num_classes, pre_train=''):
    return _retinanet('resnet34_backbone', pre_train, num_classes)


if __name__ == '__main__':
    arr = np.random.random(size=(4, 3, 2))
    res = np.max(arr)

    inputs = torch.rand(4, 3, 400, 400)

    pre_train = '/workshop/weihule/data/weights/resnet/resnet50-acc76.322.pth'
    if not os.path.exists(pre_train):
        pre_train = '/nfs/home57/weihule/data/weights/resnet/resnet50-acc76.322.pth'
    retina_model = resnet50_retinanet(num_classes=20,
                                      pre_train=pre_train)
    outputs = retina_model(inputs)
    for p in outputs:
        for i in p:
            print(i.shape)
