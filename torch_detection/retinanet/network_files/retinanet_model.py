import os
import sys
import torch
import torch.nn as nn
import numpy as np

BASE_DIR = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
                 backbone_pretrain_path='',
                 out_channels=256,
                 num_anchors=9,
                 num_classes=80):
        super(RetinaNet, self).__init__()
        self.out_channels = out_channels
        self.num_anchors = num_anchors
        self.num_classes = num_classes

        self.backbone = resnet.__dict__[backbone_type](
            **{'pre_train_path': backbone_pretrain_path}
        )

        self.fpn = FPN(*self.backbone.out_channels,
                       out_channels=self.out_channels)
        self.cls_head = RetinaClsHead(in_channels=self.out_channels,
                                      num_anchors=self.num_anchors,
                                      num_classes=self.num_classes)
        self.reg_head = RetinaRegHead(in_channels=self.out_channels,
                                      num_anchors=self.num_anchors)
        self.feature_sizes = list()
        self.batch_size = ''
        self.anchors = RetinaAnchors()

    def forward(self, inputs):
        self.batch_size = inputs.shape[0]
        features = self.backbone(inputs)    # backbone输出的三个特征提取层
        features = self.fpn(features)       # fpn之后输出的五个特征提取层

        cls_heads, reg_heads = list(), list()
        for feature in features:
            self.feature_sizes.append([feature.shape[3], feature.shape[2]])
            cls_head = self.cls_head(feature)
            reg_head = self.reg_head(feature)
            cls_heads.append(cls_head)
            reg_heads.append(reg_head)

        batch_anchors = self.anchors(self.batch_size, self.feature_sizes)
        # for cls_head, reg_head, anchor in zip(cls_heads, reg_heads, batch_anchors):
        #     print(cls_head.shape, reg_head.shape, anchor.shape)
        # if input size:[B,3,640,640]
        # features shape:[[B, 256, 80, 80],[B, 256, 40, 40],[B, 256, 20, 20],[B, 256, 10, 10],[B, 256, 5, 5]]
        # cls_heads shape:[[B, 57600, 80],[B, 14400, 80],[B, 3600, 80],[B, 900, 80],[B, 225, 80]]
        # reg_heads shape:[[B, 57600, 4],[B, 14400, 4],[B, 3600, 4],[B, 900, 4],[B, 225, 4]]
        # batch_anchors shape:[[B, 57600, 4],[B, 14400, 4],[B, 3600, 4],[B, 900, 4],[B, 225, 4]]
        return [cls_heads, reg_heads, batch_anchors]


def _retinanet(backbone_type, backbone_pretrained_path, num_classes, **kwargs):
    model = RetinaNet(backbone_type,
                      backbone_pretrain_path=backbone_pretrained_path,
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

    inputs = torch.rand(4, 3, 640, 640)
    # print(os.getcwd())
    # print(resnet.__dict__['resnet50_backbone'])
    # retina = RetinaNet(
    #     backbone_type='resnet50_backbone',
    #     backbone_pretrain_path='',
    # )
    # retina(inputs)
    pre_train = '/workshop/weihule/data/weights/resnet/resnet50-acc76.322.pth'
    if not os.path.exists(pre_train):
        pre_train = '/nfs/home57/weihule/data/weights/resnet/resnet50-acc76.322.pth'
    retina_model = resnet50_retinanet(num_classes=20,
                                      pre_train=pre_train)
    outputs = retina_model(inputs)
    for p in outputs:
        for i in p:
            print(i.shape)
