import os
import sys

import torch
import torch.nn as nn

import darknet
from fpn import YoloV3FPNHead

__all__ = [
    'darknet53_yolov3'
]


class YoloV3(nn.Module):
    def __init__(self,
                 backbone_type,
                 backbone_pretrained_path='',
                 per_level_num_anchors=3,
                 num_classes=80):
        super(YoloV3, self).__init__()
        self.per_level_num_anchors = per_level_num_anchors
        self.num_classes = num_classes

        self.backbone = darknet.__dict__[backbone_type](
            **{'pretrained_path': backbone_pretrained_path}
        )

        if backbone_type == 'darknet53backbone':
            self.fpn = YoloV3FPNHead(self.backbone.out_channels)

    def forward(self, inputs):
        features = self.backbone(inputs)
        # for p in features:
        #     print(p.shape)
        features = self.fpn(features)
        # for p in features:
        #     print(p.shape)

        obj_reg_cls_heads = []
        for feature in features:
            # feature shape: [B,H,W,3,85]
            # obj_head: feature[:, :, :, :, :1], shape[B,H,W,3,1]
            # reg_head: feature[:, :, :, :, 1:5], shape[B,H,W,3,4]
            # cls_head: feature[:, :, :, :, 5:], shape[B,H,W,3,80]
            obj_reg_cls_heads.append(feature)
        del feature

        # if input size:[B,3,416,416]
        # features shape:[[B, 255, 52, 52],[B, 255, 26, 26],[B, 255, 13, 13]]
        # [B, h, w, 3, 85]
        # obj_reg_cls_heads shape:[[B, 52, 52, 3, 85],[B, 26, 26, 3, 85],[B, 13, 13, 3, 85]]
        return obj_reg_cls_heads

        # return features


def _yolov3(backbone_type, backbone_pretrained_path):
    model = YoloV3(backbone_type,
                   backbone_pretrained_path)
    return model


def darknet53_yolov3(backbone_pretrained_path=''):
    return _yolov3('darknet53backbone', backbone_pretrained_path)


if __name__ == "__main__":
    pre_weight = '/workshop/weihule/data/weights/yolo/darknet53-acc76.836.pth'
    # yolo_model = YoloV3(backbone_type='darknet53backbone',
    #                     backbone_pretrained_path=pre_weight)

    yolo_model = darknet53_yolov3(pre_weight)
    inputs = torch.rand(4, 3, 416, 416)
    res = yolo_model(inputs)
    for p in res:
        print(p.shape)

