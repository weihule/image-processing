import os
import sys
import torch
import torch.nn as nn

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from torch_detection.yolo.network_files import yolov4backbone
from torch_detection.yolo.network_files.fpn import Yolov4FPNHead


__all__ = [
    'cspdarknet53_yolov4'
]


class YOLOV4(nn.Module):
    def __init__(self,
                 backbone_type,
                 backbone_pretrained_path='',
                 act_type='leakyrelu',
                 per_level_num_anchors=3,
                 num_classes=80):
        super(YOLOV4, self).__init__()
        if backbone_type not in ['yolov4cspdarknettinybackbone', 'yolov4cspdarknet53backbone']:
            raise ValueError('Wrong backbone type')
        self.per_level_num_anchors = per_level_num_anchors
        self.num_classes = num_classes

        self.backbone = yolov4backbone.__dict__[backbone_type](**{
            'pretrained_path': backbone_pretrained_path,
            'act_type': act_type
        })

        if backbone_type == 'yolov4cspdarknet53backbone':
            self.fpn = Yolov4FPNHead(inplanes=self.backbone.out_channels,
                                     per_level_num_anchors=self.per_level_num_anchors,
                                     num_classes=self.num_classes,
                                     act_type='leakyrelu')

    def forward(self, inputs):
        features = self.backbone(inputs)

        del inputs

        features = self.fpn(features)

        obj_reg_cls_heads = []
        for feature in features:
            # feature shape is [B, H, W, 3, (1+4+num_classes)]

            # obj_head: [:, :, :, :, 0:1], shape: [B, H, W, 3, 1]
            # reg_head: [:, :, :, :, 1:5], shape: [B, H, W, 3, 4]
            # cls_head: [:, :, :, :, 5:], shape: [B, H, W, 3, 80]

            obj_reg_cls_heads.append(feature)
        del feature

        # if inputs size: [B, 3, 608, 608]
        # obj_reg_cls_heads shape is [[[B, 76, 76, 3, (1+4+num_classes)], ...]]
        return [obj_reg_cls_heads]


def _yolov4(backbone_type, backbone_pretrained_path, **kwargs):
    model = YOLOV4(backbone_type=backbone_type,
                   backbone_pretrained_path=backbone_pretrained_path,
                   **kwargs)
    return model


def cspdarknet53_yolov4(backbone_pretrained_path, **kwargs):

    return _yolov4('yolov4cspdarknet53backbone',
                   backbone_pretrained_path=backbone_pretrained_path,
                   **kwargs)


if __name__ == "__main__":
    net = cspdarknet53_yolov4(backbone_pretrained_path=None,
                              act_type='leakyrelu',
                              per_level_num_anchors=3,
                              num_classes=20)
    inputs = torch.randn(size=(4, 3, 608, 608))
    outputs = net(inputs)
    preds = outputs[0]
    for p in preds:
        print(p.shape)



