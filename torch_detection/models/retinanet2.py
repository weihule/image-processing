import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
# BASE_DIR = D:\workspace\code\image-processing
sys.path.append(BASE_DIR)

from torch_detection.models import backbones
from torch_detection.utils.util import load_state_dict


__all__ = [
    "resnet18_retinanet",
    "resnet34_retinanet",
    "resnet50_retinanet",
    "resnet101_retinanet",
    "resnet152_retinanet"
]


class RetinaFPN(nn.Module):
    def __init__(self, inplanes, planes, use_p5=False):
        """

        Args:
            inplanes: if resnet50, [512, 1024, 2048], if resnet 34, [128, 256, 512]
            planes: 256
            use_p5: bool
        """
        super(RetinaFPN, self).__init__()
        self.use_p5 = use_p5
        self.P3_1 = nn.Conv2d(inplanes[0],
                              planes,
                              kernel_size=1,
                              stride=1,
                              padding=0)
        self.P3_2 = nn.Conv2d(planes,
                              planes,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        self.P4_1 = nn.Conv2d(inplanes[1],
                              planes,
                              kernel_size=1,
                              stride=1,
                              padding=0)
        self.P4_2 = nn.Conv2d(planes,
                              planes,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        self.P5_1 = nn.Conv2d(inplanes[2],
                              planes,
                              kernel_size=1,
                              stride=1,
                              padding=0)
        self.P5_2 = nn.Conv2d(planes,
                              planes,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        self.P6 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=2,
            padding=1) if self.use_p5 else nn.Conv2d(
                inplanes[2], planes, kernel_size=3, stride=2, padding=1)
        self.P7 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1))

    def forward(self, inputs):
        """

        Args:
            inputs: if resnet50, [B, 512, 80, 80]
                                 [B, 1024, 40, 40]
                                 [B, 2048, 20, 20]
        Returns:

        """
        [C3, C4, C5] = inputs
        P5 = self.P5_1(C5)  # [B, 2048, 20, 20] -> [B, 256, 20, 20]
        P4 = self.P4_1(C4)  # [B, 1024, 40, 40] -> [B, 256, 40, 40]
        P4 = F.interpolate(P5,
                           size=(P4.shape[2], P4.shape[3]),
                           mode="bilinear",
                           align_corners=True) + P4
        P3 = self.P3_1(C3)  # [B, 512, 80, 80] -> [B, 256, 80, 80]
        P3 = F.interpolate(P4,
                           size=(P3.shape[2], P3.shape[3]),
                           mode="bilinear",
                           align_corners=True) + P3
        del C3, C4
        P5 = self.P5_2(P5)  # [B, 256, 20, 20] -> [B, 256, 20, 20]
        P4 = self.P4_2(P4)  # [B, 256, 40, 40] -> [B, 256, 40, 40]
        P3 = self.P3_2(P3)  # [B, 256, 80, 80] -> [B, 256, 80, 80]

        # if use_p5: [B, 256, 20, 20] -> [B, 256, 10, 10]
        # else:      [B, 2048, 20, 20] -> [B, 256, 10, 10]
        P6 = self.P6(P5) if self.use_p5 else self.P6(C5)
        del C5
        P7 = self.P7(P6)    # [B, 256, 10, 10] -> [B, 256, 5, 5]

        return [P3, P4, P5, P6, P7]


class RetinaClsHead(nn.Module):
    def __init__(self, inplanes, num_anchors, num_classes, num_layers=4):
        super(RetinaClsHead, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(
                nn.Conv2d(inplanes, inplanes, 3, 1, 1)
            )
            layers.append(nn.ReLU(inplace=True))
        self.cls_head = nn.Sequential(*layers)
        self.cls_out = nn.Conv2d(inplanes,
                                 num_anchors * num_classes,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0)

        prior = 0.01
        b = -math.log((1 - prior) / prior)
        self.cls_out.bias.data.fill_(b)

    def forward(self, x):
        x = self.cls_head(x)    # [B, C, H, W]
        x = self.cls_out(x)     # [B, 9 * num_classes, H, W]
        x = self.sigmoid(x)

        return x


class RetinaRegHead(nn.Module):
    def __init__(self, inplanes, num_anchors, num_layers=4):
        super(RetinaRegHead, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(
                nn.Conv2d(inplanes, inplanes, 3, 1, 1))
            layers.append(nn.ReLU(inplace=True))
        self.reg_head = nn.Sequential(*layers)
        self.reg_out = nn.Conv2d(inplanes,
                                 num_anchors * 4,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0)

    def forward(self, x):
        x = self.reg_head(x)    # [B, C, H, W]
        x = self.reg_out(x)     # [B, 9 * 4, H, W]

        return x


class RetinaNet(nn.Module):
    def __init__(self,
                 backbone_type,
                 backbone_pretrained_path='',
                 planes=256,
                 num_anchors=9,
                 num_classes=80
                 ):
        super(RetinaNet, self).__init__()
        self.planes = planes
        self.num_anchors = num_anchors
        self.num_classes = num_classes

        self.backbone = backbones.__dict__[backbone_type]()
        # TODO: 加载backbone的预训练权重信息
        self.backbone = load_state_dict(backbone_pretrained_path, self.backbone)

        self.fpn = RetinaFPN(self.backbone.out_channels,
                             self.planes,
                             use_p5=False)
        self.cls_head = RetinaClsHead(self.planes,
                                      self.num_anchors,
                                      self.num_classes,
                                      num_layers=4)
        self.reg_head = RetinaRegHead(self.planes,
                                      self.num_anchors,
                                      num_layers=4)

    def forward(self, inputs):
        features = self.backbone(inputs)
        del inputs

        # [B, 256, 80, 80], 40, 20, 10, 5
        features = self.fpn(features)

        cls_heads, reg_heads = [], []
        for feature in features:
            cls_head = self.cls_head(feature)
            # [N,9*num_classes,H,W] -> [N,H,W,9*num_classes] -> [N,H,W,9,num_classes]
            cls_head = cls_head.permute(0, 2, 3, 1).contiguous()
            cls_head = cls_head.view(cls_head.shape[0], cls_head.shape[1],
                                     cls_head.shape[2], -1, self.num_classes)
            cls_heads.append(cls_head)

            reg_head = self.reg_head(feature)
            # [N, 9*4,H,W] -> [N,H,W,9*4] -> [N,H,W,9,4]
            reg_head = reg_head.permute(0, 2, 3, 1).contiguous()
            reg_head = reg_head.view(reg_head.shape[0], reg_head.shape[1],
                                     reg_head.shape[2], -1, 4)
            reg_heads.append(reg_head)

        del features
        # if input size:[B,3,640,640]
        # features shape:[[B, 256, 80, 80],[B, 256, 40, 40],[B, 256, 20, 20],[B, 256, 10, 10],[B, 256, 5, 5]]
        # cls_heads shape:[[B, 80, 80, 9, 80],[B, 40, 40, 9, 80],[B, 20, 20, 9, 80],[B, 10, 10, 9, 80],[B, 5, 5, 9, 80]]
        # reg_heads shape:[[B, 80, 80, 9, 4],[B, 40, 40, 9, 4],[B, 20, 20, 9, 4],[B, 10, 10, 9, 4],[B, 5, 5, 9, 4]]
        return [cls_heads, reg_heads]


def _retinanet(backbone_type, backbone_pretrained_path, **kwargs):
    model = RetinaNet(backbone_type,
                      backbone_pretrained_path=backbone_pretrained_path,
                      **kwargs)

    return model


def resnet18_retinanet(backbone_pretrained_path='', **kwargs):
    return _retinanet('resnet18backbone',
                      backbone_pretrained_path=backbone_pretrained_path,
                      **kwargs)


def resnet34_retinanet(backbone_pretrained_path='', **kwargs):
    return _retinanet('resnet34backbone',
                      backbone_pretrained_path=backbone_pretrained_path,
                      **kwargs)


def resnet50_retinanet(backbone_pretrained_path='', **kwargs):
    return _retinanet('resnet50backbone',
                      backbone_pretrained_path=backbone_pretrained_path,
                      **kwargs)


def resnet101_retinanet(backbone_pretrained_path='', **kwargs):
    return _retinanet('resnet101backbone',
                      backbone_pretrained_path=backbone_pretrained_path,
                      **kwargs)


def resnet152_retinanet(backbone_pretrained_path='', **kwargs):
    return _retinanet('resnet152backbone',
                      backbone_pretrained_path=backbone_pretrained_path,
                      **kwargs)


if __name__ == "__main__":
    i = torch.randn((4, 3, 640, 640))
    pre_weight = r"D:\workspace\data\training_data\resnet50\resnet50-acc76.264.pth"
    model = resnet50_retinanet(backbone_pretrained_path=pre_weight,
                               num_classes=20)
    cls, reg = model(i)
    for c, r in zip(cls, reg):
        print(c.shape, r.shape)





