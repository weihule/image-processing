import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import backbones


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
        x = self.reg_head(x)
        x = self.reg_out(x)

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
        # TODO: 考虑下怎么加载backbone的权重信息

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
            cls_head = self.cls_
