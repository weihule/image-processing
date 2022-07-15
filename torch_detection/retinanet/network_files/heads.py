import os
import sys
import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class RetinaClsHead(nn.Module):
    def __init__(self,
                 in_channels,
                 num_anchors,
                 num_classes,
                 num_layers=4):
        super(RetinaClsHead, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(
                nn.Conv2d(in_channels, in_channels, 3, 1, 1)
            ),
            layers.append(nn.ReLU(inplace=True))
        self.num_classes = num_classes
        self.cls_head = nn.Sequential(*layers)
        self.cls_out = nn.Conv2d(in_channels,
                                 num_anchors*num_classes,
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
        x = self.cls_head(x)
        x = self.cls_out(x)     # [B, num_anchors*num_classes, H, W]
        b, _, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()    # [B, H, W, num_anchors*num_classes]
        x = x.reshape((b, -1, self.num_classes))    # [B, H*W*num_anchors, num_classes]
        x = self.sigmoid(x)

        return x


class RetinaRegHead(nn.Module):
    def __init__(self, in_channels, num_anchors, num_layers=4):
        super(RetinaRegHead, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(
                nn.Conv2d(in_channels,
                          in_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1))
            layers.append(nn.ReLU(inplace=True))
        self.reg_head = nn.Sequential(*layers)
        self.reg_out = nn.Conv2d(in_channels,
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
        x = self.reg_out(x)    # [B, num_anchors*4, H, W]
        b, _, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()    # [B, H, W, num_anchors*4]
        x = x.reshape((b, -1, 4))    # [B, H*W*num_anchors, 4]

        return x

    # def __init__(self, in_channels, num_anchors, num_layers=4):
    #     super(RetinaRegHead, self).__init__()
    #     layers = []
    #     for _ in range(num_layers):
    #         layers.append(
    #             nn.Conv2d(in_channels,
    #                       in_channels,
    #                       kernel_size=3,
    #                       stride=1,
    #                       padding=1))
    #         layers.append(nn.ReLU(inplace=True))
    #     self.reg_head = nn.Sequential(*layers)
    #     self.reg_out = nn.Conv2d(in_channels,
    #                              num_anchors * 4,
    #                              kernel_size=3,
    #                              stride=1,
    #                              padding=1)
    #
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.normal_(m.weight, std=0.01)
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, val=0)
    #
    # def forward(self, x):
    #     x = self.reg_head(x)
    #     b, c, h, w = x.shape
    #     x = self.reg_out(x)
    #
    #     return x
