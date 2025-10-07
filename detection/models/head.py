import os
import sys
import math
import torch
import torch.nn as nn

from .backbones.network_blocks import DWConv, BaseConv

__all__ = [
    'RetinaClsHead',
    'RetinaRegHead',
    'YOLOXHead'
]


class RetinaClsHead(nn.Module):
    def __init__(self, inplanes, num_anchors, num_classes, num_layers=4):
        super(RetinaClsHead, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(
                nn.Conv2d(inplanes,
                          inplanes,
                          kernel_size=3,
                          stride=1,
                          padding=1))
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
        x = self.cls_head(x)
        x = self.cls_out(x)
        x = self.sigmoid(x)

        return x


class RetinaRegHead(nn.Module):

    def __init__(self, inplanes, num_anchors, num_layers=4):
        super(RetinaRegHead, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(
                nn.Conv2d(inplanes,
                          inplanes,
                          kernel_size=3,
                          stride=1,
                          padding=1))
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


class YOLOXHead(nn.Module):
    def __init__(self,
                 num_classes,
                 width=1.0,
                 strides=(8, 16, 32),
                 in_channels=(256, 512, 1024),
                 act='silu',
                 depthwise=False):
        super(YOLOXHead, self).__init__()
        self.n_anchors = 1
        self.num_classes = num_classes
        self.decode_in_features = True

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()
        Conv = DWConv if depthwise else BaseConv

        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(in_channels=int(in_channels[i] * width),
                         out_channels=int(256 * width),
                         ksize=1,
                         stride=1,
                         groups=1,
                         act=act)
            )
            self.cls_convs.append(
                nn.Sequential(*[
                    Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act),
                    Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act)
                ])
            )
            self.reg_convs.append(
                nn.Sequential(*[
                    Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act),
                    Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act)
                ])
            )
            self.cls_preds.append(
                nn.Conv2d(int(256 * width), self.n_anchors * num_classes, kernel_size=1, stride=1, padding=0)
            )
            self.reg_preds.append(
                nn.Conv2d(int(256 * width), 4, kernel_size=1, stride=1, padding=0)
            )
            self.obj_preds.append(
                nn.Conv2d(int(256 * width), self.n_anchors * 1, kernel_size=1, stride=1, padding=0)
            )

        self.use_l1 = False
        self.strides = strides

    def initialize_biases(self, prior_prob):
        for conv in self.cls_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, xin):
        """
        if   YOLOFPN
        Args:
            xin: [b, 256, h/8, w/8], [b, 512, h/16, w/16], [b, 1024, h/32, w/32]
        Returns:

        """
        outputs = []
        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.strides, xin)
        ):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x

            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)

            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_out = self.obj_preds[k](reg_feat)

            outputs.append(torch.cat([reg_output, obj_out, cls_output], dim=1))

        # [[b, 4+1+num_classes, h/8, w/8], [b, 4+1+num_classes, h/16, w/16], [b, 4+1+num_classes, h/32, w/32]]
        return outputs


