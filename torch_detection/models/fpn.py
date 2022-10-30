import os
import sys
import torch.nn as nn
import torch.nn.functional as F

from .backbones.network_blocks import BaseConv, CSPLayer, DWConv

__all__ = [
    'RetinaFPN',
    'YOLOFPN'
]


class RetinaFPN(nn.Module):
    def __init__(self, inplanes, planes, use_p5=False):
        super(RetinaFPN, self).__init__()
        # inplanes:[C3_inplanes,C4_inplanes,C5_inplanes]
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
        [C3, C4, C5] = inputs

        P5 = self.P5_1(C5)
        P4 = self.P4_1(C4)
        P4 = F.interpolate(P5,
                           size=(P4.shape[2], P4.shape[3]),
                           mode='bilinear',
                           align_corners=True) + P4
        P3 = self.P3_1(C3)
        P3 = F.interpolate(P4,
                           size=(P3.shape[2], P3.shape[3]),
                           mode='bilinear',
                           align_corners=True) + P3

        del C3, C4

        P5 = self.P5_2(P5)
        P4 = self.P4_2(P4)
        P3 = self.P3_2(P3)

        P6 = self.P6(P5) if self.use_p5 else self.P6(C5)

        del C5

        P7 = self.P7(P6)

        return [P3, P4, P5, P6, P7]


class YOLOFPN(nn.Module):
    """
    YOLOFPN module. Draknet 53 is the default backbone of this model
    """
    def __Init__(self,
                 depth=53,
                 in_features=('dark3', 'dark4', 'dark5')):
        super(YOLOFPN, self).__init__()

        self.in_features = in_features

        # out 1
        self.out1_cbl = BaseConv(512, 256, 1, stride=1, act='lrelu')
        self.out1 = self._make_embedding([256, 512], 512 + 256)

        # out 2
        self.out2_cbl = BaseConv(256, 128, 1, stride=1, act='lrelu')
        self.out2 = self._make_embedding([128, 256], 256 + 128)

        # upsample
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def _make_cbl(self, _in, _out, ks):
        return BaseConv(_in, _out, ks, stride=1, act='lrelu')

    def _make_embedding(self, filters_list, in_filters):
        m = nn.Sequential(
            *[
                self._make_cbl(in_filters, filters_list[0], 1),
                self._make_cbl(filters_list[0], filters_list[1], 3),
                self._make_cbl(filters_list[1], filters_list[0], 1),
                self._make_cbl(filters_list[0], filters_list[1], 3),
                self._make_cbl(filters_list[1], filters_list[0], 1),
            ]
        )
        return m

    def forward(self, inputs):
        # if darknet53, inputs shape is [h, w, 3]
        # [b, 256, h/8, w/8], [b, 512, h/16, w/16], [b, 512, h/32, w/32]
        x2, x1, x0 = [inputs[k] for k in inputs.keys()]

        # yolo branch 1
        x1_in = self.out1_cbl(x0)            # [b, 128, h/32, w/32]
        x1_in = self.upsample(x1_in)         # [b, 256, h/16, w/16]
        x1_in = torch.cat([x1_in, x1], 1)    # [b, 256 + 512, h/16, w/16]
        out_dark4 = self.out1(x1_in)         # [b, 256, h/16, w/16]

        # yolo branch 2
        x2_in = self.out2_cbl(out_dark4)      # [b, 128, h/16, w/16]
        x2_in = self.upsample(x2_in)         # [b, 128, h/8, w/8]
        x2_in = torch.cat([x2_in, x2], 1)    # [b, 128 + 256, h/8, w/8]
        out_dark3 = self.out2(x2_in)         # [b, 128, h/8, w/8]

        outputs = (out_dark3, out_dark4, x0)

        return outputs


class YOLOPAFPN(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model
    """
    def __init__(self,
                 depth=1.0,
                 width=1.0,
                 in_features=('dark3', 'dark4', 'dark5'),
                 in_channels=(256, 512, 1024),
                 depthwise=False,
                 act='silu'):
        super(YOLOPAFPN, self).__init__()


if __name__ == "__main__":
    import torch
    ups = nn.Upsample(scale_factor=2, mode='nearest')
    ins = torch.randn(4, 3, 12, 12)
    outs = ups(ins)
    print(outs.shape)
