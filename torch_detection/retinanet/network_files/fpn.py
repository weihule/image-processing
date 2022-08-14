import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    def __init__(self, inplanes, planes, use_p5=False):
        super(FPN, self).__init__()
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


# class FPN(nn.Module):
#     def __init__(self, c3_in_channels, c4_in_channels, c5_in_channels, out_channels):
#         super(FPN, self).__init__()
#         self.p3_1 = nn.Conv2d(in_channels=c3_in_channels,
#                               out_channels=out_channels,
#                               kernel_size=1,
#                               stride=1)
#         self.p3_2 = nn.Conv2d(in_channels=out_channels,
#                               out_channels=out_channels,
#                               kernel_size=3,
#                               stride=1,
#                               padding=1)
#         self.p4_1 = nn.Conv2d(in_channels=c4_in_channels,
#                               out_channels=out_channels,
#                               kernel_size=1,
#                               stride=1)
#         self.p4_2 = nn.Conv2d(in_channels=out_channels,
#                               out_channels=out_channels,
#                               kernel_size=3,
#                               stride=1,
#                               padding=1)
#         self.p5_1 = nn.Conv2d(in_channels=c5_in_channels,
#                               out_channels=out_channels,
#                               kernel_size=1,
#                               stride=1)
#         self.p5_2 = nn.Conv2d(in_channels=out_channels,
#                               out_channels=out_channels,
#                               kernel_size=3,
#                               stride=1,
#                               padding=1)
#         self.p6 = nn.Conv2d(in_channels=out_channels,
#                             out_channels=out_channels,
#                             kernel_size=3,
#                             stride=2,
#                             padding=1)
#         self.p7 = nn.Sequential(
#             nn.ReLU(),
#             nn.Conv2d(out_channels, out_channels, 3, 2, 1)
#         )
#
#     def forward(self, inputs):
#         [c3, c4, c5] = inputs
#
#         p5 = self.p5_1(c5)
#         p4 = self.p4_1(c4)
#         # 意思是将 p5 上采样, 上采样之后的形状为 B, C, p4.shape[2], p4.shape[3]
#         p4 += F.interpolate(p5, size=(p4.shape[2], p4.shape[3]), mode='nearest')
#         p3 = self.p3_1(c3)
#         p3 += F.interpolate(p4, size=(p3.shape[2], p3.shape[3]), mode='nearest')
#
#         p6 = self.p6(p5)
#         p7 = self.p7(p6)
#
#         p5 = self.p5_2(p5)
#         p4 = self.p5_2(p4)
#         p3 = self.p5_2(p3)
#
#         del c3, c4, c5
#
#         return [p3, p4, p5, p6, p7]

