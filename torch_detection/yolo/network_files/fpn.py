import os
import sys
from torch import Tensor
import torch.nn as nn
import torch
import math
import torch.nn.functional as F

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from torch_detection.yolo.network_files.darknet import ConvBnActBlock


class SPP(nn.Module):
    """
    Spatial pyramid pooling layer used in Yolov3-SPP
    """

    def __init__(self, kernels=None):
        super(SPP, self).__init__()
        if kernels is None:
            self.kernels = [5, 9, 13]
        else:
            self.kernels = kernels

        self.maxpool_layers = nn.ModuleList([
            nn.MaxPool2d(kernel_size=kernel, stride=1, padding=kernel // 2)
            for kernel in self.kernels
        ])

    def forward(self, x):
        out = torch.cat([x] + [layer(x) for layer in self.maxpool_layers], dim=1)

        return out


class YoloV3FPNHead(nn.Module):
    def __init__(self,
                 inplanes,
                 per_level_num_anchors=3,
                 num_classes=80,
                 act_type='leakyrelu'):
        super(YoloV3FPNHead, self).__init__()
        self.per_level_num_anchors = per_level_num_anchors

        # inplanes: [c3_inplanes, c4_inplanes, c5_inplanes]
        # 这三个输出特征图的宽高分别为(w/8，h/8),(w/16,h/16),(w/32,h/32)
        # 如果输入是(640, 640, 3)
        # 输出是 (B, 256, 80, 80), (B, 512, 40, 40), (B, 1024, 20, 20)
        p5_1_layers = []
        for i in range(5):
            # i = (0, 2, 4)
            if i % 2 == 0:
                p5_1_layers.append(ConvBnActBlock(inplanes[2],
                                                  inplanes[2] // 2,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0))
            # i = (1, 3)
            else:
                p5_1_layers.append(ConvBnActBlock(inplanes[2] // 2,
                                                  inplanes[2],
                                                  kernel_size=3,
                                                  stride=1,
                                                  padding=1))
        # 经过self.p5_1之后,channel变为inplanes[2]//2, 并进行上采样并拼接
        self.p5_1 = nn.Sequential(*p5_1_layers)

        # self.p5_2 和 self.p5_pred_conv获得yolo_head最后的输出特征
        self.p5_2 = ConvBnActBlock(inplanes[2] // 2,
                                   inplanes[2],
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
        self.p5_pred_conv = nn.Conv2d(inplanes[2],
                                      per_level_num_anchors * (1 + 4 + num_classes),
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      bias=False)
        self.p5_up_conv = ConvBnActBlock(inplanes[2] // 2,
                                         inplanes[1] // 2,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0)

        self.p4_1_layers = []
        # 先添加一层
        self.p4_1_layers.append(ConvBnActBlock(inplanes[1] + inplanes[1] // 2,
                                               inplanes[1] // 2,
                                               kernel_size=1,
                                               stride=1,
                                               padding=0))
        for i in range(1, 5):
            if i % 2 == 0:
                self.p4_1_layers.append(ConvBnActBlock(inplanes[1],
                                                       inplanes[1] // 2,
                                                       kernel_size=1,
                                                       stride=1,
                                                       padding=0))
            else:
                self.p4_1_layers.append(ConvBnActBlock(inplanes[1] // 2,
                                                       inplanes[1],
                                                       kernel_size=3,
                                                       stride=1,
                                                       padding=1))

        # 经过self.p4_1之后,channel变为inplanes[1]//2, 并进行上采样并拼接
        self.p4_1 = nn.Sequential(*self.p4_1_layers)

        # self.p4_2 和 self.p4_pred_conv获得yolo_head最后的输出特征
        self.p4_2 = ConvBnActBlock(inplanes[1] // 2,
                                   inplanes[1],
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
        self.p4_pred_conv = nn.Conv2d(inplanes[1],
                                      per_level_num_anchors * (1 + 4 + num_classes),
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      bias=False)

        self.p4_up_conv = ConvBnActBlock(inplanes[1] // 2,
                                         inplanes[0] // 2,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0)

        self.p3_1_layers = []
        # 先添加一层
        self.p3_1_layers.append(ConvBnActBlock(inplanes[0] + inplanes[0] // 2,
                                               inplanes[0] // 2,
                                               kernel_size=1,
                                               stride=1,
                                               padding=0))
        for i in range(1, 5):
            if i % 2 == 0:
                self.p3_1_layers.append(ConvBnActBlock(inplanes[0],
                                                       inplanes[0] // 2,
                                                       kernel_size=1,
                                                       stride=1,
                                                       padding=0))
            else:
                self.p3_1_layers.append(ConvBnActBlock(inplanes[0] // 2,
                                                       inplanes[0],
                                                       kernel_size=3,
                                                       stride=1,
                                                       padding=1))
        # 经过self.p3_1之后,channel变为inplanes[0]//2, 并进行上采样并拼接
        self.p3_1 = nn.Sequential(*self.p3_1_layers)

        # self.p3_2 和 self.p3_pred_conv获得yolo_head最后的输出特征
        self.p3_2 = ConvBnActBlock(inplanes[0] // 2,
                                   inplanes[0],
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
        self.p3_pred_conv = nn.Conv2d(inplanes[0],
                                      per_level_num_anchors * (1 + 4 + num_classes),
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs: Tensor):
        [c3, c4, c5] = inputs

        p5 = self.p5_1(c5)
        del c5
        p5_up_sample = F.interpolate(self.p5_up_conv(p5),
                                     size=(c4.shape[2], c4.shape[3]),
                                     mode='bilinear',
                                     align_corners=True)
        c4 = torch.cat([c4, p5_up_sample], dim=1)
        del p5_up_sample
        p4 = self.p4_1(c4)
        p4_up_sample = F.interpolate(self.p4_up_conv(p4),
                                     size=(c3.shape[2], c3.shape[3]),
                                     mode='bilinear',
                                     align_corners=True)
        c3 = torch.cat([c3, p4_up_sample], dim=1)
        del p4_up_sample
        p3 = self.p3_1(c3)
        del c3

        p5 = self.p5_2(p5)
        p5 = self.p5_pred_conv(p5)

        p4 = self.p4_2(p4)
        p4 = self.p4_pred_conv(p4)

        p3 = self.p3_2(p3)
        p3 = self.p3_pred_conv(p3)

        # p5 shape: [B,255,H,W] -> [B,H,W,255] -> [B,H,W,3,255]
        p5 = p5.permute(0, 3, 2, 1).contiguous()
        p5 = p5.view(p5.shape[0], p5.shape[1], p5.shape[2],
                     self.per_level_num_anchors, -1)

        # p4 shape: [B,255,H,W] -> [B,H,W,255] -> [B,H,W,3,255]
        p4 = p4.permute(0, 3, 2, 1).contiguous()
        p4 = p4.view(p4.shape[0], p4.shape[1], p4.shape[2],
                     self.per_level_num_anchors, -1)

        # p3 shape: [B,255,H,W] -> [B,H,W,255] -> [B,H,W,3,255]
        p3 = p3.permute(0, 3, 2, 1).contiguous()
        p3 = p3.view(p3.shape[0], p3.shape[1], p3.shape[2],
                     self.per_level_num_anchors, -1)

        p5[:, :, :, :, :3] = self.sigmoid(p5[:, :, :, :, :3])
        p5[:, :, :, :, 5:] = self.sigmoid(p5[:, :, :, :, 5:])

        p4[:, :, :, :, :3] = self.sigmoid(p4[:, :, :, :, :3])
        p4[:, :, :, :, 5:] = self.sigmoid(p4[:, :, :, :, 5:])

        p3[:, :, :, :, :3] = self.sigmoid(p3[:, :, :, :, :3])
        p3[:, :, :, :, 5:] = self.sigmoid(p3[:, :, :, :, 5:])

        return [p3, p4, p5]


# yolov4 url = 'https://zhuanlan.zhihu.com/p/342570549'
class Yolov4FPNHead(nn.Module):
    def __init__(self,
                 inplanes,
                 per_level_num_anchors=3,
                 num_classes=80,
                 act_type='leakyrelu'):
        super(Yolov4FPNHead, self).__init__()

        # inplanes: [C3_inplanes, C4_inplanes, C5_inplanes]
        # [256, 512, 1024]
        self.per_level_num_anchors = per_level_num_anchors

        # 通过 p5_block1 之后, W和H并未变化, 通道变为原来的一半
        p5_block1 = nn.Sequential(*[
            ConvBnActBlock(inplanes=inplanes[2],
                           planes=inplanes[2] // 2,
                           kernel_size=1,
                           stride=1,
                           padding=0,
                           groups=1,
                           has_bn=True,
                           has_act=True,
                           act_type=act_type) if idx % 2 == 0
            else ConvBnActBlock(inplanes=inplanes[2] // 2,
                                planes=inplanes[2],
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                groups=1,
                                has_bn=True,
                                has_act=True,
                                act_type=act_type)
            for idx in range(3)
        ])
        p5_spp_block = SPP(kernels=[5, 9, 13])
        p5_block2 = nn.Sequential(
            ConvBnActBlock(inplanes=inplanes[2] * 2,
                           planes=inplanes[2] // 2,
                           kernel_size=1,
                           stride=1,
                           padding=0,
                           groups=1,
                           has_bn=True,
                           has_act=True,
                           act_type=act_type),
            ConvBnActBlock(inplanes=inplanes[2] // 2,
                           planes=inplanes[2],
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           groups=1,
                           has_bn=True,
                           has_act=True,
                           act_type=act_type),
            ConvBnActBlock(inplanes=inplanes[2],
                           planes=inplanes[2] // 2,
                           kernel_size=1,
                           stride=1,
                           padding=0,
                           groups=1,
                           has_bn=True,
                           has_act=True,
                           act_type=act_type))
        self.P5_1 = nn.Sequential(p5_block1, p5_spp_block, p5_block2)
        self.P5_up_conv = ConvBnActBlock(inplanes=inplanes[2] // 2,
                                         planes=inplanes[1] // 2,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0,
                                         groups=1,
                                         has_bn=True,
                                         has_act=True,
                                         act_type=act_type)
        self.P4_cat_conv = ConvBnActBlock(inplanes=inplanes[1],
                                          planes=inplanes[1] // 2,
                                          kernel_size=1,
                                          stride=1,
                                          padding=0,
                                          groups=1,
                                          has_bn=True,
                                          has_act=True,
                                          act_type=act_type)
        self.P4_1 = nn.Sequential(*[
            ConvBnActBlock(inplanes=inplanes[1],
                           planes=inplanes[1] // 2,
                           kernel_size=1,
                           stride=1,
                           padding=0,
                           groups=1,
                           has_bn=True,
                           has_act=True,
                           act_type=act_type) if idx % 2 == 0
            else ConvBnActBlock(inplanes=inplanes[1] // 2,
                                planes=inplanes[1],
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                groups=1,
                                has_bn=True,
                                has_act=True,
                                act_type=act_type)
            for idx in range(5)])
        self.P4_up_conv = ConvBnActBlock(inplanes=inplanes[1] // 2,
                                         planes=inplanes[0] // 2,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0,
                                         groups=1,
                                         has_bn=True,
                                         has_act=True,
                                         act_type=act_type)
        self.P3_cat_conv = ConvBnActBlock(inplanes=inplanes[0],
                                          planes=inplanes[0] // 2,
                                          kernel_size=1,
                                          stride=1,
                                          padding=0,
                                          groups=1,
                                          has_bn=True,
                                          has_act=True,
                                          act_type=act_type)
        self.P3_1 = nn.Sequential(*[
            ConvBnActBlock(inplanes=inplanes[0],
                           planes=inplanes[0] // 2,
                           kernel_size=1,
                           stride=1,
                           padding=0,
                           groups=1,
                           has_bn=True,
                           has_act=True,
                           act_type=act_type) if idx % 2 == 0
            else ConvBnActBlock(inplanes=inplanes[0] // 2,
                                planes=inplanes[0],
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                groups=1,
                                has_bn=True,
                                has_act=True,
                                act_type=act_type)
            for idx in range(5)
        ])
        self.P3_out_conv = ConvBnActBlock(inplanes=inplanes[0] // 2,
                                          planes=inplanes[0],
                                          kernel_size=3,
                                          stride=1,
                                          padding=1,
                                          groups=1,
                                          has_bn=True,
                                          has_act=True,
                                          act_type=act_type)
        self.P3_down_conv = ConvBnActBlock(inplanes=inplanes[0] // 2,
                                           planes=inplanes[1] // 2,
                                           kernel_size=3,
                                           stride=2,
                                           padding=1,
                                           groups=1,
                                           has_bn=True,
                                           has_act=True,
                                           act_type=act_type)
        self.P4_2 = nn.Sequential(*[
            ConvBnActBlock(inplanes=inplanes[1],
                           planes=inplanes[1] // 2,
                           kernel_size=1,
                           stride=1,
                           padding=0,
                           groups=1,
                           has_bn=True,
                           has_act=True,
                           act_type=act_type) if idx % 2 == 0
            else ConvBnActBlock(inplanes=inplanes[1] // 2,
                                planes=inplanes[1],
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                groups=1,
                                has_bn=True,
                                has_act=True,
                                act_type=act_type)
            for idx in range(5)
        ])
        self.P4_out_conv = ConvBnActBlock(inplanes=inplanes[1] // 2,
                                          planes=inplanes[1],
                                          kernel_size=3,
                                          stride=1,
                                          padding=1,
                                          groups=1,
                                          has_bn=True,
                                          has_act=True,
                                          act_type=act_type)
        self.P4_down_conv = ConvBnActBlock(inplanes=inplanes[1] // 2,
                                           planes=inplanes[2] // 2,
                                           kernel_size=3,
                                           stride=2,
                                           padding=1,
                                           groups=1,
                                           has_bn=True,
                                           has_act=True,
                                           act_type=act_type)
        self.P5_2 = nn.Sequential(*[
            ConvBnActBlock(inplanes=inplanes[2],
                           planes=inplanes[2] // 2,
                           kernel_size=1,
                           stride=1,
                           padding=0,
                           groups=1,
                           has_bn=True,
                           has_act=True,
                           act_type=act_type) if idx % 2 == 0
            else ConvBnActBlock(inplanes=inplanes[2] // 2,
                                planes=inplanes[2],
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                groups=1,
                                has_bn=True,
                                has_act=True,
                                act_type=act_type)
            for idx in range(5)
        ])
        self.P5_out_conv = ConvBnActBlock(inplanes=inplanes[2] // 2,
                                          planes=inplanes[2],
                                          kernel_size=3,
                                          stride=1,
                                          padding=1,
                                          groups=1,
                                          has_bn=True,
                                          has_act=True,
                                          act_type=act_type)
        self.P5_pred_conv = nn.Conv2d(in_channels=inplanes[2],
                                      out_channels=per_level_num_anchors*(1+4+num_classes),
                                      kernel_size=1,
                                      padding=0,
                                      stride=1,
                                      bias=True)

        self.P4_pred_conv = nn.Conv2d(in_channels=inplanes[1],
                                      out_channels=per_level_num_anchors*(1+4+num_classes),
                                      kernel_size=1,
                                      padding=0,
                                      stride=1,
                                      bias=True)

        self.P3_pred_conv = nn.Conv2d(in_channels=inplanes[0],
                                      out_channels=per_level_num_anchors*(1+4+num_classes),
                                      kernel_size=1,
                                      padding=0,
                                      stride=1,
                                      bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        # if inputs is [608, 608, 3]
        # strides is [8, 16, 32]
        # inputs is [[B, 256, 76, 76], [B, 512, 38, 38], [B, 1024, 19, 19]]
        [C3, C4, C5] = inputs

        # 经过 self.Px_1 之后, 通道数都变为原来的 1/2

        # [B, 512, 19, 19]
        P5 = self.P5_1(C5)
        del C5

        # self.P5_up_conv(P5) shape is [B, 256, 19, 19]
        # P5_upsample shape is [B, 256, 38, 38]
        P5_upsample = F.interpolate(self.P5_up_conv(P5),
                                    size=(C4.shape[2], C4.shape[3]),
                                    mode='bilinear',
                                    align_corners=True)

        # self.P4_cat_conv(C4) shape is [B, 256, 38, 38]
        # C4 shape is [B, 512, 38, 38], 仍保持原来的shape
        C4 = torch.cat([self.P4_cat_conv(C4), P5_upsample], dim=1)
        del P5_upsample

        # P4 shape is [B, 256, 38, 38]
        P4 = self.P4_1(C4)
        del C4

        # P4 shape is [B, 128, 76, 76]
        P4_upsample = F.interpolate(self.P4_up_conv(P4),
                                    size=(C3.shape[2], C3.shape[3]),
                                    mode='bilinear',
                                    align_corners=True)

        # self.P3_cat_conv(C3) shape is [B, 128, 76, 76]
        # C3 shape is [B, 256, 76, 76], 仍保持原来的shape
        C3 = torch.cat([self.P3_cat_conv(C3), P4_upsample], dim=1)
        del P4_upsample

        # P3 shape is [B, 128, 76, 76]
        P3 = self.P3_1(C3)
        del C3

        # P3_out shape is [B, 256, 76, 76]
        P3_out = self.P3_out_conv(P3)

        # P3_out shape is [B, 3*(1+4+num_classes), 76, 76]
        P3_out = self.P3_pred_conv(P3_out)

        # P4 shape is [B, 512, 38, 38]
        P4 = torch.cat([P4, self.P3_down_conv(P3)], dim=1)
        del P3

        # P4 shape is [B, 256, 38, 38]
        P4 = self.P4_2(P4)

        # P4_out shape is [B, 512, 38, 38]
        P4_out = self.P4_out_conv(P4)

        # P4_out shape is [B, 255, 38, 38]
        P4_out = self.P4_pred_conv(P4_out)

        P5 = torch.cat([P5, self.P4_down_conv(P4)], dim=1)
        del P4

        # P5 shape is [B, 512, 19, 19]
        P5 = self.P5_2(P5)

        # P5_out shape is [B, 1024, 19, 19]
        P5_out = self.P5_out_conv(P5)

        # P5_out shape is [B, 255, 19, 19]
        P5_out = self.P5_pred_conv(P5_out)
        del P5

        # P3_out shape: [B, 255, H, W] -> [B, H, W, 255] -> [B, H, W, 3, 85]
        P3_out = P3_out.permute(0, 2, 3, 1).contiguous()
        P3_out = P3_out.view(P3_out.shape[0], P3_out.shape[1], P3_out.shape[2],
                             self.per_level_num_anchors, -1)

        # P4_out shape: [B, 255, H, W] -> [B, H, W, 255] -> [B, H, W, 3, 85]
        P4_out = P4_out.permute(0, 2, 3, 1).contiguous()
        P4_out = P4_out.view(P4_out.shape[0], P4_out.shape[1], P4_out.shape[2],
                             self.per_level_num_anchors, -1)

        # P5_out shape: [B, 255, H, W] -> [B, H, W, 255] -> [B, H, W, 3, 85]
        P5_out = P5_out.permute(0, 2, 3, 1).contiguous()
        P5_out = P5_out.view(P5_out.shape[0], P5_out.shape[1], P5_out.shape[2],
                             self.per_level_num_anchors, -1)

        P3_out[:, :, :, :, 0:3] = torch.sigmoid(P3_out[:, :, :, :, 0:3])
        P3_out[:, :, :, :, 5:] = torch.sigmoid(P3_out[..., 5:])
        P4_out[:, :, :, :, 0:3] = torch.sigmoid(P4_out[:, :, :, :, 0:3])
        P4_out[:, :, :, :, 5:] = torch.sigmoid(P4_out[..., 5:])
        P5_out[:, :, :, :, 0:3] = torch.sigmoid(P5_out[:, :, :, :, 0:3])
        P5_out[:, :, :, :, 5:] = torch.sigmoid(P5_out[..., 5:])

        # [[B, h1, w1, 3, (1+4+NUM_classes)], ...]
        return [P3_out, P4_out, P5_out]


if __name__ == "__main__":
    # from darknet import darknet53backbone
    # pre_weight = '/workshop/weihule/data/weights/yolo/darknet53-acc76.836.pth'
    # darknet = darknet53backbone(pretrained_path=pre_weight)
    # inputs = torch.rand(4, 3, 416, 416)
    # fpn_inputs = darknet(inputs)
    #
    # yolo_fpn = YoloV3FPNHead(inplanes=[256, 512, 1024])
    #
    # fpn_outputs = yolo_fpn(fpn_inputs)
    #
    # for i in fpn_outputs:
    #     print(i.shape)

    v4_fpn = Yolov4FPNHead(inplanes=[256, 512, 1024],
                           per_level_num_anchors=3,
                           num_classes=20)
    inputs = [torch.randn((4, 256, 76, 76)), torch.randn((4, 512, 38, 38)), torch.randn((4, 1024, 19, 19))]
    res = v4_fpn(inputs)
    for p in res:
        print(p.shape)
