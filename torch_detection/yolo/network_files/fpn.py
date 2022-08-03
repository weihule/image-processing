import os
import sys
from torch import Tensor
import torch.nn as nn
import torch
import math
import torch.nn.functional as F

BASE_DIR = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
from torch_detection.yolo.network_files.darknet import ConvBnActBlock


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
                                                  inplanes[2]//2,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0))
            # i = (1, 3)
            else:
                p5_1_layers.append(ConvBnActBlock(inplanes[2]//2,
                                                  inplanes[2],
                                                  kernel_size=3,
                                                  stride=1,
                                                  padding=1))
        # 经过self.p5_1之后,channel变为inplanes[2]//2, 并进行上采样并拼接
        self.p5_1 = nn.Sequential(*p5_1_layers)

        # self.p5_2 和 self.p5_pred_conv获得yolo_head最后的输出特征
        self.p5_2 = ConvBnActBlock(inplanes[2]//2,
                                   inplanes[2],
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
        self.p5_pred_conv = nn.Conv2d(inplanes[2],
                                      per_level_num_anchors*(1+4+num_classes),
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      bias=False)
        self.p5_up_conv = ConvBnActBlock(inplanes[2]//2,
                                         inplanes[1]//2,
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
                                                       inplanes[1]//2,
                                                       kernel_size=1,
                                                       stride=1,
                                                       padding=0))
            else:
                self.p4_1_layers.append(ConvBnActBlock(inplanes[1]//2,
                                                       inplanes[1],
                                                       kernel_size=3,
                                                       stride=1,
                                                       padding=1))

        # 经过self.p4_1之后,channel变为inplanes[1]//2, 并进行上采样并拼接
        self.p4_1 = nn.Sequential(*self.p4_1_layers)

        # self.p4_2 和 self.p4_pred_conv获得yolo_head最后的输出特征
        self.p4_2 = ConvBnActBlock(inplanes[1]//2,
                                   inplanes[1],
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
        self.p4_pred_conv = nn.Conv2d(inplanes[1],
                                      per_level_num_anchors*(1+4+num_classes),
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      bias=False)

        self.p4_up_conv = ConvBnActBlock(inplanes[1]//2,
                                         inplanes[0]//2,
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
        self.p3_2 = ConvBnActBlock(inplanes[0]//2,
                                   inplanes[0],
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
        self.p3_pred_conv = nn.Conv2d(inplanes[0],
                                      per_level_num_anchors*(1+4+num_classes),
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


if __name__ == "__main__":
    from darknet import darknet53backbone
    pre_weight = '/workshop/weihule/data/weights/yolo/darknet53-acc76.836.pth'
    darknet = darknet53backbone(pretrained_path=pre_weight)
    inputs = torch.rand(4, 3, 416, 416)
    fpn_inputs = darknet(inputs)
    for i in fpn_inputs:
        print(i.shape)

    yolo_fpn = YoloV3FPNHead(inplanes=[256, 512, 1024])

    fpn_outputs = yolo_fpn(fpn_inputs)

    for i in fpn_outputs:
        print(i.shape)








