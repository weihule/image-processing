import os
import sys
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

base_dir = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__))))
sys.path.append(base_dir)
from utils.util import get_logger, load_state_dict

__all__ = [
    'darknet53backbone'
]


class ActivationBlock(nn.Module):
    def __init__(self, act_type='leakyrelu', inplace=True):
        super(ActivationBlock, self).__init__()
        assert act_type in ['silu', 'relu', 'leakyrelu'], 'Unsupport activation function!'
        if act_type == 'silu':
            self.act = nn.SiLU(inplace=inplace)
        elif act_type == 'relu':
            self.act = nn.ReLU(inplace=inplace)
        elif act_type == 'leakyrelu':
            self.act = nn.LeakyReLU(0.1, inplace=inplace)

    def forward(self, x):
        x = self.act(x)

        return x


class ConvBnActBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size,
                 stride,
                 padding,
                 groups=1,
                 has_bn=True,
                 has_act=True,
                 act_type='leakyrelu'):
        super(ConvBnActBlock, self).__init__()
        bias = False if has_bn else True

        self.layer = nn.Sequential(
            nn.Conv2d(inplanes,
                      planes,
                      kernel_size,
                      stride=stride,
                      padding=padding,
                      groups=groups,
                      bias=bias),
            nn.BatchNorm2d(planes) if has_bn else nn.Sequential(),
            ActivationBlock(act_type=act_type, inplace=True)
            if has_act else nn.Sequential(),
        )

    def forward(self, x):
        x = self.layer(x)

        return x


class Darknet19Block(nn.Module):
    def __init__(self,
                 inplanes,
                 planes,
                 layer_num,
                 use_maxpool=False,
                 act_type='leakyrelu'):
        super(Darknet19Block, self).__init__()
        self.use_maxpool = use_maxpool
        layers = []
        for i in range(0, layer_num):
            if i % 2 == 0:
                layers.append(
                    ConvBnActBlock(inplanes,
                                   planes,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   groups=1,
                                   has_bn=True,
                                   has_act=True,
                                   act_type=act_type))
            else:
                layers.append(
                    ConvBnActBlock(planes,
                                   inplanes,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0,
                                   groups=1,
                                   has_bn=True,
                                   has_act=True,
                                   act_type=act_type))

        self.Darknet19Block = nn.Sequential(*layers)
        if self.use_maxpool:
            self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.Darknet19Block(x)

        if self.use_maxpool:
            x = self.MaxPool(x)

        return x


class Darknet53Block(nn.Module):
    def __init__(self, inplanes, act_type='leakyrelu'):
        super(Darknet53Block, self).__init__()
        squeezed_planes = int(inplanes // 2)
        self.conv = nn.Sequential(
            ConvBnActBlock(
                inplanes=inplanes,
                planes=squeezed_planes,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1),
            ConvBnActBlock(
                inplanes=squeezed_planes,
                planes=inplanes,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=1)
        )

    def forward(self, x: Tensor):
        x = self.conv(x)
        return x


class Darknet53Backbone(nn.Module):
    def __init__(self, act_type='leakyrelu'):
        super(Darknet53Backbone, self).__init__()

        self.conv1 = ConvBnActBlock(3,
                                    32,
                                    kernel_size=3,
                                    padding=1,
                                    stride=1)
        self.conv2 = ConvBnActBlock(32,
                                    64,
                                    kernel_size=3,
                                    padding=1,
                                    stride=2)
        self.block1 = self._make_layer(64, 1)
        self.conv3 = ConvBnActBlock(64,
                                    128,
                                    kernel_size=3,
                                    padding=1,
                                    stride=2)
        self.block2 = self._make_layer(128, 2)
        self.conv4 = ConvBnActBlock(128,
                                    256,
                                    kernel_size=3,
                                    padding=1,
                                    stride=2)
        self.block3 = self._make_layer(256, 8)
        self.conv5 = ConvBnActBlock(256,
                                    512,
                                    kernel_size=3,
                                    padding=1,
                                    stride=2)
        self.block4 = self._make_layer(512, 8)
        self.conv6 = ConvBnActBlock(512,
                                    1024,
                                    kernel_size=3,
                                    padding=1,
                                    stride=2)
        self.block5 = self._make_layer(1024, 4)

        self.out_channels = [256, 512, 1024]

    def forward(self, x: Tensor):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.block1(x)
        x = self.conv3(x)
        x = self.block2(x)
        x = self.conv4(x)

        c3 = self.block3(x)
        c4 = self.conv5(c3)
        c4 = self.block4(c4)
        c5 = self.conv6(c4)
        c5 = self.block5(c5)

        del x

        # 这三个输出特征图的宽高分别为(w/8，h/8),(w/16,h/16),(w/32,h/32)
        # 如果输入是(640, 640, 3)
        # 输出是 (B, 256, 80, 80), (B, 256, 40, 40), (B, 256, 20, 20)
        return [c3, c4, c5]

    @staticmethod
    def _make_layer(inplanes, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(Darknet53Block(inplanes))
        return nn.Sequential(*layers)


def darknet53backbone(pretrained_path='', **kwargs):
    model = Darknet53Backbone(**kwargs)

    if pretrained_path:
        load_state_dict(pretrained_path, model)
    else:
        print('no backbone pretrained model!')

    return model


if __name__ == "__main__":
    pre_weight = '/workshop/weihule/data/weights/yolo/darknet53-acc76.836.pth'
    darknet = darknet53backbone(pretrained_path=pre_weight)
    inputs = torch.rand(4, 3, 640, 640)
    res = darknet(inputs)
    for p in res:
        print(p.shape)
