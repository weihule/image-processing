import os
import sys

import torch
import torch.nn as nn

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from torch_classification.backbones.yolov4backbone import CSPDarkNetTinyBlock, CSPDarkNetBlock, ConvBnActBlock
from torch_detection.utils.util import load_state_dict


__all__ = [
    'yolov4cspdarknettinybackbone',
    'yolov4cspdarknet53backbone',
]


class CSPDarknetTiny(nn.Module):

    def __init__(self, planes=[64, 128, 256, 512], act_type='leakyrelu'):
        super(CSPDarknetTiny, self).__init__()
        self.conv1 = ConvBnActBlock(3,
                                    32,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True,
                                    act_type=act_type)
        self.conv2 = ConvBnActBlock(32,
                                    planes[0],
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True,
                                    act_type=act_type)
        self.block1 = CSPDarkNetTinyBlock(planes[0],
                                          planes[0],
                                          act_type=act_type)
        self.block2 = CSPDarkNetTinyBlock(planes[1],
                                          planes[1],
                                          act_type=act_type)
        self.block3 = CSPDarkNetTinyBlock(planes[2],
                                          planes[2],
                                          act_type=act_type)
        self.conv3 = ConvBnActBlock(planes[3],
                                    planes[3],
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True,
                                    act_type=act_type)

        self.out_channels = [planes[2], planes[3]]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x, _ = self.block1(x)
        x, _ = self.block2(x)
        x, x1 = self.block3(x)
        x2 = self.conv3(x)

        # if input shape is [640, 640, 3]
        # [[B, 256, 40, 40], [B, 512, 20, 20]]
        return x1, x2


class CSPDarknet53(nn.Module):

    def __init__(self,
                 inplanes=32,
                 planes=None,
                 act_type='leakyrelu'):
        super(CSPDarknet53, self).__init__()
        if planes is None:
            self.planes = [64, 128, 256, 512, 1024]
        else:
            self.planes = planes

        self.conv1 = ConvBnActBlock(3,
                                    inplanes,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True,
                                    act_type=act_type)
        self.block1 = CSPDarkNetBlock(inplanes,
                                      self.planes[0],
                                      num_blocks=1,
                                      reduction=False,
                                      act_type=act_type)
        self.block2 = CSPDarkNetBlock(self.planes[0],
                                      self.planes[1],
                                      num_blocks=2,
                                      reduction=True,
                                      act_type=act_type)
        self.block3 = CSPDarkNetBlock(self.planes[1],
                                      self.planes[2],
                                      num_blocks=8,
                                      reduction=True,
                                      act_type=act_type)
        self.block4 = CSPDarkNetBlock(self.planes[2],
                                      self.planes[3],
                                      num_blocks=8,
                                      reduction=True,
                                      act_type=act_type)
        self.block5 = CSPDarkNetBlock(self.planes[3],
                                      self.planes[4],
                                      num_blocks=4,
                                      reduction=True,
                                      act_type=act_type)

        self.out_channels = [self.planes[2], self.planes[3], self.planes[4]]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        C3 = self.block3(x)
        C4 = self.block4(C3)
        C5 = self.block5(C4)

        # if input shape is [640, 640, 3]
        # [[B, 256, 80, 80], [B, 512, 40, 40], [B, 1024, 20, 20]]
        return [C3, C4, C5]


def yolov4cspdarknettinybackbone(pretrained_path='', **kwargs):
    model = CSPDarknetTiny(**kwargs)

    if pretrained_path:
        load_state_dict(pretrained_path, model)
    else:
        print('no backbone pretrained model!')

    return model


def yolov4cspdarknet53backbone(pretrained_path='', **kwargs):
    model = CSPDarknet53(**kwargs)

    if pretrained_path:
        load_state_dict(pretrained_path, model)
    else:
        print('no backbone pretrained model!')

    return model


if __name__ == "__main__":
    from thop import profile, clever_format
    pre_weight_path1 = '/workshop/weihule/data/weights/yolov4backbone/darknet53_857.pth'
    pre_weight_tiny_path1 = '/workshop/weihule/data/weights/yolov4backbone/yolov4cspdarknettiny-acc64.368.pth'

    pre_weight_path2 = 'D:\\workspace\\data\\weights\\yolov4backbone\\darknet53_857.pth'
    pre_weight_tiny_path2 = 'D:\\workspace\\data\\weights\\yolov4backbone\\yolov4cspdarknettiny-acc64.368.pth'

    pre_weight_path = pre_weight_path2
    pre_weight_tiny_path = pre_weight_tiny_path2
    net = yolov4cspdarknet53backbone(pretrained_path=pre_weight_path)
    net_tiny = yolov4cspdarknettinybackbone(pretrained_path=pre_weight_tiny_path)
    inputs = torch.randn(size=(4, 3, 608, 608))
    res = net(inputs)
    for p in res:
        print(p.shape)

    res_tiny = net_tiny(inputs)
    for p in res_tiny:
        print(p.shape)




