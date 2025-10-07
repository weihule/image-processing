import os
import sys
import torch
import torch.nn as nn
sys.path.append(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))


__all__ = [
    'resnet18backbone',
    'resnet34halfbackbone',
    'resnet34backbone',
    'resnet50halfbackbone',
    'resnet50backbone',
    'resnet101backbone',
    'resnet152backbone',
]


class ConvBnActBlock(nn.Module):
    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size,
                 stride,
                 padding,
                 groups=1,
                 has_bn=True,
                 has_act=True):
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
            nn.ReLU(inplace=True) if has_act else nn.Sequential(),
        )

    def forward(self, x):
        x = self.layer(x)

        return x


class BasicBlock(nn.Module):
    """
    构建resnet18和resnet34所用的残差块
    """
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.downsample = True if stride != 1 or inplanes != planes * 1 else False

        self.conv1 = ConvBnActBlock(inplanes,
                                    planes,
                                    kernel_size=3,
                                    stride=stride,
                                    padding=1,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True)
        self.conv2 = ConvBnActBlock(planes,
                                    planes,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=1,
                                    has_bn=True,
                                    has_act=False)
        self.relu = nn.ReLU(inplace=True)

        if self.downsample:
            self.downsample_conv = ConvBnActBlock(inplanes,
                                                  planes,
                                                  kernel_size=1,
                                                  stride=stride,
                                                  padding=0,
                                                  groups=1,
                                                  has_bn=True,
                                                  has_act=False)

    def forward(self, x):
        inputs = x

        x = self.conv1(x)
        x = self.conv2(x)

        if self.downsample:
            inputs = self.downsample_conv(inputs)

        x = x + inputs
        x = self.relu(x)

        return x


class Bottleneck(nn.Module):
    """
    构建resnet50、resnet100和resnet152所用的残差块
    """
    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.downsample = True if stride != 1 or inplanes != planes * 4 else False

        self.conv1 = ConvBnActBlock(inplanes,
                                    planes,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True)
        self.conv2 = ConvBnActBlock(planes,
                                    planes,
                                    kernel_size=3,
                                    stride=stride,
                                    padding=1,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True)
        self.conv3 = ConvBnActBlock(planes,
                                    planes * 4,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1,
                                    has_bn=True,
                                    has_act=False)
        self.relu = nn.ReLU(inplace=True)

        if self.downsample:
            self.downsample_conv = ConvBnActBlock(inplanes,
                                                  planes * 4,
                                                  kernel_size=1,
                                                  stride=stride,
                                                  padding=0,
                                                  groups=1,
                                                  has_bn=True,
                                                  has_act=False)

    def forward(self, x):
        inputs = x

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        if self.downsample:
            inputs = self.downsample_conv(inputs)

        x = x + inputs
        x = self.relu(x)

        return x


class ResNetBackbone(nn.Module):
    def __init__(self, block, layer_nums, inplanes=64):
        """
        Args:
            block: BasicBlock or BottleNeck
            layer_nums: if resnet50, [3, 4, 6, 3]
            inplanes: 64
        """
        super(ResNetBackbone, self).__init__()
        self.block = block
        self.layer_nums = layer_nums
        self.inplanes = inplanes
        self.planes = [inplanes, inplanes * 2, inplanes * 4, inplanes * 8]
        self.expansion = 1 if block is BasicBlock else 4

        self.conv1 = ConvBnActBlock(3,
                                    self.inplanes,
                                    kernel_size=7,
                                    stride=2,
                                    padding=3,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(self.block,
                                      self.planes[0],
                                      self.layer_nums[0],
                                      stride=1)
        self.layer2 = self.make_layer(self.block,
                                      self.planes[1],
                                      self.layer_nums[1],
                                      stride=2)
        self.layer3 = self.make_layer(self.block,
                                      self.planes[2],
                                      self.layer_nums[2],
                                      stride=2)
        self.layer4 = self.make_layer(self.block,
                                      self.planes[3],
                                      self.layer_nums[3],
                                      stride=2)

        # self.planes = [64, 128, 256, 512]
        self.out_channels = [
            self.planes[1] * self.expansion,
            self.planes[2] * self.expansion,
            self.planes[3] * self.expansion,
        ]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, block, planes, layer_nums, stride):
        layers = []
        for i in range(0, layer_nums):
            if i == 0:
                layers.append(block(self.inplanes, planes, stride))
            else:
                layers.append(block(self.inplanes, planes))
            self.inplanes = planes * self.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        # if x: [B, 3, 64, 640] and model is resnet50
        x = self.conv1(x)        # [B, 64, 320, 320]
        x = self.maxpool1(x)     # [B, 64, 160, 160]

        x = self.layer1(x)       # [B, 256, 160, 160]
        C3 = self.layer2(x)      # [B, 512, 80, 80]
        C4 = self.layer3(C3)     # [B, 1024, 40, 40]
        C5 = self.layer4(C4)     # [B, 2048, 20, 20]

        del x

        return [C3, C4, C5]


def _resnetbackbone(block, layers, inplanes, **kwargs):
    model = ResNetBackbone(block, layers, inplanes)

    return model


def resnet18backbone(**kwargs):
    model = _resnetbackbone(BasicBlock, [2, 2, 2, 2],
                            64,
                            **kwargs)

    return model


def resnet34halfbackbone(**kwargs):
    model = _resnetbackbone(BasicBlock, [3, 4, 6, 3],
                            32,
                            **kwargs)

    return model


def resnet34backbone(**kwargs):
    model = _resnetbackbone(BasicBlock, [3, 4, 6, 3],
                            64,
                            **kwargs)

    return model


def resnet50halfbackbone(**kwargs):
    model = _resnetbackbone(Bottleneck, [3, 4, 6, 3],
                            32,
                            **kwargs)

    return model


def resnet50backbone(**kwargs):
    model = _resnetbackbone(Bottleneck, [3, 4, 6, 3],
                            64,
                            **kwargs)

    return model


def resnet101backbone(**kwargs):
    model = _resnetbackbone(Bottleneck, [3, 4, 23, 3],
                            64,
                            **kwargs)

    return model


def resnet152backbone(**kwargs):
    model = _resnetbackbone(Bottleneck, [3, 8, 36, 3],
                            64,
                            **kwargs)

    return model


if __name__ == '__main__':
    net = resnet50backbone()
    print("Model size: {:.5f}M".format(sum(p.numel() for p in net.parameters()) / 1000000.0))
    inputs_ = torch.randn(4, 3, 640, 640)
    outputs_ = net(inputs_)
    for p in outputs_:
        print(p.shape)

