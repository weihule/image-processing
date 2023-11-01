import os
import sys
import torch
import torch.nn as nn

__all__ = [
    'yolov4_csp_darknet_tiny',
    'yolov4_csp_darknet53'
]

url = 'https://zhuanlan.zhihu.com/p/342570549'


class ActivationBlock(nn.Module):
    def __init__(self, act_type='leakyrelu', inplace=True):
        super(ActivationBlock, self).__init__()
        assert act_type in ['silu', 'relu', 'leakyrelu'], 'Unsupported activation type!'

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
            nn.Conv2d(inplanes, planes,
                      kernel_size=kernel_size, stride=stride, padding=padding,
                      groups=groups, bias=bias),
            nn.BatchNorm2d(planes) if has_bn else nn.Sequential(),
            ActivationBlock(act_type=act_type, inplace=True)
            if has_act else nn.Sequential()
        )

    def forward(self, x):
        x = self.layer(x)

        return x


class ResBlock(nn.Module):
    def __init__(self, inplanes, planes, squeeze=False, act_type='leakyrelu'):
        super(ResBlock, self).__init__()
        squeeze_planes = max(1, int(inplanes // 2)) if squeeze else inplanes
        self.conv = nn.Sequential(
            ConvBnActBlock(inplanes=inplanes,
                           planes=squeeze_planes,
                           kernel_size=1,
                           stride=1,
                           padding=0,
                           groups=1,
                           has_bn=True,
                           has_act=True,
                           act_type=act_type),
            ConvBnActBlock(inplanes=squeeze_planes,
                           planes=planes,
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           groups=1,
                           has_bn=True,
                           has_act=True,
                           act_type=act_type))

    def forward(self, x):
        x = x + self.conv(x)

        return x


class CSPDarkNetTinyBlock(nn.Module):
    def __init__(self, inplanes, planes, act_type='leakyrelu'):
        super(CSPDarkNetTinyBlock, self).__init__()
        self.planes = planes

        self.conv1 = ConvBnActBlock(inplanes=inplanes,
                                    planes=planes,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True,
                                    act_type=act_type)
        self.conv2 = ConvBnActBlock(inplanes=planes // 2,
                                    planes=planes // 2,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True,
                                    act_type=act_type)
        self.conv3 = ConvBnActBlock(inplanes=planes // 2,
                                    planes=planes // 2,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True,
                                    act_type=act_type)
        self.conv4 = ConvBnActBlock(inplanes=planes,
                                    planes=planes,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True,
                                    act_type=act_type)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x1 = self.conv1(x)

        # 沿着 通道 这个维度, 将tensor分成 self.planes // 2份
        _, x = torch.split(x1, self.planes // 2, dim=1)

        x2 = self.conv2(x)
        x = self.conv3(x2)

        x = torch.cat([x, x2], dim=1)

        x3 = self.conv4(x)
        x = torch.cat([x1, x3], dim=1)

        x = self.maxpool(x)

        return x, x3


class CSPDarkNetBlock(nn.Module):
    def __init__(self,
                 inplanes,
                 planes,
                 num_blocks,
                 reduction=True,
                 act_type='leakyrelu'):
        super(CSPDarkNetBlock, self).__init__()
        self.front_conv = ConvBnActBlock(inplanes=inplanes,
                                         planes=planes,
                                         kernel_size=3,
                                         stride=2,
                                         padding=1,
                                         groups=1,
                                         has_bn=True,
                                         has_act=True,
                                         act_type=act_type)
        blocks = nn.Sequential(*[
            ResBlock(planes//2 if reduction else planes,
                     planes//2 if reduction else planes,
                     squeeze=not reduction) for _ in range(num_blocks)
        ])
        self.left_conv = nn.Sequential(
            ConvBnActBlock(inplanes=planes,
                           planes=planes // 2 if reduction else planes,
                           kernel_size=1,
                           stride=1,
                           padding=0,
                           groups=1,
                           has_bn=True,
                           has_act=True,
                           act_type=act_type),
            blocks,
            ConvBnActBlock(inplanes=planes // 2 if reduction else planes,
                           planes=planes // 2 if reduction else planes,
                           kernel_size=1,
                           stride=1,
                           padding=0,
                           groups=1,
                           has_bn=True,
                           has_act=True,
                           act_type=act_type))
        self.right_conv = ConvBnActBlock(inplanes=planes,
                                         planes=planes//2 if reduction else planes,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0,
                                         groups=1,
                                         has_bn=True,
                                         has_act=True,
                                         act_type=act_type)
        self.out_conv = ConvBnActBlock(inplanes=planes if reduction else planes * 2,
                                       planes=planes,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0,
                                       groups=1,
                                       has_bn=True,
                                       has_act=True,
                                       act_type=act_type)

    def forward(self, x):
        # input shape is [256, 256, 3]
        # block1

        # shape is [B, 64, 128, 128]
        x = self.front_conv(x)

        left = self.left_conv(x)
        
        right = self.right_conv(x)

        x = torch.cat([left, right], dim=1)

        del left, right

        x = self.out_conv(x)

        return x


class CSPDarkNetTiny(nn.Module):
    def __init__(self,
                 planes=None,
                 act_type='leakyrelu',
                 num_classes=1000):
        super(CSPDarkNetTiny, self).__init__()
        self.num_classes = num_classes
        if planes is None:
            self.planes = [64, 128, 256, 512]
        else:
            self.planes = planes

        self.conv1 = ConvBnActBlock(3, 32,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True,
                                    act_type=act_type)
        self.conv2 = ConvBnActBlock(32, self.planes[0],
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True,
                                    act_type=act_type)
        self.block1 = CSPDarkNetTinyBlock(inplanes=self.planes[0],
                                          planes=self.planes[0],
                                          act_type=act_type)
        self.block2 = CSPDarkNetTinyBlock(inplanes=self.planes[1],
                                          planes=self.planes[1],
                                          act_type=act_type)
        self.block3 = CSPDarkNetTinyBlock(inplanes=self.planes[2],
                                          planes=self.planes[2],
                                          act_type=act_type)
        self.conv3 = ConvBnActBlock(self.planes[3], self.planes[3],
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True,
                                    act_type=act_type)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.planes[3], self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x, _ = self.block1(x)
        x, _ = self.block2(x)
        x, _ = self.block3(x)
        x = self.conv3(x)

        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x


class CSPDarknet53(nn.Module):
    def __init__(self,
                 inplanes=32,
                 planes=None,
                 act_type='leakyrelu',
                 num_classes=1000):
        super(CSPDarknet53, self).__init__()
        self.num_classes = num_classes
        if planes is None:
            self.planes = [64, 128, 256, 512, 1024]

        self.conv1 = ConvBnActBlock(3, inplanes,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True,
                                    act_type=act_type)
        self.block1 = CSPDarkNetBlock(inplanes=inplanes,
                                      planes=self.planes[0],
                                      num_blocks=1,
                                      reduction=False,
                                      act_type=act_type)
        self.block2 = CSPDarkNetBlock(inplanes=self.planes[0],
                                      planes=self.planes[1],
                                      num_blocks=2,
                                      reduction=True,
                                      act_type=act_type)
        self.block3 = CSPDarkNetBlock(inplanes=self.planes[1],
                                      planes=self.planes[2],
                                      num_blocks=8,
                                      reduction=True,
                                      act_type=act_type)
        self.block4 = CSPDarkNetBlock(inplanes=self.planes[2],
                                      planes=self.planes[3],
                                      num_blocks=8,
                                      reduction=True,
                                      act_type=act_type)
        self.block5 = CSPDarkNetBlock(inplanes=self.planes[3],
                                      planes=self.planes[4],
                                      num_blocks=4,
                                      reduction=True,
                                      act_type=act_type)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.planes[4], self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x


def yolov4_csp_darknet_tiny(**kwargs):
    model = CSPDarkNetTiny(**kwargs)

    return model


def yolov4_csp_darknet53(**kwargs):
    model = CSPDarknet53(**kwargs)

    return model


if __name__ == "__main__":
    from thop import profile, clever_format
    
    inputs = torch.randn((4, 3, 256, 256))
    print(inputs.shape)

    # model1 = yolov4_csp_darknet_tiny(num_classes=5)
    # macs1, prarms1 = profile(model1, inputs=inputs, verbose=False)
    # macs1, prarms1 = clever_format([macs1, prarms1], '%.3f')
    # print('model1', macs1, prarms1)

    model2 = yolov4_csp_darknet53(num_classes=5)
    # macs2, prarms2 = profile(model2, inputs=inputs, verbose=False)
    # macs2, prarms2 = clever_format([macs2, prarms2], '%.3f')
    # print('model2', macs2, prarms2)
    outputs = model2(inputs)
    print(outputs.shape)


