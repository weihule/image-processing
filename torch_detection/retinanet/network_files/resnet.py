import os
import torch
import torch.nn as nn


# __all__ = [
#     'resnet50_backbone',
#     'resnet34_backbone'
# ]
#
# class BasicBlock(nn.Module):
#     expansion = 1
#
#     def __init__(self, in_channel, out_channel, stride=1, downsample=None, use_se=False, **kwargs):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channel, out_channel,
#                                kernel_size=3, stride=stride,
#                                padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channel)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(out_channel, out_channel,
#                                kernel_size=3, stride=1,
#                                padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channel)
#         self.downsample = downsample
#
#     def forward(self, x):
#         identity = x
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         out += identity
#         out = self.relu(out)
#         # if self.use_se:
#         #     out = self.se(out)
#
#         return out
#
#
# class Bottleneck(nn.Module):
#     expansion = 4
#
#     def __init__(self, in_channel, out_channel, stride=1, downsample=None):
#         super(Bottleneck, self).__init__()
#
#         self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
#                                kernel_size=1, stride=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channel)
#         # -----------------------------------------
#         self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
#                                kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channel)
#         # -----------------------------------------
#         self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel * self.expansion,
#                                kernel_size=1, stride=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#
#     def forward(self, x):
#         identity = x
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         out += identity
#         out = self.relu(out)
#
#         return out
#
#
# class ResNet(nn.Module):
#     def __init__(self,
#                  block,
#                  blocks_num,
#                  in_channel=64
#                  ):
#         super(ResNet, self).__init__()
#         self.in_channel = in_channel
#
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.in_channel,
#                                kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(self.in_channel)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.channels = [in_channel, in_channel * 2, in_channel * 4, in_channel * 8]
#         self.expansion = 1 if block is BasicBlock else 4
#
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, self.channels[0], blocks_num[0], stride=1)
#         self.layer2 = self._make_layer(block, self.channels[1], blocks_num[1], stride=2)
#         self.layer3 = self._make_layer(block, self.channels[2], blocks_num[2], stride=2)
#         self.layer4 = self._make_layer(block, self.channels[3], blocks_num[3], stride=2)
#
#         self.out_channels = [
#             self.channels[1] * self.expansion,
#             self.channels[2] * self.expansion,
#             self.channels[3] * self.expansion,
#         ]
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#     def _make_layer(self, block, channel, block_num, stride=1):
#         downsample = None
#         """
#        对于layer2, 3, 4来说, stride == 2时, 即第一层需要进行下采样
#         对于layer1来说, 所有层都不需要进行下采样, 但是其第一层
#         需要进行卷积操作, stride == 1, 但是self.in_channel == 64,
#         但是 channel * block.expansion == 256, 这两者不相等，但是
#         仍然有shortcut连接, 所以仍然需要右边进行一个卷积操作来调整channel
#
#         另外, 这里的stride是为了layer的第一层设置的, 传到了block里
#         其余层的stride都是固定的, 已经在block中设置了
#         """
#         if stride != 1 or self.in_channel != channel * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.in_channel, channel * block.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(channel * block.expansion)
#             )
#         layers = []
#
#         # 先把第一层单独添加进去
#         layers.append(block(self.in_channel,
#                             channel,
#                             stride=stride,
#                             downsample=downsample
#                             )
#                       )
#
#         # 每个layer的第一层之后, self.in_channel就变成channel的倍数了
#         self.in_channel = channel * block.expansion
#
#         for _ in range(1, block_num):
#             layers.append(block(self.in_channel,
#                                 channel
#                                 ))
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         x = self.layer1(x)
#         c3 = self.layer2(x)
#         c4 = self.layer3(c3)
#         c5 = self.layer4(c4)
#
#         del x
#
#         return [c3, c4, c5]
#
#
# def load_state_dict(model, pre_trained_path, excluded_layer_name):
#     if not os.path.exists(pre_trained_path):
#         print('No pretrained model file!')
#         return
#     state_dict = torch.load(pre_trained_path, map_location=torch.device('cpu'))
#     filtered_state_dict = {
#         name: weight
#         for name, weight in state_dict.items()
#         if name in model.state_dict() and not any(excluded_name in name for excluded_name in excluded_layer_name)
#         and weight.shape == model.state_dict()[name].shape
#     }
#     if len(filtered_state_dict) == 0:
#         print('No pretrained parameters to load!')
#     else:
#         model.load_state_dict(filtered_state_dict, strict=False)
#
#     return
#
#
# def _resnet_backbone(block, layers, pre_train_path):
#     model = ResNet(block, layers)
#     if pre_train_path:
#         load_state_dict(model, pre_train_path, '')
#     else:
#         print('no backbone pretrained model!')
#
#     return model
#
#
# def resnet50_backbone(pre_train_path=''):
#     model = _resnet_backbone(Bottleneck,
#                              [3, 4, 6, 3],
#                              pre_train_path)
#     return model
#
#
# def resnet34_backbone(pre_train_path=''):
#     model = _resnet_backbone(BasicBlock,
#                              [3, 4, 6, 3],
#                              pre_train_path)
#     return model


__all__ = [
    'resnet18_backbone',
    'resnet34_halfbackbone',
    'resnet34_backbone',
    'resnet50_halfbackbone',
    'resnet50_backbone',
    'resnet101_backbone',
    'resnet152_backbone',
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
        x = self.conv1(x)
        x = self.maxpool1(x)

        x = self.layer1(x)
        C3 = self.layer2(x)
        C4 = self.layer3(C3)
        C5 = self.layer4(C4)

        del x

        return [C3, C4, C5]


def _resnetbackbone(block, layers, inplanes, pretrained_path=''):
    model = ResNetBackbone(block, layers, inplanes)

    if pretrained_path:
        # load_state_dict(pretrained_path, model)
        pass
    else:
        print('no backbone pretrained model!')

    return model


def resnet18_backbone(pretrained_path=''):
    model = _resnetbackbone(BasicBlock, [2, 2, 2, 2],
                            64,
                            pretrained_path=pretrained_path)

    return model


def resnet34_halfbackbone(pretrained_path=''):
    model = _resnetbackbone(BasicBlock, [3, 4, 6, 3],
                            32,
                            pretrained_path=pretrained_path)

    return model


def resnet34_backbone(pretrained_path=''):
    model = _resnetbackbone(BasicBlock, [3, 4, 6, 3],
                            64,
                            pretrained_path=pretrained_path)

    return model


def resnet50_halfbackbone(pre_train_path=''):
    model = _resnetbackbone(Bottleneck, [3, 4, 6, 3],
                            32,
                            pretrained_path=pre_train_path)

    return model


def resnet50_backbone(pre_train_path=''):
    model = _resnetbackbone(Bottleneck, [3, 4, 6, 3],
                            64,
                            pretrained_path=pre_train_path)

    return model


def resnet101_backbone(pretrained_path=''):
    model = _resnetbackbone(Bottleneck, [3, 4, 23, 3],
                            64,
                            pretrained_path=pretrained_path)

    return model


def resnet152_backbone(pretrained_path=''):
    model = _resnetbackbone(Bottleneck, [3, 8, 36, 3],
                            64,
                            pretrained_path=pretrained_path)

    return model


# if __name__ == "__main__":
#     print('hhh')
#     inputs = torch.rand(4, 3, 600, 600)
#     model = resnet50_backbone(pre_train_path='')
#     o3, o4, o5 = model(inputs)
#     print(o3.shape, o4.shape, o5.shape)
