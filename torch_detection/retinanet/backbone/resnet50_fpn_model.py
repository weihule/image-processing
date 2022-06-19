import os
from sys import modules
import torch
import torch.nn as nn
from torch import Tensor
from torchvision.ops.misc import FrozenBatchNorm2d
import torch.nn.functional as F
from collections import OrderedDict
from typing import Tuple, List, Dict
from feature_pyramid_network import IntermediateLayerGetter, BackboneWithFPN, FeaturePyramidNetwork, LastLevelMaxPool
from custome_resnet50_fpn import LayerGetter

class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, down_sample=None, norm_layer=None):
        super(BottleNeck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = norm_layer(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = norm_layer(out_channel)
        self.conv3 = nn.Conv2d(out_channel, out_channel * self.expansion,
                               kernel_size=1, stride=1, bias=False)
        self.bn3 = norm_layer(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.down_sample = down_sample

    def forward(self, x: Tensor):
        identity = x
        if self.down_sample is not None:
            identity = self.down_sample(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))

        x += identity
        x = self.relu(x)

        return x


class ResNet(nn.Module):
    def __init__(self, block, block_nums, num_classes=1000, include_top=True, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.include_top = include_top
        self.in_channel = 64
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, channel=64, block_num=block_nums[0], stride=1)
        self.layer2 = self._make_layer(block, channel=128, block_num=block_nums[1], stride=2)
        self.layer3 = self._make_layer(block, channel=256, block_num=block_nums[2], stride=2)
        self.layer4 = self._make_layer(block, channel=512, block_num=block_nums[3], stride=2)
        if self.include_top:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        norm_layer = self._norm_layer
        down_sample = None
        if stride != 1 or channel != channel * block.expansion:
            down_sample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion)
            )
        layers = OrderedDict()
        layers.update({'block1': block(self.in_channel, channel,
                                       stride=stride, down_sample=down_sample, norm_layer=norm_layer)})

        self.in_channel = channel * block.expansion
        for i in range(block_num - 1):
            k = 'block' + str(i + 2)
            layers.update({k: block(self.in_channel, channel, norm_layer=norm_layer)})

        return nn.Sequential(layers)

    def forward(self, x: Tensor):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avg_pool(x)
            x = torch.flatten(x, start_dim=1)
            x = self.fc(x)

        return x


def overwrite_eps(model, eps):
    for m in model.modules():
        if isinstance(m, FrozenBatchNorm2d):
            m.eps = eps


def resnet50_fpn_backbone(pretrain_path="",
                          norm_layer=FrozenBatchNorm2d,
                          trainable_layers=3,   # 可以设置成1-5之间的数
                          returned_layers=None,
                          extra_blocks=None):
    """
    resnet50_fpn backbone
    :param pretrain_path: resnet50的预训练权重
    :param norm_layer: 如果自己的GPU显存很大可以设置很大的batch_size, 那么自己可以传入正常的BatchNorm2d层
    :param trainable_layers: 指定需要训练的层结构
    :param returned_layers: 指定需要返回的层结构
    :param extra_blocks: 在输出的特征图基础上额外添加的层结构
    :return:
    """
    resnet_backbone = ResNet(BottleNeck, [3, 4, 6, 3],
                             num_classes=1000, include_top=True, norm_layer=norm_layer)
    if isinstance(norm_layer, FrozenBatchNorm2d):
        overwrite_eps(resnet_backbone, 0.0)
    if pretrain_path != "":
        assert os.path.exists(pretrain_path), '{} is not exist'.format(pretrain_path)
        resnet_backbone.load_state_dict(torch.load(pretrain_path), strict=False)

    assert 1 <= trainable_layers <= 5, \
        'trainable_layers is not valid, expected 1 between and 5, got {}'.format(trainable_layers)
    layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]

    # 如果所有都需要训练，那么切记需要加上conv1之后的bn1
    if trainable_layers == 5:
        layers_to_train.append('bn1')

    # freeze layers
    for name, parameter in resnet_backbone.named_parameters():
        # 只需要训练在 layers_to_train 中的层结构
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad = False

    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()

    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]
    assert min(returned_layers) > 0 and max(returned_layers) < 5

    # {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
    return_layers = {f'layer{k}': str(idx) for idx, k in enumerate(returned_layers)}

    # in_channel 为layer4的输出特征矩阵channel = 512*4
    in_channels_stage2 = resnet_backbone.in_channel // 8  # 256

    # 记录resnet50提供给fpn的特征层channels [256, 512, 1024, 2048]
    in_channels_list = [in_channels_stage2 * 2 ** (num-1) for num in returned_layers]

    # 通过fpn后得到的每个特征层的channel
    out_channels = 256

    return BackboneWithFPN(resnet_backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks)


if __name__ == "__main__":
    resnet50 = ResNet(BottleNeck, [3, 4, 6, 3], 10)
    # for name, module in resnet50.named_children():
    #     print(name, module)
    return_layers = {'layer2': '1', 'layer3': '2', 'layer4': '3'}
    # ilg = IntermediateLayerGetter(resnet50, return_layers)
    # input_data = torch.rand(4, 3, 640, 640)
    # out = ilg(input_data)
    # for name, i in out.items():
    #     print(name, i.shape)

    lg = LayerGetter(resnet50, return_layers)
    input_data = torch.rand(4, 3, 640, 640)
    out = lg(input_data)
    c3_channel, c4_channel, c5_channel = '', '', ''
    fpn_input = list()

    # input_data = torch.rand(4, 3, 12, 12)
    # up_sample_layer = nn.Upsample(scale_factor=2, mode='nearest')
    # out_put = up_sample_layer(input_data)
    # print(out_put.shape)
    # res = resnet50_fpn_backbone()
    # print(res)

    arr = [1, 2, 3]
    a, b, c = arr
    print(a, b, c)
