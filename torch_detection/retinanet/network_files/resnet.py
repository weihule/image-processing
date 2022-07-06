import os
import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, use_se=False, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel,
                               kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channel, out_channel,
                               kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        if self.use_se:
            out = self.se(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        # # default : width = out_channel
        # width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel * self.expansion,
                               kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.in_channel,
                               kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0], stride=1)
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        """ 
       对于layer2, 3, 4来说, stride == 2时, 即第一层需要进行下采样
        对于layer1来说, 所有层都不需要进行下采样, 但是其第一层
        需要进行卷积操作, stride == 1, 但是self.in_channel == 64, 
        但是 channel * block.expansion == 256, 这两者不相等，但是
        仍然有shortcut连接, 所以仍然需要右边进行一个卷积操作来调整channel

        另外, 这里的stride是为了layer的第一层设置的, 传到了block里
        其余层的stride都是固定的, 已经在block中设置了
        """
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion)
            )
        layers = []

        # 先把第一层单独添加进去
        layers.append(block(self.in_channel,
                            channel,
                            stride=stride,
                            downsample=downsample,
                            groups=self.groups,
                            width_per_group=self.width_per_group,
                            )
                      )

        # 每个layer的第一层之后, self.in_channel就变成channel的倍数了
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group

                                ))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        c3 = self.layer2(x)
        c4 = self.layer3(x)
        c5 = self.layer4(x)

        del x

        return [c3, c4, c5]


def load_state_dict(model, pre_trained_path, excluded_layer_name):
    if not os.path.exists(pre_trained_path):
        print('No pretrained model file!')
        return
    state_dict = torch.load(pre_trained_path, map_location=torch.device('cpu'))
    filtered_state_dict = {
        name: weight
        for name, weight in state_dict.items()
        if name in model.state_dict() and not any(excluded_name in name for excluded_name in excluded_layer_name)
        and weight.shape == model.state_dict()[name].shape
    }
    if len(filtered_state_dict) == 0:
        print('No pretrained parameters to load!')
    else:
        model.load_state_dict(filtered_state_dict, strict=False)

    return


def _resnet_backbone(block, layers, in_channels, pre_train_path):
    model = ResNet(block, layers)
    if pre_train_path:
        load_state_dict(model, pre_train_path, '')
    else:
        print('no backbone pretrained model!')

    return model
