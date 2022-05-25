import os
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.functional import adaptive_avg_pool2d
from typing import Callable, List


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)     # 自适应池化之后并展平
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batch_size, num_channels, height, width = x.size()
    channel_per_group = num_channels // groups  # 向下取整, 每个组所对应的channel个数

    # reshape
    # batch_size, num_channnels, height, width -> batch_size, groups, chennel_per_group, height, width
    x = x.view(batch_size, groups, channel_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batch_size, -1, height, width)

    return x


class InvertedResidual(nn.Module):
    def __init__(self, input_c: int, output_c: int, stride: int):
        super(InvertedResidual, self).__init__()
        if stride not in [1, 2]:
            raise ValueError("illegal stride value.")
        self.stride = stride

        assert output_c % 2 == 0
        branch_features = output_c // 2
        # 当stride为1时，input_c应该是branch_features的两倍
        # '<<'是位运算，可以理解为x2的快速方法
        # 这里的逻辑是：当stride==1时，判断 input_c 是否等于 branch_features * 2
        assert (self.stride != 1) or (input_c == branch_features << 1)

        if self.stride == 2:
            # 没有进行 channel_split 的左分支
            self.branch1 = nn.Sequential(
                self.depthwise_conv(input_c, input_c, kernel_s=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(input_c),
                nn.Conv2d(input_c, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True)
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=input_c if self.stride > 1 else branch_features,
                        out_channels=branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_s=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            SELayer(branch_features)
        )

    @staticmethod
    def depthwise_conv(input_c: int, output_c: int,
                        kernel_s: int, stride: int = 1,
                        padding: int = 0, bias: bool=False,
                        ) -> nn.Conv2d:
        return nn.Conv2d(input_c, output_c, kernel_size=kernel_s, 
                    stride=stride, padding=padding, bias=bias, groups=input_c)

    def forward(self, x: Tensor) -> Tensor:
        if self.stride == 1:
            x1, x2 = x.chunk(chunks=2, dim=1)   # B C H W, 沿channel维度分成两块
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)
        return out


class ShuffleNetV2(nn.Module):
    def __init__(self,
                stages_repeats: List[int],
                stages_out_channels: List[int],
                num_classes: int=1000,
                inverted_residual: Callable[..., nn.Module] = InvertedResidual):
        super(ShuffleNetV2, self).__init__()

        if len(stages_repeats) != 3:
            raise ValueError("expected stages_repeats as list of 3 positive ints")
        if len(stages_out_channels) != 5:
            raise ValueError("expected stages_out_channels as list of 5 positive ints")
        self._stage_out_channels = stages_out_channels

        # input RGB image
        input_channels = 3
        output_channels = self._stage_out_channels[0]

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        input_channels = output_channels    # 3 -> 24
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage2: nn.Sequential
        self.stage3: nn.Sequential
        self.stage4: nn.Sequential

        stage_names = ["stage{}".format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [inverted_residual(input_channels, output_channels, 2)]
            for _ in range(repeats-1):
                seq.append(inverted_residual(output_channels, output_channels, 1))
            
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

            # # 上一行代码就相当于
            # self.stage2 = nn.Sequential(*seq)
            # self.stage3 = nn.Sequential(*seq)
            # self.stage4 = nn.Sequential(*seq)

        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        
        self.fc = nn.Linear(output_channels, num_classes)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)   # B, C, 7, 7

        x = adaptive_avg_pool2d(x, (1, 1))  # B, C, 1, 1
        x = x.flatten(start_dim=1)  # B, C
        x = self.fc(x)

        return x
    
    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def shufflenet_v2_x1_0(num_classes=1000):
    model = ShuffleNetV2(stages_repeats=[4, 8, 4],
                        stages_out_channels=[24, 116, 232, 464, 1024],
                        num_classes=num_classes)

    return model


def shufflenet_v2_x0_5(num_classes=1000):
    model = ShuffleNetV2(stages_repeats=[4, 8, 4],
                        stages_out_channels=[24, 48, 96, 192, 1024],
                        num_classes=num_classes)

    return model


if __name__ == "__main__":
    # tensor_arr = torch.tensor(arr)
    # print(tensor_arr, tensor_arr.shape)
    # torch_transpose = tensor_arr.transpose(0, 1)
    # torch_transpose = torch.transpose(tensor_arr, 0, 1)
    # print(torch_transpose, torch_transpose.shape)

    # model = shufflenet_v2_x1_0(num_classes=5)
    # print(model)

    arr = torch.randint(1, 10, size=(4, 3, 2, 2))
    b, c, _, _ = arr.size()

    layer = nn.AdaptiveAvgPool2d(1)
    arr_view = layer(arr)
    arr_view = arr.view(b, c)
    arr_flatten = torch.flatten(arr, start_dim=1)
    print(arr_view.shape, arr_flatten.shape)


