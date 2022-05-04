import os
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn

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
                nn.BatchNorm2d(input_c),
                nn.ReLU(inplace=True)
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=input_c if self.stride > 1 else  branch_features, 
                        out_channels=branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            self.depthwise_conv(branch_features, branch_features, kernel_s=3, stride=self.stride),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True)
        )

    @staticmethod
    def depthwise_conv(input_c:int, output_c:int, 
                        kernel_s:int, stride: int=1,
                        padding: int = 0, bias: bool=False,
                        ) -> nn.Conv2d:
        return nn.Conv2d(input_c, output_c, kernel_size=kernel_s, 
                    stride=stride, padding=padding, bias=bias, groups=input_c)

    def forward(self, x: Tensor) -> Tensor:
        if self.stride == 1:
            x1, x2 = x.chunk(chunks=2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out)

class ShuffleNetV2(nn.Module):
    def __init__(self):
        super(ShuffleNetV2, self).__init__()

if __name__ == "__main__":
    # arr = np.arange(24).reshape((2, 3, 4))
    # print(arr, arr.shape)
    # np_transpose = arr.transpose((1, 0, 2))
    # np_transpose = np.transpose(arr, (1, 0, 2))
    # print(np_transpose, np_transpose.shape)

    # tensor_arr = torch.tensor(arr)
    # print(tensor_arr, tensor_arr.shape)
    # torch_transpose = tensor_arr.transpose(0, 1)
    # torch_transpose = torch.transpose(tensor_arr, 0, 1)
    # print(torch_transpose, torch_transpose.shape)

    # arr = torch.tensor(np.arange(24).reshape((6, 2, 2)))   # b,c,h,w -> b, 3, 2, h, w
    # print(arr)
    # x = arr.view(3, 2, 2, 2)
    # x = torch.transpose(x, 1, 2).contiguous()
    # x = x.view(-1, 2, 2)
    # x1, x2 = torch.chunk(arr, chunks=2, dim=0)
    # x1, x2, x3 = arr.chunk(chunks=4, dim=0)
    # print(x1, x2, x3)

    arr1 = torch.randn((2, 3))
    arr2 = torch.randn((2, 3))
    print(arr1)
    print(arr2)
    # arr = torch.cat((arr1, arr2), dim=0)
    arr = torch.stack((arr1, arr2), dim=0)
    print(arr, arr.shape)