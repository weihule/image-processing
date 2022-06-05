import os
from pyparsing import Opt
import torch
import torch.nn as nn
from torch import Tensor
from torchvision import transforms, datasets
from custom_dataset import CustomDataset
from typing import Optional, Callable
from collections import OrderedDict


def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


class ConvBNActivation(nn.Sequential):
    def __init__(self, 
                    in_planes: int,
                    out_planes: int,
                    kernel_size: int=3,
                    stride: int=1,
                    groups: int=1,
                    norm_layer: Optional[Callable[..., nn.Module]]=None,
                    activation_layer: Optional[Callable[..., nn.Module]]=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.SiLU

        super(ConvBNActivation, self).__init__(
            nn.Conv2d(in_channels=in_planes,
                        out_channels=out_planes,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        groups=groups,
                        bias=False),
            norm_layer(out_planes),
            activation_layer()
        )



class SqueezeExcitation(nn.Module):
    def __init__(self, input_c, output_c, squeeze_factor=4):
        super(SqueezeExcitation, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(input_c, input_c//squeeze_factor),
            nn.SiLU(),
            nn.Linear(input_c//squeeze_factor, output_c),
            nn.Sigmoid()
        )
    
    def forward(self, x: Tensor):
        b, c, _, _ = x.shape
        y = self.avg_pool(x)    # [B, C, 1, 1]
        y = torch.flatten(y, start_dim=1)   # [B, C*1*1]
        y = self.fc(y).view(b, c, 1, 1)

        return  x * y.expand_as(x)

if __name__ == "__main__":
    res = _make_divisible(16.8)
    print(res)
    print(torch.__version__)