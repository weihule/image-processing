import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Basic convolutional block
    convolution (bias discarded) + batch normalization + relu6

        in_c: number of input channels
        out_c: number of output channels
        k: kernel size
        s: stride
        p: padding
        g: number of blocked connections from input channels to output channels(default: 1)
    """
    def __init__(self, in_c, out_c, k, s, p, g):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return F.relu6(self.bn(self.conv(x)))


class BottleNeck(nn.Module):
    """
    1*1 升维
    3*3 DW
    1*1 降维
    """
    def __init__(self, in_channels, out_channels, expansion_factor, stride):
        super(BottleNeck, self).__init__()
        mid_channels = in_channels * expansion_factor
        self.use_residual = stride == 1 and in_channels == out_channels


