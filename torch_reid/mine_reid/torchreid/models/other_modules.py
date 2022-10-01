import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'HorizontalPooling',
    'FRelu',
    'activate_function'
]


class HorizontalPooling(nn.Module):
    def __init__(self):
        super(HorizontalPooling, self).__init__()

    def forward(self, x):
        x_width = x.shape[3]

        return F.max_pool2d(x, kernel_size=(1, x_width))


class FRelu(nn.Module):
    def __init__(self, in_channels):
        super(FRelu, self).__init__()
        self.conv_frelu = nn.Conv2d(in_channels, in_channels,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=in_channels)
        self.bn_frelu = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        x1 = self.conv_frelu(x)
        x1 = self.bn_frelu(x1)
        x = torch.maximum(x, x1)

        return x


def activate_function(act_name, channels=None):
    if act_name == 'relu':
        return nn.ReLU(inplace=True)
    elif act_name == 'frelu':
        return FRelu(channels)
    else:
        raise KeyError(f'Unknown activate function {act_name}')


if __name__ == "__main__":
    arr1 = torch.randn(3, 4)
    arr2 = torch.randn(3, 4)
    print(arr1)
    print(arr2)
    arr = torch.maximum(arr1, arr2)
    print(arr, arr.shape)


