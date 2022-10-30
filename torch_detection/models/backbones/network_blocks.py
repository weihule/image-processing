import torch
import torch.nn as nn


__all__ = [
    'SiLU',
    'get_activation',
    'BaseConv',
    'DWConv',
    'Bottleneck',
    'ResLayer',
    'SPPBottleneck',
    'CSPLayer',
    'Focus'
]


class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act='silu'):
        super(BaseConv, self).__init__()
        padding = (ksize - 1) // 2
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=ksize,
                              stride=stride,
                              padding=padding,
                              groups=groups,
                              bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(name=act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class DWConv(nn.Module):
    """
    Depthwise Conv + Conv
    """
    def __init__(self, in_channels, out_channels, ksize, stride=1, act='silu'):
        super(DWConv, self).__init__()
        self.dconv = BaseConv(in_channels=in_channels,
                              out_channels=in_channels,
                              ksize=ksize,
                              groups=in_channels,
                              stride=stride,
                              act=act)
        self.pconv = BaseConv(in_channels=in_channels,
                              out_channels=out_channels,
                              ksize=1,
                              groups=1,
                              stride=1,
                              act=act)

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


class Bottleneck(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 shortcut=True,
                 expansion=0.5,
                 depthwise=False,
                 act="silu", ):
        super(Bottleneck, self).__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels=in_channels,
                              out_channels=hidden_channels,
                              ksize=1,
                              groups=1,
                              stride=1,
                              act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y += x

        return y


class ResLayer(nn.Module):
    """
    Residual layer with 'in_channels' inputs
    """
    def __init__(self, in_channels):
        super(ResLayer, self).__init__()
        mid_channels = in_channels // 2
        self.layer1 = BaseConv(in_channels=in_channels,
                               out_channels=mid_channels,
                               ksize=1,
                               groups=1,
                               stride=1,
                               act='lrelu')
        self.layer2 = BaseConv(in_channels=mid_channels,
                               out_channels=in_channels,
                               ksize=3,
                               groups=1,
                               stride=1,
                               act='lrelu')

    def forward(self, x):
        out = self.layer2(self.layer1(x))
        return x + out


class SPPBottleneck(nn.Module):
    """
    Spatial pyramid pooling layer used in YOLOv3-SPP
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(5, 9, 13),
                 activation='silu'):
        super(SPPBottleneck, self).__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels=in_channels,
                              out_channels=hidden_channels,
                              ksize=1,
                              groups=1,
                              stride=1,
                              act=activation)
        self.m = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_size]
        )
        conv2_channels = hidden_channels * (len(kernel_size) + 1)
        self.conv2 = BaseConv(in_channels=conv2_channels,
                              out_channels=out_channels,
                              ksize=1,
                              groups=1,
                              stride=1,
                              act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)

        return x


class CSPLayer(nn.Module):
    """
    C3 in yolov5, CSP Bottleneck with 3 convolutions
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 n=1,
                 shortcut=True,
                 expansion=0.5,
                 depthwise=False,
                 act='silu'):
        super(CSPLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [Bottleneck(hidden_channels,
                                  hidden_channels,
                                  shortcut,
                                  1.0,
                                  depthwise,
                                  act=act) for _ in range(n)]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat([x_1, x_2], dim=1)    # [b, hidden_channels+hidden_channels, h, w]
        return self.conv3(x)                # [b, out_channels, h, w]


class Focus(nn.Module):
    """
    focus width and height information into channel space
    """
    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act='silu'):
        super(Focus, self).__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride=stride, act=act)

    def forward(self, x):
        """
        Args:
            x: [b, c, h, w]
        Returns:
        """
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat([patch_top_left, patch_bot_left, patch_top_right, patch_bot_right], dim=1)
        # print('====', x.shape)

        return self.conv(x)


if __name__ == "__main__":
    focus = Focus(3, 32, ksize=3, stride=1)
    dwconv = DWConv(3, 32, ksize=3, stride=1)
    inputs = torch.randn(4, 3, 640, 640)
    outs = focus(inputs)
    outs2 = dwconv(inputs)
    print(outs.shape)
    print(outs2.shape)
