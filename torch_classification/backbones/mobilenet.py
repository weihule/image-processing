import os
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'mobilenetv2_x1_0',
    'mobilenetv2_x1_4'
]

# download url: https://download.pytorch.org/models/mobilenet_v2-b0353104.pth


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

    def __init__(self, in_c, out_c, k, s=1, p=0, g=1):
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

    def __init__(self, in_channels, out_channels, expansion_factor, stride=1):
        super(BottleNeck, self).__init__()
        mid_channels = in_channels * expansion_factor
        self.use_residual = stride == 1 and in_channels == out_channels
        self.conv1 = ConvBlock(in_c=in_channels,
                               out_c=mid_channels,
                               k=1, s=1, p=0, g=1)
        self.dwconv2 = ConvBlock(in_c=mid_channels,
                                 out_c=mid_channels,
                                 k=3, s=stride, p=1, g=mid_channels)
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1,
                      stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        m = self.conv1(x)
        m = self.dwconv2(m)
        m = self.conv3(m)
        if self.use_residual:
            return m + x
        else:
            return m


class MobileNetV2(nn.Module):
    def __init__(self,
                 num_classes,
                 width_mult=1.,
                 fc_dims=None,
                 dropout_p=None,
                 **kwargs):
        super(MobileNetV2, self).__init__()
        self.in_channels = int(32 * width_mult)
        self.feature_dim = int(1280 * width_mult) if width_mult > 1 else 1280

        # construct layers
        self.conv1 = ConvBlock(3, self.in_channels, 3, s=2, p=1)
        self.conv2 = self._make_layer(BottleNeck, 1, int(16 * width_mult), 1, 1)
        self.conv3 = self._make_layer(BottleNeck, 6, int(24 * width_mult), 2, 2)
        self.conv4 = self._make_layer(BottleNeck, 6, int(32 * width_mult), 3, 2)
        self.conv5 = self._make_layer(BottleNeck, 6, int(64 * width_mult), 4, 2)
        self.conv6 = self._make_layer(BottleNeck, 6, int(96 * width_mult), 3, 1)
        self.conv7 = self._make_layer(BottleNeck, 6, int(160 * width_mult), 3, 2)
        self.conv8 = self._make_layer(BottleNeck, 6, int(320 * width_mult), 1, 1)
        self.conv9 = nn.Conv2d(self.in_channels, self.feature_dim, kernel_size=1)

        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, num_classes)
        )
        self._init_params()

    def _make_layer(self, block, t, c, n, s):
        """
        Args:
            block: BottleNeck
            t: expansion factor
            c: output channel
            n: number of blocks
            s: stride of first layer
        Returns:
        """
        layers = []
        layers.append(block(in_channels=self.in_channels,
                            out_channels=c,
                            expansion_factor=t,
                            stride=s))
        self.in_channels = c
        for _ in range(1, n):
            layers.append(block(in_channels=self.in_channels,
                                out_channels=c,
                                expansion_factor=t,
                                stride=1))

        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        v = self.global_avgpool(x)
        v = v.view(v.shape[0], -1)

        y = self.classifier(v)

        return y
    

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3,
                 stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__()


def mobilenetv2_x1_0(num_classes, **kwargs):
    model = MobileNetV2(num_classes=num_classes,
                        width_mult=1,
                        **kwargs)

    return model


def mobilenetv2_x1_4(num_classes, **kwargs):
    model = MobileNetV2(
        num_classes,
        width_mult=1.4,
        **kwargs
    )

    return model


if __name__ == "__main__":
    from thop import profile
    from thop import clever_format
    inputs = torch.randn(4, 3, 224, 224)
    net = mobilenetv2_x1_0(num_classes=5)
    macs, params = profile(net, (inputs, ), verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print('macs: {}, params: {}'.format(macs, params))

    outputs = net(inputs)
    print('outputs.shape: {}'.format(outputs.shape))
