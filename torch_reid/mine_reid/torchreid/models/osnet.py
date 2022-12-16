from __future__ import print_function, absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F

from other_modules import activate_function, HorizontalPooling
from other_modules import attention_module

__all__ = [
    'osnet_x1_0',
    'osnet_x0_75'
]


class ConvLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 groups=1,
                 IN=False):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              padding=padding,
                              stride=stride,
                              groups=groups,
                              bias=False)
        if IN:
            self.bn = nn.InstanceNorm2d(out_channels, affine=True)
        else:
            self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class Conv1x1(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 groups=1,
                 act_func='relu'):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act_func = activate_function(act_func, out_channels)
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act_func(x)
        return x


class Conv1x1Linear(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Conv1x1Linear, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)

        return x


class Conv3x3(nn.Module):
    """
    3x3 convolution + bn + relu
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 groups=1,
                 act_func='relu'):
        super(Conv3x3, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act_func = activate_function(act_func, out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act_func(x)

        return x


class LightConv3x3(nn.Module):
    """
    Lightweight 3x3 convolution

    1x1 (linear) + dw 3x3 (nolinear)
    """

    def __init__(self, in_channels, out_channels, act_func=None):
        super(LightConv3x3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False,
                               groups=out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act_func = activate_function(act_name=act_func,
                                          channels=out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.act_func(x)

        return x


class ChannelGate(nn.Module):
    """
    A mini-network that generates channel-wise gates conditioned on input tensor
    """

    def __init__(self,
                 in_channels,
                 num_gates=None,
                 return_gates=False,
                 gate_activation='sigmoid',
                 reduction=16,
                 layer_norm=False,
                 act_func='relu'):
        super(ChannelGate, self).__init__()
        if num_gates is None:
            num_gates = in_channels
        self.return_gates = return_gates
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 降维
        self.fc1 = nn.Conv2d(in_channels,
                             in_channels // reduction,
                             kernel_size=1,
                             bias=True,
                             padding=0)
        self.norm1 = None
        if layer_norm:
            self.norm1 = nn.LayerNorm([in_channels // reduction, 1, 1])
        # self.relu = nn.ReLU(inplace=True)
        self.act_func = activate_function(act_func, in_channels // reduction)

        self.fc2 = nn.Conv2d(in_channels // reduction,
                             num_gates,
                             kernel_size=1,
                             bias=True,
                             padding=0)
        if gate_activation == 'sigmoid':
            self.gate_activation = nn.Sigmoid()
        elif gate_activation == 'relu':
            self.gate_activation = activate_function(act_name='relu',
                                                     channels=num_gates)
        elif gate_activation == 'linear':
            self.gate_activation = None
        elif gate_activation == 'frelu':
            self.gate_activation = activate_function(act_name='frelu',
                                                     channels=num_gates)
        else:
            raise RuntimeError(f'Unknown gate activation {gate_activation}')

    def forward(self, x):
        inputs = x
        x = self.global_avgpool(x)
        x = self.fc1(x)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.act_func(x)
        x = self.fc2(x)
        if self.gate_activation is not None:
            x = self.gate_activation(x)
        if self.return_gates:
            return x
        return inputs * x


class OSBlock(nn.Module):
    """
    Omni-scale feature learning block
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 act_func='relu',
                 IN=False,
                 bottleneck_reduction=4,
                 attention=None,
                 **kwargs):
        super(OSBlock, self).__init__()
        mid_channels = out_channels // bottleneck_reduction
        self.conv1 = Conv1x1(in_channels, mid_channels, act_func='relu')
        self.conv2a = LightConv3x3(mid_channels, mid_channels, act_func='relu')
        self.conv2b = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels, act_func='relu'),
            LightConv3x3(mid_channels, mid_channels, act_func='relu')
        )
        self.conv2c = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels, act_func='relu'),
            LightConv3x3(mid_channels, mid_channels, act_func='relu'),
            LightConv3x3(mid_channels, mid_channels, act_func='relu')
        )
        self.conv2d = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels, act_func='relu'),
            LightConv3x3(mid_channels, mid_channels, act_func='relu'),
            LightConv3x3(mid_channels, mid_channels, act_func='relu'),
            LightConv3x3(mid_channels, mid_channels, act_func='relu')
        )
        self.gate = ChannelGate(mid_channels, act_func='relu')
        self.conv3 = Conv1x1Linear(mid_channels, out_channels)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)
        self.IN = None
        if IN:
            self.IN = nn.InstanceNorm2d(out_channels, affine=True)
        self.act_fun = activate_function(act_func, out_channels)
        if attention:
            self.attention = attention_module(attention, channel=out_channels)
        else:
            self.attention = None

    def forward(self, x):
        identity = x.clone()
        x1 = self.conv1(x)
        x2a = self.conv2a(x1)
        x2b = self.conv2b(x1)
        x2c = self.conv2c(x1)
        x2d = self.conv2d(x1)
        x2 = self.gate(x2a) + self.gate(x2b) + self.gate(x2c) + self.gate(x2d)
        x3 = self.conv3(x2)
        if self.attention is not None:
            x3 = self.attention(x3)

        if self.downsample is not None:
            identity = self.downsample(identity)
        out = x3 + identity
        if self.IN is not None:
            out = self.IN(out)
        out = self.act_fun(out)

        return out


class OSNet(nn.Module):
    def __init__(self,
                 num_classes,
                 blocks,
                 layers,
                 channels,
                 feature_dim=512,
                 act_func='relu',
                 attention=None,
                 loss='softmax',
                 IN=False,
                 aligned=False,
                 **kwargs):
        super(OSNet, self).__init__()
        num_blocks = len(blocks)
        # blocks [OSBlock, OSBlock, OSBlock]
        # layers [2, 2, 2]
        assert num_blocks == len(layers)
        # channels [64, 256, 384, 512]
        assert num_blocks == len(channels) - 1
        self.loss = loss
        self.feature_dim = feature_dim
        self.aligned = aligned
        self.horizontal_pool = HorizontalPooling()

        # convolutional backbone
        self.conv1 = ConvLayer(3,
                               channels[0],
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               IN=IN)
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        # [conv2, transition]
        self.conv2 = self._make_layers(
            blocks[0],
            layers[0],
            channels[0],
            channels[1],
            reduce_spatial_size=True,
            act_func=act_func,
            attention=attention,
            IN=IN
        )
        # [conv2, transition]
        self.conv3 = self._make_layers(
            blocks[1],
            layers[1],
            channels[1],
            channels[2],
            reduce_spatial_size=True,
            act_func=act_func,
            attention=attention,
        )
        # conv4
        self.conv4 = self._make_layers(
            blocks[2],
            layers[2],
            channels[2],
            channels[3],
            reduce_spatial_size=False,
            act_func=act_func,
            attention=attention
        )
        self.conv5 = Conv1x1(channels[3], channels[3])
        if attention:
            self.conv5_atten = attention_module(attention, channel=channels[3])
        else:
            self.conv5_atten = None
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # fully connected layer
        self.fc = self._construct_fc_layer(fc_dims=self.feature_dim,
                                           input_dim=channels[3],
                                           dropout_p=None)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

        if self.aligned:
            self.aligned_bn = nn.BatchNorm2d(self.feature_dim)
            self.aligned_relu = nn.ReLU(inplace=True)
            self.aligned_conv1 = nn.Conv2d(self.feature_dim, 128, kernel_size=1, stride=1, padding=0, bias=True)

        self._init_params()

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        if self.conv5_atten:
            x = self.conv5_atten(x)
        return x

    def forward(self, x):
        x = self.featuremaps(x)  # [b, 512, 16, 8]

        if not self.training:
            lf = self.horizontal_pool(x)  # [b, 512, 16, 1]

        if self.aligned and self.training:
            print('---', x.shape)
            lf = self.aligned_bn(x)
            lf = self.aligned_relu(lf)
            lf = self.horizontal_pool(lf)  # [b, 512, 16, 1]
            lf = self.aligned_conv1(lf)  # [b, 128, 16, 1]

        if self.aligned or not self.training:
            lf = lf.view(lf.shape[:3])  # [b, 128, 16]
            lf = lf / torch.pow(lf, 2).sum(dim=1, keepdim=True).clamp(min=1e-12).sqrt()

        f = F.avg_pool2d(x, x.shape[2:])  # [b, 512, 1, 1]
        f = f.reshape(f.shape[0], -1)  # [b, 512]
        if self.fc is not None:
            f = self.fc(f)

        # 测试阶段使用 f 和 lf
        if not self.training:
            return f, lf
        y = self.classifier(f)  # [b, num_classes]

        # grad cam 可视化
        if not self.training and self.loss == 'grad_cam':
            return y

        if self.loss not in ['softmax', 'softmax_cent', 'softmax_trip', 'softmax_trip_cent']:
            raise KeyError(f'Unsupported {self.loss} loss type')

        # 训练阶段并且使用aligned
        if self.aligned and self.training:
            return y, f, lf
        else:
            print('执行这里')
            return y, f

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        """

        Args:
            fc_dims: 512
            input_dim: channels[-1]
            dropout_p: (float)

        Returns:

        """
        if fc_dims is None or fc_dims < 0:
            self.feature_dim = input_dim
            return None

        if isinstance(fc_dims, int):
            fc_dims = [fc_dims]

        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim

        self.feature_dim = fc_dims[-1]

        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
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

    def _make_layers(self,
                     block,
                     layer,
                     in_channels,
                     out_channels,
                     reduce_spatial_size,
                     act_func='relu',
                     attention=None,
                     IN=False):
        """
        layer 是 block 的重复次数
        """
        layers = []
        layers.append(block(in_channels, out_channels, act_func=act_func, attention=attention, IN=IN))
        for i in range(1, layer):
            layers.append(block(out_channels, out_channels, act_func=act_func, attention=attention, IN=IN))

        # 宽高减半
        if reduce_spatial_size:
            layers.append(
                nn.Sequential(
                    Conv1x1(out_channels, out_channels),
                    nn.AvgPool2d(2, stride=2)
                )
            )

        return nn.Sequential(*layers)


def osnet_x1_0(num_classes=1000, act_func='relu', loss='softmax', aligned=False, **kwargs):
    model = OSNet(num_classes,
                  blocks=[OSBlock, OSBlock, OSBlock],
                  layers=[2, 2, 2],
                  channels=[64, 256, 384, 512],
                  feature_dim=512,
                  act_func=act_func,
                  loss=loss,
                  aligned=aligned,
                  **kwargs)

    return model


def osnet_x0_75(num_classes=1000, act_func='relu', loss='softmax', aligned=False, **kwargs):
    model = OSNet(num_classes,
                  blocks=[OSBlock, OSBlock, OSBlock],
                  layers=[2, 2, 2],
                  channels=[48, 192, 288, 384],
                  feature_dim=512,
                  act_func=act_func,
                  loss=loss,
                  aligned=aligned,
                  **kwargs)

    return model


if __name__ == "__main__":
    arr = torch.randn(4, 3, 256, 128)
    flag_align = True
    model = osnet_x0_75(num_classes=751, act_func='relu', loss='softmax_trip', aligned=flag_align)

    if flag_align:
        outs, local_features = model(arr)
    else:
        outs, features, local_feature = model(arr)
