from __future__ import print_function, absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F

from .other_modules import activate_function, HorizontalPooling
from .other_modules import attention_module
from .odconv import ODConv2d

__all__ = [
    'odosnet_x1_0',
    'odosnet_x0_75',
    'odosnet_x0_5'
]


class ConvLayer(nn.Module):
    """Convolution layer (conv + bn + relu)."""

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            groups=1,
            IN=False
    ):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            groups=groups
        )
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


# basic layers
class ODConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, groups=1, IN=False):
        super(ODConvLayer, self).__init__()
        self.odconv = ODConv2d(in_planes=in_channels,
                               out_planes=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               dilation=1,
                               groups=groups,
                               reduction=0.0625,
                               kernel_num=4)
        if IN:
            self.bn = nn.InstanceNorm2d(out_channels, affine=True)
        else:
            self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.odconv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class Conv1x1(nn.Module):
    """1x1 convolution + bn + relu."""

    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            1,
            stride=stride,
            padding=0,
            bias=False,
            groups=groups
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ODConv1x1(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(ODConv1x1, self).__init__()
        self.odconv = ODConv2d(in_planes=in_channels,
                               out_planes=out_channels,
                               kernel_size=1,
                               stride=stride,
                               padding=0,
                               dilation=1,
                               groups=groups,
                               reduction=0.0625,
                               kernel_num=4)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.odconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1x1Linear(nn.Module):
    """1x1 convolution + bn (w/o non-linearity)."""

    def __init__(self, in_channels, out_channels, stride=1):
        super(Conv1x1Linear, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1, stride=stride, padding=0, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class ODConv1x1Linear(nn.Module):
    """1x1 convolution + bn (w/o non-linearity)."""

    def __init__(self, in_channels, out_channels, stride=1):
        super(ODConv1x1Linear, self).__init__()
        self.odconv = ODConv2d(in_planes=in_channels,
                               out_planes=out_channels,
                               kernel_size=1,
                               stride=stride,
                               padding=0,
                               dilation=1,
                               groups=1,
                               reduction=0.0625,
                               kernel_num=4)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.odconv(x)
        x = self.bn(x)
        return x


class ODLightConv3x3(nn.Module):
    """
    Light wight 3x3 convolution
    1x1 (linear) + dw 3x3 (no_linear)
    """

    def __init__(self, in_channels, out_channels):
        super(ODLightConv3x3, self).__init__()
        self.conv1 = ODConv2d(in_planes=in_channels,
                              out_planes=out_channels,
                              kernel_size=1,
                              stride=1,
                              padding=0,
                              dilation=1,
                              groups=1,
                              reduction=0.0625,
                              kernel_num=4)
        self.conv2 = ODConv2d(in_planes=out_channels,
                              out_planes=out_channels,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              dilation=1,
                              groups=out_channels,
                              reduction=0.0625,
                              kernel_num=4)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.relu(self.bn(x))

        return x


# build blocks for omni-scale feature learning
class ChannelGate(nn.Module):
    """
    A mini-network that generates channel-wise gates conditions on input tensor
    """

    def __init__(self,
                 in_channels,
                 num_gates=None,
                 return_gates=False,
                 gate_activation='sigmoid',
                 reduction=16,
                 layer_num=False):
        super(ChannelGate, self).__init__()
        if num_gates is None:
            num_gates = in_channels
        self.return_gates = return_gates
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        # self.fc1 = nn.Conv2d(in_channels,
        #                      in_channels // reduction,
        #                      kernel_size=1,
        #                      bias=True,
        #                      padding=0)
        self.fc1 = ODConv2d(in_planes=in_channels,
                            out_planes=in_channels // reduction,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            dilation=1,
                            groups=1,
                            reduction=0.0625,
                            kernel_num=4)
        self.norm1 = None
        if layer_num:
            self.norm1 = nn.LayerNorm([in_channels // reduction, 1, 1])
        self.relu = nn.ReLU(inplace=True)
        # self.fc2 = nn.Conv2d(in_channels // reduction,
        #                      num_gates,
        #                      kernel_size=1,
        #                      bias=True,
        #                      padding=0)
        self.fc2 = ODConv2d(in_planes=in_channels // reduction,
                            out_planes=num_gates,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            dilation=1,
                            groups=1,
                            reduction=0.0625,
                            kernel_num=4)
        if gate_activation == 'sigmoid':
            self.gate_activation = nn.Sigmoid()
        elif gate_activation == 'relu':
            self.gate_activation = nn.ReLU(inplace=True)
        elif gate_activation == 'linear':
            self.gate_activation = None
        else:
            raise RuntimeError("Unknown gate activation: {}".format(gate_activation))

    def forward(self, x):
        inputs = x
        x = self.global_avgpool(x)
        x = self.fc1(x)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if self.gate_activation is not None:
            x = self.gate_activation(x)
        if self.return_gates:
            return x
        return inputs * x


class ODOSBlock(nn.Module):
    """Omni-scale feature learning block."""

    def __init__(
            self,
            in_channels,
            out_channels,
            IN=False,
            bottleneck_reduction=4,
            **kwargs
    ):
        super(ODOSBlock, self).__init__()
        mid_channels = out_channels // bottleneck_reduction
        self.conv1 = ODConv1x1(in_channels, mid_channels)
        self.conv2a = ODLightConv3x3(mid_channels, mid_channels)
        self.conv2b = nn.Sequential(
            ODLightConv3x3(mid_channels, mid_channels),
            ODLightConv3x3(mid_channels, mid_channels),
        )
        self.conv2c = nn.Sequential(
            ODLightConv3x3(mid_channels, mid_channels),
            ODLightConv3x3(mid_channels, mid_channels),
            ODLightConv3x3(mid_channels, mid_channels),
        )
        self.conv2d = nn.Sequential(
            ODLightConv3x3(mid_channels, mid_channels),
            ODLightConv3x3(mid_channels, mid_channels),
            ODLightConv3x3(mid_channels, mid_channels),
            ODLightConv3x3(mid_channels, mid_channels),
        )
        # self.gate = ChannelGate(mid_channels)
        self.conv3 = ODConv1x1Linear(mid_channels, out_channels)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = ODConv1x1Linear(in_channels, out_channels)
        self.IN = None
        if IN:
            self.IN = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2a = self.conv2a(x1)
        x2b = self.conv2b(x1)
        x2c = self.conv2c(x1)
        x2d = self.conv2d(x1)
        # x2 = self.gate(x2a) + self.gate(x2b) + self.gate(x2c) + self.gate(x2d)
        x2 = x2a + x2b + x2c + x2d
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = x3 + identity
        if self.IN is not None:
            out = self.IN(out)
        return F.relu(out)


class ODOSNet(nn.Module):
    """Omni-Scale Network.

    Reference:
        - Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ICCV, 2019.
        - Zhou et al. Learning Generalisable Omni-Scale Representations
          for Person Re-Identification. TPAMI, 2021.
    """

    def __init__(
            self,
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
            **kwargs
    ):
        super(ODOSNet, self).__init__()
        num_blocks = len(blocks)
        assert num_blocks == len(layers)
        assert num_blocks == len(channels) - 1
        self.loss = loss
        self.feature_dim = feature_dim

        self.aligned = aligned
        self.horizontal_pool = HorizontalPooling()

        # convolutional backbone
        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3, IN=IN)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._make_layer(
            blocks[0],
            layers[0],
            channels[0],
            channels[1],
            reduce_spatial_size=True,
            IN=IN
        )
        self.conv3 = self._make_layer(
            blocks[1],
            layers[1],
            channels[1],
            channels[2],
            reduce_spatial_size=True
        )
        self.conv4 = self._make_layer(
            blocks[2],
            layers[2],
            channels[2],
            channels[3],
            reduce_spatial_size=False
        )
        self.conv5 = ODConv1x1(channels[3], channels[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        # fully connected layer
        self.fc = self._construct_fc_layer(
            self.feature_dim, channels[3], dropout_p=None
        )
        # identity classification layer
        self.classifier = nn.Linear(self.feature_dim, num_classes)

        if self.aligned:
            self.aligned_bn = nn.BatchNorm2d(channels[3])
            self.aligned_relu = nn.ReLU(inplace=True)
            self.aligned_conv1 = nn.Conv2d(channels[3], 128, kernel_size=1, stride=1, padding=0, bias=True)

        self._init_params()

    def _make_layer(
            self,
            block,
            layer,
            in_channels,
            out_channels,
            reduce_spatial_size,
            IN=False
    ):
        layers = []

        layers.append(block(in_channels, out_channels, IN=IN))
        for i in range(1, layer):
            layers.append(block(out_channels, out_channels, IN=IN))

        if reduce_spatial_size:
            layers.append(
                nn.Sequential(
                    Conv1x1(out_channels, out_channels),
                    nn.AvgPool2d(2, stride=2)
                )
            )

        return nn.Sequential(*layers)

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
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

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

    def forward(self, x):
        x = self.featuremaps(x)  # [b, channels[3], 16, 8]
        if not self.training:
            lf = self.horizontal_pool(x)  # [b, channels[3], 16, 1]

        if self.aligned and self.training:
            lf = self.aligned_bn(x)
            lf = self.aligned_relu(lf)
            lf = self.horizontal_pool(lf)  # [b, 512, 16, 1]
            lf = self.aligned_conv1(lf)  # [b, 128, 16, 1]

        if self.aligned or not self.training:
            lf = lf.view(lf.shape[:3])  # [b, 128, 16]
            lf = lf / torch.pow(lf, 2).sum(dim=1, keepdim=True).clamp(min=1e-12).sqrt()

        f = F.avg_pool2d(x, x.shape[2:])  # [b, channels[3], 1, 1]
        f = f.reshape(f.shape[0], -1)  # [b, channels[3]]
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
            return y, f


def odosnet_x1_0(num_classes=1000, act_func='relu', attention=None, loss='softmax', aligned=False, **kwargs):
    model = ODOSNet(num_classes,
                    blocks=[ODOSBlock, ODOSBlock, ODOSBlock],
                    layers=[2, 2, 2],
                    channels=[64, 256, 384, 512],
                    feature_dim=512,
                    act_func=act_func,
                    attention=attention,
                    loss=loss,
                    aligned=aligned,
                    **kwargs)

    return model


def odosnet_x0_75(num_classes=1000, act_func='relu', attention=None, loss='softmax', aligned=False, **kwargs):
    model = ODOSNet(num_classes,
                    blocks=[ODOSBlock, ODOSBlock, ODOSBlock],
                    layers=[2, 2, 2],
                    channels=[48, 192, 288, 384],
                    feature_dim=512,
                    act_func=act_func,
                    attention=attention,
                    loss=loss,
                    aligned=aligned,
                    **kwargs)

    return model


def odosnet_x0_5(num_classes=1000, act_func='relu', attention=None, loss='softmax', aligned=False, **kwargs):
    model = ODOSNet(num_classes,
                    blocks=[ODOSBlock, ODOSBlock, ODOSBlock],
                    layers=[2, 2, 2],
                    channels=[32, 128, 192, 256],
                    feature_dim=512,
                    act_func=act_func,
                    attention=attention,
                    loss=loss,
                    aligned=aligned,
                    **kwargs)

    return model


if __name__ == "__main__":
    model_1_0 = odosnet_x1_0(num_classes=751, act_func='relu', attention=None, loss='softmax', aligned=False)
    model_0_75 = odosnet_x0_75(num_classes=751, act_func='relu', attention=None, loss='softmax', aligned=False)
    model_0_5 = odosnet_x0_5(num_classes=751, act_func='relu', attention=None, loss='softmax', aligned=False)

    arr_inputs = torch.randn(4, 3, 128, 256)
    preds, features = model_1_0(arr_inputs)
    preds2, features2 = model_0_75(arr_inputs)
    preds3, features3 = model_0_5(arr_inputs)
    print(preds.shape, features.shape)
    print(preds2.shape, features2.shape)
    print(preds3.shape, features3.shape)
    print(
        f"model_1_0 size: {sum(p.numel() for p in model_1_0.parameters()) / 1000000.0:.3f}M, {len(model_1_0.state_dict())} layers")
    print(
        f"model_0_75 size: {sum(p.numel() for p in model_0_75.parameters()) / 1000000.0:.3f}M, {len(model_0_75.state_dict())} layers")
    print(
        f"model_0_5 size: {sum(p.numel() for p in model_0_5.parameters()) / 1000000.0:.3f}M, {len(model_0_5.state_dict())} layers")
