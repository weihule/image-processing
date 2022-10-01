from __future__ import print_function, absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F

from other_modules import activate_function

__all__ = [
    'osnet_x1_0',
    'osnet_x0_75'
]


class ConvLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
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
    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
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
    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(Conv3x3, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class LightConv3x3(nn.Module):
    """
    Lightweight 3x3 convolution

    1x1 (linear) + dw 3x3 (nolinear)
    """
    def __init__(self, in_channels, out_channels, act_func):
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
        # self.act_func = nn.ReLU(inplace=True)
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
                 layer_norm=False):
        super(ChannelGate, self).__init__()
        if num_gates is None:
            num_gates = in_channels
        self.return_gates = return_gates
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 降维
        self.fc1 = nn.Conv2d(in_channels,
                             in_channels//reduction,
                             kernel_size=1,
                             bias=True,
                             padding=0)
        self.norm1 = None
        if layer_norm:
            self.norm1 = nn.LayerNorm([in_channels // reduction, 1, 1])
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels//reduction,
                             num_gates,
                             kernel_size=1,
                             bias=True,
                             padding=0)
        if gate_activation == 'sigmoid':
            self.gate_activation = nn.Sigmoid()
        elif gate_activation == 'relu':
            self.gate_activation = nn.ReLU(inplace=True)
        elif gate_activation == 'linear':
            self.gate_activation = None
        else:
            raise RuntimeError(f'Unknown gate activation {gate_activation}')

    def forward(self, x):
        inputs = x.clone()
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


class OSBlock(nn.Module):
    """
    Omni-scale feature learning block
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 act_func,
                 IN=False,
                 bottleneck_reduction=4,
                 **kwargs):
        super(OSBlock, self).__init__()
        mid_channels = out_channels // bottleneck_reduction
        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2a = LightConv3x3(mid_channels, mid_channels, act_func)
        self.conv2b = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels, act_func),
            LightConv3x3(mid_channels, mid_channels, act_func)
        )
        self.conv2c = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels, act_func),
            LightConv3x3(mid_channels, mid_channels, act_func),
            LightConv3x3(mid_channels, mid_channels, act_func)
        )
        self.conv2d = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels, act_func),
            LightConv3x3(mid_channels, mid_channels, act_func),
            LightConv3x3(mid_channels, mid_channels, act_func),
            LightConv3x3(mid_channels, mid_channels, act_func)
        )
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)
        self.IN = None
        if IN:
            self.IN = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):
        identity = x.clone()
        x1 = self.conv1(x)
        x2a = self.conv2a(x1)
        x2b = self.conv2b(x1)
        x2c = self.conv2c(x1)
        x2d = self.conv2d(x1)
        x2 = self.gate(x2a) + self.gate(x2b) + self.gate(x2c) + self.gate(x2d)
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = x3 + identity
        if self.IN is not None:
            out = self.IN(out)

        return F.relu(out)


class OSNet(nn.Module):
    def __init__(self,
                 num_classes,
                 blocks,
                 layers,
                 channels,
                 feature_dim=512,
                 act_func='relu',
                 loss='softmax',
                 IN=False,
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
        )
        # conv4
        self.conv4 = self._make_layers(
            blocks[2],
            layers[2],
            channels[2],
            channels[3],
            reduce_spatial_size=False,
            act_func=act_func,
        )
        self.conv5 = Conv1x1(channels[3], channels[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # fully connected layer
        self.fc = self._construct_fc_layer(fc_dims=self.feature_dim,
                                           input_dim=channels[3],
                                           dropout_p=None)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

        self._init_params()

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

    def forward(self, x, return_featuremaps=False):
        x = self.featuremaps(x)     # [b, 512, 16, 8]
        if return_featuremaps:
            return x
        v = self.global_avgpool(x)
        v = v.view(v.shape[0], -1)  # [b, 512]
        if self.fc is not None:
            v = self.fc(v)

        # 测试阶段(输出分类器之前的特征)
        if not self.training:
            return v

        y = self.classifier(v)
        if self.loss == 'softmax':
            return y
        elif self.loss == 'softmax_cent':
            return y, v
        elif self.loss == 'softmax_trip':
            return y, v
        else:
            raise KeyError(f'Unsupported {self.loss}')

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        """

        Args:
            fc_dims: (int) 512
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
                     act_func,
                     IN=False):
        """
        layer 是 block 的重复次数
        """
        layers = []
        layers.append(block(in_channels, out_channels, act_func=act_func, IN=IN))
        for i in range(1, layer):
            layers.append(block(out_channels, out_channels, act_func=act_func, IN=IN))

        # 宽高减半
        if reduce_spatial_size:
            layers.append(
                nn.Sequential(
                    Conv1x1(out_channels, out_channels),
                    nn.AvgPool2d(2, stride=2)
                )
            )

        return nn.Sequential(*layers)


def osnet_x1_0(num_classes=1000, act_func='relu', loss='softmax', **kwargs):
    model = OSNet(num_classes,
                  blocks=[OSBlock, OSBlock, OSBlock],
                  layers=[2, 2, 2],
                  channels=[64, 256, 384, 512],
                  feature_dim=512,
                  act_func=act_func,
                  loss=loss,
                  **kwargs)

    return model


def osnet_x0_75(num_classes=1000, act_func='relu', loss='softmax', **kwargs):
    model = OSNet(num_classes,
                  blocks=[OSBlock, OSBlock, OSBlock],
                  layers=[2, 2, 2],
                  channels=[48, 192, 288, 384],
                  feature_dim=512,
                  act_func=act_func,
                  loss=loss,
                  **kwargs)

    return model


if __name__ == "__main__":
    arr = torch.randn(4, 3, 256, 128)
    model = osnet_x0_75(num_classes=751, act_func='relu', loss='softmax_trip')
    outs, features = model(arr)
    print(outs.shape, features.shape, model.feature_dim)

    # arr = torch.arange(24).reshape((2, 2, 2, 3)).float()
    # print(arr)
    #
    # bn = nn.BatchNorm2d(2)
    # bn_out = bn(arr)
    # print(bn_out)
    #
    # test = torch.tensor([0, 1, 2, 3, 4, 5,
    #                      12, 13, 14, 15, 16, 17]).float()
    #
    # test = (test - torch.mean(test)) / torch.var(test, unbiased=False).sqrt()
    # print(test)


