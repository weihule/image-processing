from __future__ import division, absolute_import
import warnings
import torch
from torch import nn
from torch.nn import functional as F

from .other_modules import activate_function, HorizontalPooling
from .other_modules import attention_module
from .other_modules import SCBottleneck  # 自校准卷积
from .osnet_origin import ConvLayer, Conv1x1, LightConv3x3, ChannelGate, OSBlock

__all__ = [
    'sc_osnet_x1_0_origin'
]


class SCOSNet(nn.Module):
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
            loss='softmax',
            act_func='relu',
            attention=None,
            aligned=False,
            IN=False,
            **kwargs
    ):
        super(SCOSNet, self).__init__()
        num_blocks = len(blocks)
        assert num_blocks == len(layers)
        assert num_blocks == len(channels) - 1
        self.loss = loss
        self.feature_dim = feature_dim

        # convolutional backbone
        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3, IN=IN)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._make_layer(
            blocks[0],
            layers[0],
            channels[0],
            channels[1],
            reduce_spatial_size=True,
            IN=IN,
            attention=attention,
            act_func=act_func,
        )
        self.conv3 = self._make_layer(
            blocks[1],
            layers[1],
            channels[1],
            channels[2],
            reduce_spatial_size=True,
            attention=attention,
            act_func=act_func,
        )
        # self.conv4 = self._make_layer(
        #     blocks[2],
        #     layers[2],
        #     channels[2],
        #     channels[3],
        #     reduce_spatial_size=False,
        #     attention=attention,
        #     act_func=act_func,
        # )
        # self.conv5 = Conv1x1(channels[3], channels[3])
        self.conv4 = nn.Sequential(*[SCBottleneck(channels[2], int(channels[2] / 4),
                                                  stride=1,
                                                  downsample=None,
                                                  cardinality=1,
                                                  bottleneck_width=32,
                                                  avd=False,
                                                  dilation=1,
                                                  is_first=False,
                                                  norm_layer=nn.BatchNorm2d),
                                     SCBottleneck(channels[2], int(channels[2] / 4),
                                                  stride=1,
                                                  downsample=None,
                                                  cardinality=1,
                                                  bottleneck_width=32,
                                                  avd=False,
                                                  dilation=1,
                                                  is_first=False,
                                                  norm_layer=nn.BatchNorm2d)
                                     ])
        self.conv5 = Conv1x1(channels[2], channels[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        # fully connected layer
        self.fc = self._construct_fc_layer(
            self.feature_dim, channels[3], dropout_p=None
        )
        # identity classification layer
        self.classifier = nn.Linear(self.feature_dim, num_classes)

        self.aligned = aligned
        self.horizon_pool = HorizontalPooling()
        if self.aligned:
            self.aligned_bn = nn.BatchNorm2d(channels[3])
            self.aligned_relu = nn.ReLU(inplace=True)
            self.aligned_conv = nn.Conv2d(channels[3], 128,
                                          kernel_size=1,
                                          stride=1,
                                          padding=0,
                                          bias=True)

        self._init_params()

    def _make_conv4(self):
        pass

    def _make_layer(
            self,
            block: OSBlock,
            layer,
            in_channels,
            out_channels,
            reduce_spatial_size,
            IN=False,
            attention=None,
            act_func='relu',
    ):
        layers = []

        layers.append(block(in_channels, out_channels, IN=IN))
        for i in range(1, layer):
            layers.append(block(out_channels, out_channels, IN=IN, attention=attention, act_func=act_func, ))

        if reduce_spatial_size:
            layers.append(
                nn.Sequential(
                    Conv1x1(out_channels, out_channels),
                    nn.AvgPool2d(2, stride=2)
                )
            )

        return nn.Sequential(*layers)

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
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

    def forward(self, x, return_featuremaps=False):
        x = self.featuremaps(x)  # [B, channels[3], 16, 8]
        # TODO: 这里只有在训练阶段使用lf, 测试阶段不输出lf(和原版不同)
        if self.aligned and self.training:
            lf = self.aligned_bn(x)
            lf = self.aligned_relu(lf)  # [B, 128, 16, 8]
            lf = self.horizon_pool(lf)  # [B, 128, 16, 1]
            lf = self.aligned_conv(lf)
            lf = lf.view(lf.shape[0:3])  # [B, 128, 16]
            # 这里相当于是对lf做了通道方向的归一化
            lf = lf / torch.pow(lf, 2).sum(dim=1, keepdim=True).clamp(min=1e-12).sqrt()
            print(lf.shape)
        if return_featuremaps:
            return x
        f = self.global_avgpool(x)
        f = f.view(f.size(0), -1)

        if self.fc is not None:
            f = self.fc(f)

        if not self.training:
            return f
        y = self.classifier(f)
        if self.loss == 'softmax':
            return y
        elif self.loss == 'softmax_trip':
            if self.aligned:
                return y, f, lf
            else:
                return y, f
        elif self.loss == 'softmax_trip_cent':
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


def sc_osnet_x1_0_origin(num_classes=1000, act_func='relu', attention=None, loss='softmax_trip', aligned=False,
                         **kwargs):
    model = SCOSNet(num_classes,
                    blocks=[OSBlock, OSBlock, OSBlock],
                    layers=[2, 2, 2],
                    channels=[64, 256, 384, 512],
                    feature_dim=512,
                    act_func=act_func,
                    attention=attention,
                    loss=loss,
                    aligned=aligned,
                    **kwargs)

    return model
