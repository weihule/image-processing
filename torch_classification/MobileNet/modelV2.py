import torch
import torch.nn as nn
from torch import Tensor, batch_norm

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

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor):
        y = self.avg(x)     # [B, C, 1, 1]
        y = torch.flatten(y, start_dim=1)   # [B, C*1*1]
        y = self.fc(y)

        return x * y.expand_as(x)

class ConvBNRelu(nn.Sequential):
    """
    这个类继承自nn.Sequential, 只有一个初始化,
    需要在初始化的时候就做好 Sequential 的工作
    """
    def __init__(self, in_channel, out_channel,
                kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2    # 向下取整
        super(ConvBNRelu, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, groups=groups),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )  

class InvertedResidual(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        hidden_channel = in_channel * expand_ratio
        self.short_cut = (stride == 1 and in_channel == out_channel)

        layers = list()
        # 如果expand_ratio即扩展因子为1, PW卷积既不升维也不降维，所以可以去掉
        if expand_ratio != 1:
            # 1x1 pointwise conv
            layers.append(ConvBNRelu(in_channel, hidden_channel, kernel_size=1))
        

# class InvertedResidual(nn.Module):
#     def __init__(self, in_channel, out_channel, stride, expand_ratio):
#         super(InvertedResidual, self).__init__()
#         hidden_channel = expand_ratio * in_channel
#         self.short_cut = (stride==1 and in_channel==out_channel)

#         layers = []
#         # expand_ratio不为1时，有 conv1*1 这一块
#         if expand_ratio != 1:
#             # 1x1 pointwise conv
#             layers.append(ConvBNReLU(in_channel, hidden_channel, kernel_size=1))
#         layers.extend([
#             # 3x3 depthwise conv
#             ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
#             # 1x1 pointwise conv( 这里用的是线性激活，所以不能有ReLU，加到BN层就好 )
#             nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
#             nn.BatchNorm2d(out_channel)
#         ])

#         self.conv = nn.Sequential(*layers)

#     def forward(self, x):
#         if self.short_cut:
#             return x + self.conv(x)
#         else:
#             return self.conv(x)

# class MobileNetV2(nn.Module):
#     def __init__(self, num_classes=1000, alpha=1.0, round_nearest=8):
#         super(MobileNetV2, self).__init__()
#         input_channel = _make_divisible(32 * alpha, round_nearest)
#         last_channel = _make_divisible(1280 * alpha, round_nearest)

#         inverted_residual_setting = [
#             # t, c, n, s
#             [1, 16, 1, 1],
#             [6, 24, 2, 2],
#             [6, 32, 3, 2],
#             [6, 64, 4, 2],
#             [6, 96, 3, 1],
#             [6, 160, 3, 2],
#             [6, 320, 1, 1],
#         ]

#         features = []
#         # conv1 layer
#         features.append(ConvBNReLU(3, input_channel, stride=2))
#         # building inverted residual blocks
#         for t,c,n,s in inverted_residual_setting:
#             output_channel = _make_divisible(c*alpha, round_nearest)
#             for i in range(n):
#                 stride = s if i == 0 else 1
#                 features.append(InvertedResidual(input_channel, output_channel, stride=stride, expand_ratio=t))
#                 input_channel = output_channel
        
#         # building last several layers
#         features.append(ConvBNReLU(input_channel, last_channel, kernel_size=1))

#         # combine feature layers
#         self.features = nn.Sequential(*features)

#         # building classifier
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.classifier = nn.Sequential(
#             nn.Dropout(0.3),
#             nn.Linear(last_channel, num_classes)
#         )

#         # weight initialization
#         for m in self.modules():
#             if isinstance(m ,nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode="fan_out")
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.ones_(m.weight)
#                 nn.init.zeros_(m.bias)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.zeros_(m.bias)

#     def forward(self, x):
#         x = self.features(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x, start_dim=1)
#         x = self.classifier(x)

#         return x


if __name__ == "__main__":
    input_feature = torch.randn(1, 3, 12, 12)
    layer = ConvBNRelu(3, 6, stride=2)
    x = layer(input_feature)
    print(layer) 
    print(x.shape)





