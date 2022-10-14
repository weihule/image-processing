import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'HorizontalPooling',
    'FRelu',
    'activate_function',
    'attention_module'
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
        res = torch.maximum(x, x1)

        return res


def activate_function(act_name, channels=None):
    if act_name == 'relu':
        return nn.ReLU(inplace=True)
    elif act_name == 'frelu':
        return FRelu(channels)
    elif act_name == 'prelu':
        return nn.PReLU()
    else:
        raise KeyError(f'Unknown activate function {act_name}')

# =========== SE ================


class SEAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEAttention, self).__init__()

        # 返回1*1特征图，通道不变
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape

        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        return y.expand_as(x)*x
# =========== SE ================


# =========== CBAM ================
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))

        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channel, channel//reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel//reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x)  # [b, c, 1, 1]
        max_out = self.max_pool(x)  # [b, c, 1, 1]

        return self.sigmoid(avg_out + max_out)  # [b, c, 1, 1]


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in [3, 7], 'Kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)    # [b, 1, h, w]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [b, 1, h, w]
        y = torch.cat([avg_out, max_out], dim=1)        # [b, 2, h, w]
        y = self.conv(y)                                # [b, 1, h, w]

        return self.sigmoid(y)


class CBAMAttention(nn.Module):
    def __init__(self, channel):
        super(CBAMAttention, self).__init__()
        self.ca = ChannelAttention(channel)
        self.sa = SpatialAttention(kernel_size=7)

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x

        return x
# =========== CBAM ================


# =========== TripletAttention ================


class ZPool(nn.Module):
    def __init__(self):
        super(ZPool, self).__init__()

    def forward(self, x):
        x1, _ = torch.max(x, dim=1, keepdim=True)   # [b, 1, h, c]
        x2 = torch.mean(x, dim=1, keepdim=True)     # [b, 1, h, c]
        outs = torch.cat([x1, x2], dim=1)           # [b, 2, h, w]

        return outs


class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = nn.Conv2d(2, 1, kernel_size, 1, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_compress = self.compress(x)   # [b, 2, h, w]
        x_out = self.conv(x_compress)
        x_out = self.sigmoid(x_out)     # [b, 1, h, w]

        # # [b, c, h, w]
        return x_out * x


class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = AttentionGate()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()    # [b, h, c, w]
        x_out1 = self.cw(x_perm1)                       # [b, 1(h), c, w]
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()        # [b, c, 1(h), w]

        x_perm2 = x.permute(0, 3, 2, 1).contiguous()    # [b, w, h, c]
        x_out2 = self.cw(x_perm2)                       # [b, 1(w), h, c]
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()        # [b, c, h, 1(w)]

        if not self.no_spatial:
            x_out = self.hw(x)                          # [b, 1(c), h, w]
            x_out = 1/3 * (x_out + x_out11 + x_out21)
        else:
            x_out = 1/2 * (x_out11 + x_out21)

        return x_out


class NAM(nn.Module):
    def __init__(self, channels):
        super(NAM, self).__init__()
        self.channels = channels
        self.bn2 = nn.BatchNorm2d(self.channels, affine=True)

    def forward(self, x):
        residual = x
        x = self.bn2(x)
        weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())      # [channel]
        # 交换维度是为了 weight_bn 进行广播操作 -> [1, 1, 1, channel]
        # [b, c, h, w] -> [b, h, w, c]
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = torch.sigmoid(x) * residual

        return x


def attention_module(attention_name, channel=None):
    if attention_name == 'se_attention':
        return SEAttention(channel)
    elif attention_name == 'cbam_attention':
        return CBAMAttention(channel)
    elif attention_name == 'triplet_attention':
        return TripletAttention(no_spatial=True)
    elif attention_name == 'nam_attention':
        return NAM(channel)
    else:
        raise KeyError(f'Unknown {attention_name}')


if __name__ == "__main__":
    arr = torch.randn(4, 512, 16, 16)
    cbam = CBAMAttention(channel=512)
    se = SEAttention(channel=512)
    trip = TripletAttention(no_spatial=True)
    nam = attention_module('nam_attention', channel=512)
    outs_cbam = cbam(arr)
    outs_se = se(arr)
    outs_trip = trip(arr)
    outs_nam = nam(arr)
    print(outs_nam.shape)

    a = torch.arange(16).reshape(2, 2, 2, 2)
    b = torch.tensor([3, 4])
    ab = torch.mul(a, b)
    print(a)
    print(ab)


