import torch.nn as nn
from network_blocks import BaseConv, CSPLayer, SPPBottleneck, DWConv, Focus, ResLayer


class Darknet(nn.Module):
    depth2blocks = {21: [1, 2, 2, 1], 53: [2, 8, 8, 4]}
    def __init__(self,
                 depth,
                 in_channels=3,
                 stem_out_channels=32,
                 out_features=('dark3', 'dark4', 'dark5')):
        """
        Args:
            depth: (int) depth of darknet used in model, usually use [21, 53] for this param.
            in_channels:
            stem_out_channels:
            out_features:
        """
        super(Darknet, self).__init__()
        assert out_features, 'please provide output features of darknet'
        self.out_features = out_features
        self.stem = nn.Sequential(
            BaseConv(in_channels, stem_out_channels, ksize=3, stride=1, act='lrelu'),
            *self.make_group_layer(stem_out_channels, num_blocks=1, stride=2)
        )
        in_channels = stem_out_channels * 2    # 64
        num_blocks = Darknet.depth2blocks[depth]
        # create darknet with 'stem_out_channels' and 'num_blocks' layers
        # to make model structure more clear, we don't use 'for' statement in python
        self.dark2 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[0], stride=2)
        )
        in_channels *= 2    # 128
        self.dark3 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[1], stride=2)
        )
        in_channels *= 2    # 256
        self.dark4 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[2], stride=2)
        )
        in_channels *= 2    # 512

        self.dark5 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[1], stride=2),
            *self.make_spp_block([in_channels, in_channels * 2], in_channels * 2)
        )

    def make_group_layer(self, in_channels, num_blocks, stride=1):
        """
        starts with conv layer then has 'num_blocks' 'reslayer'
        [b, c, h, w] -> [b, 2c, ]
        Args:
            in_channels:
            num_blocks:
            stride:
        Returns:
        """
        return [
            BaseConv(in_channels, in_channels * 2, ksize=3, stride=stride, act='lrelu'),
            *[(ResLayer(in_channels * 2)) for _ in range(num_blocks)]
        ]

    def make_spp_block(self, filter_list, in_filters):
        m = nn.Sequential(
            *[BaseConv(in_filters, filter_list[0], 1, stride=1, act='lrelu'),
              BaseConv(filter_list[0], filter_list[1], 3, stride=1, act='lrelu'),
              SPPBottleneck(in_channels=filter_list[1],
                            out_channels=filter_list[0],
                            activation='lrelu'),
              BaseConv(filter_list[0], filter_list[1], 3, stride=1, act='lrelu'),
              BaseConv(filter_list[1], filter_list[0], 1, stride=1, act='lrelu')]
        )

        return m

    def forward(self, x):
        outputs = {}
        x = self.stem(x)   # [b, 64, h/2, w/2]
        outputs['stem'] = x

        x = self.dark2(x)  # [b, 128, h/4, w/4]
        outputs['dark2'] = x

        x = self.dark3(x)  # [b, 256, h/8, w/8]
        outputs['dark3'] = x

        x = self.dark4(x)  # [b, 512, h/16, w/16]
        outputs['dark4'] = x

        x = self.dark5(x)  # [b, 512, h/32, w/32]
        outputs['dark5'] = x

        return {k: v for k, v in outputs.items() if k in self.out_features}


class CSPDarknet(nn.Module):
    def __init__(self,
                 dep_num,
                 wid_num,
                 out_features=('dark3', 'dark4', 'dark5'),
                 depthwise=False,
                 act='silu'):
        super(CSPDarknet, self).__init__()


if __name__ == "__main__":
    import torch
    dark = Darknet(depth=53)
    ins = torch.randn(4, 3, 640, 640)
    outs = dark(ins)
    for k, v in outs.items():
        print(k, v.shape)

