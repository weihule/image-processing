import os
import torch.nn as nn

from . import backbones
from .head import YOLOXHead
from .fpn import YOLOFPN, YOLOPAFPN


class YOLOX(nn.Module):
    def __init__(self,
                 backbone_type='cspdark53backbone',
                 pre_train_load_dir=None,
                 num_classes=80):
        super(YOLOX, self).__init__()
        self.backbone = backbones.__dict__[backbone_type](
            **{'pre_train_load_dir': pre_train_load_dir}
        )
        self.fpn = YOLOPAFPN()
        self.head = YOLOXHead(num_classes=num_classes)
        self.in_features = ("dark3", "dark4", "dark5"),

    def forward(self, x):
        # out_features content features of [dark3, dark4, dark5]
        out_features = self.backbone(x)

        # darknet53: [b, 256, h/8, w/8], [b, 512, h/16, w/16], [b, 512, h/32, w/32]
        # cspdarknet53: [b, 256, h/8, w/8], [b, 512, h/16, w/16], [b, 1024, h/32, w/32]
        features = [out_features[f] for f in self.in_features]

        # [b, 256, h/8, w/8], [b, 512, h/16, w/16], [b, 1024, h/32, w/32]
        fpn_outs = self.fpn(features)

        # [[b, 4+1+num_classes, h/8, w/8], [b, 4+1+num_classes, h/16, w/16], [b, 4+1+num_classes, h/32, w/32]]
        out_puts = self.head(fpn_outs)

