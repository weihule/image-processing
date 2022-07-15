import os
import sys
import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class YoloV3FPNHead(nn.Module):
    def __init__(self,
                 inplanes,
                 per_level_num_anchors=3,
                 num_classes=80,
                 act_type='leakyrelu'):
        super(YoloV3FPNHead, self).__init__()
        self.per_level_num_anchors = per_level_num_anchors

        # inplanes:[c3_inplanes, c4_inplanes, c5_inplanes]
        p5_1_layers = []
