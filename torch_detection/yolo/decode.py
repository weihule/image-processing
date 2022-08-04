import os
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import batched_nms

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(BASE_DIR)
from torch_detection.yolo.network_files.anchor import YoloV3Anchors


class YoloV4Decoder:
    def __init__(self,
                 anchor_sizes=None,
                 strides=None,
                 per_level_num_anchors=3,
                 max_object_num=100,
                 min_score_threshold=0.05,
                 topn=1000,
                 nms_type='python_nms',
                 nms_threshold=0.5):
        assert nms_type in ['torch_nms', 'python_nms',
                            'diou_python_nms'], 'wrong nms type!'
        if anchor_sizes is None:
            self.anchor_sizes = [[10, 13], [16, 30], [33, 23], [30, 61],
                                 [62, 45], [59, 119], [116, 90], [156, 198],
                                 [373, 326]]
        else:
            self.anchor_sizes = anchor_sizes

        if strides is None:
            self.strides = [8, 16, 32]
        else:
            self.strides = strides
        self.per_level_num_anchors = per_level_num_anchors
        self.max_object_num = max_object_num
        self.min_score_threshold = min_score_threshold
        self.topn = topn
        self.nms_type = nms_type
        self.nms_threshold = nms_threshold
        self.anchors = YoloV3Anchors(anchor_sizes=self.anchor_sizes,
                                     strides=self.strides)


class DecodeMethod:
    def __init__(self,
                 max_object_num=100,
                 min_score_threshold=0.5,
                 topn=100,
                 nms_type='python_nms',
                 nms_threshold=0.5):
        self.max_object_num = max_object_num
        self.min_score_threshold = min_score_threshold
        self.topn = topn
        self.nms_type = nms_type


if __name__ == "__main__":
    print(BASE_DIR)
