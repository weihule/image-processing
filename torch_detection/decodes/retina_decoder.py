import os
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decode_method import DetNMSMethod, DecodeMethod


class RetinaDecoder:
    def __init__(self,
                 areas=([32, 32], [64, 64], [128, 128], [256, 256], [512, 512]),
                 ratios=(0.5, 1, 2),
                 scales=(2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)),
                 strides=(8, 16, 32, 64, 128),
                 max_object_num=100,
                 min_score_threshold=0.05,
                 topn=1000,
                 nms_type='python_nms',
                 nms_threshold=0.5
                 ):
        assert nms_type in ['torch_nms',
                            'python_nms',
                            'diou_python_nms'], 'wrong nms type!'

        self.decode_function = DecodeMethod(
            max_object_num=max_object_num,
            min_score_threshold=min_score_threshold,
            topn=topn,
            nms_type=nms_type,
            nms_threshold=nms_threshold)

    def __call__(self, preds):
