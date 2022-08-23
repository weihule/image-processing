import os
import sys
import torch
import torch.nn as nn

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from torch_detection.yolo.network_files.darknet import ConvBnActBlock

