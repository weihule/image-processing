import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "network_files"))
import torch
import torch.nn as nn
import numpy as np
from network_files.fpn import FPN
from network_files.heads import RetinaClsHead, RetinaRegHead


class RetinaNet(nn.Module):
    def __init__(self,
                 backbone_type,
                 backbone_pretrain_path='',
                 ):

if __name__ == '__main__':
    arr = np.random.random(size=(4, 3, 2))
    res = np.max(arr)

    print(os.getcwd())
