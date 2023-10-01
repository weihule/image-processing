import os
import sys
import argparse
import functools
import time

import torch
from torch.utils.data import DataLoader

from utils.util import Logger, AverageMeter, set_seed


def main(cfgs):
    assert torch.cuda.is_available(), "need gpu to train network!"
    torch.cuda.empty_cache()

    gpus_type = torch.cuda.get_device_name()
    gpus_num = torch.cuda.device_count()

    set_seed(cfgs["seed"])

    local_rank = int(os.environ['LOCAL_RANK'])
    # start init process
    torch.distributions
