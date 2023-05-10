import os
import math
import random
import time
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.cuda import amp
import torch.nn.functional as F
from torch.backends import cudnn
from torch.optim import Adam, SGD, lr_scheduler
from torch.utils.data import DataLoader

from losses import CELoss
from utils.util import get_logger, load_state_dict
from new_config import cfg


def main(logger):
    torch.cuda.empty_cache()

    if cfg.seed:
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)

    gpus = torch.cuda.device_count()
    logger.info(f'use {gpus} gpus')

    cudnn.benchmark = True
    cudnn.enabled = True
    start_time = time.time()

    device = torch.device(cfg.device)
    logger.info("start loading data")

    train_loader = DataLoader(dataset=cfg.train_dataset,
                              batch_size=cfg.batch_size,
                              shuffle=True,
                              num_workers=4,
                              collate_fn=cfg.collater)

    val_loader = DataLoader(dataset=cfg.val_dataset,
                            batch_size=cfg.batch_size,
                            shuffle=False,
                            num_workers=4,
                            collate_fn=cfg.collater)

    for d in val_loader:
        i = d["image"]
        l = d["label"]
        print(i.shape, l)
        break


if __name__ == "__main__":
    if not os.path.exists(cfg.save_root):
        os.mkdir(cfg.save_root)
    if not os.path.exists(cfg.log):
        os.mkdir(cfg.log)
    if not os.path.exists(cfg.checkpoints):
        os.mkdir(cfg.checkpoints)
    if not os.path.exists(cfg.pth_path):
        os.mkdir(cfg.pth_path)

    logger_writer = get_logger('resnet', cfg.log)
    main(logger_writer)
