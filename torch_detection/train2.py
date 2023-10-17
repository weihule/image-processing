import os
import sys
import argparse
import functools
import time

import torch
import torch.distributed
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.util import Logger, AverageMeter, set_seed, worker_seed_init_fn
from datasets.coco import CocoDetection
from datasets.voc import VOCDetection
from datasets.transform import *
from datasets.collater import DetectionCollater
from config import Cfg


def main():
    assert torch.cuda.is_available(), "need gpu to train network!"
    torch.cuda.empty_cache()
    print("start")
    cfg = Cfg()
    log_dir = os.path.join(cfg.work_dir, 'log')
    checkpoint_dir = os.path.join(cfg.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    sys.stdout = Logger(os.path.join(cfg.save_dir, 'train.log'))

    gpus_type = torch.cuda.get_device_name()
    gpus_num = torch.cuda.device_count()

    set_seed(cfg.seed)

    os.makedirs(
        checkpoint_dir) if not os.path.exists(checkpoint_dir) else None
    os.makedirs(log_dir) if not os.path.exists(log_dir) else None

    batch_size, num_workers = cfg.batch_size, cfg.num_workers
    assert cfg.batch_size % cfg.gpus_num == 0, 'config.batch_size is not divisible by config.gpus_num!'
    assert cfg.num_workers % cfg.gpus_num == 0, 'config.num_workers is not divisible by config.gpus_num!'
    batch_size = int(cfg.batch_size // cfg.gpus_num)
    num_workers = int(cfg.num_workers // cfg.gpus_num)

    train_loader = DataLoader(cfg.train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True,
                              num_workers=num_workers,
                              collate_fn=cfg.train_collater)
    # test_loader = DataLoader(cfg.test_dataset,
    #                          batch_size=batch_size,
    #                          shuffle=False,
    #                          pin_memory=True,
    #                          num_workers=num_workers,
    #                          collate_fn=cfg.test_collater)

    train_dataset = cfg.train_dataset
    print(len(train_dataset))
    d = train_dataset[2]
    print(type(d))


if __name__ == "__main__":
    main()


