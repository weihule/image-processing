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


# 设置环境变量
os.environ['LOCAL_RANK'] = '1'
os.environ['RANK'] = '0'  # 第一个进程的排名
os.environ['WORLD_SIZE'] = '4'  # 总共两个进程
os.environ['MASTER_ADDR'] = 'localhost'  # 主机地址
os.environ['MASTER_PORT'] = '12345'  # 主机端口


def main(cfgs):
    assert torch.cuda.is_available(), "need gpu to train network!"
    torch.cuda.empty_cache()
    print("start")

    log_dir = os.path.join(cfgs["work_dir"], 'log')
    checkpoint_dir = os.path.join(cfgs["work_dir"], 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    sys.stdout = Logger(os.path.join(cfgs["save_dir"], 'train.log'))

    gpus_type = torch.cuda.get_device_name()
    gpus_num = torch.cuda.device_count()

    set_seed(cfgs["seed"])

    local_rank = int(os.environ['LOCAL_RANK'])
    # start init process
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(local_rank)
    torch.distributed.new_group(list(range(gpus_num)))

    if local_rank == 0:
        os.makedirs(
            checkpoint_dir) if not os.path.exists(checkpoint_dir) else None
        os.makedirs(log_dir) if not os.path.exists(log_dir) else None

    torch.distributed.barrier()

    batch_size, num_workers = cfgs["batch_size"], cfgs["num_workers"]
    assert cfgs["batch_size"] % cfgs["gpus_num"] == 0, 'config.batch_size is not divisible by config.gpus_num!'
    assert cfgs["num_workers"] % cfgs["gpus_num"] == 0, 'config.num_workers is not divisible by config.gpus_num!'
    batch_size = int(cfgs["batch_size"] // cfgs["gpus_num"])
    num_workers = int(cfgs["num_workers"] // cfgs["gpus_num"])

    init_fn = functools.partial(worker_seed_init_fn,
                                num_workers=num_workers,
                                local_rank=local_rank,
                                seed=cfgs["seed"])

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, shuffle=True)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              pin_memory=True,
                              drop_last=True,
                              num_workers=num_workers,
                              collate_fn=train_collater,
                              sampler=train_sampler,
                              worker_init_fn=init_fn)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             pin_memory=True,
                             num_workers=num_workers,
                             collate_fn=test_collater)


if __name__ == "__main__":
    cfg_dict = {
        "seed": 0
    }
    main(cfg_dict)


