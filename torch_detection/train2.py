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


# 设置环境变量
os.environ['LOCAL_RANK'] = '0'
# os.environ['RANK'] = '0'  # 第一个进程的排名
# os.environ['WORLD_SIZE'] = '4'  # 总共两个进程
# os.environ['MASTER_ADDR'] = 'localhost'  # 主机地址
# os.environ['MASTER_PORT'] = '1234'  # 主机端口

os.environ['MASTER_ADDR'] = '127.0.0.1'  # 主服务器的 IP 地址
os.environ['MASTER_PORT'] = '7001'  # 主服务器的端口
os.environ['WORLD_SIZE'] = '4'  # 总进程数
os.environ['RANK'] = '0'  # 当前进程的排名


def main():
    assert torch.cuda.is_available(), "need gpu to train network!"
    torch.cuda.empty_cache()
    print("start")

    log_dir = os.path.join(Cfg.work_dir, 'log')
    checkpoint_dir = os.path.join(Cfg.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    sys.stdout = Logger(os.path.join(Cfg.save_dir, 'train.log'))

    gpus_type = torch.cuda.get_device_name()
    gpus_num = torch.cuda.device_count()

    set_seed(Cfg.seed)

    local_rank = int(os.environ['LOCAL_RANK'])
    print("---")
    # start init process
    # torch.distributed.init_process_group(backend='nccl')
    # torch.cuda.set_device(local_rank)
    # torch.distributed.new_group(list(range(gpus_num)))
    #
    # if local_rank == 0:
    #     os.makedirs(
    #         checkpoint_dir) if not os.path.exists(checkpoint_dir) else None
    #     os.makedirs(log_dir) if not os.path.exists(log_dir) else None
    #
    # torch.distributed.barrier()

    batch_size, num_workers = Cfg.batch_size, Cfg.num_workers
    assert Cfg.batch_size % Cfg.gpus_num == 0, 'config.batch_size is not divisible by config.gpus_num!'
    assert Cfg.num_workers % Cfg.gpus_num == 0, 'config.num_workers is not divisible by config.gpus_num!'
    batch_size = int(Cfg.batch_size // Cfg.gpus_num)
    num_workers = int(Cfg.num_workers // Cfg.gpus_num)

    init_fn = functools.partial(worker_seed_init_fn,
                                num_workers=num_workers,
                                local_rank=local_rank,
                                seed=Cfg.seed)

    train_sampler = torch.utils.data.distributed.DistributedSampler(Cfg.train_dataset, shuffle=True)
    train_loader = DataLoader(Cfg.train_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              pin_memory=True,
                              drop_last=True,
                              num_workers=num_workers,
                              collate_fn=Cfg.train_collater,
                              sampler=train_sampler,
                              worker_init_fn=init_fn)
    test_loader = DataLoader(Cfg.test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             pin_memory=True,
                             num_workers=num_workers,
                             collate_fn=Cfg.test_collater)


if __name__ == "__main__":
    main()


