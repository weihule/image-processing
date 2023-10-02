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
from datasets.voc import VOCDataset
from datasets.transform import *
from datasets.collater import DetectionCollater

# 设置环境变量
os.environ['LOCAL_RANK'] = '1'
os.environ['RANK'] = '0'  # 第一个进程的排名
os.environ['WORLD_SIZE'] = '4'  # 总共两个进程
os.environ['MASTER_ADDR'] = 'localhost'  # 主机地址
os.environ['MASTER_PORT'] = '12345'  # 主机端口

COCO2017_path = '/root/autodl-tmp/COCO2017'
VOCdataset_path = '/root/autodl-tmp/VOCdataset'


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

    input_image_size = [640, 640]
    train_dataset = CocoDetection(COCO2017_path,
                                  set_name='train2017',
                                  transform=transforms.Compose([
                                      RandomHorizontalFlip(prob=0.5),
                                      RandomCrop(prob=0.5),
                                      RandomTranslate(prob=0.5),
                                      YoloStyleResize(
                                          resize=input_image_size[0],
                                          divisor=32,
                                          stride=32,
                                          multi_scale=True,
                                          multi_scale_range=[0.8, 1.0]),
                                      Normalize(),
                                  ]))

    test_dataset = CocoDetection(COCO2017_path,
                                 set_name='val2017',
                                 transform=transforms.Compose([
                                     YoloStyleResize(
                                         resize=input_image_size[0],
                                         divisor=32,
                                         stride=32,
                                         multi_scale=False,
                                         multi_scale_range=[0.8, 1.0]),
                                     Normalize(),
                                 ]))
    train_collater = DetectionCollater()
    test_collater = DetectionCollater()


if __name__ == "__main__":
    cfg_dict = {
        "seed": 0
    }
    main(cfg_dict)


