import os
import sys
import argparse
import random
import shutil
import time
import warnings
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torchvision import transforms
from utils.custom_dataset import COCODataPrefetcher, collater
from utils.losses import RetinaLoss
from utils.retina_decode import RetinaNetDecoder
from network_files.model import RetinaNet
from config import Config
from utils.util import get_logger
from pycocotools.cocoeval import COCOeval

warnings.filterwarnings('ignore')


def parse_args():
    parse = argparse.ArgumentParser(
        description='PyTorch COCO Detection Training'
    )
    parse.add_argument('--lr', type=float, default=Config.lr)
    parse.add_argument('--epochs', type=int, default=Config.epochs)
    parse.add_argument('--batch_size', type=int, default=Config.batch_size)
    parse.add_argument('--pre_trained', type=bool, default=Config.pre_trained)
    parse.add_argument('--num_classes', type=int, default=Config.num_classes)
    parse.add_argument('--input_image_size', type=int, default=Config.input_image_size)
    parse.add_argument('--num_workers', type=int, default=Config.num_workers)
    parse.add_argument('--resume', type=str, default=Config.resume)
    parse.add_argument('--checkpoint_path', type=str, default=Config.checkpoint_path)
    parse.add_argument('--log', type=str, default=Config.log)
    parse.add_argument('--seed', type=int, default=Config.seed)

    return parse.parse_args()


def mian(logger, args):
    if not torch.cuda.is_available():
        raise Exception('need gpu to train network')

    torch.cuda.empty_cache()

    if args.seed is not None:
        random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True

    gpus = torch.cuda.device_count()
    logger.info(f'use {gpus} gpus')
    logger.info(f'args: {args} gpus')

    cudnn.benchmark = True
    cudnn.enabled = True
    start_time = time.time()

    # dataset and dataloader
    logger.info('start loading data')
    train_loader = DataLoader(Config.train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              collate_fn=collater)
    logger.info('finish loading data')


if __name__ == "__main__":
    args = parse_args()
    print(args.seed)

    print(torch.cuda.device_count())
