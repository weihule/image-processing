import os
import math
import random
import time
import yaml
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.cuda import amp
import torch.nn.functional as F
from torch.backends import cudnn
from torch.optim import Adam, SGD, lr_scheduler
from torch.utils.data import DataLoader

from datasets.data_manager import init_dataset
from datasets.transform import transform_func
from datasets.collater import Collater

from backbones.model_manager import init_model

from losses import CELoss
from utils.optimers import init_optimizer
from utils.schedulers import init_scheduler
from utils.util import get_logger, load_state_dict

with open("cfg.yaml", "r", encoding="utf-8") as fr:
    cfgs = yaml.load(fr, Loader=yaml.FullLoader)


def main(logger):
    torch.cuda.empty_cache()

    random.seed(cfgs["seed"])
    torch.manual_seed(cfgs["seed"])
    torch.cuda.manual_seed(cfgs["seed"])
    torch.cuda.manual_seed_all(cfgs["seed"])
    cudnn.deterministic = True

    gpus = torch.cuda.device_count()
    logger.info(f'use {gpus} gpus')

    cudnn.benchmark = True
    cudnn.enabled = True
    start_time = time.time()

    device = torch.device(cfgs["device"])

    logger.info("start loading data")
    transform = transform_func(image_size=cfgs["image_size"],
                               use_random_erase=True)
    train_dataset = init_dataset(name=cfgs["dataset_name"],
                                 root_dir=cfgs[cfgs["mode"]]["root_dir"],
                                 set_name=cfgs["train_set_name"],
                                 class_file=cfgs[cfgs["mode"]]["class_file"],
                                 transform=transform["train"])
    print(len(train_dataset))
    collater = Collater(mean=cfgs["mean"],
                        std=cfgs["std"])
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=cfgs["batch_size"],
                              shuffle=True,
                              num_workers=cfgs["num_workers"],
                              collate_fn=collater)

    model = init_model(backbone_type=cfgs["backbone_type"],
                       num_classes=cfgs["num_classes"])
    model.to(device)

    load_state_dict(saved_model_path=cfgs[cfgs["mode"]]["pre_weight_path"],
                    model=model)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(params, lr=cfgs["lr"], weight_decay=4e-5)


if __name__ == "__main__":
    if not os.path.exists(r"D:\Desktop\test"):
        os.mkdir(r"D:\Desktop\test")
    logger_writer = get_logger('test', r"D:\Desktop\test")
    main(logger_writer)


