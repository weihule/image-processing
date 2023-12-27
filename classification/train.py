import os
import math
import random
import time
import yaml
from tqdm import tqdm

import torch
import torch.nn as nn
from pathlib import Path
import torch.nn.functional as F
from torch.backends import cudnn
from torch.optim import Adam, SGD, lr_scheduler
from torch.utils.data import DataLoader

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.device_count()

from datasets.data_manager import init_dataset
from datasets.transform import transform_func
from datasets.collater import Collater

from backbones.model_manager import init_model

from losses import CELoss
from utils.optimers import init_optimizer
from utils.schedulers import init_scheduler
from utils.util import get_logger, load_state_dict, make_dir


def run():
    # 解析并获取配置文件中的内容
    with open("cfg.yaml", "r", encoding='utf-8') as fr:
        cfgs = yaml.load(fr, Loader=yaml.FullLoader)

    # 日志和权重文件等的保存根路径
    total_save_root = cfgs[cfgs["mode"]]["save_root"]
    make_dir(total_save_root)

    # 各个模型对应的保存根目录
    model_name = cfgs["backbone_type"]
    save_root = os.path.join(total_save_root, model_name)
    make_dir(save_root)

    # 日志文件保存路径
    log = os.path.join(save_root, 'log')
    make_dir(log)

    # checkpoints文件保存路径
    checkpoints = os.path.join(save_root, 'checkpoints')
    make_dir(checkpoints)

    # 权重文件保存路径
    pth_path = os.path.join(save_root, 'pths')
    make_dir(pth_path)

    logger_writer = get_logger(name=model_name, log_dir=log)
    main(logger=logger_writer, cfgs=cfgs)


def train(cfgs, logger, model, train_loader, criterion, optimizer, scheduler, epoch, device):
    # 设置训练模式
    model.train()

    mean_loss = 0.
    iter_idx = 1
    for ds in tqdm(train_loader):
        images, labels = ds["image"], ds["label"]
        images, labels = images.to(device), labels.to(device)
        # 梯度清零
        optimizer.zero_grad()

        # [B, num_classes]
        preds = model(images)
        loss = criterion(preds, labels)
        mean_loss += loss.item()

        # 损失回传
        loss.backward()

        # 梯度更新
        optimizer.step()

        if iter_idx % cfgs["print_interval"] == 0 or iter_idx == len(train_loader):
            lr_value = optimizer.state_dict()['param_groups'][0]['lr']
            logger.info(
                f"train epoch {epoch:3d}, iter [{iter_idx:5d}, {len(train_loader)}], loss: {loss.item():.3f} lr:{lr_value}")
        iter_idx += 1

    mean_loss = mean_loss / len(train_loader)
    scheduler.step()

    return mean_loss


def evaluate_acc(model, val_dataset_len, val_loader, device):
    # 验证模式
    model.eval()
    model.to(device)
    correct = 0
    for ds in tqdm(val_loader):
        images, labels = ds["image"], ds["label"]
        images, labels = images.to(device), labels.to(device)
        preds = model(images)
        preds = F.softmax(preds, dim=-1)
        _, max_indices = torch.max(preds, dim=-1)
        correct += torch.eq(labels, max_indices).sum().item()
    val_acc = round(correct / val_dataset_len, 4)
    return val_acc


def main(cfgs, logger):
    torch.cuda.empty_cache()

    # 设置相同的随机种子, 确保实验结果可复现
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

    # 数据加载
    logger.info("start loading data")
    transform = transform_func(image_size=cfgs["image_size"],
                               use_random_erase=False)
    collater = Collater(mean=cfgs["mean"],
                        std=cfgs["std"])
    train_dataset = init_dataset(name=cfgs["dataset_name"],
                                 root_dir=cfgs[cfgs["mode"]]["root_dir"],
                                 set_name=cfgs["train_set_name"],
                                 class_file=cfgs[cfgs["mode"]]["class_file"],
                                 transform=transform["train"])
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=cfgs["batch_size"],
                              shuffle=True,
                              num_workers=cfgs["num_workers"],
                              collate_fn=collater)

    val_dataset = init_dataset(name=cfgs["dataset_name"],
                               root_dir=cfgs[cfgs["mode"]]["root_dir"],
                               set_name=cfgs["val_set_name"],
                               class_file=cfgs[cfgs["mode"]]["class_file"],
                               transform=transform["val"])
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=cfgs["batch_size"],
                            shuffle=False,
                            num_workers=cfgs["num_workers"],
                            collate_fn=collater)

    # 设置模型
    model = init_model(backbone_type=cfgs["backbone_type"],
                       num_classes=cfgs["num_classes"])
    model.to(device)

    # 载入预训练权重
    load_state_dict(saved_model_path=cfgs[cfgs["mode"]]["pre_weight_path"],
                    model=model)

    # 设置优化器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = init_optimizer(optim="sgd",
                               params=params,
                               lr=cfgs["lr"],
                               weight_decay=cfgs["weight_decay"],
                               momentum=cfgs["momentum"])
    scheduler = init_scheduler(scheduler="cosine_annealing_lr",
                               optimizer=optimizer,
                               T_max=cfgs["epochs"])

    # 损失函数
    criterion = CELoss(use_custom=False)

    best_acc = 0.
    start_epoch = 1

    # 断点重续
    resume = os.path.join(cfgs[cfgs["mode"]]["save_root"],
                          cfgs["backbone_type"],
                          "checkpoints",
                          "resume.pth")
    if os.path.exists(resume):
        logger.info(f"start resume model from {resume}")
        checkpoint = torch.load(resume, map_location=torch.device('cpu'))
        start_epoch += checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info(f"epoch {checkpoint['epoch']}, best_acc: {checkpoint['best_acc']}, loss: {checkpoint['loss']}")
        logger.info('finish resume model !')

    pths_dir = os.path.join(cfgs[cfgs["mode"]]["save_root"],
                            cfgs["backbone_type"],
                            "pths")
    logger.info(f"Starting training from the {start_epoch} epoch")
    for epoch in range(start_epoch, cfgs["epochs"] + 1):
        mean_loss = train(cfgs=cfgs,
                          logger=logger,
                          model=model,
                          train_loader=train_loader,
                          criterion=criterion,
                          optimizer=optimizer,
                          scheduler=scheduler,
                          epoch=epoch,
                          device=device)
        logger.info(f"train: epoch: {epoch}, loss: {mean_loss:.3f}")
        if epoch % cfgs["save_interval"] == 0 or epoch == cfgs["epochs"]:
            val_acc = evaluate_acc(model=model,
                                   val_dataset_len=len(val_dataset),
                                   val_loader=val_loader,
                                   device=device)
            logger.info(f"epoch = {epoch}, val_acc = {val_acc}")
            print('epoch: {}  mean_loss: {:.3f} val_acc: {:.3f}%'.format(epoch, mean_loss, val_acc * 100))

            if val_acc > best_acc:
                # 先删除历史权重
                for i in Path(pths_dir).glob("*.pth"):
                    i.unlink(missing_ok=True)
                best_acc = val_acc
                best_weight_name = cfgs["backbone_type"] + "-" + str(best_acc) + ".pth"
                best_weight_path = os.path.join(pths_dir,
                                                best_weight_name)
                torch.save(model.state_dict(), best_weight_path)

                torch.save({
                    'epoch': epoch,
                    'best_acc': best_acc,
                    'loss': mean_loss,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()
                }, resume)
        train_time = (time.time() - start_time) / 60
        logger.info(f'finish training, total training time: {train_time:.2f} mins')


if __name__ == "__main__":
    run()
