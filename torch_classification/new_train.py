import os
import math
import random
import time
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import math

import torch
import torch.nn as nn
from torch.cuda import amp
import torch.nn.functional as F
from torch.backends import cudnn
from torch.optim import Adam, SGD, lr_scheduler
from torch.utils.data import DataLoader

from losses import CELoss
from datasets.data_manager import init_dataset
from backbones.model_manager import init_model
from utils.util import get_logger, load_state_dict
from utils.optimers import init_optimizer
from utils.schedulers import init_scheduler
from new_config import cfg


def train(logger, model, train_loader, criterion, optimizer, scheduler, epoch, device):
    model.train()
    mean_loss = 0.00
    iter_idx = 1
    for datas in tqdm(train_loader):
        images, labels = datas['image'],  datas['label']
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        preds = model(images)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        mean_loss += loss.item()

    mean_loss = mean_loss / len(train_loader)
    scheduler.step()

    return mean_loss


def evaluate_acc(model, val_loader, device):
    model.eval()
    model = model.to(device)
    correct = 0.
    all_samples = 0.
    with torch.no_grad():
        for datas in tqdm(val_loader, desc="val"):
            images, labels = datas['image'], datas['label']
            images, labels = images.to(device), labels.to(device)
            preds = model(images)
            preds = F.softmax(preds, dim=-1)
            _, max_indices = torch.max(preds, dim=1)
            correct += torch.eq(max_indices, labels).sum().item()
            all_samples += len(labels)
        val_acc = round((correct / all_samples), 3)

    return val_acc


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

    train_dataset, val_dataset = init_dataset(name=cfg.dataset_name,
                                              root_dir=cfg.dataset_path,
                                              transform_dict=cfg.transform_dict)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=cfg.batch_size,
                              shuffle=True,
                              num_workers=4,
                              collate_fn=cfg.collater)

    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=cfg.batch_size,
                            shuffle=False,
                            num_workers=4,
                            collate_fn=cfg.collater)

    model = init_model(backbone_type=cfg.backbone_type,
                       num_classes=cfg.num_classes)
    model = model.to(device)

    load_state_dict(saved_model_path=cfg.pre_weight_path,
                    model=model)

    if cfg.freeze_layer:
        for name, param in model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = init_optimizer(optim="adam",
                               params=params,
                               lr=cfg.lr,
                               weight_decay=4E-5)
    scheduler = init_scheduler(scheduler="cosine_annealing_lr",
                               optimizer=optimizer,
                               step_size=cfg.step_size,
                               gamma=cfg.gamma,
                               T_max=cfg.T_max)
    criterion = CELoss(use_custom=False)

    best_acc = 0.
    start_epoch = 1

    # resume training
    if os.path.exists(cfg.resume):
        logger.info(f'start resume model from {cfg.resume}')
        checkpoint = torch.load(cfg.resume, map_location=torch.device('cpu'))
        start_epoch += checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info(f'finish resume model from {cfg.resume}')
        logger.info(f"epoch {checkpoint['epoch']}, best_acc: {checkpoint['best_acc']}, loss: {checkpoint['loss']}")
        logger.info(f'finish resume model from {cfg.resume}')

    print('start_epoch = ', start_epoch)
    for epoch in range(start_epoch, cfg.epochs+1):
        mean_loss = train(logger=logger,
                          model=model,
                          train_loader=train_loader,
                          criterion=criterion,
                          optimizer=optimizer,
                          scheduler=scheduler,
                          epoch=epoch,
                          device=device)
        logger.info(f"train: epoch: {epoch}, loss: {mean_loss:.3f}")
        if epoch % cfg.save_interval == 0 or epoch == cfg.epochs:
            val_acc = evaluate_acc(model, val_loader, device)
            logger.info(f"epoch {epoch}, val_acc {val_acc}")

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), os.path.join(cfg.save_root, 'pths', str(best_acc)+'.pth'))

            torch.save({
                'epoch': epoch,
                'best_acc': best_acc,
                'loss': mean_loss,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }, os.path.join(cfg.checkpoints, 'latest.pth'))
    train_time = (time.time() - start_time) / 60
    logger.info(f'finish training, total training time: {train_time:.2f} mins')


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
