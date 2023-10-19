import os
import sys
import argparse
import functools
import time

import torch
import torch.distributed
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.coco import CocoDetection
from datasets.voc import VOCDetection
from datasets.transform import *
from datasets.collater import DetectionCollater
from utils.util import Logger, AverageMeter, set_seed, worker_seed_init_fn
from utils.optimizers import build_optimizer
from utils.schedulers import Scheduler
from utils.evaluate_coco import evaluate_coco_detection


def train_detection(train_loader: DataLoader, model, criterion, optimizer, config):
    losses = AverageMeter()
    model.train()

    iters = len(train_loader.dataset) // config.batch_size
    iter_index = 1

    for _, data in enumerate(train_loader):
        images, targets = data['image'], data['annots']
        images, targets = images.cuda(), targets.cuda()

        if torch.any(torch.isinf(images)) or torch.any(torch.isinf(targets)):
            continue

        if torch.any(torch.isnan(images)) or torch.any(torch.isnan(targets)):
            continue

        if torch.sum(images) == 0:
            continue

        outs_tuple = model(images)
        loss_value = criterion(outs_tuple, targets)

        loss = sum(loss_value.values())

        inf_nan_flag = False
        for k, v in loss_value.items():
            if torch.any(torch.isinf(v)) or torch.any(torch.isnan(v)):
                inf_nan_flag = True

        if torch.any(torch.isinf(loss)) or torch.any(torch.isnan(loss)):
            inf_nan_flag = True

        if loss == 0. or inf_nan_flag:
            optimizer.zero_grad()
            continue

        if hasattr(config, 'accumulation_steps') and config.accumulation_steps > 1:
            loss = loss / config.accumulation_steps

        loss.backward()

        losses.update(loss.item(), images.size(0))

        if hasattr(config, 'accumulation_steps') and config.accumulation_steps > 1:
            if iter_index % config.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        else:
            optimizer.step()
            optimizer.zero_grad()

        iter_index += 1

    avg_loss = losses.avg

    return avg_loss


def test_detection(test_loader: DataLoader, model, criterion, decoder, config):
    assert config.eval_type in ['COCO', 'VOC']

    func_dict = {
        'COCO': evaluate_coco_detection,
        # 'VOC': evaluate_voc_detection,
    }
    result_dict = func_dict[config.eval_type](test_loader, model, criterion,
                                              decoder, config)

    return result_dict


def main():
    assert torch.cuda.is_available(), "need gpu to train network!"
    import config as cfg
    torch.cuda.empty_cache()
    print("start")
    log_dir = os.path.join(cfg.work_dir, 'log')
    print("work_dir = ", cfg.work_dir)
    checkpoint_dir = os.path.join(cfg.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    sys.stdout = Logger(os.path.join(cfg.save_dir, 'train.log'))

    gpus_type = torch.cuda.get_device_name()
    gpus_num = torch.cuda.device_count()

    set_seed(0)

    # os.makedirs(
    #     checkpoint_dir) if not os.path.exists(checkpoint_dir) else None
    # os.makedirs(log_dir) if not os.path.exists(log_dir) else None

    batch_size, num_workers = cfg.batch_size, cfg.num_workers
    assert batch_size % gpus_num == 0, 'config.batch_size is not divisible by config.gpus_num!'
    assert num_workers % gpus_num == 0, 'config.num_workers is not divisible by config.gpus_num!'
    batch_size = int(batch_size // gpus_num)
    num_workers = int(num_workers // gpus_num)

    train_loader = DataLoader(cfg.train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True,
                              num_workers=num_workers,
                              collate_fn=cfg.train_collater)
    test_loader = DataLoader(cfg.test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             pin_memory=True,
                             num_workers=num_workers,
                             collate_fn=cfg.test_collater)
    model = cfg.model.cuda()
    decoder = cfg.decoder
    criterion = cfg.criterion.cuda()

    optimizer = build_optimizer(cfg.optimizer, model)
    scheduler = Scheduler(cfg)

    start_epoch, train_time = 0, 0
    best_metric, metric, test_loss = 0, 0, 0
    if os.path.exists(resume_model):
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        used_time = checkpoint['time']
        train_time += used_time

        best_metric = checkpoint['best_metric']
        test_loss = checkpoint['test_loss']
        lr = checkpoint['lr']

        print(f'resuming model from {resume_model}. '
              f'resume_epoch: {saved_epoch:0>3d}, '
              f'used_time: {used_time:.3f} hours, '
              f'best_metric: {best_metric:.3f}%, '
              f'test_loss: {test_loss}, '
              f'lr: {lr:.6f}')

    for epoch in range(start_epoch, cfg.epochs, 1):
        per_epoch_start_time = time.time()
        print(f'epoch {epoch:0>3d} lr: {scheduler.current_lr:.6f}')

        torch.cuda.empty_cache()
        train_loss = train_detection(train_loader,
                                     model,
                                     criterion,
                                     optimizer,
                                     cfg)
        print(f'train: epoch {epoch:0>3d}, train_loss: {train_loss:.4f}')
        torch.cuda.empty_cache()

        if epoch in cfg.eval_epoch or epoch == cfg.epochs:
            result_dict = test_detection(test_loader,
                                         model,
                                         criterion,
                                         decoder,
                                         cfg)
            print(f"eval: epoch: {epoch:0>3d}\n")

            for k, v in result_dict.items():
                if k == cfg.save_model_metric:
                    metric = v
                elif k == 'test_loss':
                    test_loss = v

        torch.cuda.empty_cache()

        train_time += (time.time() - per_epoch_start_time) / 3600

        if metric > best_metric and metric <= 100:
            best_metric = metric
            torch.save(model.module.state_dict(),
                       os.path.join(checkpoint_dir, 'best.pth'))

        torch.save(
            {
                'epoch': epoch,
                'time': train_time,
                'best_metric': best_metric,
                'test_loss': test_loss,
                'lr': scheduler.current_lr,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(checkpoint_dir, 'latest.pth'))

        print(f'until epoch: {epoch:0>3d}, best_metric: {best_metric:.3f}%')

    print(f'train done. model: {cfg.network}, '
          f'train time: {train_time:.3f} hours, '
          f'best_metric: {best_metric:.3f}%')


if __name__ == "__main__":
    main()
