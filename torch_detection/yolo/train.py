import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import sys
import time
import argparse
import warnings
import numpy as np
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from torch_detection.utils.util import get_logger
from losses import YoloV4Loss
from config_yolo import Config
from network_files.yolov3 import darknet53_yolov3

warnings.filterwarnings('ignore')


def train(train_loader, model, criterion, optimizer, scheduler, epoch, logger, Config):
    conf_losses, cls_losses, reg_losses = list(), list(), list()
    losses_list = list()
    model.train()
    iters = len(train_loader.dataset) // Config.batch_size

    iter_index = 1
    train_bar = tqdm(train_loader)
    for datas in train_bar:
        images, annotations = datas['img'], datas['annot']
        images, annotations = images.cuda().float(), annotations.cuda()

        if Config.apex:
            scaler = amp.GradScaler()
            auto_cast = amp.autocast

            # 前向过程(model + loss)开启 autocast
            with auto_cast():
                obj_reg_cls_heads = model(images)
                # for i, p, j in zip(cls_heads, reg_heads, batch_anchors):
                #     print(i.shape, p.shape, j.shape)
                loss_dict = criterion(obj_reg_cls_heads, annotations)
                cls_loss, reg_loss, conf_loss = loss_dict['cls_loss'], loss_dict['reg_loss'], loss_dict['conf_loss']
                losses = cls_loss + reg_loss + conf_loss

                if losses == 0. or torch.any(torch.isinf(losses)) or torch.any(torch.isnan(losses)):
                    optimizer.zero_grad()
                    continue

            # 反向传播在autocast上下文之外
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            scaler.step(optimizer)
            scaler.update()
        else:
            obj_reg_cls_heads = model(images)
            loss_dict = criterion(obj_reg_cls_heads, annotations)
            cls_loss, reg_loss, conf_loss = loss_dict['cls_loss'], loss_dict['reg_loss'], loss_dict['conf_loss']
            losses = cls_loss + reg_loss + conf_loss
            if cls_losses == 0.0 or reg_loss == 0.0:
                continue
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            optimizer.zero_grad()

        conf_losses.append(conf_loss.item())
        cls_losses.append(cls_loss.item())
        reg_losses.append(reg_loss.item())
        losses_list.append(losses.item())

        if iter_index % Config.print_interval == 0:
            logger.info(
                f"train: epoch {epoch:3d}, iter [{iter_index:5d}, {iters:5d}], \
                conf_loss: {conf_loss.item():.2f}, cls_loss: {cls_loss.item():.2f}, reg_loss: {reg_loss.item():.2f}, \
                total_loss: {losses.item():.2f}"
            )
        iter_index += 1
        # print(f"epoch: {epoch}, iter_index: {iter_index}/{iters}")

        # break

    # scheduler.step()
    scheduler.step(np.mean(losses_list))

    return np.mean(conf_losses), np.mean(cls_losses), np.mean(reg_losses), np.mean(losses_list)


def main(logger):
    if not torch.cuda.is_available():
        raise Exception('need gpu to train network')

    if Config.seed is not None:
        random.seed(Config.seed)
        torch.cuda.manual_seed(Config.seed)
        torch.cuda.manual_seed_all(Config.seed)
        cudnn.deterministic = True

    gpus = torch.cuda.device_count()
    logger.info(f'use {gpus} gpus')
    logger.info(f'Config: {Config} gpus')

    cudnn.benchmark = True
    cudnn.enabled = True
    start_time = time.time()

    # dataset and dataloader
    logger.info('start loading data')
    train_loader = DataLoader(Config.train_dataset,
                              batch_size=Config.batch_size,
                              shuffle=True,
                              num_workers=Config.num_workers,
                              pin_memory=True,
                              collate_fn=Config.collater,
                              prefetch_factor=6)
    # train_loader = DataLoader(Config.val_dataset,
    #                           batch_size=Config.batch_size,
    #                           shuffle=False,
    #                           num_workers=Config.num_workers,
    #                           pin_memory=True,
    #                           collate_fn=Config.collater,
    #                           prefetch_factor=4)
    logger.info('finish loading data')

    pre_weight = '/workshop/weihule/data/weights/yolo/darknet53-acc76.836.pth'
    # pre_weight = None
    model = darknet53_yolov3(pre_weight)

    criterion = YoloV4Loss().cuda()

    model = model.cuda()
    # optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr)
    #
    # lf = lambda x: ((1 + math.cos(x * math.pi / Config.epochs)) / 2) * (1 - Config.lrf) + Config.lrf  # cosine
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           patience=3,
                                                           verbose=True)

    best_map = 0.0
    start_epoch = 1
    # resume training
    if os.path.exists(Config.resume):
        logger.info(f'start resuming model from {Config.resume}')
        checkpoint = torch.load(Config.resume, map_location=torch.device('cpu'))
        start_epoch += checkpoint['epoch']
        best_map = checkpoint['best_map']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info(
            f"finish resuming model from {Config.resume}, \
            epoch: {checkpoint['epoch']}, best_map: {checkpoint['best_map']}"

            f"loss: {checkpoint['loss']:3f}, \
            cls_loss: {checkpoint['cls_loss']:2f}, reg_loss: {checkpoint['reg_loss']:2f}"
        )

    if not os.path.exists(Config.checkpoint_path):
        os.mkdir(Config.checkpoint_path)

    logger.info('start training')
    print('start training...')
    for epoch in range(start_epoch, Config.epochs + 1):
        print(epoch)
        torch.cuda.empty_cache()
        conf_losses, cls_losses, reg_losses, losses = train(train_loader=train_loader,
                                                            model=model,
                                                            criterion=criterion,
                                                            optimizer=optimizer,
                                                            scheduler=scheduler,
                                                            epoch=epoch,
                                                            logger=logger,
                                                            Config=Config)
        logger.info(
            f"train: epoch {epoch:3d}, conf_loss: {conf_losses:.2f}, cls_loss: {cls_losses:.2f}, reg_loss: {reg_losses:.2f}, loss: {losses:.2f}"
        )
        torch.cuda.empty_cache()
        # break

        if epoch % 5 == 0 or epoch == Config.epochs:
            #         all_eval_result = evaluate_voc(Config.val_dataset, model, decoder)
            #         logger.info(f'eval done.')
            #         if all_eval_result is not None:
            #             logger.info(
            #                 f"val: epoch: {epoch:0>5d}, \
            #                     IoU=0.5:0.95,area=all,maxDets=100,mAP:{all_eval_result[0]:.3f}, \
            #                     IoU=0.5,area=all,maxDets=100,mAP:{all_eval_result[1]:.3f}, \
            #                     IoU=0.75,area=all,maxDets=100,mAP:{all_eval_result[2]:.3f}, \
            #                     IoU=0.5:0.95,area=small,maxDets=100,mAP:{all_eval_result[3]:.3f}, \
            #                     IoU=0.5:0.95,area=medium,maxDets=100,mAP:{all_eval_result[4]:.3f}, \
            #                     IoU=0.5:0.95,area=large,maxDets=100,mAP:{all_eval_result[5]:.3f}, \
            #                     IoU=0.5:0.95,area=all,maxDets=1,mAR:{all_eval_result[6]:.3f}, \
            #                     IoU=0.5:0.95,area=all,maxDets=10,mAR:{all_eval_result[7]:.3f}, \
            #                     IoU=0.5:0.95,area=all,maxDets=100,mAR:{all_eval_result[8]:.3f}, \
            #                     IoU=0.5:0.95,area=small,maxDets=100,mAR:{all_eval_result[9]:.3f}, \
            #                     IoU=0.5:0.95,area=medium,maxDets=100,mAR:{all_eval_result[10]:.3f}, \
            #                     IoU=0.5:0.95,area=large,maxDets=100,mAR:{all_eval_result[11]:.3f}"
            #             )
            #             if all_eval_result[0] > best_map:
            #                 torch.save(model.state_dict(), os.path.join(Config.checkpoints, 'best.pth'))
            #                 best_map = all_eval_result[0]
            #
            torch.save(
                {
                    'epoch': epoch,
                    'best_map': best_map,
                    'cls_loss': cls_losses,
                    'reg_loss': reg_losses,
                    'loss': losses,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, os.path.join(Config.checkpoint_path, 'latest.pth')
            )
    logger.info(f'finish training, best_map: {best_map:.3f}')
    training_time = (time.time() - start_time) / 3600
    print('finish training')
    logger.info(
        f'finish training, total training time: {training_time:.2f} hours'
    )


if __name__ == "__main__":
    logger = get_logger('yolo', Config.log)
    main(logger=logger)
