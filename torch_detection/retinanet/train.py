import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '4'
import sys
import argparse
import random
import shutil
import time
import math
import warnings
import json

import numpy as np
import torch
import torch.nn as nn
from thop import profile
from thop import clever_format
# from apex import amp
from torch.cuda import amp
from torch.cuda.amp import GradScaler,autocast
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
    parse.add_argument('--lrf', type=float, default=Config.lrf)
    parse.add_argument('--epochs', type=int, default=Config.epochs)
    parse.add_argument('--batch_size', type=int, default=Config.batch_size)
    parse.add_argument('--pre_trained', type=bool, default=Config.pre_trained)
    parse.add_argument('--num_classes', type=int, default=Config.num_classes)
    parse.add_argument('--input_image_size', type=int, default=Config.input_image_size)
    parse.add_argument('--num_workers', type=int, default=Config.num_workers)
    parse.add_argument('--resume', type=str, default=Config.resume)
    parse.add_argument('--checkpoint_path', type=str, default=Config.checkpoint_path)
    parse.add_argument('--log', type=str, default=Config.log)
    parse.add_argument('--evaluate', type=str, default=Config.evaluate)
    parse.add_argument('--seed', type=int, default=Config.seed)
    parse.add_argument('--apex', type=bool, default=Config.apex)
    parse.add_argument('--print_interval', type=int, default=Config.print_interval)

    return parse.parse_args()


def validate(val_dataset, model, decoder):
    # model = model.module
    model.eval()
    with torch.no_grad():
        all_eval_result = evaluate_coco(val_dataset, model, decoder)


def evaluate_coco(val_dataset, model, decoder):
    results, image_ids = list(), list()
    for index in range(len(val_dataset)):
        datas = val_dataset[index]
        scale = datas['scale']
        val_inputs = torch.from_numpy(datas['img']).cuda().permute(2, 0, 1).unsqueeze(dim=0)
        print(index, val_inputs.shape)
        cls_heads, reg_heads, batch_anchors = model(val_inputs)

        # for pred_cls_head, pred_reg_head, pred_batch_anchor in zip(
        #         cls_heads, reg_heads, batch_anchors
        # ):
        #     print(pred_cls_head.shape, pred_reg_head.shape, pred_batch_anchor.shape)
        scores, classes, boxes = decoder(cls_heads, reg_heads, batch_anchors)
        scores, classes, boxes = scores.cpu(), classes.cpu(), boxes.cpu()
        boxes /= scale

        # make sure decode batch_size is 1
        # scores shape: [1, max_detection_num]
        # classes shape: [1, max_detection_num]
        # boxes shape: [1, max_detection_num, 4]
        if scores.shape[0] != 1:
            raise ValueError('batch_size num expected 1, but got {}'.format(scores.shape[0]))
        scores = scores.squeeze(0)
        classes = classes.squeeze(0)
        boxes = boxes.squeeze(0)

        # for coco_eval,we need [x_min,y_min,w,h] format pred boxes
        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

        # 每个样本(anchor)记录一次
        for object_score, object_class, object_box in zip(scores, classes, boxes):
            object_score = float(object_score)
            object_class = float(object_class)
            object_box = object_box.tolist()

            if object_class == -1:
                continue

            image_result = {
                'image_id': val_dataset.image_ids[index],
                'category_id': val_dataset.find_category_id_from_coco_label(object_class),
                'score': object_score,
                'bbox': object_box
            }
            results.append(image_result)
        # 每张图片记录一次
        image_ids.append(val_dataset.image_ids[index])

        print('{}/{}'.format(index, len(val_dataset)))

    if len(results) == 0:
        print('No target detected in test set images')
        return

    json.dump(results, open('{}_box_results.json'.format(val_dataset.set_name), 'w'), indent=4)

    # load results in COCO evaluation tool
    coco = val_dataset.coco
    coco_pred = coco.loadRes('{}_box_results.json'.format(val_dataset.set_name))
    coco_eval = COCOeval(coco, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    all_eval_result = coco_eval.stats

    return all_eval_result


def train(train_loader, model, criterion, optimizer, scheduler, epoch, logger, args, scaler):
    cls_losses, reg_losses, losses = list(), list(), list()
    model.train()
    iters = len(train_loader.dataset) // args.batch_size
    pre_fetcher = COCODataPrefetcher(train_loader)
    images, annotations = pre_fetcher.next()
    iter_index = 1

    while images is not None:
        images, annotations = images.cuda().float(), annotations.cuda()
        cls_heads, reg_heads, batch_anchors = model(images)
        cls_loss, reg_loss = criterion(cls_heads, reg_heads, batch_anchors, annotations)
        loss = cls_loss + reg_loss
        if cls_losses == 0.0 or reg_loss == 0.0:
            optimizer.zero_grad()
            continue

        if args.apex:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            optimizer.zero_grad()

        cls_losses.append(cls_loss.item())
        reg_losses.append(reg_loss.item())
        losses.append(loss.item())

        images, annotations = pre_fetcher.next()

        if iter_index % args.print_interval == 0:
            logger.info(
                f"train: epoch {epoch:3d}, iter [{iter_index:5d}, {iters:5d}], \
                cls_loss: {cls_loss.item():.2f}, reg_loss: {reg_loss.item():.2f}, total_loss: {loss.item():.2f}"
            )
        iter_index += 1

    scheduler.step()

    return np.mean(cls_losses), np.mean(reg_losses), np.mean(losses)


def main(logger, args):
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

    model = RetinaNet(num_classes=args.num_classes)

    flops_input = torch.rand(1, 3, args.input_image_size, args.input_image_size)
    flops, params = profile(model, inputs=(flops_input, ))
    flops, params = clever_format([flops, params], '%.3f')

    criterion = RetinaLoss(image_w=args.input_image_size,
                           image_h=args.input_image_size,
                           ).cuda()
    decoder = RetinaNetDecoder(image_w=args.input_image_size,
                               image_h=args.input_image_size).cuda()
    model = model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = GradScaler()

    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # if args.apex:
    #     model, optimizer = amp.

    best_map = 0.0
    start_epoch = 1
    # resume training
    if os.path.exists(args.resume):
        logger.info(f'start resuming model from {args.resume}')
        checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
        start_epoch += checkpoint['epoch']
        best_map = checkpoint['best_map']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info(
            f"finish resuming model from {args.resume}, \
            epoch: {checkpoint['epoch']}, best_map: {checkpoint['best_map']}"
            
            f"loss: {checkpoint['loss']:3f}, \
            cls_loss: {checkpoint['cls_loss']:2f}, reg_loss: {checkpoint['reg_loss']:2f}"
        )

    if not os.path.exists(args.checkpoint_path):
        os.mkdir(args.checkpoint_path)

    logger.info('start training')
    for epoch in range(start_epoch, args.epochs + 1):
        cls_losses, reg_losses, losses = train(train_loader=train_loader,
                                                model=model,
                                                criterion=criterion,
                                                optimizer=optimizer,
                                                scheduler=scheduler,
                                                epoch=epoch,
                                                logger=logger,
                                                args=args,
                                                scaler=scaler)
        logger.info(
            f"train: epoch {epoch:3d}, cls_loss: {cls_losses:.2f}, reg_loss: {reg_losses:.2f}, loss: {losses:.2f}"
        )

        if epoch % 5 == 0 or epoch == args.epochs:
            all_eval_result = validate(Config.val_dataset, model, decoder)
            logger.info(f'eval done.')



if __name__ == "__main__":
    inputs1 = torch.rand(4, 3, 640, 640).cuda()
    inputs2 = torch.rand(4, 3, 640, 640).cuda()
    inputs = [inputs1, inputs2]

    args = parse_args()
    model = RetinaNet(num_classes=args.num_classes)
    model = model.cuda()
    reti_decoder = RetinaNetDecoder(image_w=args.input_image_size, image_h=args.input_image_size)

    res = evaluate_coco(Config.val_dataset, model, reti_decoder)
    print(res)

    # for inp in inputs:
    #     print('hhh')
    #     pred_cls_heads, pred_reg_heads, pred_batch_anchors = model(inp)
    #     for pred_cls_head, pred_reg_head, pred_batch_anchor in zip(
    #             pred_cls_heads, pred_reg_heads, pred_batch_anchors
    #     ):
    #         print(pred_cls_head.shape, pred_reg_head.shape, pred_batch_anchor.shape)

