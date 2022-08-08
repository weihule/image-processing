import os
import sys
import random
import time
import warnings
import numpy as np
from tqdm import tqdm
import torch
from torch.cuda import amp
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from losses import RetinaLoss
from retina_decode import RetinaNetDecoder
from network_files.retinanet_model import resnet50_retinanet

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(BASE_DIR)

from torch_detection.retinanet.config import Config
from torch_detection.utils.custom_dataset import collater
from torch_detection.utils.util import get_logger

warnings.filterwarnings('ignore')


def train(train_loader, model, criterion, optimizer, scheduler, epoch, logger, args):
    cls_losses, reg_losses, losses = list(), list(), list()
    model.train()
    iters = len(train_loader.dataset) // Config.batch_size

    iter_index = 1
    train_bar = tqdm(train_loader)
    for datas in train_bar:
        images, annotations = datas['img'], datas['annot']
        # print(images.shape, annotations.shape)
        images, annotations = images.cuda().float(), annotations.cuda()
        optimizer.zero_grad()

        if Config.apex:
            scaler = amp.GradScaler()
            auto_cast = amp.autocast
            with auto_cast():
                reg_cls_heads = model(images)
                cls_loss, reg_loss = criterion(cls_heads, reg_heads, batch_anchors, annotations)
                loss = cls_loss + reg_loss
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            scaler.step(optimizer)
            scaler.update()
        else:
            cls_heads, reg_heads, batch_anchors = model(images)
            cls_loss, reg_loss = criterion(cls_heads, reg_heads, batch_anchors, annotations)
            loss = cls_loss + reg_loss
            if cls_losses == 0.0 or reg_loss == 0.0:
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

        images, annotations = pre_fetcher.next()

        cls_losses.append(cls_loss.item())
        reg_losses.append(reg_loss.item())
        losses.append(loss.item())

        if iter_index % Config.print_interval == 0:
            logger.info(
                f"train: epoch {epoch:3d}, iter [{iter_index:5d}, {iters:5d}], \
                cls_loss: {cls_loss.item():.2f}, reg_loss: {reg_loss.item():.2f}, total_loss: {loss.item():.2f}"
            )
        iter_index += 1
        print(f"epoch: {epoch}, iter_index: {iter_index}/{iters}")

        # break

    # scheduler.step()
    scheduler.step(np.mean(losses))

    return np.mean(cls_losses), np.mean(reg_losses), np.mean(losses)


def main(logger):
    if not torch.cuda.is_available():
        raise Exception('need gpu to train network')

    torch.cuda.empty_cache()

    if Config.seed is not None:
        random.seed(Config.seed)
        torch.cuda.manual_seed(Config.seed)
        torch.cuda.manual_seed_all(Config.seed)
        cudnn.deterministic = True

    gpus = torch.cuda.device_count()
    logger.info(f'use {gpus} gpus')

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
                              collate_fn=collater)
    logger.info('finish loading data')

    pre_train = '/workshop/weihule/data/weights/resnet/resnet50-acc76.322.pth'
    if not os.path.exists(pre_train):
        pre_train = '/nfs/home57/weihule/data/weights/resnet/resnet50-acc76.322.pth'

    model = resnet50_retinanet(num_classes=20, pre_train=pre_train)

    # flops_input = torch.rand(1, 3, Config.input_image_size, Config.input_image_size)
    # flops, params = profile(model, inputs=(flops_input,))
    # flops, params = clever_format([flops, params], '%.3f')
    # logger.info(f"model: resnet50_backbone, flops: {flops}, params: {params}")

    criterion = RetinaLoss(image_w=Config.input_image_size,
                           image_h=Config.input_image_size,
                           ).cuda()
    decoder = RetinaNetDecoder(image_w=Config.input_image_size,
                               image_h=Config.input_image_size).cuda()
    model = model.cuda()
    # optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr)
    #
    # lf = lambda x: ((1 + math.cos(x * math.pi / Config.epochs)) / 2) * (1 - Config.lrf) + Config.lrf  # cosine
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           patience=3,
                                                           verbose=True)

    # if Config.apex:
    #     model, optimizer = amp.

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
        cls_losses, reg_losses, losses = train(train_loader=train_loader,
                                               model=model,
                                               criterion=criterion,
                                               optimizer=optimizer,
                                               scheduler=scheduler,
                                               epoch=epoch,
                                               logger=logger,
                                               args=args)
        logger.info(
            f"train: epoch {epoch:3d}, cls_loss: {cls_losses:.2f}, reg_loss: {reg_losses:.2f}, loss: {losses:.2f}"
        )
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
    logger_writer = get_logger(__name__, Config.log)
    main(logger=logger_writer)

    # range_loader = tqdm(range(10000))
    # c = 0
    # for i in range_loader:
    #     c = i
    #     time.sleep(0.01)
    # range_loader.desc = f'this is {c}'

    # flops, params = profile(model, inputs=(inputs[0],))
    # flops, params = clever_format([flops, params], '%.3f')

    # reti_decoder = RetinaNetDecoder(image_w=Config.input_image_size, image_h=Config.input_image_size)
    #
    # res = evaluate_coco(Config.val_dataset, model, reti_decoder)
    # print(res)

    # for inp in inputs:
    #     pred_cls_heads, pred_reg_heads, pred_batch_anchors = model(inp)
    #     for pred_cls_head, pred_reg_head, pred_batch_anchor in zip(
    #             pred_cls_heads, pred_reg_heads, pred_batch_anchors
    #     ):
    #         print(pred_cls_head.shape, pred_reg_head.shape, pred_batch_anchor.shape)
