import os
import sys
import argparse
import random
import time
import warnings
import cv2
import numpy as np
import torch
from torch.cuda import amp
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from losses import RetinaLoss
from retina_decode import RetinaNetDecoder
from config import Config
from network_files.retinanet_model import resnet50_retinanet

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(BASE_DIR)

from torch_detection.utils.custom_dataset import DataPrefetcher, collater
from torch_detection.utils.util import get_logger

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


def train(train_loader, model, criterion, optimizer, scheduler, epoch, logger, args):
    cls_losses, reg_losses, losses = list(), list(), list()
    model.train()
    iters = len(train_loader.dataset) // args.batch_size
    pre_fetcher = DataPrefetcher(train_loader)
    images, annotations = pre_fetcher.next()

    iter_index = 1
    # train_bar = tqdm(train_loader)
    # for datas in train_bar:
    while images is not None:
        # images, annotations = datas['img'], datas['annot']
        # print(images.shape, annotations.shape)
        images, annotations = images.cuda().float(), annotations.cuda()
        # print('in train', images.shape, annotations.shape)
        optimizer.zero_grad()

        if args.apex:
            scaler = amp.GradScaler()
            auto_cast = amp.autocast
            with auto_cast():
                cls_heads, reg_heads, batch_anchors = model(images)
                # for i, p, j in zip(cls_heads, reg_heads, batch_anchors):
                #     print(i.shape, p.shape, j.shape)
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

        if iter_index % args.print_interval == 0:
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
                              pin_memory=True,
                              collate_fn=collater)
    logger.info('finish loading data')

    pre_train = '/workshop/weihule/data/weights/resnet/resnet50-acc76.322.pth'
    if not os.path.exists(pre_train):
        pre_train = '/nfs/home57/weihule/data/weights/resnet/resnet50-acc76.322.pth'

    model = resnet50_retinanet(num_classes=20, pre_train=pre_train)

    # flops_input = torch.rand(1, 3, args.input_image_size, args.input_image_size)
    # flops, params = profile(model, inputs=(flops_input,))
    # flops, params = clever_format([flops, params], '%.3f')
    # logger.info(f"model: resnet50_backbone, flops: {flops}, params: {params}")

    criterion = RetinaLoss(image_w=args.input_image_size,
                           image_h=args.input_image_size,
                           ).cuda()
    decoder = RetinaNetDecoder(image_w=args.input_image_size,
                               image_h=args.input_image_size).cuda()
    model = model.cuda()
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    #
    # lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           patience=3,
                                                           verbose=True)

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
    print('start training...')
    for epoch in range(start_epoch, args.epochs + 1):
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

        if epoch % 5 == 0 or epoch == args.epochs:
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
    #                 torch.save(model.state_dict(), os.path.join(args.checkpoints, 'best.pth'))
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
                }, os.path.join(args.checkpoint_path, 'latest.pth')
            )
    logger.info(f'finish training, best_map: {best_map:.3f}')
    training_time = (time.time() - start_time) / 3600
    print('finish training')
    logger.info(
        f'finish training, total training time: {training_time:.2f} hours'
    )


def test_make_grid():
    val_loader = DataLoader(Config.train_dataset,
                            batch_size=16,
                            shuffle=False,
                            num_workers=2,
                            prefetch_factor=4,
                            pin_memory=True,
                            collate_fn=collater)
    mean = np.array([[[0.471, 0.448, 0.408]]])
    std = np.array([[[0.234, 0.239, 0.242]]])
    # datas是一个dict, key就是定义的三个key, 对应的value已经打了batch
    datas = next(iter(val_loader))
    batch_annots = datas['annot']
    batch_images = datas['img']

    c = 0
    for img, annot in zip(batch_images, batch_annots):
        c += 1
        img, annot = img.numpy(), annot.numpy()
        img = img.transpose(1, 2, 0)    # [c, h, w] -> [h, w, c] RGB

        img = (img * std + mean) * 255.
        img = np.uint8(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)   # RGB -> BGR
        for point in annot:
            point = np.int32(point[:4])
            cv2.rectangle(img, [point[0], point[1]], [point[2], point[3]], (0, 255, 0), 1)
        cv2.imwrite(str(c)+'.png', img)

    # img = make_grid(batch_images, nrow=8)
    # img = img.numpy()
    # img = img.transpose(1, 2, 0)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # cv2.imwrite('test.jpg', img)


if __name__ == "__main__":
    # args = parse_args()
    # logger = get_logger(__name__, args.log)
    # main(logger=logger, args=args)

    test_make_grid()

    # range_loader = tqdm(range(10000))
    # c = 0
    # for i in range_loader:
    #     c = i
    #     time.sleep(0.01)
    # range_loader.desc = f'this is {c}'

    # flops, params = profile(model, inputs=(inputs[0],))
    # flops, params = clever_format([flops, params], '%.3f')

    # reti_decoder = RetinaNetDecoder(image_w=args.input_image_size, image_h=args.input_image_size)
    #
    # res = evaluate_coco(Config.val_dataset, model, reti_decoder)
    # print(res)

    # for inp in inputs:
    #     pred_cls_heads, pred_reg_heads, pred_batch_anchors = model(inp)
    #     for pred_cls_head, pred_reg_head, pred_batch_anchor in zip(
    #             pred_cls_heads, pred_reg_heads, pred_batch_anchors
    #     ):
    #         print(pred_cls_head.shape, pred_reg_head.shape, pred_batch_anchor.shape)
