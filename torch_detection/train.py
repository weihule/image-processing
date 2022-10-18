import argparse
import datetime
import os
import random
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.cuda import amp
from torch.utils.data import DataLoader
from torchvision import transforms

import datasets
import losses
import models
import metrics
from models.decoder import RetinaDecoder
from datasets.collater_func import MultiScaleCollater
from datasets.data_transfrom import RandomFlip
from util.avgmeter import AverageMeter
from util.optimizers import init_optimizer
from util.schedulers import init_scheduler, adjust_learning_rate
from util.utils import save_checkpoints


def parse_args():
    parser = argparse.ArgumentParser(description='some orders')
    # Datasets
    parser.add_argument('--dataset_name', type=str, default='voc')
    parser.add_argument('--num_classes', type=int, default=80)
    parser.add_argument('--root_dir', type=str)
    parser.add_argument('--image_root_dir', type=str)
    parser.add_argument('--annotation_root_dir', type=str)
    parser.add_argument('-j', '--workers', default=4, type=int)
    parser.add_argument('--resize', type=int, default=640)
    parser.add_argument('--pin_memory', type=bool, default=True)

    # Optimization options
    parser.add_argument('--optim', type=str, default='adam', help="optimization algorithm (see optimizers.py)")
    parser.add_argument('--max_epoch', default=50, type=int, help="maximum epochs to run")
    parser.add_argument('--start_epoch', default=0, type=int, help="manual epoch number (useful on restarts)")
    parser.add_argument('--train_batch', default=16, type=int, help="train batch size")
    parser.add_argument('--test_batch', default=16, type=int, help="test batch size")
    parser.add_argument('--lr', '--learning_rate', default=0.0001, type=float, help="initial learning rate")
    parser.add_argument('--step_size', default=20, type=int,
                        help="stepsize to decay learning rate (>0 means this is enabled)")
    parser.add_argument('--gamma', default=0.1, type=float, help="learning rate decay")
    parser.add_argument('--weight_decay', default=5e-04, type=float, help="weight decay")

    # Architecture
    parser.add_argument('-a', '--arch', type=str, default='resnet50_retinanet', choices=models.get_names())
    parser.add_argument('--pre_train_load_dir', type=str, default=None, help='save dir of pretrain weight')
    parser.add_argument('--apex', action='store_true')
    parser.add_argument('--num_workers', type=int, default=4)

    # Miscs
    parser.add_argument('--print_freq', type=int, default=50, help="print frequency")
    parser.add_argument('--seed', type=int, default=3407, help="manual seed")
    parser.add_argument('--resume', type=str, default='checkpoint.pth')
    parser.add_argument('--eval_step', type=int, default=-1,
                        help="run evaluation for every N epochs (set to -1 to test after training)")
    parser.add_argument('--start_eval', type=int, default=0, help="start to evaluate after specific epoch")
    parser.add_argument('--save_dir', type=str, default='log')
    parser.add_argument('--use_mosaic', action='store_true')
    parser.add_argument('--use_cpu', action='store_true', help="use cpu")
    parser.add_argument('--use_ddp', action='store_true', help="use multiple devices")
    parser.add_argument('--gpu_devices', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')

    # Eval
    parser.add_argument('--eval_voc_iou_threshold_list', type=list,
                        default=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
    parser.add_argument('--only_evaluate', action='store_true')

    args = parser.parse_args()
    return args


def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu:
        use_gpu = False

    # sys.stdout = Logger(os.path.join(args.save_dir, 'train.log'))
    print(f'==============\nArgs:{args}\n==============')
    cur_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    if use_gpu:
        print(f'Currently using GPU {args.gpu_devices}')
        # 如果网络的输入数据维度或类型上变化不大，设置 torch.backends.cudnn.benchmark = true 可以增加运行效率
        # cudnn.enabled = False
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        print('Currently using CPU (GPU is highly recommended)')

    data_transform = {
        'train': transforms.Compose([
            RandomFlip(flip_prob=0.5)
        ]),
        'test': None
    }

    train_dataset = datasets.init_dataset(args.dataset_name,
                                          root_dir=args.root_dir,
                                          image_root_dir=args.image_root_dir,
                                          annotation_root_dir=args.annotation_root_dir,
                                          resize=args.resize,
                                          use_mosaic=args.use_mosaic,
                                          transform=data_transform['train'])

    test_dataset = datasets.init_dataset(args.dataset_name,
                                         root_dir=args.root_dir,
                                         image_root_dir=args.image_root_dir,
                                         annotation_root_dir=args.annotation_root_dir,
                                         resize=args.resize,
                                         use_mosaic=False,
                                         transform=data_transform['test'])
    # VOC
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # COCO
    # mean = [0.471, 0.448, 0.408]
    # std = [0.234, 0.239, 0.242]
    collate_fn = MultiScaleCollater(mean=mean,
                                    std=std,
                                    resize=args.resize,
                                    stride=32,
                                    use_multi_scale=True,
                                    normalize=True)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.train_batch,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=args.pin_memory,
                              prefetch_factor=6,
                              collate_fn=collate_fn,
                              drop_last=True)

    test_loader = DataLoader(test_dataset,
                             batch_size=args.train_batch,
                             shuffle=False,
                             num_workers=args.num_workers,
                             pin_memory=args.pin_memory,
                             prefetch_factor=6,
                             collate_fn=collate_fn,
                             drop_last=True)

    print(f'Initializing model: {args.arch}')
    model = models.init_model(name=args.arch,
                              num_classes=args.num_classes,
                              pre_train_load_dir=args.pre_train_load_dir
                              )
    if use_gpu and args.use_ddp is False:
        model = model.cuda()
    if use_gpu and args.use_ddp:
        model = nn.DataParallel(model).cuda()

    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))

    criterion = losses.RetinaLoss(box_loss_type='CIoU')

    optimizer = init_optimizer(args.optim,
                               params=model.parameters(),
                               lr=args.lr,
                               weight_decay=args.weight_decay)

    scheduler = init_scheduler('cosine_annealing_lr',
                               optimizer=optimizer,
                               step_size=args.step_size,
                               gamma=args.gamma,
                               args=args)

    decoder = RetinaDecoder(nms_type='python_nms',
                            nms_threshold=0.5)
    start_epoch = args.start_epoch

    start_time = time.time()
    train_time = 0.
    best_epoch = 0
    best_metric, metric, test_loss = 0., 0, 0

    if os.path.exists(args.resume):
        print('Load checkpoint from {}'.format(args.resume))
        checkpoints = torch.load(args.resume)
        model.load_state_dict(checkpoints['model_state_dict'])
        start_epoch = checkpoints['epoch']
        optimizer.load_state_dict(checkpoints['optimizer_model_state_dict'])
        scheduler.load_state_dict(checkpoints['scheduler_state_dict'])
        best_metric = checkpoints['best_metric']

    if args.only_evaluate:
        res_dict = test(args.dataset_name,
                        test_loader=test_loader,
                        model=model,
                        criterion=criterion,
                        decoder=decoder,
                        args=args)
        for k, v in res_dict.items():
            if 'per_class_ap' in k:
                continue
            print(k, v)
        return

    print('===> Start training')
    for epoch in range(start_epoch, args.max_epoch):
        start_train_time = time.time()
        # 使用warm up加余弦退火学习率调整
        adjust_learning_rate(optimizer,
                             current_epoch=epoch,
                             max_epoch=args.max_epoch,
                             lr_min=1e-12,
                             lr_max=args.lr,
                             warmup_epoch=0,
                             warmup=False)
        train(epoch=epoch,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              trainloader=train_loader,
              use_gpu=use_gpu,
              args=args)
        train_time += round(time.time() - start_train_time)

        scheduler.step()

        if (epoch + 1) > args.start_eval and (epoch + 1) % args.eval_step == 0 or (epoch + 1) == args.max_epoch:
            res_dict = test(args.dataset_name,
                            test_loader=test_loader,
                            model=model,
                            criterion=criterion,
                            decoder=decoder,
                            args=args)
            for k, v in res_dict:
                if 'per_class_ap' in k:
                    continue
                print(k, v)
            max_map = res_dict['IoU=0.50,area=all,maxDets=100,mAP']
            is_best = max_map >= best_metric
            if is_best:
                best_metric = max_map
                best_epoch = epoch + 1
            save_states = {
                'model_state_dict': model.state_dict(),
                'optimizer_model_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'best_metric': best_metric,
                'best_epoch': best_epoch
            }
            save_checkpoints(save_states,
                             isbest=True,
                             save_dir=args.save_dir,
                             checkpoint_name='checkpoint_ep' + str(epoch + 1) + '.pth')

        # save_states = {
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_model_state_dict': optimizer.state_dict(),
        #         'scheduler_state_dict': scheduler.state_dict(),
        #         'epoch': epoch,
        #         'best_metric': best_metric
        #     }
        # save_checkpoints(save_states,
        #                  isbest=True,
        #                  save_dir=args.save_dir,
        #                  checkpoint_name='checkpoint_ep' + str(epoch + 1) + '.pth')
        #
        # res_dict = test(args.dataset_name,
        #                 test_loader=test_loader,
        #                 model=model,
        #                 criterion=criterion,
        #                 decoder=decoder,
        #                 args=args)

    train_time = round((time.time() - start_time) / 60, 2)
    print(f'Finnished training. train time: {train_time} mins')


def train(epoch, model, criterion, optimizer, trainloader, use_gpu, args):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()

    end = time.time()
    for batch_idx, datas in enumerate(trainloader):
        images, annots = datas['img'], datas['annot']
        if use_gpu:
            images, annots = images.cuda(), annots.cuda()
        # measure data loading time
        data_time.update(time.time() - end)

        # outs_tuple = model(images)
        # loss_value = criterion(outs_tuple, annots)
        # cls_loss, reg_loss = loss_value['cls_loss'], loss_value['reg_loss']
        # loss = sum(loss_value.values())

        # 梯度清零
        optimizer.zero_grad()

        if args.apex:
            scaler = amp.GradScaler()
            auto_cast = amp.autocast
            with auto_cast():
                outs_tuple = model(images)
                loss_value = criterion(outs_tuple, annots)
                cls_loss, reg_loss = loss_value['cls_loss'], loss_value['reg_loss']
                loss = sum(loss_value.values())
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            scaler.step(optimizer)
            scaler.update()

        else:
            outs_tuple = model(images)
            loss_value = criterion(outs_tuple, annots)
            cls_loss, reg_loss = loss_value['cls_loss'], loss_value['reg_loss']
            loss = sum(loss_value.values())

            # 反向传播
            loss.backward()

            # 梯度更新
            optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        losses.update(loss.item())

        if (batch_idx + 1) % args.print_freq == 0:
            lr_value = optimizer.state_dict()['param_groups'][0]['lr']
            print(f'Epoch: [{epoch + 1}][{batch_idx + 1}/{len(trainloader)}]\t'
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Data {batch_time.val:.3f} ({data_time.avg:.3f})\t'
                  f'Loss {losses.val:.3f} ({losses.avg:.3f})\t'
                  f'cls_loss {cls_loss.item():.3f}\t'
                  f'reg_loss {reg_loss.item():.3f}\t'
                  f'Lr {lr_value}')


def test(eval_type, test_loader, model, criterion, decoder, args):
    if eval_type == 'voc':
        res_dict = metrics.evaluate_voc_detection(test_loader=test_loader,
                                                  model=model,
                                                  criterion=criterion,
                                                  decoder=decoder,
                                                  args=args)
    else:
        raise KeyError(f'Unsupported {eval_type} type')

    return res_dict


if __name__ == "__main__":
    arg_infos = parse_args()
    main(arg_infos)
    # de_bug_main()
