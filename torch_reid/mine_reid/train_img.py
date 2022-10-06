import os
import sys
import time
import argparse

import numpy as np
import random
import datetime

import torch
import torch.nn as nn
from torch.backends import cudnn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader

from torchreid import models, datas, utils, losses
from torchreid.losses import DeepSupervision
from torchreid.datas import data_manager, data_transfrom, ImageDataset
from torchreid.datas.samplers import RandomIdentitySampler
from torchreid.utils.avgmeter import AverageMeter
from torchreid.utils.eval_metrics import evaluate
from torchreid.utils.re_ranking import re_ranking, re_ranking3
from torchreid.utils.util import Logger, init_pretrained_weights, save_checkpoints


def parse_args():
    parser = argparse.ArgumentParser(description='some orders')
    # Datasets
    parser.add_argument('--root', type=str, default='data', help="root path to data directory")
    parser.add_argument('-d', '--dataset', type=str, default='market1501', choices=data_manager.get_names())
    parser.add_argument('-j', '--workers', default=4, type=int, help="number of data loading workers (default: 4)")
    parser.add_argument('--height', type=int, default=256, help="height of an image (default: 256)")
    parser.add_argument('--width', type=int, default=128, help="width of an image (default: 128)")
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--split_id', type=int, default=0, help="split index")

    # CUHK03-specific setting
    parser.add_argument('--cuhk03_labeled', action='store_true',
                        help="whether to use labeled images, if false, detected images are used (default: False)")
    parser.add_argument('--cuhk03_classic-split', action='store_true',
                        help="whether to use classic split by Li et al. CVPR'14 (default: False)")
    parser.add_argument('--use_metric_cuhk03', action='store_true',
                        help="whether to use cuhk03-metric (default: False)")

    # Optimization options
    parser.add_argument('--optim', type=str, default='adam', help="optimization algorithm (see optimizers.py)")
    parser.add_argument('--max_epoch', default=180, type=int, help="maximum epochs to run")
    parser.add_argument('--start_epoch', default=0, type=int, help="manual epoch number (useful on restarts)")
    parser.add_argument('--train_batch', default=16, type=int, help="train batch size")
    parser.add_argument('--test_batch', default=32, type=int, help="test batch size")
    parser.add_argument('--lr', '--learning_rate', default=0.0003, type=float, help="initial learning rate")
    parser.add_argument('--step_size', default=20, type=int,
                        help="stepsize to decay learning rate (>0 means this is enabled)")
    parser.add_argument('--gamma', default=0.1, type=float, help="learning rate decay")
    parser.add_argument('--weight_decay', default=5e-04, type=float, help="weight decay")
    parser.add_argument('--margin', type=float, default=0.3, help="margin for triplet loss")
    parser.add_argument('--num_instances', type=int, default=4, help="number of instances per identity")
    parser.add_argument('--htri_only', action='store_true', help="if this is True, only htri loss is used in training")
    parser.add_argument('--weight_cent', type=float, default=0.0005, help="weight for center loss")

    # Architecture
    parser.add_argument('-a', '--arch', type=str, default='resnet50', choices=models.get_names())
    parser.add_argument('--pre_train_load_dir', type=str, default=None, help='save dir of pretrain weight')
    parser.add_argument('--loss_type', type=str, default='softmax', help='Determine the model output type',
                        choices=['softmax_trip', 'softmax_cent', 'softmax'])
    parser.add_argument('--apex', action='store_true')

    # Miscs
    parser.add_argument('--print_freq', type=int, default=10, help="print frequency")
    parser.add_argument('--seed', type=int, default=3407, help="manual seed")
    parser.add_argument('--resume', type=str, default='checkpoint.pth')
    parser.add_argument('--evaluate', action='store_true', help="evaluation only")
    parser.add_argument('--eval_step', type=int, default=-1,
                        help="run evaluation for every N epochs (set to -1 to test after training)")
    parser.add_argument('--start_eval', type=int, default=0, help="start to evaluate after specific epoch")
    parser.add_argument('--save_dir', type=str, default='log')
    parser.add_argument('--use_cpu', action='store_true', help="use cpu")
    parser.add_argument('--use_ddp', action='store_true', help="use multiple devices")
    parser.add_argument('--gpu_devices', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')

    parser.add_argument('--reranking', action='store_true', help='result re_ranking')
    parser.add_argument('--test_distance', type=str, default='global', help='test distance type')
    parser.add_argument('--aligned', action='store_true')

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

    if args.evaluate is False:
        sys.stdout = Logger(os.path.join(args.save_dir, 'train.log'))
    # else:
    #     sys.stdout = Logger(os.path.join(args.save_dir, 'test.log'))
    print(f'==============\nArgs:{args}\n==============')
    print('****************mine_reid****************')
    cur_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(args.loss_type, '--------', cur_time)

    if use_gpu:
        print(f'Currently using GPU {args.gpu_devices}')
        torch.backends.cudnn.enabled = True  # 说明设置为使用非确定性算法
        # 如果网络的输入数据维度或类型上变化不大，设置 torch.backends.cudnn.benchmark = true 可以增加运行效率
        cudnn.enabled = False
        cudnn.benchmark = False
        cudnn.deterministic = True
    else:
        print('Currently using CPU (GPU is highly recommended)')

    print('Initializing dataset {}'.format(args.dataset))
    dataset = data_manager.init_img_dataset(name=args.dataset,
                                            root=args.root,
                                            split_id=args.split_id,
                                            cuhk03_labeled=args.cuhk03_labeled,
                                            cuhk03_classic_split=args.cuhk03_classic_split)
    transform_train = transforms.Compose([
        data_transfrom.Random2DErasing(sh=0.25),
        data_transfrom.Random2DTranslation(args.height, args.width),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.Resize((args.height, args.width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if args.loss_type == 'softmax_trip':
        sampler = RandomIdentitySampler(dataset.train, num_instances=args.num_instances)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    def worker_init_fn(worker_id):
        np.random.seed(args.seed)
        random.seed(args.seed)

    # 如果使用了sampler, 对于market1501中, num_train_pids有751个, 也就是有751个行人
    # id, 每个行人id取 num_instances 张图片, 也就是有 751 * num_instances 张图片
    # 这些图片的实际id编码就是 [0,0,0,0,1,1,1,3,3,3,3, ...]
    # 所以train_batch就必须是num_instances的倍数
    train_loader = DataLoader(ImageDataset(dataset.train, transform_train),
                              batch_size=args.train_batch,
                              shuffle=shuffle,
                              sampler=sampler,
                              num_workers=args.num_workers,
                              pin_memory=args.pin_memory,
                              prefetch_factor=6,
                              worker_init_fn=worker_init_fn,
                              drop_last=True)
    query_loader = DataLoader(ImageDataset(dataset.query, transform_test),
                              batch_size=args.test_batch,
                              shuffle=False,
                              num_workers=args.num_workers,
                              pin_memory=args.pin_memory,
                              drop_last=False)
    gallery_loader = DataLoader(ImageDataset(dataset.gallery, transform_test),
                                batch_size=args.test_batch,
                                shuffle=False,
                                num_workers=args.num_workers,
                                pin_memory=args.pin_memory,
                                drop_last=False)

    print(f'Initializing model: {args.arch}')
    model = models.init_model(name=args.arch,
                              num_classes=dataset.num_train_pids,
                              loss=args.loss_type,
                              aligned=args.aligned,
                              act_func='prelu',
                              attention=None)
    # model = ResNet50(num_classes=dataset.num_train_pids)
    init_pretrained_weights(model, args.pre_train_load_dir)
    if use_gpu and args.use_ddp is False:
        model = model.cuda()
    if use_gpu and args.use_ddp:
        model = nn.DataParallel(model).cuda()

    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))

    criterion_xent = losses.CrossEntropyLabelSmooth(num_classes=dataset.num_train_pids, use_gpu=use_gpu)
    if args.aligned:
        criterion_trip = losses.TripletAlignedLoss(margin=args.margin)
    else:
        criterion_trip = losses.TripletLoss(margin=args.margin)
    optimizer_model = utils.init_optimizer(args.optim,
                                           params=model.parameters(),
                                           lr=args.lr,
                                           weight_decay=args.weight_decay)
    # ======== center_loss ========
    if args.loss_type == 'softmax_cent':
        criterion_cent = losses.CenterLoss(num_classes=dataset.num_train_pids, feat_dim=model.feature_dim)
        optimizer_cent = utils.init_optimizer('sgd',
                                              params=criterion_cent.parameters(),
                                              lr=args.lr,
                                              weight_decay=args.weight_decay)
    else:
        criterion_cent = None
        optimizer_cent = None
    # ======== center_loss ========

    scheduler = utils.init_scheduler('cosine_annealing_lr',
                                     optimizer=optimizer_model,
                                     step_size=args.step_size,
                                     gamma=args.gamma,
                                     args=args)
    start_epoch = args.start_epoch

    start_time = time.time()
    train_time = 0.
    best_rank1 = -np.inf
    best_epoch = 0

    if os.path.exists(args.resume):
        print('Load checkpoint from {}'.format(args.resume))
        checkpoints = torch.load(args.resume)
        model.load_state_dict(checkpoints['model_state_dict'])
        start_epoch = checkpoints['epoch']
        optimizer_model.load_state_dict(checkpoints['optimizer_model_state_dict'])
        scheduler.load_state_dict(checkpoints['scheduler_state_dict'])
        best_rank1 = checkpoints['best_rank1']

    print('===> Start training')
    for epoch in range(start_epoch, args.max_epoch):
        start_train_time = time.time()
        # 使用warm up加余弦退火学习率调整
        utils.adjust_learning_rate(optimizer_model,
                                   current_epoch=epoch,
                                   max_epoch=args.max_epoch,
                                   lr_min=1e-12,
                                   lr_max=args.lr,
                                   warmup_epoch=10,
                                   warmup=True)
        train(epoch=epoch,
              model=model,
              criterion_trip=criterion_trip,
              criterion_xent=criterion_xent,
              criterion_cent=criterion_cent,
              optimizer_model=optimizer_model,
              optimizer_cent=optimizer_cent,
              trainloader=train_loader,
              use_gpu=use_gpu,
              args=args)
        train_time += round(time.time() - start_train_time)

        scheduler.step()

        if (epoch + 1) > args.start_eval and (epoch + 1) % args.eval_step == 0 or (epoch + 1) == args.max_epoch:
            reranking = True if epoch+1 == args.max_epoch else False
            rank1 = test(model, query_loader, gallery_loader, use_gpu, args, reranking=reranking)
            is_best = rank1 >= best_rank1
            if is_best:
                best_rank1 = rank1
                best_epoch = epoch + 1
            save_checkpoints({
                'model_state_dict': model.state_dict(),
                'optimizer_model_state_dict': optimizer_model.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'rank1': rank1,
                'epoch': epoch,
                'best_rank1': best_rank1
            }, is_best, args.save_dir, 'checkpoint_ep' + str(epoch + 1) + '.pth')

    print(f'==> Best Rank-1 {best_rank1:.2f}, achieved at epoch {best_epoch}')
    train_time = round((time.time() - start_time) / 60, 2)
    print(f'Finnished training. train time: {train_time} mins')


def train(epoch, model, criterion_trip, criterion_xent, criterion_cent, optimizer_model, optimizer_cent, trainloader,
          use_gpu, args):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()

    end = time.time()
    for batch_idx, (imgs, pids, _) in enumerate(trainloader):
        if use_gpu:
            imgs, pids = imgs.cuda(), pids.cuda()
        # measure data loading time
        data_time.update(time.time() - end)

        # outputs shape is [batch_size, num_classes]
        # features shape is [batch_size, feat_dim]
        # local feature shape is [batch_size, 128, 8]
        if not args.aligned:
            outputs, features = model(imgs)
            local_features = None
        else:
            outputs, features, local_features = model(imgs)
        if args.htri_only:
            loss = criterion_trip(features, pids)
        elif args.loss_type == 'softmax':
            xent_loss = criterion_xent(outputs, pids)
            loss = xent_loss
        elif args.loss_type == 'softmax_cent':
            xent_loss = criterion_xent(outputs, pids)
            cent_loss = criterion_cent(features, pids) * args.weight_cent
            loss = xent_loss + cent_loss
        elif args.loss_type == 'softmax_trip' and not args.aligned:
            xent_loss = criterion_xent(outputs, pids)
            htri_loss = criterion_trip(features, pids)
            # TODO: 如果写成 htri_loss + xent_loss 就不收敛了
            loss = xent_loss + htri_loss
        elif args.loss_type == 'softmax_trip' and args.aligned:
            xent_loss = criterion_xent(outputs, pids)
            htri_loss, local_htri_loss = criterion_trip(features, pids, local_features)
            loss = xent_loss + htri_loss + local_htri_loss
        else:
            raise KeyError(f'Unknown {args.loss_type} type')
        # 梯度清零
        optimizer_model.zero_grad()
        if optimizer_cent:
            optimizer_cent.zero_grad()

        # 反向传播
        loss.backward()

        # 梯度更新
        optimizer_model.step()
        if optimizer_cent:
            # remove the impact of weight_cent in learning centers
            for param in criterion_cent.parameters():
                param.grad.data *= (1. / args.weight_cent)
            optimizer_cent.step()

        batch_time.update(time.time() - end)
        end = time.time()

        losses.update(loss.item(), pids.shape[0])

        if (batch_idx + 1) % args.print_freq == 0:
            lr_value = optimizer_model.state_dict()['param_groups'][0]['lr']
            print(f'Epoch: [{epoch + 1}][{batch_idx + 1}/{len(trainloader)}]\t'
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Data {batch_time.val:.3f} ({data_time.avg:.3f})\t'
                  f'Loss {losses.val:.3f} ({losses.avg:.3f})\t'
                  f'xent_loss {xent_loss.item():.3f} htri_loss {htri_loss.item():.3f} \t'
                  f'Lr {lr_value}')


def test(model, queryloader, galleryloader, use_gpu, args, ranks=None, reranking=False):
    if ranks is None:
        ranks = [1, 5, 10, 20]
    else:
        ranks = ranks
    batch_time = AverageMeter()

    model.eval()
    with torch.no_grad():
        qf, lqf, q_pids, q_camids = [], [], [], []
        for batch_idx, (imgs, pids, camids) in enumerate(queryloader):
            if use_gpu:
                imgs = imgs.cuda()

            end = time.time()
            # features: [batch_size, 2048], local_feature: [batch_size, 128, 8]
            features, local_feature = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            qf.append(features)
            lqf.append(local_feature)
            q_pids.extend(pids)
            q_camids.extend(camids)
        qf = torch.cat(qf, 0)            # [query_num, 2048]
        lqf = torch.cat(lqf, 0)          # [query_num, 128, 8]
        q_pids = np.asarray(q_pids)      # [query_num]
        q_camids = np.asarray(q_camids)  # [query_num]

        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.shape[0], qf.shape[1]))

        gf, lgf, g_pids, g_camids = [], [], [], []
        for batch_idx, (imgs, pids, camids) in enumerate(galleryloader):
            if use_gpu:
                imgs = imgs.cuda()

            end = time.time()
            features, local_feature = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            gf.append(features)
            lgf.append(local_feature)
            g_pids.extend(pids)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0)            # [gallery_num, 2048]
        lgf = torch.cat(lgf, 0)          # [gallery_num, 128, 8]
        g_pids = np.asarray(g_pids)      # [gallery_num]
        g_camids = np.asarray(g_camids)  # [gallery_num]

        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.shape[0], gf.shape[1]))

    print("==> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, args.test_batch))

    # TODO: 给表示图片的向量做单位化，单位化后向量的模为1，这点很重要
    qf = 1. * qf / (torch.norm(qf, p=2, dim=-1, keepdim=True) + 1e-12)
    gf = 1. * gf / (torch.norm(gf, p=2, dim=-1, keepdim=True) + 1e-12)

    m, n = qf.shape[0], gf.shape[0]
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).T
    distmat = torch.addmm(distmat, qf, gf.T, beta=1, alpha=-2)
    distmat = distmat.numpy()

    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, use_metric_cuhk03=args.use_metric_cuhk03)

    print("Results ----------")
    print("mAP: {:.2%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.2%}".format(r, cmc[r - 1]))
    print("------------------")

    if reranking:
        if args.test_distance == 'global':
            # dis_mat = re_ranking(q_fs, g_fs, k1=20, k2=6, lambda_value=0.3)

            # 欧式距离矩阵
            # q_q_dist = euclidean_dist(q_fs, q_fs)   # [3368, 3368]
            # q_g_dist = euclidean_dist(q_fs, g_fs)   # [3368, 15913]
            # g_g_dist = euclidean_dist(g_fs, g_fs)   # [15913, 15913]

            # 余弦距离矩阵
            q_q_dist = np.dot(qf, qf.T)   # [3368, 3368]
            q_g_dist = np.dot(qf, gf.T)   # [3368, 15913]
            g_g_dist = np.dot(gf, gf.T)   # [15913, 15913]
            dismat = re_ranking3(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6)
            print('Compute CMC and mAP for re_ranking')
            cmc, mAP, _ = evaluate(dismat, q_pids, g_pids, q_camids, g_camids)
            print("Results(RK) ----------")
            print(f'mAP(RK): {mAP:.2%}')
            print('CMC curve(RK)')
            for r in ranks:
                print(f'Rank-{r:2d}: {cmc[r - 1]:.2%}')
    return cmc[0]


class DeBugModel(nn.Module):
    def __init__(self, num_classes=3):
        super(DeBugModel, self).__init__()
        self.conv = nn.Conv2d(3, 10, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(10)
        self.relu = nn.ReLU(inplace=True)
        self.global_avg = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(10, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.global_avg(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)

        return x


import matplotlib.pyplot as plt
import math


def de_bug_main():
    datasets = [torch.randn(4, 3, 16, 16) for _ in range(10)]
    epochs = 100
    model = DeBugModel(num_classes=3)
    model = model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, weight_decay=5e-04, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    lr_list = []

    for epoch in range(epochs):
        model.train()
        utils.adjust_learning_rate(optimizer,
                                   current_epoch=epoch,
                                   max_epoch=epochs,
                                   lr_min=1e-12,
                                   lr_max=0.0001,
                                   warmup_epoch=10,
                                   warmup=True)
        for p in datasets:
            p = p.cuda()
            outputs = model(p)
            optimizer.zero_grad()
            loss = torch.tensor(0.8, requires_grad=True)
            loss.backward()
            optimizer.step()
        print(len(optimizer.param_groups))
        print('[', epoch, ']', optimizer.state_dict()['param_groups'][0]['lr'], '***', scheduler.get_lr())
        # print(optimizer.state_dict())
        # print(scheduler.state_dict())
        scheduler.step()
        lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
    plt.plot(range(epochs), lr_list, color='r')
    plt.show()


if __name__ == "__main__":
    arg_infos = parse_args()
    main(arg_infos)
    # de_bug_main()


