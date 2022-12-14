import os
import sys
import time
import matplotlib.pyplot as plt

import numpy as np
from tqdm import tqdm

import torch
import torchvision
import torch.nn as nn
from torch.backends import cudnn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader

from torchreid import models, datas, utils, losses
from torchreid.datas import data_manager, data_transfrom, ImageDataset
from torchreid.utils.avgmeter import AverageMeter
from torchreid.utils.eval_metrics import evaluate
from torchreid.utils.re_ranking import re_ranking, re_ranking2, re_ranking3, euclidean_dist
from torchreid.utils.util import mkdir_if_missing


class ResNet50(nn.Module):
    def __init__(self, num_classes, loss=None, pre_trained=None, **kwargs):
        super(ResNet50, self).__init__()
        if loss is None:
            self.loss = {'xent'}
        else:
            self.loss = loss

        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.classifier = nn.Linear(2048, num_classes)
        self.feat_dim = 2048  # feature dimension

    def forward(self, x):
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)
        if self.training is False:
            return f
        y = self.classifier(f)

        return y, f


def main(config):
    gpu_devices = config['gpu_devices']
    save_dir = config['save_dir']
    dataset = config['dataset']
    root = config['root']
    split_id = config['split_id']
    cuhk03_labeled = config['cuhk03_labeled']
    cuhk03_classic_split = config['cuhk03_classic_split']
    use_metric_cuhk03 = config['use_metric_cuhk03']
    arch = config['arch']
    loss_type = config['loss_type']
    height = config['height']
    width = config['width']
    test_batch = config['test_batch']
    pth_path = config['pth_path']
    aligned = config['aligned']
    reranking = config['reranking']
    test_distance = config['test_distance']
    visualize = config['visualize']

    # os.environ['CUDA_VISIBLE_DEVICES'] = gpu_devices
    device = torch.device('cuda')
    use_gpu = torch.cuda.is_available()

    # sys.stdout = Logger(os.path.join(config['save_dir'], 'test.log'))

    if use_gpu:
        # torch.backends.cudnn.enabled = True 说明设置为使用非确定性算法
        # 如果网络的输入数据维度或类型上变化不大，设置 torch.backends.cudnn.benchmark = true 可以增加运行效率
        cudnn.enabled = True
        cudnn.benchmark = True
    else:
        print('Currently using CPU (GPU is highly recommended)')

    dataset = data_manager.init_img_dataset(name=dataset,
                                            root=root,
                                            split_id=split_id,
                                            cuhk03_labeled=cuhk03_labeled,
                                            cuhk03_classic_split=cuhk03_classic_split)

    transform_test = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    query_loader = DataLoader(ImageDataset(dataset.query, transform_test),
                              batch_size=test_batch,
                              shuffle=False,
                              num_workers=4,
                              pin_memory=True,
                              drop_last=False)
    gallery_loader = DataLoader(ImageDataset(dataset.gallery, transform_test),
                                batch_size=test_batch,
                                shuffle=False,
                                num_workers=4,
                                pin_memory=True,
                                drop_last=False)

    print(f'Initializing model: {arch}')
    model = models.init_model(name=arch,
                              num_classes=dataset.num_train_pids,
                              loss=loss_type,
                              aligned=aligned,
                              act_func='prelu',
                              attention=None)
    load_dicts = torch.load(pth_path, map_location='cpu')

    model.load_state_dict(load_dicts)
    # model = model.cuda()
    model = model.to(device)

    # 测试状态
    model.eval()

    ranks = [1, 5, 10, 20]
    batch_time = AverageMeter()

    model.eval()
    with torch.no_grad():
        q_fs, q_pids, q_camids = [], [], []
        for _, (imgs, pids, camids) in enumerate(query_loader):
            if use_gpu:
                imgs = imgs.cuda()

            end = time.time()
            # if resnet50
            # outputs: [b, num_classes], features: [b, 2048], local_feature: [b, 128, 8]
            outputs, features, local_feature = model(imgs)
            batch_time.update(time.time() - end)

            features = features.detach().cpu()
            q_fs.append(features)
            q_pids.extend(pids)
            q_camids.extend(camids)
        q_fs = torch.cat(q_fs, dim=0)  # if resnet50, shape is [num_query, 2048]
        q_pids = np.array(q_pids)
        q_camids = np.array(q_camids)
        print(f'Extracted features for query set, obtained {q_fs.shape[0]}-by-{q_fs.shape[1]} matrix')

        g_fs, g_pids, g_camids = [], [], []
        for _, (imgs, pids, camids) in enumerate(gallery_loader):
            if use_gpu:
                imgs = imgs.cuda()

            end = time.time()
            features, local_feature = model(imgs)  # if resnet50, shape is [batch_size, 2048]
            batch_time.update(time.time() - end)

            features = features.detach().cpu()
            g_fs.append(features)
            g_pids.extend(pids)
            g_camids.extend(camids)
        g_fs = torch.cat(g_fs, dim=0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)
        print(f'Extracted features for gallery set, obtained {g_fs.shape[0]}-by-{g_fs.shape[1]} matrix')

    # print("==> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, test_batch)
    # feature normalization
    q_fs = 1. * q_fs / (torch.norm(q_fs, p=2, dim=-1, keepdim=True) + 1e-12)
    g_fs = 1. * g_fs / (torch.norm(g_fs, p=2, dim=-1, keepdim=True) + 1e-12)

    m, n = q_fs.shape[0], g_fs.shape[0]
    # dis_mat shape is [m, n]
    dis_mat = torch.pow(q_fs, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(g_fs, 2).sum(dim=1, keepdim=True).expand(n, m).T
    dis_mat = torch.addmm(dis_mat, mat1=q_fs, mat2=g_fs.T, beta=1, alpha=-2)
    dis_mat = dis_mat.numpy()

    print('Compute CMC and mAP')
    cmc, mAP, g_pids_sorted = evaluate(dis_mat, q_pids, g_pids, q_camids, g_camids,
                                       need_indices=visualize,
                                       use_metric_cuhk03=config['use_metric_cuhk03'])
    if visualize:
        visualization(dataset, g_pids_sorted, save_dir, re_rank=reranking)
    print("Results ----------")
    print(f'mAP: {mAP:.2%}')
    print('CMC Curve')
    for p in ranks:
        print(f'Rank-{p:2d}: {cmc[p - 1]:.2%}')
    print('-----------------')

    if reranking:
        if test_distance == 'global':
            if test_distance == 'global':
                # dis_mat = re_ranking(q_fs, g_fs, k1=20, k2=6, lambda_value=0.3)

                # 欧式距离矩阵
                # q_q_dist = euclidean_dist(q_fs, q_fs)   # [3368, 3368]
                # q_g_dist = euclidean_dist(q_fs, g_fs)   # [3368, 15913]
                # g_g_dist = euclidean_dist(g_fs, g_fs)   # [15913, 15913]

                # 余弦距离矩阵
                q_q_dist = np.dot(q_fs, q_fs.T)  # [3368, 3368]
                q_g_dist = np.dot(q_fs, g_fs.T)  # [3368, 15913]
                g_g_dist = np.dot(g_fs, g_fs.T)  # [15913, 15913]
                dis_mat = re_ranking3(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6)
            print('Compute CMC and mAP for re_ranking')
            cmc, mAP, g_pids_sorted = evaluate(dis_mat, q_pids, g_pids, q_camids, g_camids,
                                               need_indices=visualize,
                                               use_metric_cuhk03=config['use_metric_cuhk03'])
            if visualize:
                visualization(dataset, g_pids_sorted, save_dir, re_rank=reranking)
            print("Results(RK) ----------")
            print(f'mAP(RK): {mAP:.2%}')
            print('CMC curve(RK)')
            for r in ranks:
                print(f'Rank-{r:2d}: {cmc[r - 1]:.2%}')


def imshow(path, title=None):
    im = plt.imread(path)
    plt.imshow(im)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def visualization(dataset, g_pids_indices, save_dir, re_rank=False):
    print('Top 10 images as follow:')
    if re_rank:
        save_dir = os.path.join(save_dir, 're_rank')
    else:
        save_dir = os.path.join(save_dir, 'no_re_rank')
    mkdir_if_missing(save_dir)

    for idx in range(dataset.num_query_pids):
        q_path = dataset.query[idx][0]
        q_pid = dataset.query[idx][1]
        fig = plt.figure(figsize=(16, 4))
        ax = plt.subplot(1, 11, 1)
        ax.axis('off')
        imshow(q_path, 'query')

        # 没有在gallery中找到匹配的行人
        if (g_pids_indices[idx] == -1).all():
            print(f'{q_path.split(os.sep)[-1]} has no matched in gallery')
            continue

        for i, g_pid_idx in enumerate(g_pids_indices[idx][:10]):
            ax = plt.subplot(1, 11, i + 2)
            ax.axis('off')
            g_path = dataset.gallery[int(g_pid_idx)][0]
            g_pid = dataset.gallery[int(g_pid_idx)][1]
            imshow(g_path)

            # print(q_label, g_label, gids[i])
            if g_pid == q_pid:
                ax.set_title('%d' % (i + 1), color='green')
            else:
                ax.set_title('%d' % (i + 1), color='red')

        fig.savefig(os.path.join(save_dir, q_path.split(os.sep)[-1]))
        plt.close()

        if idx == 20:
            print('save finished!')
            break


if __name__ == "__main__":
    configs = {
        'gpu_devices': '0',
        'save_dir': 'D:\\Desktop\\reid_infer',
        'dataset': 'market1501',
        'root': 'D:\\workspace\\data\\dl\\reid',
        'split_id': 0,
        'cuhk03_labeled': False,
        'cuhk03_classic_split': False,
        'use_metric_cuhk03': False,
        'arch': 'osnet_x1_0',
        'loss_type': 'softmax_trip',
        'height': 256,
        'width': 128,
        'test_batch': 4,
        'pth_path': 'D:\\workspace\\data\\reid_data\\osnet_softmax_epoch50_9273\\best_model.pth',
        'aligned': True,
        'reranking': True,
        'test_distance': 'global',
        'visualize': True
    }

    main(configs)
