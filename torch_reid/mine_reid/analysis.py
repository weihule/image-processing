import os
import sys
import time
import argparse

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
        self.feat_dim = 2048     # feature dimension

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

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_devices
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
    # model = models.init_model(name=arch,
    #                           num_classes=dataset.num_train_pids,
    #                           loss=config['loss_type'])
    model = ResNet50(num_classes=dataset.num_train_pids)
    model.load_state_dict(torch.load('D:\\Desktop\\best_model.pth')['state_dict'])
    model = model.cuda()
    
    ranks = [1, 5, 10, 20]
    batch_time = AverageMeter()

    model.eval()
    with torch.no_grad():
        q_fs, q_pids, q_camids = [], [], []
        for _, (imgs, pids, camids) in enumerate(query_loader):
            if use_gpu:
                imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)  # if resnet50, shape is [batch_size, 2048]
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
            features = model(imgs)  # if resnet50, shape is [batch_size, 2048]
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

    m, n = q_fs.shape[0], g_fs.shape[0]
    # dis_mat shape is [m, n]
    dis_mat = torch.pow(q_fs, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(g_fs, 2).sum(dim=1, keepdim=True).expand(n, m).T
    dis_mat = torch.addmm(dis_mat, mat1=q_fs, mat2=g_fs.T, beta=1, alpha=-2)
    dis_mat = dis_mat.numpy()

    print('Compute CMC and mAP')
    cmc, mAP = evaluate(dis_mat, q_pids, g_pids, q_camids, g_camids, use_metric_cuhk03=config['use_metric_cuhk03'])
    print("Results ----------")
    print(f'mAP: {mAP:.2%}')
    print('CMC Curve')
    for p in ranks:
        print(f'Rank-{p:2d}: {cmc[p - 1]:.2%}')
    print('-----------------')
    

if __name__ == "__main__":
    configs = {
        'gpu_devices': '0',
        'save_dir': '',
        'dataset': 'market1501',
        'root': 'D:\\workspace\\data\\dl',
        'split_id': 0,
        'cuhk03_labeled': False,
        'cuhk03_classic_split': False,
        'use_metric_cuhk03': False,
        'arch': 'resnet50',
        'loss_type': {'xent', 'htri'},
        'height': 256,
        'width': 128,
        'test_batch': 16,
        'pth_path': 'D:\\Desktop\\best_model.pth'
    }
    
    main(configs)
    
    