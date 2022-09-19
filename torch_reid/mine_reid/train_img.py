import os
import sys
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from thop import profile, clever_format

from torchreid.datas import data_manager, data_transfrom, ImageDataset
from torchreid import models, datas, utils, losses
from torchreid.datas.samplers import RandomIdentitySampler


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
    parser.add_argument('--stepsize', default=20, type=int,
                        help="stepsize to decay learning rate (>0 means this is enabled)")
    parser.add_argument('--gamma', default=0.1, type=float, help="learning rate decay")
    parser.add_argument('--weight_decay', default=5e-04, type=float, help="weight decay")
    parser.add_argument('--margin', type=float, default=0.3, help="margin for triplet loss")
    parser.add_argument('--num_instances', type=int, default=4, help="number of instances per identity")
    parser.add_argument('--loss_type', type=str, default='cent', help="type of loss function")
    parser.add_argument('--htri_only', action='store_true', default=False,
                        help="if this is True, only htri loss is used in training")

    # Architecture
    parser.add_argument('-a', '--arch', type=str, default='resnet50', choices=models.get_names())
    parser.add_argument('--pre_trained', type=str)
    parser.add_argument('--pre_train_load_dir', type=str, help='save dir of pretrain weight')
    parser.add_argument('--loss_type', type=str, default='softmax', help='Determine the model output type')
    parser.add_argument('--apex', action='store_true', default=False)

    # Miscs
    parser.add_argument('--print_freq', type=int, default=10, help="print frequency")
    parser.add_argument('--seed', type=int, default=1, help="manual seed")
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true', help="evaluation only")
    parser.add_argument('--eval_step', type=int, default=-1,
                        help="run evaluation for every N epochs (set to -1 to test after training)")
    parser.add_argument('--start_eval', type=int, default=0, help="start to evaluate after specific epoch")
    parser.add_argument('--save_dir', type=str, default='log')
    parser.add_argument('--use_cpu', action='store_true', help="use cpu")
    parser.add_argument('--use_ddp', type=bool, default=False, help="use multiple devices")
    parser.add_argument('--gpu_devices', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')

    args = parser.parse_args()
    return args


def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu:
        use_gpu = False

    print('Initializing dataset {}'.format(args.dataset))
    dataset = data_manager.init_img_dataset(name=args.dataset,
                                            root=args.root,
                                            split_id=args.split_id,
                                            cuhk03_labeled=args.cuhk03_labeled,
                                            cuhk03_classic_split=args.cuhk03_classic_split)
    transform_train = transforms.Compose([
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

    if args.loss_type == 'xent_htri':
        sampler = RandomIdentitySampler(dataset.train, num_instances=args.num_instances)
        shuffle = False
    elif args.loss_type == 'cent':
        sampler = None
        shuffle = True
    else:
        sampler = None
        shuffle = False
    train_loader = DataLoader(ImageDataset(dataset.train, transform_train),
                              batch_size=args.train_batch,
                              shuffle=shuffle,
                              sampler=sampler,
                              num_workers=args.num_workers,
                              pin_memory=args.pin_memory,
                              prefetch_factor=6,
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

    for datas in tqdm(train_loader):
        imgs, pids, camids = datas[0], datas[1], datas[2]

    model = models.init_model(name=args.arch,
                              num_classes=5,
                              loss='softmax',
                              pretrained=True,
                              pre_train_load_dir='D:\\workspace\\data\\classification_data\\resnet')
    if use_gpu and args.use_ddp is False:
        model = model.cuda()
    if use_gpu and args.use_ddp:
        model = nn.DataParallel(model).cuda()
    macs, params = profile(model, (torch.randn(1, 3, args.height, args.width), ))
    macs, params = clever_format([macs, params], '%.3f')
    print('model size: {}'.format(params))


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


def de_bug_main(inputs):
    datasets = [torch.randn(4, 3, 16, 16) for _ in range(10)]
    model = DeBugModel(num_classes=3)
    model = model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, weight_decay=5e-04)
    scheduler = utils.init_scheduler('steplr', optimizer, step_size=2, gamma=0.1)
    epochs = 10

    for epoch in range(epochs):
        model.train()
        for p in datasets:
            outputs = model(p)
            optimizer.zero_grad()
            loss = torch.tensor(0.8, requires_grad=True)
            optimizer.step()
        scheduler.step()



if __name__ == "__main__":
    # arg_infos = parse_args()
    # main(arg_infos)
    img = torch.randn(4, 3, 16, 16)

