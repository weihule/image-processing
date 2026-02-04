import os
import random
import argparse
import numpy as np
from pathlib import Path
import yaml
from tqdm import tqdm
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils import data
from torchvision import transforms
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DataParallel, DistributedDataParallel

import network
from datasets import VOCSegmentation
from utils import ext_transforms as et
from utils.loss import FocalLoss
from utils.utils import DictWrapper, set_bn_momentum, Denormalize, save_ckpt, load_checkpoint
from utils.scheduler import PolyLR
from metrics import StreamSegMetrics
from utils.visualizer import Visualizer


def load_config(config_path):
    with open(config_path, "r", encoding='utf-8') as fr:
        config = yaml.safe_load(fr)
    return config

def setup_distributed():
    """使用 torchrun 初始化分布式环境"""
    # 检查是否在分布式环境中
    if not torch.distributed.is_available():
        raise RuntimeError("Requires distributed package to be available")

    # torchrun 会自动设置这些环境变量
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    # 初始化分布式进程组
    torch.distributed.init_process_group(backend='nccl', init_method='env://')

    # 设置当前进程使用的GPU
    torch.cuda.set_device(local_rank)

    return {
        'rank': rank,
        'world_size': world_size,
        'local_rank': local_rank,
    }

def is_main_process():
    """检查是否是主进程"""
    return int(os.environ.get('RANK', 0)) == 0

def get_rank():
    """获取当前进程的全局rank"""
    return int(os.environ.get('RANK', 0))

def get_world_size():
    """获取总进程数"""
    return int(os.environ.get('WORLD_SIZE', 1))

def get_local_rank():
    """获取当前进程的本地rank"""
    return int(os.environ.get('LOCAL_RANK', 0))

def barrier():
    """同步所有进程"""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

def get_device():
    """获取当前进程应该使用的设备"""
    if torch.cuda.is_available():
        return torch.device(f'cuda:{get_local_rank()}')
    else:
        return torch.device('cpu')

def cleanup_distributed():
    """清理分布式环境"""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

def get_dataset(opts):
    if opts.dataset.name == 'voc':
        train_transform = et.ExtCompose([
            et.ExtRandomScale((0.5, 2.0)),
            et.ExtRandomCrop(size=(opts.dataset.crop_size, opts.dataset.crop_size), pad_if_needed=True),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        if opts.dataset.crop_val:
            val_transform = et.ExtCompose([
                et.ExtResize(opts.dataset.crop_size),
                et.ExtCenterCrop(opts.dataset.crop_size),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        else:
            val_transform = et.ExtCompose([
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])

        train_dataset = VOCSegmentation(root=opts.dataset.data_root,
                                        year=opts.dataset.voc.year,
                                        image_set='train',
                                        download=opts.dataset.download,
                                        transform=train_transform)
        val_dataset = VOCSegmentation(root=opts.dataset.data_root,
                                      year=opts.dataset.voc.year,
                                      image_set='val',
                                      download=False,
                                      transform=val_transform)
    return train_dataset, val_dataset

def create_dataloader(dataset, batch_size, shuffle, opts, is_train=True):
    """创建数据加载器"""
    sampler = DistributedSampler(
        dataset,
        num_replicas=get_world_size(),
        rank=get_rank(),
        shuffle=shuffle,
        seed=opts.system.random_seed
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=opts.system.num_workers,
        drop_last=is_train,
        pin_memory=True
    )
    return dataloader, sampler

def validate(opts, model, loader, device, metrics: StreamSegMetrics, ret_samples_ids=None):
    """验证函数"""
    metrics.reset()
    ret_samples = []

    if opts.training.save_val_results and is_main_process():
        Path(opts.training.save_val_path).mkdir(parents=True, exist_ok=True)
        denorm = Denormalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        img_id = 0
    model.eval()
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation", disable=not is_main_process())

        for i, (images, labels) in enumerate(pbar):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            print(outputs.shape)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()

            break

def setup_model_and_training(opts, device):
    """设置模型、优化器、调度器和损失函数"""
    # 设置模型
    model = network.modeling.__dict__[opts.model.name](
        num_classes=opts.model.num_classes,
        output_stride=opts.model.output_stride
    )

    if opts.model.separable_conv and 'plus' in opts.model.name:
        network.convert_to_separable_conv(model.classifier)
    set_bn_momentum(model.backbone, momentum=0.01)

    # 设置优化器
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr':  float(opts.training.optimizer.lr_backbone_ratio) *
                                                       float(opts.training.optimizer.lr)},
        {'params': model.classifier.parameters(), 'lr': float(opts.training.optimizer.lr)},
    ],
        lr=float(opts.training.optimizer.lr),
        momentum=float(opts.training.optimizer.momentum),
        weight_decay=float(opts.training.optimizer.weight_decay)
    )

    # 设置学习率调度器
    if opts.training.lr_scheduler.policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=opts.training.lr_scheduler.step_size,
            gamma=0.1
        )
    elif opts.training.lr_scheduler.policy == 'poly':
        scheduler = PolyLR(
            optimizer=optimizer,
            max_iters=opts.training.epoches
        )
    else:
        raise ValueError(f"Unsupported LR scheduler policy: {opts.training.lr_scheduler.policy}")

    # 设置损失函数
    if opts.training.loss.type.lower() == 'focal_loss':
        criterion = FocalLoss(ignore_index=255, size_average=True)
    elif opts.training.loss.type.lower() == 'cross_loss':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    else:
        raise ValueError(f"Unsupported loss type: {opts.training.loss.type}")

    return model, optimizer, scheduler, criterion

def setup_environment(opts):
    """设置训练环境"""
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.system.gpu_id
    print(f"CUDA_VISIBLE_DEVICES 已设置为: {opts.system.gpu_id}")

    # 检查实际可见设备
    if torch.cuda.is_available():
        num_visible = torch.cuda.device_count()
        visible_devices = opts.system.gpu_id.split(',')
        print(f"✅ 检测到 {num_visible} 个可见GPU (物理ID: {', '.join(visible_devices)})")
        device = torch.device('cuda')
    else:
        print("⚠️  未检测到CUDA设备，将使用CPU训练（速度极慢）")
        device = torch.device('cpu')
    print(f"Device: {device}")

    # 设置随机种子
    torch.manual_seed(opts.system.random_seed)
    np.random.seed(opts.system.random_seed)
    random.seed(opts.system.random_seed)

    # 设置可视化
    vis = None
    if opts.visualization.enable_vis:
        experiment_id = f"{opts.model.name}_{opts.dataset.name}"
        vis = Visualizer(logdir="./runs",
                         env="main",
                         id=experiment_id)
        print(
            f"✅ TensorBoard visualization enabled. Run 'tensorboard --logdir {opts.visualization.vis_logdir}' to view.")

    return device, vis


def main(config_path="./config.yaml"):
    # 加载配置
    config = load_config(config_path)
    opts = DictWrapper(config)
    print(opts)

    # 设置环境
    device, vis = setup_environment(opts)
    train_ds, val_ds = get_dataset(opts)
    print(device, vis)
    print(train_ds, val_ds)
    train_loader = DataLoader(dataset=train_ds,
                              batch_size=opts.training.batch_size,
                              shuffle=True,
                              num_workers=opts.system.num_workers,
                              drop_last=True)
    val_loader = DataLoader(dataset=val_ds,
                            batch_size=opts.training.batch_size,
                            shuffle=False,
                            num_workers=opts.system.num_workers)
    print(f"Dataset: {opts.dataset.name}, Train set: {len(train_ds)}, Val set: {len(val_ds)}")

    # 设置模型和训练组件
    model, optimizer, scheduler, criterion = setup_model_and_training(opts, device)
    metrics = StreamSegMetrics(opts.model.num_classes)

    # 设置并行训练
    model = nn.DataParallel(model)
    model.to(device)

    # 加载检查点
    model, optimizer, scheduler, cur_itrs, best_score = load_checkpoint(
        opts, model, optimizer, scheduler, device
    )

    # 仅测试模式
    if opts.training.test_only:
        model.eval()
        val_score, _ = validate(opts, model, val_loader, device, metrics)
        print(metrics.to_str(val_score))
        return


if __name__ == "__main__":
    main()



