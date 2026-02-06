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

def validate(opts, model, loader, device, metrics: StreamSegMetrics, criterion=None, ret_samples_ids=None):
    """验证函数"""
    metrics.reset()
    ret_samples = []

    total_loss = 0.0
    num_batches = 0

    save_results = opts.training.save_val_results and is_main_process()
    if save_results:
        save_root = opts.training.save_val_path or "./results"
        Path(save_root).mkdir(parents=True, exist_ok=True)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_id = 0
    model.eval()
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation", disable=not is_main_process())

        for i, (images, labels) in enumerate(pbar):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)

            if criterion is not None:
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                num_batches += 1

            if ret_samples_ids is not None and i in ret_samples_ids:
                ret_samples.append((images[0].detach().cpu(), targets[0], preds[0]))

            if save_results:
                for j in range(images.shape[0]):
                    img = images[j].detach().cpu().numpy().transpose(1, 2, 0)
                    img = (img * std + mean) * 255.0
                    img = img.clip(0, 255).astype(np.uint8)
                    target = VOCSegmentation.decode_target(targets[j]).astype(np.uint8)
                    pred = VOCSegmentation.decode_target(preds[j]).astype(np.uint8)

                    Image.fromarray(img).save(Path(save_root) / f"img_{img_id}.png")
                    Image.fromarray(target).save(Path(save_root) / f"label_{img_id}.png")
                    Image.fromarray(pred).save(Path(save_root) / f"pred_{img_id}.png")
                    img_id += 1

    val_score = metrics.get_results()
    val_loss = (total_loss / num_batches) if num_batches > 0 else None
    return val_score, ret_samples, val_loss

def setup_model_and_training(opts, total_itrs):
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
            max_iters=total_itrs
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

    # 设置tensorboard
    vis = Visualizer(logdir=opts.log_set.logdir,
                     env=opts.log_set.env,
                     exp_id=opts.log_set.exp_id)
    print(
        f"✅ TensorBoard enabled. Run 'tensorboard --logdir {opts.log_set.logdir}' to view.")

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
                            batch_size=opts.training.val_batch_size,
                            shuffle=False,
                            num_workers=opts.system.num_workers)
    total_itrs = int(opts.training.epoches) * len(train_loader)
    print(f"Dataset: {opts.dataset.name}, Train set: {len(train_ds)}, Val set: {len(val_ds)}")

    # 设置模型和训练组件
    model, optimizer, scheduler, criterion = setup_model_and_training(opts, total_itrs)
    metrics = StreamSegMetrics(opts.model.num_classes)

    # 设置并行训练
    model = nn.DataParallel(model)
    model.to(device)

    # 加载检查点
    model, optimizer, scheduler, cur_itrs, best_score = load_checkpoint(
        opts, model, optimizer, scheduler, device
    )
    if hasattr(scheduler, "max_iters"):
        scheduler.max_iters = total_itrs

    # 记录超参数
    hparams = {
        "model": opts.model.name,
        "output_stride": opts.model.output_stride,
        "num_classes": opts.model.num_classes,
        "loss": opts.training.loss.type,
        "optimizer": "SGD",
        "lr": float(opts.training.optimizer.lr),
        "lr_backbone_ratio": float(opts.training.optimizer.lr_backbone_ratio),
        "momentum": float(opts.training.optimizer.momentum),
        "weight_decay": float(opts.training.optimizer.weight_decay),
        "batch_size": int(opts.training.batch_size),
        "epochs": int(opts.training.epoches),
        "scheduler": opts.training.lr_scheduler.policy,
        "dataset": opts.dataset.name,
    }
    if is_main_process():
        vis.vis_text("config", yaml.safe_dump(config, sort_keys=False, allow_unicode=True))

    # 仅测试模式
    if opts.training.test_only:
        model.eval()
        val_score, _, val_loss = validate(opts, model, val_loader, device, metrics, criterion=criterion)
        print(metrics.to_str(val_score))
        if val_loss is not None and is_main_process():
            vis.vis_scalar("Loss/val", val_loss, step=cur_itrs)
            vis.vis_scalars({
                "Mean IoU": val_score["Mean IoU"],
                "Overall Acc": val_score["Overall Acc"],
                "Mean Acc": val_score["Mean Acc"],
                "FreqW Acc": val_score["FreqW Acc"],
            }, step=cur_itrs)
        if is_main_process():
            vis.vis_hparams(hparams, {"best_miou": float(val_score["Mean IoU"])})
            vis.close()
        return

    # ===== Training =====
    # 如果恢复训练，基于迭代数估算起始 epoch
    start_epoch = cur_itrs // len(train_loader) if len(train_loader) > 0 else 0

    for epoch in range(start_epoch, opts.training.epoches):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{opts.training.epoches}",
                    disable=not is_main_process())

        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            # print(images.shape, images.device)
            # print(labels.shape, labels.device)

            optimizer.zero_grad()
            outputs = model(images)
            # print(f"outputs.shape: {outputs.shape} outputs.device: {outputs.device}")
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            cur_itrs += 1

            # poly 策略按 iteration 调整
            if opts.training.lr_scheduler.policy == "poly":
                scheduler.step()

            if is_main_process():
                if cur_itrs % opts.training.print_interval == 0:
                    vis.vis_scalar("Loss/train", loss.item(), step=cur_itrs)
                    vis.vis_scalar("LR", optimizer.param_groups[0]["lr"], step=cur_itrs)
                    pbar.set_postfix({
                        "loss": f"{loss.item():.4f}",
                        "lr": f"{optimizer.param_groups[0]['lr']:.6f}"
                    })

            # 验证
            if opts.training.val_interval > 0 and cur_itrs % opts.training.val_interval == 0:
                val_score, _, val_loss = validate(
                    opts, model, val_loader, device, metrics, criterion=criterion
                )

                if is_main_process():
                    if val_loss is not None:
                        vis.vis_scalar("Loss/val", val_loss, step=cur_itrs)
                    vis.vis_scalars({
                        "Mean IoU": val_score["Mean IoU"],
                        "Overall Acc": val_score["Overall Acc"],
                        "Mean Acc": val_score["Mean Acc"],
                        "FreqW Acc": val_score["FreqW Acc"],
                    }, step=cur_itrs)
                    print(metrics.to_str(val_score))

                    # 保存最优模型
                    if val_score["Mean IoU"] > best_score:
                        best_score = float(val_score["Mean IoU"])
                        save_ckpt(
                            path=os.path.join(opts.log_set.logdir, "best_ckpt.pth"),
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            cur_itrs=cur_itrs,
                            best_score=best_score
                        )
                model.train()

        # step 策略按 epoch 调整
        if opts.training.lr_scheduler.policy == "step":
            scheduler.step()

    if is_main_process():
        vis.vis_hparams(hparams, {"best_miou": float(best_score)})
        vis.close()


if __name__ == "__main__":
    main()



