import os
import argparse
from loguru import logger
from tqdm import tqdm
from pathlib import Path
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split

from unet import UNet
from utils.data_loading import BasicDataset, CarvanDataset
from utils.dice_score import dice_loss


def train_model(
        dirs,
        model,
        device,
        epochs,
        batch_size,
        learning_rate,
        val_percent=0.1,
        save_checkpoint=True,
        img_scale=0.5,
        amp=False,
        weight_decay=1e-8,
        momentum=0.999,
        gradient_clipping=1.0,
):
    dir_img, dir_mask, img_scale = dirs['dir_img'], dirs['dir_mask'], dirs['dir_checkpoint']
    # 1. Create Dataset
    try:
        dataset = CarvanDataset(dir_img, dir_mask, img_scale)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale)
    
    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], 
                                      generator=torch.Generator().manual_seed(0))
    
    # 3. Create Data Loader
    train_loader = DataLoader(train_set, 
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=os.cpu_count(),
                              pin_memory=True)
    val_loader = DataLoader(val_set,
                            batch_size=batch_size,
                            shuffle=False,
                            drop_last=True,
                            num_workers=os.cpu_count(),
                            pin_memory=True)
    
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(
            dict(epochs=epochs, batch_size=batch_size, 
                 learning_rate=learning_rate,
                 val_percent=val_percent, 
                 save_checkpoint=save_checkpoint, 
                 img_scale=img_scale, amp=amp)
        )

    logger.info(f"""
        ---- Starting training: ----
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    """)

    # Adam
    # optimizer = optim.Adam(
    #     model.parameters(),
    #     lr=0.001,           # 学习率
    #     betas=(0.9, 0.999), # (β1, β2)：一阶矩和二阶矩的衰减率
    #     eps=1e-08,          # 数值稳定性
    #     weight_decay=0      # L2正则化（建议用AdamW）
    # )
    # AdamW（修正版Adam，更好的权重衰减）
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay   # 解耦的权重衰减
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        'max',  # 监控的指标越大越好(这里用的指标是Dice系数， Dcie越大模型效果越好)
        patience=5
    )
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch["mask"]

                assert images.shape[1] == model.n_channel, f"""
                    Network has been defined with {model.n_channels} input channels, 
                    but loaded images have {images.shape[1]} channels. 
                    """
                
                # torch.channels_last优化计算速度，使其物理存储变成[NHWC]，但是实际shape还是[NCHW]
                images = images.to(device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        """
                        model的输出是[N, n_classes, H, W]
                        真实标签 [batch_size, H, W]
                        问题：形状不匹配, 所以移除所有大小为1的维度
                        """
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())