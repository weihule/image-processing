import argparse
from pathlib import Path
import loguru
import os
import random
import sys
import cv2
import numpy as np
from tqdm import tqdm
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, random_split

from datasets.transform import transform
from datasets.voc import ImageDataSet, dataset_collate
from models.unet import UNet
from losses import dice_loss, dice_coeff, multiclass_dice_coeff

from utils.utils import weights_init

class GenPaths(object):
    def __init__(self):
        self.root = r"D:\workspace\data\images\Seg\people"
        self.train_txt = r"D:\workspace\data\images\Seg\people\ImageSets\Segmentation\train.txt"
        self.val_txt = r"D:\workspace\data\images\Seg\people\ImageSets\Segmentation\val.txt"

    def __call__(self):
        train_image_paths, train_mask_paths = self.load_paths(self.train_txt, self.root)
        val_image_paths, val_mask_paths = self.load_paths(self.val_txt, self.root)
        return train_image_paths, train_mask_paths, val_image_paths, val_mask_paths

    @staticmethod
    def load_paths(txt_file, root):
        image_paths, mask_paths = [], []
        with open(txt_file, "r", encoding="utf-8") as fr:
            for line in fr:
                filename = line.strip()  # 去除首尾空格或换行符
                image_paths.append(Path(root) / "JPEGImages" / f"{filename}.jpg")
                mask_paths.append(Path(root) / "SegmentationClass" / f"{filename}.png")
        return image_paths, mask_paths

def main():
    gp = GenPaths()
    train_image_paths, train_mask_paths, val_image_paths, val_mask_paths = gp()
    input_size = [640, 640]
    batch_size = 8
    num_workers = 4
    num_classes = 10
    pretrained = False
    backbone = 'resnet50'
    fp16 = True
    freeze_backbone_weight = True

    # 学习率等的设置
    optimizer_type      = "adam"
    momentum            = 0.9
    weight_decay        = 0
    lr_decay_type = 'cos'

    # 损失函数设置
    #   建议选项：
    #   种类少（几类）时，设置为True
    #   种类多（十几类）时，如果batch_size比较大（10以上），那么设置为True
    #   种类多（十几类）时，如果batch_size比较小（10以下），那么设置为False
    dice_loss = True
    focal_loss = False
    #   是否给不同种类赋予不同的损失权值，默认是平衡的。
    #   设置的话，注意设置成numpy形式的，长度和num_classes一样。
    #   如：
    #   num_classes = 3
    #   cls_weights = np.array([1, 2, 3], np.float32)
    cls_weights     = np.ones([num_classes], np.float32)

    # 输出文件保存设置
    save_period = 15    # 训练多少个epoch多存一次权重
    log_dir = r'D:\workspace\weight_data\train_weight\test_seg'

    transform_func = transform(input_size=input_size)
    train_ida = ImageDataSet(train_image_paths, train_mask_paths,
                             input_size=input_size,
                             num_classes=num_classes,
                             transform=transform_func['train'])
    val_ida = ImageDataSet(val_image_paths, val_mask_paths,
                           input_size=input_size,
                           num_classes=num_classes,
                           transform=transform_func['val'])

    train_dataloader = DataLoader(train_ida, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=True,
                                  drop_last=True,
                                  collate_fn=dataset_collate)
    val_dataloader = DataLoader(val_ida, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=True,
                                  drop_last=True,
                                  collate_fn=dataset_collate)
    # for sample in train_dataloader:
    #     images, masks, seg_labels = sample['image'], sample['mask'], sample['seg_label']
    #     print(images)
    #     print(images.shape, masks.shape, seg_labels.shape)
    #     break
    model = UNet(num_classes=num_classes,
                 pretrained=pretrained,
                 backbone=backbone)
    if not pretrained:
        weights_init(model)

    if fp16:
        scaler = GradScaler()
    else:
        scaler = None

    if freeze_backbone_weight:
        model.freeze_backbone()


def test():
    # voc = VOCSegmentation(voc_root=r"D:\workspace\data\VOCdataset")
    # sample_ = voc[10]
    # image_, target_ = sample_["image"], sample_["target"]
    # print(image_.size)
    img = cv2.imread(r'D:\workspace\data\images\VOCdevkit\VOC2012\SegmentationClass\2007_000032.png')
    print(img.shape)
    print(type(img))

if __name__ == "__main__":
    main()
    # test()
