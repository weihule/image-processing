import argparse
from pathlib import Path
import loguru
import os
import random
import sys
import cv2
from tqdm import tqdm
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader, random_split

from datasets.transform import transform
from datasets.voc import ImageDataSet, dataset_collate
from models import UNet
from losses import dice_loss, dice_coeff, multiclass_dice_coeff

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
    batch_size = 4
    num_workers = 4
    num_classes = 10
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
    for sample in train_dataloader:
        images, masks, seg_labels = sample['image'], sample['mask'], sample['seg_label']
        print(images)
        print(images.shape, masks.shape, seg_labels.shape)
        break


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
