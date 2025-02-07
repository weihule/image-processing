import argparse
from pathlib import Path
import logging
import os
import random
import sys
from tqdm import tqdm
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader, random_split

from datasets.transform import transform
from datasets.voc import ImageDataSet
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
    transform_func = transform()
    train_ida = ImageDataSet(train_image_paths, train_mask_paths, transform=transform_func['train'])
    val_ida = ImageDataSet(val_image_paths, val_mask_paths, transform=transform_func['val'])

    sample = train_ida[1]


if __name__ == "__main__":
    main()
