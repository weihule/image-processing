import argparse
from pathlib import Path
import logging
import os
import random
import sys
from tqdm import tqdm
import wandb
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader, random_split

from datasets import CarDataset
from models import UNet

# dir_img = r'/root/autodl-tmp/car/img'
# dir_mask = r'/root/autodl-tmp/car/mask'
# dir_checkpoint = './checkpoints/'

dir_img = r'D:\workspace\data\dl\car\img'
dir_mask = r'D:\workspace\data\dl\car\mask'
dir_checkpoint = './checkpoints/'


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', type=int, default=2, help='Batch size')
    parser.add_argument('--learning-rate', '-l', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    return parser.parse_args()


def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    pass


def main(args):
    logger.add("my_log.log", rotation="500 MB", level="INFO")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device {device}")

    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model.to(device)

    logger.info(f'Network:\n'
                f'\t{model.n_channels} input channels\n'
                f'\t{model.n_classes} output channels (classes)\n'
                f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict["mask_values"]
        model.load_state_dict(state_dict)
        logging.info(f"Model loaded from {args.load}")

    dataset = CarDataset(images_dir=dir_img, mask_dir=dir_mask, mask_suffix="_mask")

    # split into train / validation partitions
    val_percent = 0.1
    n_val = int(0.1 * len(dataset))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val],
                                      generator=torch.Generator().manual_seed(0))

    # create data loader
    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True,
                              drop_last=False)
    val_loader = DataLoader(val_set,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True,
                            drop_last=False)

    experiment = wandb.init(project="U-Net", resume="")
    for ds in train_loader:
        image = ds["image"]
        mask = ds["mask"]
        print(image.shape, mask.shape)


def run():
    args = get_args()
    main(args)


if __name__ == "__main__":
    run()
