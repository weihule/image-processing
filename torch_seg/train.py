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

from datasets import CarDataset
from models import UNet
from losses import dice_loss, dice_coeff, multiclass_dice_coeff

# dir_img = r'/root/autodl-tmp/car/img'
# dir_mask = r'/root/autodl-tmp/car/mask'
# dir_checkpoint = './checkpoints/'

dir_img = r'/home/8TDISK/weihule/data/car/img'
dir_mask = r'/home/8TDISK/weihule/data/car/mask'
dir_checkpoint = '/home/8TDISK/weihule/training_data/seg/unet_1.0/'


# dir_img = r'D:\workspace\data\dl\car\img'
# dir_mask = r'D:\workspace\data\dl\car\mask'
# dir_checkpoint = './checkpoints/'


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', type=float, default=1e-5, help='Learning rate')
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
        train_loader,
        criterion,
        optimizer,
        scheduler,
        epoch,
        grad_scaler,
        amp: bool = False,
        gradient_clipping: float = 1.0,
):
    model.train()
    epoch_loss = 0
    for batch in tqdm(train_loader):
        images, true_masks = batch['image'], batch['mask']
        assert images.shape[1] == model.n_channels, \
            f'Network has been defined with {model.n_channels} input channels, ' \
            f'but loaded images have {images.shape[1]} channels. Please check that ' \
            'the images are loaded correctly.'
        images = images.to(device=device)
        true_masks = true_masks.to(device=device)

        with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
            masks_pred = model(images)
            print(masks_pred.shape, true_masks.shape)
            if model.n_classes == 1:
                loss = criterion(masks_pred.squeeze(1), true_masks.float())
                loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)),
                                  true_masks.float(),
                                  multiclass=False)
            else:
                loss = criterion(masks_pred, true_masks)
                loss += dice_loss(
                    F.softmax(masks_pred, dim=1).float(),
                    F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                    multiclass=True
                )
        optimizer.zero_grad(set_to_none=True)
        grad_scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        grad_scaler.step(optimizer)
        grad_scaler.update()

        epoch_loss += loss.item()
        # break
    return epoch_loss


@torch.inference_mode()
def evaluate(model, dataloader, device, amp):
    # 验证模式
    model.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']
            image, mask_true = image.to(device), mask_true.to(device)

            mask_pred = model(image)

            if model.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5)
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < model.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                mask_true = F.one_hot(mask_true, model.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), model.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

    return dice_score / max(num_val_batches, 1)


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

    save_checkpoint = True
    img_scale = 0.5

    logger.info(f'''Starting training:
        Epochs:          {args.epochs}
        Batch size:      {args.batch_size}
        Learning rate:   {args.learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device}
        Images scaling:  {img_scale}
        Mixed Precision: {args.amp}
    ''')

    # set up the optimizer and scheduler
    weight_decay = 1e-8
    momentum = 0.99
    optimizer = optim.Adam(model.parameters(),
                           lr=args.learning_rate,
                           weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    save_interval = 2
    epochs = 5
    for epoch in range(1, epochs + 1):
        mean_loss = train_model(
            model=model,
            device=device,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            grad_scaler=grad_scaler,
            epoch=epoch,
        )
        if epoch % save_interval == 0 or epoch == epochs:
            val_score = evaluate(model, val_loader, device, args.amp)
            scheduler.step(val_score)
            logger.info('Validation Dice score: {}'.format(val_score))

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, str(Path(dir_checkpoint) / 'checkpoint_epoch{}.pth'.format(epoch)))
            logger.info(f'Checkpoint {epoch} saved!')


def run():
    args = get_args()
    main(args)


def test(num):
    num = num - 5
    return num


if __name__ == "__main__":
    run()
