import os
import math
import random
import time
from tqdm import tqdm

import torch
from torch.cuda import amp
import torch.nn.functional as F
from torch.backends import cudnn
from torch.optim import Adam, SGD, lr_scheduler
from torch.utils.data import DataLoader

from config import Config
from losses import CELoss
from utils.util import get_logger, load_state_dict


def train(logger, model, train_loader, criterion, optimizer, scheduler, epoch, device):
    model.train()
    mean_loss = 0.
    iter_idx = 1
    for datas in tqdm(train_loader):
        images = datas['image']
        labels = datas['label']
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        if Config.apex:
            scaler = amp.GradScaler()
            auto_cast = amp.autocast
            with auto_cast():
                preds = model(images)
                loss = criterion(preds, labels)
                mean_loss += loss.item()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = model(images)
            loss = criterion(preds, labels)
            mean_loss += loss.item()
            loss.backward()

            optimizer.step()
        if iter_idx % Config.print_interval == 0 or iter_idx == len(train_loader):
            # 'train epoch {epoch:3d}, iter [{iter_idx:5d}, {len(train_loader)}], loss: {loss}'
            lr_value = optimizer.state_dict()['param_groups'][0]['lr']
            logger.info(f"train epoch {epoch:3d}, iter [{iter_idx:5d}, {len(train_loader)}], loss: {loss.item():.3f} lr:{lr_value}")
        iter_idx += 1

    mean_loss = mean_loss / len(train_loader)
    print('epoch: {}  mean_loss: {:.3f}'.format(epoch, mean_loss))
    scheduler.step()

    return mean_loss


def evaluate_acc(model, val_loader, device):
    model.eval()
    model = model.to(device)
    correct = 0
    with torch.no_grad():
        for datas in tqdm(val_loader):
            images = datas['image']
            labels = datas['label']
            images, labels = images.to(device), labels.to(device)
            preds = model(images)   # [batch_size, num_classes]
            preds = F.softmax(preds, dim=1)
            _, max_indices = torch.max(preds, dim=1)    # [batch_size]
            correct += torch.eq(max_indices, labels).sum().item()
    val_acc = round(correct / len(Config.val_dataset), 3)

    return val_acc


def main(logger):
    torch.cuda.empty_cache()

    if Config.seed:
        random.seed(Config.seed)
        torch.manual_seed(Config.seed)
        torch.cuda.manual_seed(Config.seed)
        torch.cuda.manual_seed_all(Config.seed)
        cudnn.deterministic = True

    gpus = torch.cuda.device_count()
    logger.info(f'use {gpus} gpus')

    cudnn.benchmark = True
    cudnn.enabled = True
    start_time = time.time()

    device = Config.device

    logger.info('start loading data')
    train_loader = DataLoader(Config.train_dataset,
                              batch_size=Config.batch_size,
                              shuffle=True,
                              num_workers=Config.num_workers,
                              pin_memory=True,
                              collate_fn=Config.cls_collater,
                              prefetch_factor=6)
    logger.info('finish loading data')
    val_loader = DataLoader(Config.val_dataset,
                            batch_size=Config.batch_size,
                            shuffle=False,
                            num_workers=Config.num_workers,
                            pin_memory=True,
                            collate_fn=Config.cls_collater,
                            prefetch_factor=4)

    model = Config.model
    model = model.to(device)

    load_state_dict(Config.pre_weight_path, model, [])

    # 冻结特征提取层的参数
    if Config.freeze_layer:
        for name, param in model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(params, lr=Config.lr, weight_decay=4E-5)
    lf = lambda x: ((1 + math.cos(x * math.pi / Config.epochs)) / 2) * (1 - Config.lrf) + Config.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    criterion = CELoss(use_custom=False)

    best_acc = 0.
    start_epoch = 1

    # resume training
    if os.path.exists(Config.resume):
        logger.info(f'start resume model from {Config.resume}')
        checkpoint = torch.load(Config.resume, map_location=torch.device('cpu'))
        start_epoch += checkpoint['epoch']
        model.load_state_dict(checkpoint['model_load_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info(f'finish resume model from {Config.resume}')
        logger.info(f"epoch {checkpoint['epoch']}, best_acc: {checkpoint['best_acc']}, loss: {checkpoint['loss']}")
        logger.info(f'finish resume model from {Config.resume}')

    logger.info('start training')
    print('start_epoch = ', start_epoch)
    for epoch in range(start_epoch, Config.epochs+1):
        mean_loss = train(logger=logger,
                          model=model,
                          train_loader=train_loader,
                          criterion=criterion,
                          optimizer=optimizer,
                          scheduler=scheduler,
                          epoch=epoch,
                          device=device)
        logger.info(f"train: epoch: {epoch}, loss: {mean_loss:.3f}")
        if epoch % Config.save_interval == 0 or epoch == Config.epochs:
            val_acc = evaluate_acc(model, val_loader, device)
            logger.info(f"epoch {epoch}, val_acc {val_acc}")
            print('epoch: {} acc: {:.3f}'.format(epoch+1, val_acc))

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), os.path.join(Config.save_root, 'pths', 'best.pth'))

            torch.save({
                'epoch': epoch,
                'best_acc': best_acc,
                'loss': mean_loss,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }, os.path.join(Config.checkpoints, 'latest.pth'))
    train_time = (time.time() - start_time) / 60
    logger.info(f'finish training, total training timr: {train_time:.2f} mins')


if __name__ == "__main__":
    if not os.path.exists(Config.save_root):
        os.mkdir(Config.save_root)
    if not os.path.exists(Config.log):
        os.mkdir(Config.log)
    if not os.path.exists(Config.checkpoints):
        os.mkdir(Config.checkpoints)
    if not os.path.exists(Config.pth_path):
        os.mkdir(Config.pth_path)

    logger_writer = get_logger('resnet', Config.log)
    main(logger_writer)

