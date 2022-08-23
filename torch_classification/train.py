import os
import math
import torch
from collections import OrderedDict
import torch.nn as nn
from torch.cuda import amp
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import transforms, datasets
from torch.optim import Adam, SGD, lr_scheduler
from torch.utils.data import DataLoader
from utils.datasets import collater
from config import Config
from losses import CELoss


def train(model, train_loader, criterion, optimizer, scheduler, epoch, device):
    model.train()
    mean_loss = 0.
    for datas in tqdm(train_loader):
        images, labels = datas
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

    mean_loss = mean_loss / len(train_loader)
    print('epoch: {}  mean_loss: {:.3f}'.format(epoch, mean_loss))
    scheduler.step()

    return mean_loss


def main():
    device = Config.device
    train_set = Config.train_dataset
    val_set = Config.val_dataset
    train_loader = DataLoader(train_set,
                              batch_size=Config.batch_size,
                              shuffle=True,
                              num_workers=Config.num_workers,
                              collate_fn=collater)
    val_loader = DataLoader(val_set,
                            batch_size=Config.batch_size,
                            shuffle=False,
                            num_workers=Config.num_workers,
                            collate_fn=collater)

    model = Config.model
    model = model.to(device)

    # 加载预训练权重
    if Config.pre_weight_path:
        weights_dict = torch.load(Config.pre_weight_path, map_location=device)
        load_dict = OrderedDict({k: v for k, v in weights_dict.items() if 'fc' not in k})
        model.load_state_dict(load_dict, strict=False)

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

    best_acc = 0
    for epoch in range(Config.epochs):
        mean_loss = train(model=model,
                          train_loader=train_loader,
                          criterion=criterion,
                          optimizer=optimizer,
                          scheduler=scheduler,
                          epoch=epoch,
                          device=device)

        if epoch % 2 == 0 or epoch == Config.epochs:
            model.eval()
            correct = 0
            with torch.no_grad():
                for images, labels in tqdm(val_loader):
                    images, labels = images.to(device), labels.to(device)
                    pred = model(images)
                    probs = F.softmax(pred, dim=1)
                    pred_y = torch.argmax(probs, dim=1)
                    correct += torch.eq(pred_y, labels).sum().item()
            val_acc = round(correct / len(val_set), 3)
            print('epoch: {} acc: {:.3f}'.format(epoch+1, val_acc))

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), Config.save_path + '_' + str(val_acc).replace('0.', '') + '.pth')


if __name__ == "__main__":
    main()

