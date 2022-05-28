import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import sys
import torch
import numpy as np
from tqdm import tqdm
import math
import cv2
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler, SGD, Adam
from model import shufflenet_v2_x1_0
# from torchvision.models.shufflenetv2 import shufflenet_v2_x1_0
from utils import read_split_data, train_one_epoch, evaluate
from custom_dataset import CustomDataset
import argparse
from collections import OrderedDict
import matplotlib.pyplot as plt


def main(args):
    # label_file = 'D:\\workspace\\code\\study\\torch_classification\\shuffleNet\\class_indices.txt'

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    # tb_writer = SummaryWriter()

    train_images_path, train_images_label = read_split_data(args.data_path, "train")
    val_images_path, val_images_label = read_split_data(args.data_path, "val")

    # train_set = datasets.ImageFolder(os.path.join(args.data_path, "train"))

    data_transform = {
        'train': transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        'val': transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }

    train_set = CustomDataset(train_images_path, train_images_label, data_transform['train'])
    val_set = CustomDataset(val_images_path, val_images_label, data_transform['val'])

    # train_set = datasets.ImageFolder(root=os.path.join(args.data_path, 'train'),
    #                                  transform=data_transform['train'])
    #
    # val_set = datasets.ImageFolder(root=os.path.join(args.data_path, 'val'),
    #                                  transform=data_transform['val'])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=nw,
                              collate_fn=train_set.collate_fn)
    # collate_fn=train_set.collate_fn
    val_loader = DataLoader(val_set,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=nw,
                            collate_fn=val_set.collate_fn)
    # collate_fn=val_set.collate_fn
    model = shufflenet_v2_x1_0(num_classes=args.num_classes)
    model = model.to(device)

    # for param in model.parameters():
    #     print(param, param.numel())

    if args.weights != '':
        if os.path.exists(args.weights):
            weights_dict = torch.load(args.weights, map_location=device)
            load_weights_dict = OrderedDict({k: v for k, v in weights_dict.items()
                                 if 'fc' not in k})
            # load_weights_dict = {k: v for k, v in weights_dict.items()
            #                      if 'fc' not in k}
            # load_weights_dict = {k: v for k, v in weights_dict.itmes()
            #                      if model.state_dict()[k].numel() == v.numel()}
            model.load_state_dict(load_weights_dict, strict=False)
        else:
            raise FileNotFoundError('not found weights file: {}'.format(args.weights))

    # 是否冻结权重
    if args.freeze_layers:
        for name, param in model.named_parameters():
            param.requires_grad = False if 'fc' not in name else True

    pg = [p for p in model.parameters() if p.requires_grad]
    # optimizer = SGD(pg, lr=args.lr, momentum=0.9, weight_decay=4E-5)
    optimizer = Adam(pg, lr=args.lr, weight_decay=4E-5)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    loss_function = torch.nn.CrossEntropyLoss()
    save_root = '/nfs/home57/weihule/data/weights/shufflenetv2'
    # save_root = 'D:\\workspace\\data\\weights\\shufflenetv2'
    best_acc = 0.
    for epoch in range(args.epochs):
        # train
        model.train()
        mean_loss = torch.zeros(1).to(device)
        train_loader = tqdm(train_loader)
        for step, data in enumerate(train_loader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            
            pred = model(images)
            loss = loss_function(pred, labels)
            loss.backward()
            mean_loss += loss.detach()

            train_loader.desc = '[epoch {}] loss {}'.format(epoch+1, round(loss.item(), 3))

            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, end training', loss)
                sys.exit(1)

            optimizer.step()
            optimizer.zero_grad()
        mean_loss = round(mean_loss.item() / (len(train_loader)+1), 3)
        print('mean_loss: {}'.format(mean_loss))
        # 每个epoch之后, 更新lr
        scheduler.step()
        # print(optimizer.param_groups[0]['lr'])

        # val
        model.eval()
        acc, sum_num, total_num = evaluate(model, val_loader, device)

        print('[epoch {}] accuracy {} ({}/{})'.format(epoch + 1, round(acc, 3), sum_num, total_num))
    #     tags = ['loss', 'accuracy', 'learning_rate']
    #     # tb_writer.add_scalar(tags[0], mean_loss, epoch)
    #     # tb_writer.add_scalar(tags[1], acc, epoch)
    #     # tb_writer.add_scalar(tags[2], optimizer.param_groups[0]['lr'], epoch)
    #     if acc > best_acc:
    #         best_acc = acc
    #         save_path = os.path.join(save_root, 'model-0522.pth')
    #         torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--num_classes", type=int, default=5)
    parse.add_argument("--epochs", type=int, default=50)
    parse.add_argument("--batch_size", type=int, default=128)
    parse.add_argument("--lr", type=float, default=0.001)
    parse.add_argument("--lrf", type=float, default=0.001)

    # 数据集所在根目录
    parse.add_argument("--data_path", type=str, default="/workshop/weihule/data/DL/flower")
    # parse.add_argument("--data_path", type=str, default="D:\\workspace\\data\\DL\\flower")
    # parse.add_argument("--data_path", type=str, default="/nfs/home57/weihule/data/DL/flower")

    # 预训练权重
    parse.add_argument("--weights", type=str,
                       default="/workshop/weihule/data/weights/shufflenetv2/shufflenetv2_x1_pre.pth")
    # parse.add_argument("--weights", type=str,
    #                    default="/nfs/home57/weihule/data/weights/shufflenetv2/shufflenetv2_x1_pre.pth")
    # parse.add_argument("--weights", type=str,
    #                    default="D:\\workspace\\data\\weights\\shufflenetv2\\shufflenetv2_x1_pre.pth")

    parse.add_argument("--freeze_layers", type=bool, default=True)
    parse.add_argument("--device", default="cuda:0")

    opt = parse.parse_args()
    main(opt)

    # x = [1650, 2000, 2500]
    # xp = [1625, 2920]
    # fp = [685, 699]
    # y = np.interp(x, xp, fp)
    #
    # # x = [1, 3, 5]
    # # xp = [0, 10]
    # # fp = [0, 10]
    # # y = np.interp(x, xp, fp)
    # plt.plot(xp, fp, '-o')
    # plt.plot(x, y, 'x')
    # plt.show()









