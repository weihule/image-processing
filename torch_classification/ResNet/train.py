import os
import sys
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
import torchvision
from tqdm import tqdm
import json
import glob
import random
from custom_dataset import CustomDataset
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import math
from model import resnet50, resnet18

sys.path.append(os.getcwd())


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.RandomResizedCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    }

    root = "../dataset"

    with open("./demo/class_indices.json") as f:
        cls_idx = json.load(f)
    reverse_idx = {v: k for k, v in cls_idx.items()}

    #     # train_images
    #     train_img_path = []
    #     train_img_label = []

    #     for i in glob.glob(os.path.join(root, "train/*/*")):
    #         train_img_path.append(i)

    #     random.shuffle(train_img_path)
    #     # print(train_img_path[0])
    #     for i in train_img_path:
    #         train_img_label.append(int(reverse_idx[i.split(os.sep)[-2]]))

    #    # val_images
    #     val_img_path = []
    #     val_img_label = []

    #     for i in glob.glob(os.path.join(root, "val/*/*")):
    #         val_img_path.append(i)

    #     random.shuffle(val_img_path)
    #     for i in val_img_path:
    #         val_img_label.append(int(reverse_idx[i.split(os.sep)[-2]]))

    BATCHSIZE = 4
    nw = 4
    # ------------------------------------------------------------
    # train_ds = MyDataSet(train_img_path, train_img_label, data_transform["train"])

    train_ds = datasets.ImageFolder(root=os.path.join(root, "train"), transform=data_transform["train"])
    train_loader = DataLoader(train_ds,
                              batch_size=BATCHSIZE,
                              shuffle=True,
                              num_workers=nw,
                              )
    train_num = len(train_ds)

    # validate_ds = MyDataSet(val_img_path, val_img_label, data_transform["train"])

    validate_ds = datasets.ImageFolder(root=os.path.join(root, "val"), transform=data_transform["val"])
    validate_loader = DataLoader(validate_ds,
                                 batch_size=BATCHSIZE,
                                 shuffle=False,
                                 num_workers=nw)
    val_num = len(validate_ds)

    print("using {} imgaes for training, {} images for validation".format(train_num, val_num))

    # show images
    for images, labels in train_loader:
        images, labels = next(iter(train_loader))
        print(images.size(), labels.size())
        for i in range(BATCHSIZE):
            plt.subplot(1, BATCHSIZE, i + 1)
            index = str(labels[i].item())
            plt.title(index)
            mean = torch.tensor((0.485, 0.456, 0.406))
            std = torch.tensor((0.229, 0.224, 0.225))
            mean = mean.view(3, 1, 1).expand(3, 224, 224)
            std = std.view(3, 1, 1).expand(3, 224, 224)
            images[i] = images[i] * std + mean
            plt.imshow(images[i].permute(1, 2, 0))
            plt.axis("off")
        plt.show()
        break

    net = resnet50()
    # change fc layer structure
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 5)
    net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.0001)

    EPOCHS = 55
    best_acc = 0.0
    save_path = './weights/resNet50_01.pth'
    train_steps = len(train_loader)

    for epoch in range(EPOCHS):
        # train
        net.train()
        # 每个epoch, running_loss加了train_steps次
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for data in train_bar:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = net(images)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(
                epoch, EPOCHS, loss.item()
            )

        # validata
        net.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                images, labels = val_data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                predict_y = torch.max(outputs, dim=-1)[1]
                acc += torch.eq(predict_y, labels).sum().item()

                val_bar = "valid epoch[{}/{}]".format(epoch, EPOCHS)

        val_acc = acc / val_num
        print("epoch:{}  training loss:{}  val acc:{}".format(
            epoch, running_loss / train_steps, val_acc
        ))

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), save_path)

    print("Finished Training")


def train_cifar():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 128
    epochs = 100
    lr = 0.001
    lrf = 0.001
    data_transform = {
        'train': transforms.Compose([transforms.RandomCrop(32, padding=4),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                     ]),

        'val': transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    }

    # data_path = 'D:\\workspace\\data\\DL'
    data_path = '/nfs/home57/weihule/data/DL'
    # data_path = '/workshop/weihule/data/DL'

    log_file = '/nfs/home57/weihule/data/weights/resnet/resnet18_cifar.txt'
    train_set = torchvision.datasets.CIFAR10(root=data_path,
                                             train=True,
                                             transform=data_transform['train'],
                                             download=True)
    val_set = torchvision.datasets.CIFAR10(root=data_path,
                                           train=False,
                                           transform=data_transform['val'],
                                           download=True)
    print('the number of training: {}'.format(len(train_set)))
    print('the number of validation: {}'.format(len(val_set)))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)

    model = resnet18(num_classes=10)
    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=lr)
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    loss_function = nn.CrossEntropyLoss()
    best_acc = 0.
    for epoch in range(epochs):
        model.train()
        mean_loss = 0.
        train_bar = tqdm(train_loader)
        for datas in train_bar:
            images, labels = datas
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            pred_y = model(images)
            loss = loss_function(pred_y, labels)
            loss.backward()
            mean_loss += loss.item()
            optimizer.step()

            train_bar.desc = '[epoch {}] loss {}'.format(epoch + 1, round(loss.item(), 3))

        print('mean_loss: {}'.format(round(mean_loss / (len(train_loader) + 1), 3)))

        # 每个epoch之后, 更新lr
        scheduler.step()

        # val
        model.eval()
        val_bar = tqdm(val_loader)
        correct = 0
        all_num = len(val_set)
        for images, labels in val_bar:
            images, labels = images.to(device), labels.to(device)
            pred = model(images)
            pred = torch.argmax(pred, dim=1)
            correct += torch.eq(pred, labels).sum().item()

        print('[epoch {}] accuracy {} ({}/{})'.format(epoch + 1, round(correct / all_num, 3), correct, all_num))

        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(str(round(mean_loss / (len(train_loader) + 1), 3)) + '\t' + str(round(correct / all_num, 3)) + '\n')


if __name__ == "__main__":
    # main()

    train_cifar()
