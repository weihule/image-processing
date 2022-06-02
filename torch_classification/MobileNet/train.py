import collections
import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import transforms, datasets
from torch.optim import Adam, SGD, lr_scheduler
from torch.utils.data import DataLoader
from pathlib import Path
from collections import OrderedDict
from torchvision.models.mobilenetv2 import mobilenet_v2
# from torchvision.models.mobilenetv3 import mobilenet_v3_small
from utils import read_split_data, evaluate
from modelV2 import MobileNetV2
from custom_dataste import CustomDataset


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device for training".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    image_path = "../dataset"
    BATCHSIZE = 4
    nw = min([os.cpu_count(), BATCHSIZE if BATCHSIZE > 1 else 0, 8])
    train_dataset = datasets.ImageFolder(os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    # {'daisy': 0, 'dandelion': 1, 'roses': 2, 'sunflowers': 3, 'tulips': 4}
    flower_list = train_dataset.class_to_idx

    invert_dict = {v: k for k, v in flower_list.items()}

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCHSIZE,
                              shuffle=True,
                              num_workers=nw)

    val_dataset = datasets.ImageFolder(os.path.join(image_path, "val"),
                                       transform=data_transform["val"])
    val_num = len(val_dataset)
    val_loader = DataLoader(val_dataset,
                            batch_size=BATCHSIZE,
                            shuffle=False,
                            num_workers=nw)
    print("using {} images for training, {} images for validation".format(train_num, val_num))

    # create model
    net = MobileNetV2(num_classes=5)
    net.to(device)

    # pre_weights_path = "./weights/mobilenet_v2_pre.pth"
    pre_weights_path = "./weights/mobilenet_v3_small_pre.pth"
    # 字典形式
    pre_dict = torch.load(pre_weights_path, map_location=device)
    # for k, v in pre_dict.items():
    #     print(k)
    # 只载入特征提取层的权重
    pre_dict = collections.OrderedDict({k: v for k, v in pre_dict.items() if "classifier" not in k})
    net.load_state_dict(pre_dict, strict=False)

    # freeze features weights
    for param in net.features.parameters():
        param.requires_grad = False

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.0001)

    best_acc = 0.0
    save_path = "./weights/mobileNetV3_small_01.pth"
    train_steps = len(train_loader)
    EPOCHS = 50
    for epoch in range(EPOCHS):
        # train
        net.train()
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
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     EPOCHS, loss.item())

        # validate
        net.eval()
        acc = 0.
        with torch.no_grad():
            val_bar = tqdm(val_loader)
            for data in val_bar:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                predict = torch.max(outputs, dim=-1)[1]
                acc += torch.eq(predict, labels).sum().item()

                val_bar.desc = "valid epoch {}/{}".format(epoch + 1,
                                                          EPOCHS)

        val_acc = acc / val_num
        print("epoch {}/{} loss:{:.3f} acc:{:.3f}".format(epoch + 1, EPOCHS,
                                                          running_loss / train_steps, val_acc))

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), save_path)

    print("Finished")


def main_1():
    dataset_path = '/workshop/weihule/data/DL/flower'
    pre_weight_path = '/workshop/weihule/data/weights/mobilenet/mobilenet_v2_pre.pth'
    epochs = 30
    batch_size = 128
    lr = 0.001
    lrf = 0.001
    num_workers = 4
    freeze_layer = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_images_path, train_images_label = read_split_data(dataset_path, "train")
    val_images_path, val_images_label = read_split_data(dataset_path, "val")

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
    train_loader = DataLoader(train_set, batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              collate_fn=train_set.collate_fn)
    val_loader = DataLoader(val_set, batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            collate_fn=val_set.collate_fn)

    model = MobileNetV2(num_classes=5)
    # model = mobilenet_v2()
    # model.classifier = nn.Sequential(
    #     nn.Dropout(0.2),
    #     nn.Linear(1280, 5)
    # )
    model = model.to(device)

    # # 加载预训练权重
    # if pre_weight_path:
    #     weights_dict = torch.load(pre_weight_path, map_location=device)
    #     load_dict = OrderedDict({k: v for k, v in weights_dict.items() if 'classifier' not in k})
    #     model.load_state_dict(load_dict, strict=False)
    # else:
    #     raise FileNotFoundError('not found weights file {}'.format(pre_weight_path))
    #
    # # 冻结特征提取层的参数
    # if freeze_layer:
    #     for name, param in model.named_parameters():
    #         print(name)
    #         if 'classifier' not in name:
    #             param.requires_grad = False

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(params, lr=lr, weight_decay=4E-5)
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    criterion = torch.nn.CrossEntropyLoss()

    best_acc = 0
    for epoch in range(epochs):
        model.train()
        train_bar = tqdm(train_loader)
        mean_loss = 0.
        for datas in train_bar:
            images, labels = datas
            images, labels = images.to(device), labels.to(device)
            pred_y = model(images)
            loss = criterion(pred_y, labels)
            mean_loss += loss.item()
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            train_bar.desc = 'epoch [{}/{}] loss: {:.3f}'.format(epoch+1, epochs, loss.item())
        print('epoch: {}  mean_loss: {:.3f}'.format(epoch, mean_loss / len(train_loader)))
        scheduler.step()

        model.eval()
        val_bar = tqdm(val_loader)
        correct = 0
        for images, labels in val_bar:
            images, labels = images.to(device), labels.to(device)
            pred = model(images)
            probs = F.softmax(pred, dim=1)
            pred_y = torch.argmax(probs, dim=1)
            correct += torch.eq(pred_y, labels).sum().item()
            val_bar.desc = 'epoch [{}/{}]'.format(epoch+1, epochs)
        print('epoch: {} acc: {:.3f}'.format(epoch+1, correct / len(val_set)))


if __name__ == "__main__":
    main_1()

    # main()
