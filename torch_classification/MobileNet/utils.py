import os
import sys
import json
from tqdm import tqdm
import random

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

label_file = '/nfs/home57/weihule/code/study/torch_classification/shuffleNet/class_indices.txt'
if not os.path.exists(label_file):
    label_file = '/workshop/weihule/code/study/torch_classification/shuffleNet/class_indices.txt'
if not os.path.exists(label_file):
    label_file = 'D:\\workspace\\code\\study\\torch_classification\\shuffleNet\\class_indices.txt'
label_dict = dict()
with open(label_file, "r", encoding="utf-8") as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip("\n")
        label_dict[line.split(":")[0]] = line.split(":")[1]

inverse_label_dict = {v:k for k, v in label_dict.items()}


def read_split_data(root: str, mode: str):
    # for k, v in label_dict.items():
    #     print(k, v)
    images_path = list()  # 图片路径
    images_label = list()  # 图片对应索引信息
    for cls in label_dict.keys():
        img_list = os.listdir(os.path.join(root, mode, cls))
        images_path.extend([os.path.join(root, mode, cls, fn) for fn in img_list])
        images_label.extend([int(label_dict[cls]) for _ in range(len(img_list))])

    return images_path, images_label


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    # model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    mean_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()
    data_loader = tqdm(data_loader)

    for step, data in enumerate(data_loader):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        pred = model(images)
        loss = loss_function(pred, labels)
        loss.backward()
        mean_loss = (mean_loss + loss.detach()) / (step + 1)

        data_loader.desc = '[epoch {}] mean_loss {}'.format(epoch+1, round(mean_loss.item(), 3))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, end training', loss)
            sys.exit(1)

        optimizer.step()

        return mean_loss.item()


@torch.no_grad()
def evaluate(model, data_loader, device):

    # 验证样本总个数
    total_num = len(data_loader.dataset)

    # 存储预测正确样本个数
    sum_num = torch.zeros(1).to(device)

    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        pred = model(images)
        pred = torch.argmax(pred, dim=1)
        sum_num += torch.eq(pred, labels).sum()

    return sum_num.item() / total_num, sum_num.item(), total_num


if __name__ == "__main__":
    root = "/nfs/home57/weihule/data/DL/flower"
    images_path, images_label = read_split_data(root, mode="train")
    for path, index in zip(images_path, images_label):
        print(path, index)
