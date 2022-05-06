import os
import torch 
import torch.nn as nn
import cv2
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import shufflenet_v2_x1_0
# from torchvision.models.shufflenetv2 import shufflenet_v2_x1_0
from utils import read_split_data
import argparse

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter()

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)
    model = shufflenet_v2_x1_0(num_classes=10)

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--num_classes", type=int, default=10)
    parse.add_argument("--epoches", type=int, default=10)
    parse.add_argument("--batch_size", type=int, default=64)
    parse.add_argument("--lr", type=float, default=0.001)
    parse.add_argument("--lrf", type=float, default=0.001)

    # 数据集所在根目录
    parse.add_argument("--data_path", type=str, default="/workshop/cyclone_data/process/data/DL/flower")
    # parse.add_argument("--data_path", type=str, default="/nfs/home57/weihule/data/DL/flower")

    # 预训练权重
    # parse.add_argument("--weights", type=str, default="")
    # parse.add_argument("--weights", type=str, default="")

    parse.add_argument("--freeze_layers", type=bool, default=True)
    parse.add_argument("--device", default="cuda:0")

    opt = parse.parse_args()
    main(opt)