# -*- coding: utf-8 -*-
import os
import cv2
import torch
import random
import numpy as np
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import torch.nn.functional as F
from utils.util import cal_macs_params
from backbones.model_manager import init_model
import time
from PIL import Image
from collections import Counter


labels = ['below', 'NoChamfer', 'rough', 'normal']


def inference(cfgs):
    """
    以文件夹的形式推理
    """
    if cfgs["seed"]:
        seed = cfgs["seed"]
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)  # 为cpu设置随机数种子

    device = torch.device("cuda:0")
    num_classes = cfgs["num_classes"]

    model = init_model(backbone_type=cfgs["model"],
                       num_classes=num_classes)
    model = model.to(device)

    if cfgs["train_model_path"]:
        weight_dict = torch.load(cfgs["train_model_path"], map_location=torch.device("cpu"))
        model.load_state_dict(weight_dict, strict=True)

    model.eval()

    image_paths = list()
    for img_name in os.listdir(cfgs["test_image_dir"]):
        image_paths.append(os.path.join(cfgs["test_image_dir"], img_name))
    infer_num = len(image_paths)
    image_paths = [image_paths[s: s + cfgs["batch_size"]] for
                   s in range(0, len(image_paths), cfgs["batch_size"])]

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    mean = np.asarray(mean, dtype=np.float32).reshape((1, 1, 1, 3))
    std = np.asarray(std, dtype=np.float32).reshape((1, 1, 1, 3))

    start_time = time.time()

    res = []
    names = []
    scores = []
    for batch_datas in image_paths:
        batch_images = []
        for img_path in batch_datas:
            names.append(Path(img_path).name)
            image = cv2.imread(img_path)
            image = cv2.resize(image, (cfgs["input_image_size"], cfgs["input_image_size"]))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            batch_images.append(image)
        batch_images = np.stack(batch_images, axis=0, dtype=np.float32)
        batch_images = (batch_images / 255. - mean) / std
        batch_images = batch_images.transpose(0, 3, 1, 2)

        batch_images = torch.from_numpy(batch_images).float().to(device)

        pred = model(batch_images)
        pred = F.sigmoid(pred)
        scores.append(pred)
        output = (pred > 0.5).int()

        res.append(output)

    res = torch.concatenate(res, dim=0)
    scores = torch.concatenate(scores, dim=0)
    for name, ret, score in zip(names, res.tolist(), scores.tolist()):
        pred_class = multi2name(ret, labels)
        print(f"image_name: {name:<20s} "
              f"pred_class: {pred_class:<35s} "
              f"prob: {score}")


def multi2name(ret, labels):
    name = []
    for idx, i in enumerate(ret):
        if i == 1:
            name.append(labels[idx])
    name = '-'.join(name)
    return name


def run():
    val_root = "/home/8TDISK/weihule/data/mojiao/test/rough-NoChamfer-below"
    img_resize = 448
    model_path = f"/home/8TDISK/weihule/data/training_data/mojiao_multi/resnet18/resnet18/pths/resnet18-0.8401.pth"
    infer_cfg = {
        "seed": 0,
        "model": "resnet18",
        "input_image_size": img_resize,
        "batch_size": 8,
        "train_model_path": model_path,
        "test_image_dir": val_root,
        "num_classes": 4
    }
    inference(infer_cfg)


if __name__ == "__main__":
    run()
