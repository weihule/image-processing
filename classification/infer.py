# -*- coding: utf-8 -*-
import os
import cv2
import torch
import random
import numpy as np
import json
import torch.nn.functional as F
from utils.util import cal_macs_params
from backbones.model_manager import init_model
import time
from PIL import Image

from datasets.kitchen import labels_list


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

    model = init_model(backbone_type=cfgs["model"],
                       num_classes=cfgs["num_classes"])
    model = model.to(device)

    if cfgs["train_model_path"]:
        weight_dict = torch.load(cfgs["train_model_path"], map_location=torch.device("cpu"))
        model.load_state_dict(weight_dict, strict=True)

    model.eval()

    image_paths = list()
    for img_name in os.listdir(cfgs["test_image_dir"]):
        image_paths.append(os.path.join(cfgs["test_image_dir"], img_name))
    infer_num = len(image_paths)
    image_paths = [image_paths[s: s+cfgs["batch_size"]] for
                   s in range(0, len(image_paths), cfgs["batch_size"])]

    # 加载类别索引和类别名称映射
    idx2cls = {}
    for k, v in enumerate(labels_list):
        idx2cls[k] = v
    cls2idx = {v: k for k, v in idx2cls.items()}

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    mean = np.asarray(mean, dtype=np.float32).reshape((1, 1, 1, 3))
    std = np.asarray(std, dtype=np.float32).reshape((1, 1, 1, 3))

    start_time = time.time()
    for batch_datas in image_paths:
        batch_images = []
        for img_path in batch_datas:
            image = cv2.imread(img_path)
            image = cv2.resize(image, (cfgs["input_image_size"], cfgs["input_image_size"]))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            batch_images.append(image)
        batch_images = np.stack(batch_images, axis=0, dtype=np.float32)
        batch_images = (batch_images / 255. - mean) / std
        batch_images = batch_images.transpose(0, 3, 1, 2)

        batch_images = torch.from_numpy(batch_images).float().to(device)

        output = model(batch_images)
        output = F.softmax(output, dim=1)
        pred_scores, pred_classes = torch.max(output, dim=-1)
        for image_path, pred_class, pred_score in zip(batch_datas, pred_classes, pred_scores):
            print(f"image_name: {image_path.split(os.sep)[-1]} "
                  f"pred_class: {idx2cls[pred_class.item()]} "
                  f"prob: {round(pred_score.item(), 3)}")
    fps = round(1 / ((time.time() - start_time) / infer_num), 1)
    print(f"model: {cfgs['model']} FPS: {fps}")
    cal_macs_params(model.to(torch.device("cpu")), input_size=(4, 3, 224, 224))


# -------------------------------------#
#       调用摄像头检测
# -------------------------------------#
def detect_single_image(model, image, mean, std, device):
    # 将图像转成(B, C, H, W)的四维tensor
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.expand_dims(image, axis=0).astype(np.float32)
    image = (image / 255. - mean) / std
    image = image.transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).float().to(device)

    output = model(image)
    output = F.softmax(output, dim=1)
    preds, indices = torch.sort(output, dim=-1, descending=True)
    preds, indices = preds[0].detach().cpu().numpy(), indices[0].detach().cpu().numpy()
    return preds[:3], indices[:3]


def infer_video(cfgs):
    # 模型部分
    seed = 0
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # 为cpu设置随机数种子
    device = torch.device("cuda:0")

    # 初始化模型并加载权重
    model = init_model(backbone_type=cfgs["model"],
                       num_classes=cfgs["num_classes"])
    model = model.to(device)
    if cfgs["train_model_path"]:
        weight_dict = torch.load(cfgs["train_model_path"], map_location=torch.device("cpu"))
        model.load_state_dict(weight_dict, strict=True)

    model.eval()

    # 加载类别索引和类别名称映射
    idx2cls = {}
    for k, v in enumerate(labels_list):
        idx2cls[k] = v
    cls2idx = {v: k for k, v in idx2cls.items()}

    # 处理图像部分
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    mean = np.asarray(mean, dtype=np.float32).reshape((1, 1, 1, 3))
    std = np.asarray(std, dtype=np.float32).reshape((1, 1, 1, 3))

    # capture = cv2.VideoCapture(0)
    capture=cv2.VideoCapture(cfgs["video_path"])
    # fps = 0.0
    while True:
        t1 = time.time()
        try:
            # 读取某一帧
            ref, frame = capture.read()
            print(f"frame.shape = {frame.shape}")

            # 进行检测
            top3_preds, top3_indices = detect_single_image(model, frame, mean, std, device)
            top3_names = [idx2cls[i] for i in top3_indices]

            # frame = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)

            # fps = (fps + (1. / (time.time() - t1))) / 2
            fps = 1. / (time.time() - t1)
            # print("fps= %.2f" % fps)
            frame = cv2.putText(frame, f"fps= {fps:.2f}", (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            frame = cv2.putText(frame, f"cls_name= {top3_names[0]:<15s}", (0, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("video", frame)
            # 按下esc键，退出
            c = cv2.waitKey(30) & 0xff
            if c == 27:
                capture.release()
                break
        except Exception as e:
            break


if __name__ == "__main__":
    # infer_cfg = {
    #     "seed": 0,
    #     "model": "mobilenetv2_x1_0",
    #     "num_classes": 6,
    #     "input_image_size": 224,
    #     "batch_size": 4,
    #     "train_model_path": r"/home/8TDISK/weihule/data/training_data/kitchen/mobilenetv2_x1_0/mobilenetv2_x1_0/pths/mobilenetv2_x1_0-1.0.pth",
    #     "class_file": r"D:\workspace\data\dl\flower\flower.json",
    #     "test_image_dir": r"/home/8TDISK/weihule/data/kitchen/picture",
    #     "video_path": r"/home/8TDISK/weihule/data/kitchen/video/ele_00cfd9a2b68a27a9360090e104ff8476.ts"
    # }

    infer_cfg = {
        "seed": 0,
        "model": "mobilenetv2_x1_0",
        "num_classes": 6,
        "input_image_size": 224,
        "batch_size": 4,
        "train_model_path": r"D:\workspace\data\mobilenetv2_x1_0-1.0.pth",
        "class_file": r"D:\workspace\data\dl\flower\flower.json",
        "test_image_dir": r"/home/8TDISK/weihule/data/kitchen/picture",
        "video_path": r"D:\workspace\data\kitchen\video\ele_00cfd9a2b68a27a9360090e104ff8476.ts"
    }
    # inference(infer_cfg)
    infer_video(infer_cfg)


