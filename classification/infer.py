# -*- coding: utf-8 -*-
import os
import cv2
import pandas as pd
import torch
import random
import numpy as np
import json
import torch.nn.functional as F
from utils.util import cal_macs_params
from backbones.model_manager import init_model
import time
from pathlib import Path
from PIL import Image

from datasets.kitchen import labels_list

final_ret = {"filename": [],
             "result": []}

en2zh = {
    "unfinished_chamfers": "未做倒角",
    "uneven": "不平整",
    "normal": "正常",
    "below": "低于防水胶条"
}


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
    image_paths = [image_paths[s: s + cfgs["batch_size"]] for
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
    num = 1
    for batch_datas in image_paths:
        batch_images = []
        raw_images = []
        for img_path in batch_datas:
            image = cv2.imread(img_path)
            raw_images.append(image)
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
        for raw_image, image_path, pred_class, pred_score in zip(raw_images, batch_datas, pred_classes, pred_scores):
            # print(f"image_name: {image_path.split(os.sep)[-1]} "
            #       f"pred_class: {idx2cls[pred_class.item()]} "
            #       f"prob: {round(pred_score.item(), 3)}")
            print(f"image_name: {image_path.split(os.sep)[-1]:<20s} "
                  f"pred_class: {idx2cls[pred_class.item()]:<20s} "
                  f"prob: {round(pred_score.item(), 3):<20.3f}")
            file_path = Path("/home/8TDISK/weihule/data/mojiao_dataset/show/below") / (en2zh[idx2cls[pred_class.item()]]+'-'+str(num).rjust(3, "0")+'-'+str(round(pred_score.item(), 3))+'.jpg')
            cv2.imwrite(str(file_path), raw_image)

            final_ret["filename"].append(image_path.split(os.sep)[-1])
            final_ret["result"].append(idx2cls[pred_class.item()])
            num += 1

    fps = round(1 / ((time.time() - start_time) / infer_num), 1)
    print(f"model: {cfgs['model']} FPS: {fps}")
    # print(final_ret["result"].count("normal"), len(final_ret["result"]))
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
    capture = cv2.VideoCapture(cfgs["video_path"])

    # TODO: kitchen
    final_ret["filename"].append(cfgs["video_path"].split(os.sep)[-1])
    res = set()

    # fps = 0.0
    while True:
        t1 = time.time()
        try:
            # 读取某一帧
            ref, frame = capture.read()

            # 进行检测
            top3_preds, top3_indices = detect_single_image(model, frame, mean, std, device)
            top3_names = [idx2cls[i] for i in top3_indices]

            # frame = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)

            # fps = (fps + (1. / (time.time() - t1))) / 2
            fps = 1. / (time.time() - t1)
            # print("fps= %.2f" % fps)
            frame = cv2.putText(frame, f"fps= {fps:.2f}", (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            frame = cv2.putText(frame, f"cls_name= {top3_names[0]:<15s}", (0, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0), 2)

            res.add(top3_names[0])
            # cv2.imshow("video", frame)
            # # 按下esc键，退出
            # c = cv2.waitKey(30) & 0xff
            # if c == 27:
            #     capture.release()
            #     break
        except Exception as e:
            break

    final_ret["result"].append(list(res))


def run():
    dict_map = {
        "normal": 0,
        "smoke": 1,
        "shirtless": 2,
        "rat": 4,
        "cat": 8,
        "dog": 16
    }
    image_cfg = {
        "seed": 0,
        "model": "resnet18",
        "num_classes": 4,
        "input_image_size": 448,
        "batch_size": 4,
        "train_model_path": r"/home/8TDISK/weihule/data/training_data/mojiao/resnet18/resnet18/pths/resnet18-0.7411.pth",
        "test_image_dir": r"/home/8TDISK/weihule/data/mojiao_dataset/val/below",
    }
    inference(image_cfg)
    # video_root = r"/home/8TDISK/weihule/data/kitchen/video"
    # for i in os.listdir(video_root):
    #     video_cfg = {
    #         "seed": 0,
    #         "model": "resnet18",
    #         "num_classes": 6,
    #         "input_image_size": 448,
    #         "batch_size": 4,
    #         "train_model_path": r"/home/8TDISK/weihule/data/training_data/kitchen/resnet18/resnet18/pths/resnet18-0.8.pth",
    #         "video_path": os.path.join(video_root, i)
    #     }
    #     infer_video(video_cfg)
    #     # break
    #
    # temp_list = []
    # for i in final_ret["result"]:
    #     if isinstance(i, str):
    #         temp_list.append(dict_map[i])
    #     else:
    #         temp_number = 0
    #         for sub_i in i:
    #             temp_number += dict_map[sub_i]
    #         temp_list.append(temp_number)
    # final_ret["result"] = temp_list
    #
    # pd_ret = pd.DataFrame(final_ret)
    # pd_ret.to_csv("./test.csv", index=False)
    #
    # print(len(final_ret["filename"]), len(final_ret["result"]))


if __name__ == "__main__":
    run()
