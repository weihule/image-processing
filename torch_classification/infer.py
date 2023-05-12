# -*- coding: utf-8 -*-
import os
import cv2
import torch
import random
import numpy as np
import json
import argparse
import torch.nn.functional as F
import backbones
import time
from PIL import Image


def parse_args():
    parse = argparse.ArgumentParser(description="detect image")
    parse.add_argument("--seed", type=int, default=0, help="seed")
    parse.add_argument("--model", type=str, default="resnet50", help="name of model")
    parse.add_argument("--train_num_classes", type=int, default=1000, help="class number")
    parse.add_argument("--input_image_size", type=int, default=224, help="input image size")
    parse.add_argument("--batch_size", type=int, default=4, help="batch size")
    parse.add_argument("--train_model_path", type=str,
                       default="D:\\Desktop\\resnet50-acc76.264.pth",
                       help="trained model weight path")
    # D:\\Desktop\\test
    # D:\\workspace\\data\\dl\\imagenet100\\imagenet100_val\\African-hunting-dog
    parse.add_argument("--test_image_dir", type=str,
                       default="D:\\workspace\\data\\dl\\imagenet100\\imagenet100_val\\African-hunting-dog",
                       help="test images")
    parse.add_argument("--index_path", type=str,
                       default="D:\\workspace\\code\\study\\torch_classification\\utils\\imagenet1000.json",
                       help="index_path")
    args = parse.parse_args()

    return args


def inference(args):
    """
    以文件夹的形式推理
    """
    print(f"args: {args}")

    assert args.model in backbones.__dict__.keys(), "Unsupported model!"

    if args.seed:
        seed = args.seed
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)  # 为cpu设置随机数种子

    device = torch.device("cuda:0")

    model = backbones.__dict__[args.model](
        **{"num_classes": args.train_num_classes}
    )
    model = model.to(device)

    if args.train_model_path:
        weight_dict = torch.load(args.train_model_path, map_location=torch.device("cpu"))
        model.load_state_dict(weight_dict, strict=True)

    model.eval()

    image_paths = list()
    for img_name in os.listdir(args.test_image_dir):
        image_paths.append(os.path.join(args.test_image_dir, img_name))
    image_paths = [image_paths[s: s + args.batch_size] for s in range(0, len(image_paths), args.batch_size)]

    # 加载类别索引和类别名称映射
    with open(args.index_path, "r", encoding="utf-8") as fr:
        cls2name = json.load(fr)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    mean = np.asarray(mean, dtype=np.float32).reshape((1, 1, 1, 3))
    std = np.asarray(std, dtype=np.float32).reshape((1, 1, 1, 3))
    for batch_datas in image_paths:
        batch_images = []
        for img_path in batch_datas:
            image = cv2.imread(img_path)
            image = cv2.resize(image, (args.input_image_size, args.input_image_size))
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
            print(image_path.split(os.sep)[-1],
                  pred_class.item(),
                  cls2name[str(pred_class.item())],
                  round(pred_score.item(), 3))


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


def infer_video():
    # 模型部分
    seed = 0
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # 为cpu设置随机数种子
    device = torch.device("cuda:0")
    model = backbones.__dict__["resnet50"](
        **{"num_classes": 1000}
    )
    model = model.to(device)
    train_model_path = "D:\\Desktop\\resnet50-acc76.264.pth"
    weight_dict = torch.load(train_model_path, map_location=torch.device("cpu"))
    model.load_state_dict(weight_dict, strict=True)
    model.eval()

    # 加载类别索引和类别名称映射
    index_path = "./utils/imagenet1000.json"
    with open(index_path, "r", encoding="utf-8") as fr:
        cls2name = json.load(fr)

    # 处理图像部分
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    mean = np.asarray(mean, dtype=np.float32).reshape((1, 1, 1, 3))
    std = np.asarray(std, dtype=np.float32).reshape((1, 1, 1, 3))

    capture = cv2.VideoCapture(0)  # capture=cv2.VideoCapture("1.mp4")
    # fps = 0.0
    while True:
        t1 = time.time()
        # 读取某一帧
        ref, frame = capture.read()

        # 进行检测
        top3_preds, top3_indices = detect_single_image(model, frame, mean, std, device)
        top3_names = [cls2name[str(i)] for i in top3_indices]
        print(top3_preds, top3_indices)

        # frame = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)

        # fps = (fps + (1. / (time.time() - t1))) / 2
        fps = 1. / (time.time() - t1)
        print("fps= %.2f" % fps)
        frame = cv2.putText(frame, f"fps= {fps:.2f}", (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        frame = cv2.putText(frame, f"cls_name= {top3_names[0]:<15s}", (0, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        frame = cv2.putText(frame, f"cls_name= {top3_names[1]:<15s}", (0, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("video", frame)
        # 按下esc键，退出
        c = cv2.waitKey(30) & 0xff
        if c == 27:
            capture.release()
            break


def main():
    args = parse_args()
    inference(args)


def test():
    arr = torch.tensor([[0.001, 0.005, 0.004, 0.1, 0.02]])
    print(torch.sort(arr, dim=-1))


if __name__ == "__main__":
    # main()
    infer_video()
