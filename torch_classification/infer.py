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
from torchvision import transforms


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
                       default="D:\\Desktop\\test",
                       help="test images")
    args = parse.parse_args()

    return args


def inference(args):
    print(f"args: {args}")

    assert args.model in backbones.__dict__.keys(), "Unsupported model!"

    if args.seed:
        seed = args.seed
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)    # 为cpu设置随机数种子

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
    image_paths = [image_paths[s: s+args.batch_size] for s in range(0, len(image_paths), args.batch_size)]

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
            print(image_path.split(os.sep)[-1], pred_class.item(), pred_score.item())


def main2():
    args = parse_args()
    inference(args)


def main(cfgs):
    with open('utils/flower.json', 'r', encoding='utf-8') as fr:
        infos = json.load(fr)
        class2index = infos['classes']
    index2class = {v: k for k, v in class2index.items()}

    model = cfgs['net']
    img_root = cfgs['img_root']
    model_path = cfgs['model_path']
    batch_size = cfgs['batch_infer']
    size = cfgs['size']
    device = cfgs['device']
    mean = cfgs['mean']
    std = cfgs['std']
    single_folder = cfgs['single_folder']

    mean = np.expand_dims(np.expand_dims(mean, 0), 0)
    std = np.expand_dims(np.expand_dims(std, 0), 0)

    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
    imgs_path = []
    if single_folder is False:
        for folder in os.listdir(img_root):
            for fn in os.listdir(os.path.join(img_root, folder)):
                fn_path = os.path.join(img_root, folder, fn)
                imgs_path.append(fn_path)
    if single_folder:
        for fn in os.listdir(img_root):
            fn_path = os.path.join(img_root, fn)
            imgs_path.append(fn_path)
    split_imgs = [imgs_path[p:p+batch_size] for p in range(0, len(imgs_path), batch_size)]

    model.eval()
    with torch.no_grad():
        for sub_lists in split_imgs:
            images = []
            images_name = []
            for fn in sub_lists:
                images_name.append(fn.split(os.sep)[-1])
                img = cv2.imread(fn)
                img = cv2.resize(img, (size, size))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = (img / 255. - mean) / std
                images.append(img)
            images_tensor = [torch.tensor(p, dtype=torch.float32) for p in images]
            images_tensor = torch.stack(images_tensor, dim=0).to(device)
            images_tensor = images_tensor.permute(0, 3, 1, 2).contiguous()
            preds = model(images_tensor)
            preds = F.softmax(preds, dim=1)
            probs, indices = torch.max(preds, dim=1)
            probs, indices = probs.cpu().numpy().tolist(), indices.cpu().numpy().tolist()
            for img_name, prob, idx in zip(images_name, probs, indices):
                print(img_name, round(prob, 3), index2class[idx])


if __name__ == "__main__":
    # mode = 'local'
    # if mode == 'local':
    #     imgs = 'D:\\workspace\\data\\dl\\flower\\test'
    #     model_path = 'D:\\workspace\\data\\classification_data\\mobilenetv2\\pths\\best.pth'
    # elif mode == 'company':
    #     imgs = '/workshop/weihule/data/dl/flower/test'
    #     model_path = '/workshop/weihule/data/weights/yolov4backbone/darknet53_857.pth'
    # else:
    #     imgs = '/workshop/weihule/data/dl/flower/test'
    #     model_path = ''
    #
    # backbone_type = 'mobilenetv2_x1_0'
    # model = backbones.__dict__[backbone_type](
    #     **{'num_classes': 5,
    #        'act_type': 'leakyrelu'}
    # )
    #
    # cfg_dict = {
    #     'img_root': imgs,
    #     'model_path': model_path,
    #     'net': model,
    #     'batch_infer': 8,
    #     'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    #     'size': 224,
    #     'mean': [0.485, 0.456, 0.406],
    #     'std': [0.229, 0.224, 0.225],
    #     'single_folder': False
    # }
    #
    # main(cfg_dict)

    # main2()
    test02()




