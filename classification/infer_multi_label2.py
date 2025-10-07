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

from datasets.eye_multi_label import (impression, HyperF_Type, HyperF_Area_DA,
                                      HyperF_Fovea, HyperF_ExtraFovea, HyperF_Y,
                                      HypoF_Type, HypoF_Area_DA,
                                      HypoF_Fovea, HypoF_ExtraFovea, HypoF_Y,
                                      CNV, Vascular_abnormality_DR, Pattern)

feature_dict = {
    "impression": "Impression",
    "HyperF_Type": "HyperF_Type",
    "HyperF_Area_DA": "HyperF_Area(DA)",
    "HyperF_Fovea": "HyperF_Fovea",
    "HyperF_ExtraFovea": "HyperF_ExtraFovea",
    "HyperF_Y": "HyperF_Y",
    "HypoF_Type": "HypoF_Type",
    "HypoF_Area_DA": "HypoF_Area(DA)",
    "HypoF_Fovea": "HypoF_Fovea",
    "HypoF_ExtraFovea": "HypoF_ExtraFovea",
    "HypoF_Y": "HypoF_Y",
    "CNV": "CNV",
    "Vascular_abnormality_DR": "Vascular abnormality (DR)",
    "Pattern": "Pattern"
}

feature_num_classes = {
    "impression": 23,
    "HyperF_Type": 5,
    "HyperF_Area_DA": 3,
    "HyperF_Fovea": 2,
    "HyperF_ExtraFovea": 18,
    "HyperF_Y": 4,
    "HypoF_Type": 3,
    "HypoF_Area_DA": 3,
    "HypoF_Fovea": 2,
    "HypoF_ExtraFovea": 16,
    "HypoF_Y": 4,
    "CNV": 2,
    "Vascular_abnormality_DR": 15,
    "Pattern": 14
}
feature_idx = list(feature_dict.keys())


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

    symptoms = Path(cfgs["train_model_path"]).parts[-4].split("multi_")[-1]
    num_classes = feature_num_classes[symptoms]

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

    # 加载类别索引和类别名称映射
    cls2idx = dict(zip(impression, range(len(impression))))

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    mean = np.asarray(mean, dtype=np.float32).reshape((1, 1, 1, 3))
    std = np.asarray(std, dtype=np.float32).reshape((1, 1, 1, 3))

    start_time = time.time()

    res = []
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

        pred = model(batch_images)
        pred = F.sigmoid(pred)
        output = (pred > 0.5).int()

        res.append(output)

    res = torch.concatenate(res, dim=0)
    # 将每一行转换为字符串，以便进行比较
    str_rows = ["".join(map(str, row.cpu().numpy().tolist())) for row in res]

    # 找到出现次数最多的一行
    most_frequent_row = max(str_rows, key=str_rows.count)

    # 将字符串还原为 Tensor
    result_tensor = torch.tensor([int(digit) for digit in most_frequent_row])

    indices = torch.nonzero(result_tensor).reshape((-1)).numpy().tolist()

    if not indices:
        try:
            # 找出每行中值为 1 的索引位置
            indices_ = [row.nonzero().squeeze().tolist() for row in res]
            indices_ = [row for row in indices_ if row]
            # 使用 Counter 统计元素出现次数
            counter = Counter(indices_)

            # 找到重复次数最多的元素
            most_common_element = counter.most_common(1)[0][0]
        except:
            most_common_element = 0

        indices = [most_common_element]

    names = [eval(symptoms)[i] for i in indices]

    return {symptoms: names}


def run():
    val_root = "/home/8TDISK/weihule/data/mojiao/test"
    # val_root = "/home/8TDISK/weihule/data/eye_competition/Train/Train"\
    folders = list(pd.read_csv("./sublit_sample_test.csv")["Folder"])
    writer_info = {"Impression": [],
                   "HyperF_Type": [],
                   "HyperF_Area(DA)": [],
                   "HyperF_Fovea": [],
                   "HyperF_ExtraFovea": [],
                   "HyperF_Y": [],
                   "HypoF_Type": [],
                   "HypoF_Area(DA)": [],
                   "HypoF_Fovea": [],
                   "HypoF_ExtraFovea": [],
                   "HypoF_Y": [],
                   "CNV": [],
                   "Vascular abnormality (DR)": [],
                   "Pattern": [],
                   "ID": [],
                   "Folder": []}
    for folder in tqdm(folders):
        val_dir = Path(val_root) / folder
        ID = folder.split("_")[0]
        writer_info["ID"].append(ID)
        writer_info["Folder"].append(folder)
        # 调用14个分类模型
        for k in feature_dict.keys():
            img_resize = 672
            model_dir = f"resnet50_multi_{k}"
            train_model_path = Path("/home/8TDISK/weihule/data/training_data") / model_dir / "resnet50/pths"
            train_model_path = list(train_model_path.glob("*.pth"))[0]

            infer_cfg = {
                "seed": 0,
                "model": "resnet50",
                "input_image_size": img_resize,
                "batch_size": 8,
                "train_model_path": train_model_path,
                "class_file": r"D:\workspace\data\dl\flower\flower.json",
                "test_image_dir": str(val_dir)
            }
            infer_res = inference(infer_cfg)
            temp_k = list(infer_res.keys())[0]
            temp_v = infer_res[temp_k]
            temp_v = ",".join(temp_v)

            writer_info[feature_dict[temp_k]].append(temp_v)

    writer_info = pd.DataFrame(writer_info)
    writer_info.to_csv("./submit6.csv", index=False)


if __name__ == "__main__":
    run()
