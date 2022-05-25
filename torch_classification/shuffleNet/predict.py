import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import sys
import math
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import math
import cv2
from PIL import Image
from torchvision import datasets, transforms
from model import shufflenet_v2_x1_0
from utils import inverse_label_dict


def main():
    model_path = '/nfs/home57/weihule/data/weights/shufflenetv2/model-0522.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )

    infer_batch = 8

    # 实例化模型
    model = shufflenet_v2_x1_0(num_classes=5).to(device)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))

    # 预处理数据
    img_root = '/nfs/home57/weihule/data/DL/flower/test'
    img_path = [os.path.join(img_root, fn) for fn in os.listdir(img_root)]
    iter_num = math.ceil(len(img_path)/infer_batch)
    img_path_split = np.array_split(img_path, iter_num)
    for paths in img_path_split:
        img_list = [transform(Image.open(path)).to(device) for path in paths]
        img_tensor = torch.stack(img_list, dim=0)
        pred_y = model(img_tensor)
        probability = F.softmax(pred_y, dim=1)
        probs, indexes = torch.max(probability, dim=1)
        for fn, cls, prob in zip(paths, indexes, probs):
            print(fn.split(os.sep)[-1], inverse_label_dict[str(cls.item())], prob.item())

        break


if __name__ == "__main__":
    main()

