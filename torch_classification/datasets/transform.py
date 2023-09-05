import math
import os
import numpy as np
import torch
from torchvision import transforms
import cv2
import random
from PIL import Image
from pathlib import Path

__all__ = [
    'OpenCV2PIL',
    'PIL2OpenCV',
    'TorchResize',
    'TorchRandomHorizontalFlip',
    'TorchColorJitter',
    'Random2DErasing',
    'transform'
]


class OpenCV2PIL:
    def __init__(self):
        pass

    def __call__(self, sample: dict):
        image, label = sample["image"], sample["label"]
        image = Image.fromarray(np.uint8(image))

        return {
            "image": image,
            "label": label
        }


class PIL2OpenCV:
    def __init__(self):
        pass

    def __call__(self, sample: dict):
        image, label = sample["image"], sample["label"]
        image = np.asarray(image).astype(np.float32)

        return {
            "image": image,
            "label": label
        }


class TorchResize:
    def __init__(self, resize=224):
        self.func = transforms.Resize(size=[resize, resize])

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        image = self.func(image)

        return {
            "image": image,
            "label": label
        }


class TorchRandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.func = transforms.RandomHorizontalFlip(p=prob)

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = self.func(image)

        return {
            'image': image,
            'label': label
        }


class TorchColorJitter:
    def __init__(self, brightness=0.3, contrast=0.3, saturation=0.3, hue=0):
        self.func = transforms.ColorJitter(brightness=brightness,
                                           contrast=contrast,
                                           saturation=saturation,
                                           hue=hue)

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = self.func(image)

        return {
            'image': image,
            'label': label
        }


class Random2DErasing:
    def __init__(self, p=0.5, sl=0.01, sh=0.2, r1=0.3):
        self.p = p
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, sample):
        if random.random() < self.p:
            return sample

        # 这个类放在了PIL2OpenCV下，所以这里的image已经是np.float32的RGB形式
        image, label = sample['image'], sample['label']
        image_h, image_w, _ = image.shape
        area = image_h * image_w

        target_area = random.uniform(self.sl, self.sh)*area     # 确定随机擦除区域的面积
        aspect_ratio = random.uniform(self.r1, 1 / self.r1)     # 确定随机擦除区域的ratio (h / w)

        erase_h = int(round(math.sqrt(target_area * aspect_ratio)))
        erase_w = int(round(math.sqrt(target_area / aspect_ratio)))

        if erase_w < image_w and erase_h < image_h:
            # 确定随机擦除区域的(x, y)坐标
            loc1 = random.randint(0, image_h - erase_h - 1)
            loc2 = random.randint(0, image_w - erase_w - 1)
            erase_area = (np.random.rand(erase_h, erase_w, 3) * 255.)
            image[loc1: loc1+erase_h, loc2: loc2+erase_w, :] = erase_area

        return {
            'image': image,
            'label': label
        }


transform = {
    'train': transforms.Compose([
        OpenCV2PIL(),
        TorchResize(resize=224),
        TorchRandomHorizontalFlip(prob=0.5),
        TorchColorJitter(),
        PIL2OpenCV(),
        Random2DErasing()
    ]),
    'val': transforms.Compose([
        OpenCV2PIL(),
        TorchResize(resize=400),
        PIL2OpenCV()
    ])
}


def test_erase():
    """
    测试图像随机擦除
    """
    image_dir = r"D:\Desktop\apps\out\data_dir\query"
    save_dir = r"D:\Desktop"
    sl, sh, r1=0.01, 0.2, 0.3
    for i in Path(image_dir).iterdir():
        image = cv2.imread(str(i))
        image = np.asarray(image, dtype=np.float32)
        image_h, image_w, _ = image.shape
        area = image_h * image_w

        target_area = random.uniform(sl, sh)*area     # 确定随机擦除区域的面积
        aspect_ratio = random.uniform(r1, 1 / r1)     # 确定随机擦除区域的ratio (h / w)

        erase_h = int(round(math.sqrt(target_area * aspect_ratio)))
        erase_w = int(round(math.sqrt(target_area / aspect_ratio)))

        if erase_w < image_w and erase_h < image_h:
            # 确定随机擦除区域的(x, y)坐标
            loc1 = random.randint(0, image_h - erase_h - 1)
            loc2 = random.randint(0, image_w - erase_w - 1)
            erase_area = (np.random.rand(erase_h, erase_w, 3) * 255.)
            image[loc1: loc1+erase_h, loc2: loc2+erase_w, :] = erase_area
        
        image = image.astype(np.uint8)
        cv2.imwrite(str(Path(save_dir) / i.name), image)


if __name__ == '__main__':
    test_erase()







