import numpy as np
from PIL import Image
import random
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt


def pad_if_smaller(img, size, fill=0):
    # 如果图像最小边长小于给定size，则用数值fill进行padding
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        # # 创建一个新的空白图像，用0值填充
        # new_img = Image.new('RGB', (ow+padw, oh+padh),
        #                     color=(fill, fill, fill))
        # new_img.paste(img, (0, 0))
        new_img = F.pad(img, (0, 0, padw, padh), fill=fill)
    else:
        new_img = img
    return new_img


class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        image, target = sample['image'], sample['target']
        if isinstance(image, Image.Image) and isinstance(target, Image.Image):
            raise f"type expected Image.Image, but get {type(image)} and {type(target)}"

        if np.random.uniform(0, 1) < self.prob:
            image = F.hflip(image)
            target = F.hflip(target)

        return {
            'image': image,
            'target': target,
        }


class RandomVerticalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        image, target = sample['image'], sample['target']
        if isinstance(image, Image.Image) and isinstance(target, Image.Image):
            raise f"type expected Image.Image, but get {type(image)} and {type(target)}"

        if np.random.uniform(0, 1) < self.prob:
            image = F.vflip(image)
            target = F.vflip(target)

        return {
            'image': image,
            'target': target,
        }


class RandomResize:
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            self.max_size = min_size
        else:
            self.max_size = max_size

    def __call__(self, sample):
        image, target = sample['image'], sample['target']
        size = random.randint(self.min_size, self.max_size)
        # 这里size传入的是int类型，所以是将图像的最小边长缩放到size大小
        image = F.resize(image, size)
        # 这里的interpolation注意下，在torchvision(0.9.0)以后才有InterpolationMode.NEAREST
        # 如果是之前的版本需要使用PIL.Image.NEAREST
        target = F.resize(target, size,
                          interpolation=transforms.InterpolationMode.NEAREST)

        return {
            'image': image,
            'target': target,
        }

class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        image, target = sample['image'], sample['target']
        image = pad_if_smaller(image, self.size, fill=0)
        target = pad_if_smaller(target, self.size, fill=255)

        crop_params = transforms.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)

        return {
            'image': image,
            'target': target,
        }


class ToTensor(object):
    def __call__(self, sample):
        image, target = sample['image'], sample['target']
        image = F.to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return {
            'image': image,
            'target': target,
        }


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, target = sample['image'], sample['target']
        image = F.normalize(image, mean=self.mean, std=self.std)
        return {
            'image': image,
            'target': target,
        }


if __name__ == "__main__":
    # 创建一个三维张量
    tensor = np.random.rand(2, 3, 4)

    # 使用 transpose 对张量进行转置，交换前两个维度和后两个维度
    transposed_tensor = np.transpose(tensor, (1, 2, 0))

    # 输出转置后的张量
    # print(transposed_tensor.shape)

    img_path = r"D:\workspace\data\VOCdataset\VOC2012\JPEGImages\2007_000032.jpg"
    target_path = r"D:\workspace\data\VOCdataset\VOC2012\SegmentationClass\2007_000032.png"
    # img_ = pad_if_smaller(Image.open(img_path), size=640, fill=0)

    img_ = Image.open(img_path)
    target_ = Image.open(target_path)
    print(type(img_), type(target_), img_.mode, target_.mode)

    # rc = RandomCrop(size=640)
    # d = rc({"image": img_, "target": target_})
    rr = RandomResize(min_size=280)
    d = rr({"image": img_, "target": target_})
    img_new, target_new = d["image"], d["target"]
    print(img_new, type(img_new), type(target_new))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img_)
    axes[0].axis('off')  # 隐藏坐标轴信息
    axes[1].imshow(target_)
    axes[1].axis('off')
    plt.show()

    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 5))
    axes2[0].imshow(img_new)
    axes2[0].axis('off')  # 隐藏坐标轴信息
    axes2[1].imshow(target_new)
    axes2[1].axis('off')
    plt.show()


