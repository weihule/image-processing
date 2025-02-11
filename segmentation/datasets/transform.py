import numpy as np
from PIL import Image
import random
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt

__all__ = [
    'transform'
]

# 随机缩放类，jitter为True时进行随机缩放
class ResizeImage:
    def __init__(self, target_size, jitter=False):
        self.target_size = target_size
        self.jitter = jitter

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        iw, ih = image.size
        h, w = self.target_size

        if self.jitter:
            # 计算目标尺寸相对于原图的最大缩放因子
            max_scale_w = w / iw  # 水平缩放因子
            max_scale_h = h / ih  # 垂直缩放因子

            scale_range = (0.85, 1.5)

            # 取水平和垂直的最小值，确保不会超过target尺寸
            max_scale = min(max_scale_w, max_scale_h)

            # 随机生成一个缩放因子，确保不超过最大缩放因子
            scale = random.uniform(scale_range[0], min(scale_range[1], max_scale))
            print(f"scale = {scale}")

            # 等比例缩放图像
            new_w = int(iw * scale)
            new_h = int(ih * scale)

            # 对图像和标签进行等比例缩放
            image = image.resize((new_w, new_h), Image.BICUBIC)
            mask = mask.resize((new_w, new_h), Image.NEAREST)

            # 将图像多余的部分用灰条填充
            dx = (w - new_w) // 2  # 水平居中
            dy = (h - new_h) // 2  # 垂直居中

            # 创建新图像和标签，并用灰条填充
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_mask = Image.new('L', (w, h), 0)

            new_image.paste(image, (dx, dy))
            new_mask.paste(mask, (dx, dy))
        else:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)

            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', [w, h], (128, 128, 128))
            new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

            mask = mask.resize((nw, nh), Image.NEAREST)
            new_mask = Image.new('L', [w, h], 0)
            new_mask.paste(mask, ((w - nw) // 2, (h - nh) // 2))

        return {'image': new_image, 'mask': new_mask}


# 随机颜色抖动类
class RandomColorJitter:
    def __init__(self, contrast=0.3, saturation=0.3, hue=0.3):
        self.color_jitter = transforms.ColorJitter(
            brightness=0, contrast=contrast, saturation=saturation, hue=hue
        )

    def __call__(self, sample):
        if random.uniform(0, 1) >= 0.5:
            image = self.color_jitter(sample['image'])
            return {'image': image, 'mask': sample['mask']}
        else:
            return sample


# 随机水平翻转
class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        if random.random() < self.prob:
            image = F.hflip(image)
            mask = F.hflip(mask)

        return {
            'image': image,
            'mask': mask
        }


# 做归一化
class Normalize:
    def __init__(self, mean=None, std=None):
        self.func = transforms.ToTensor()    # 归一化到 [0,1]
        if mean is not None:
            self.func2 = transforms.Normalize(mean, std)

    def __call__(self, sample):
        image = sample['image']
        image = self.func(image)
        if self.func2 is not None:
            image = self.func2(image)

        return {'image': image, 'mask': sample['mask']}


def transform(input_size):
    return {
        'train': transforms.Compose([ResizeImage(target_size=(input_size[0], input_size[1]), jitter=False),
                                  RandomColorJitter(),
                                  RandomHorizontalFlip()
                                  ]),
        'val': transforms.Compose([ResizeImage(target_size=(input_size[0], input_size[1]), jitter=False)])
    }

def test():
    # 创建一个三维张量
    tensor = np.random.rand(2, 3, 4)

    # 使用 transpose 对张量进行转置，交换前两个维度和后两个维度
    transposed_tensor = np.transpose(tensor, (1, 2, 0))

    # 输出转置后的张量
    print(transposed_tensor.shape)

    img_path = r"D:\workspace\data\images\VOCdevkit\VOC2012\JPEGImages\2007_000032.jpg"
    target_path = r"D:\workspace\data\images\VOCdevkit\VOC2012\SegmentationClass\2007_000032.png"
    raw_sample = {'image': Image.open(img_path),
              'mask': Image.open(target_path)}

    img_ = Image.open(img_path)
    target_ = Image.open(target_path)
    print(type(img_), type(target_), img_.mode, target_.mode)

    # rc = RandomCrop(size=640)
    # d = rc({"image": img_, "target": target_})
    ri = ResizeImage(target_size=(640, 640), jitter=True)
    sample = ri(raw_sample)
    x, y = sample['image'], sample['mask']
    print(f"type(x) = {type(x)} type(y) = {type(y)}")

    compose = transforms.Compose([ResizeImage(target_size=(640, 640), jitter=False),
                                  RandomColorJitter(),
                                  RandomHorizontalFlip()
                                  ])
    sample2 = compose(raw_sample)
    x2, y2 = sample2['image'], sample2['mask']
    print(f"type(x2) = {type(x2)} type(y2) = {type(y2)} x2 = {x2} y2 = {y2}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img_)
    axes[0].axis('off')  # 隐藏坐标轴信息
    axes[1].imshow(target_)
    axes[1].axis('off')
    plt.show()

    fig2, axes2 = plt.subplots(1, 4, figsize=(15, 10))
    axes2[0].imshow(x)
    axes2[0].axis('off')  # 隐藏坐标轴信息
    axes2[1].imshow(y)
    axes2[1].axis('off')
    axes2[2].imshow(x2)
    axes2[2].axis('off')
    axes2[3].imshow(y2)
    axes2[3].axis('off')
    plt.show()


if __name__ == "__main__":
    test()



