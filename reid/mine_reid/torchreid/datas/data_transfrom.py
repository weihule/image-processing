from __future__ import print_function, absolute_import
import os

import numpy as np
from PIL import Image
import random
import math
import cv2

__all__ = [
    'Random2DTranslation',
    'Random2DErasing'
]


class Random2DTranslation:
    def __init__(self, height, width, p=0.5):
        self.height = height
        self.width = width
        self.p = p

    def __call__(self, img):
        # 不做数据增广, 直接resize到需要的尺寸
        if random.random() < self.p:
            return img.resize((self.width, self.height), Image.BILINEAR)
        # 先resize, 再crop
        new_width, new_height = int(round(self.width * 1.125)), int(round(self.height * 1.125))
        resize_img = img.resize((new_width, new_height), Image.BILINEAR)
        x_max_range = new_width - self.width
        y_max_range = new_height - self.height
        x1 = int(round(random.uniform(0, x_max_range)))
        y1 = int(round(random.uniform(0, y_max_range)))
        crop_img = resize_img.crop((x1, y1, x1 + self.width, y1 + self.height))
        return crop_img


class Random2DErasing:
    """
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    """
    def __init__(self, p=0.5, sl=0.02, sh=0.3, r1=0.3):
        self.p = p
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        if random.random() < self.p:
            return img
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        img_h, img_w, c = img.shape
        area = img_h * img_w

        target_area = random.uniform(self.sl, self.sh) * area
        aspect_ratio = random.uniform(self.r1, 1 / self.r1)

        # 随机擦除区域的高和宽
        erase_h = int(round(math.sqrt(target_area * aspect_ratio)))
        erase_w = int(round(math.sqrt(target_area / aspect_ratio)))

        if erase_w < img_w and erase_h < img_h:
            loc1 = random.randint(0, img_h - erase_h - 1)
            loc2 = random.randint(0, img_w - erase_w - 1)
            # loc3 = loc1 + erase_h
            # loc4 = loc2 + erase_w
            erase_area = (np.random.rand(erase_h, erase_w, 3) * 255.).astype(np.uint8)
            img[loc1:loc1 + erase_h, loc2:loc2 + erase_w, :] = erase_area
            img = Image.fromarray(np.uint8(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
            return img

        else:
            img = Image.fromarray(np.uint8(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
            return img


if __name__ == '__main__':
    rand_trans = Random2DTranslation(height=256, width=128)
    rand_era = Random2DErasing(sh=0.25)
    root = 'D:\\workspace\\data\\dl\\market1501\\Market-1501-v15.09.15\\bounding_box_train'
    save_root = 'D:\\Desktop\\delete'
    for idx, fn in enumerate(os.listdir(root)):
        fn_path = os.path.join(root, fn)
        img = Image.open(fn_path)
        img = rand_era(img)
        save_path = os.path.join(save_root, fn)
        img.save(save_path)
        if idx == 300:
            break

    # img = (np.zeros((256, 128, 3))).astype(np.uint8)
    # h, w = 60, 25
    # img[10:10+h, 10:10+w, :] = (np.random.rand(h, w, 3) * 255).astype(np.uint8)
    # cv2.imshow('res', img)
    # cv2.waitKey(0)







