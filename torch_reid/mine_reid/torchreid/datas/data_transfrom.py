from __future__ import print_function, absolute_import
import os

import numpy as np
from PIL import Image
import random

__all__ = [
    'Random2DTranslation'
]


class Random2DTranslation:
    def __init__(self, height, width, p=0.5):
        self.height = height
        self.width = width
        self.p = p

    def __call__(self, img):
        # 不做数据增光, 直接resize到需要的尺寸
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


class RandomErasing:
    def __init__(self, p=0.5, sl=0.02, sh=0.4, r1=0.3, mean=None):
        self.p = p
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        if mean is None:
            self.mean = [0.4914, 0.5822, 0.4465]
        else:
            self.mean = mean





