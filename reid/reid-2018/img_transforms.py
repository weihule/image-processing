from __future__ import absolute_import

from PIL import Image
import random
import numpy as np


class Random2DTranslation(object):
    """
    With a probability, first increase image size to (1 + 1/8), and then perform random crop.

    Args:
        height (int): target height.
        width (int): target width.
        p (float): probability of performing this transformation. Default: 0.5.
    """
    def __init__(self, height, width, p=0.5, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.p = p
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        # 满足条件, 就不做数据增广, 直接resize到需要的尺寸
        if random.random() < self.p:
            return img.resize((self.width, self.height), self.interpolation)
        # 否则, 先resize, 再crop
        new_width, new_height = int(round(self.width * 1.125)), int(round(self.height * 1.125))
        resized_img = img.resize((new_width, new_height), self.interpolation)
        x_maxrange = new_width - self.width
        y_maxrange = new_height - self.height
        x1 = int(round(random.uniform(0, x_maxrange)))
        y1 = int(round(random.uniform(0, y_maxrange)))
        croped_img = resized_img.crop((x1, y1, x1 + self.width, y1 + self.height))
        return croped_img


class Random2DFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img):
        flip_flag = np.random.uniform(0, 1)
        if flip_flag <= 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        return img



if __name__ == '__main__':
    pass