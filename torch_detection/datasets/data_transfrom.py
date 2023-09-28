import os
import sys
import random
import cv2
import numpy as np
import torch

__all__ = [
    'Normalize',
    'RandomFlip',
    'RandomCrop',
    'RandomTranslate'
]


class Normalize:
    def __init__(self):
        self.mean = torch.tensor([[[[0.471, 0.448, 0.408]]]], dtype=torch.float32)
        self.std = torch.tensor([[[[0.234, 0.239, 0.242]]]], dtype=torch.float32)

    def __call__(self, image):
        # image shape: [h, w, 3], 这里三分量的顺序已经调整成 RGB 了
        image = (image - self.mean) / self.std
        return image


class RandomFlip:
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, sample):
        flip_flag = np.random.uniform(0, 1)
        if flip_flag < self.flip_prob:
            image = sample['img']
            # image = image[:, ::-1, :]   # 水平镜像
            image = cv2.flip(image, 1)
            annots = sample['annot']
            scale = sample['scale']
            size = sample['size']

            height, width, channel = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            annots[:, 0] = width - x2
            annots[:, 2] = width - x1

            sample = {'img': image, 'annot': annots, 'scale': scale, 'size': size}

        return sample


class RandomCrop:
    def __init__(self, crop_prob=0.35):
        self.crop_prob = crop_prob

    def __call__(self, sample):
        image, annots, scale, size = sample['img'], sample['annot'], sample['scale'], sample['size']
        if annots.shape[0] == 0:
            return sample
        prob = random.uniform(0, 1)
        if prob < self.crop_prob:
            h, w, _ = image.shape
            # 找出所有gt的最小外接矩形, shape: (4, )
            max_bbox = np.concatenate((np.min(annots[:, :2], axis=0),
                                       np.max(annots[:, 2:], axis=0)), axis=-1)
            max_left_trans, max_up_trans = max_bbox[0], max_bbox[1]
            max_right_trans, max_down_trans = w - max_bbox[2], h - max_bbox[3]

            # crop_x_min = max(0, int(max_bbox[0]-np.random.uniform(0, max_left_trans)))
            crop_x_min = int(np.random.uniform(0, max_left_trans))
            crop_y_min = int(np.random.uniform(0, max_up_trans))
            crop_x_max = max(w, int(max_bbox[2] + np.random.uniform(0, max_right_trans)))
            crop_y_max = max(h, int(max_bbox[2] + np.random.uniform(0, max_down_trans)))

            image = image[crop_y_min: crop_y_max, crop_x_min: crop_x_max]
            annots[:, [0, 2]] = annots[:, [0, 2]] - crop_x_min
            annots[:, [1, 3]] = annots[:, [1, 3]] - crop_y_min

            sample = {'img': image, 'annot': annots, 'scale': scale, 'size': size}
        return sample


class RandomTranslate:
    def __init__(self, translate_prob=0.5):
        self.translate_prob = translate_prob

    def __call__(self, sample):
        image, annots, scale = sample['img'], sample['annot'], sample['scale']

        if annots.shape[0] == 0:
            return sample

        prob = random.uniform(0, 1)
        if prob < self.translate_prob:
            h, w, _ = image.shape
            # 找出所有annots的最小外接矩形
            max_bbox = np.concatenate((np.min(annots[:, :2], axis=0),
                                       np.max(annots[:, 2:], axis=0)), axis=-1)
            max_left_trans, max_up_trans = max_bbox[0], max_bbox[1]
            max_right_trans, max_down_trans = w - max_bbox[2], h - max_bbox[3]

            tx = random.uniform(-(max_left_trans - 1), (max_right_trans - 1))
            ty = random.uniform(-(max_up_trans - 1), (max_down_trans - 1))

            M = np.array([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, M, (w, h))

            annots[:, [0, 2]] = annots[:, [0, 2]] + tx
            annots[:, [1, 3]] = annots[:, [1, 3]] + ty

        return {'img': image, 'annot': annots, 'scale': scale}


if __name__ == "__main__":
    arr = [[12, 10, 4, 5, 0], [1, 2, 47, 56, 0],
           [2, 3, 5, 8, 1]]
    arr = np.array(arr)
    print(arr[:, 0].shape)
