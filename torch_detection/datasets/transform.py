import os
import sys
import random
import cv2
import numpy as np
import torch

__all__ = [
    "RandomHorizontalFlip",
    "RandomCrop",
    "RandomTranslate",
    "Normalize",

    "YoloStyleResize",
    "RetinaStyleResize"
]


class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        image, annots, scale, size = sample['image'], sample['annots'], sample[
            'scale'], sample['size']

        if annots.shape[0] == 0:
            return sample

        if np.random.uniform(0, 1) < self.prob:
            image = image[:, ::-1, :]
            _, w, _ = image.shape
            # 水平翻转图像
            image = np.flip(image, axis=1)
            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            annots[:, 0] = w - x2
            annots[:, 1] = w - x1

        return {
            'image': image,
            'annots': annots,
            'scale': scale,
            'size': size,
        }


class RandomCrop:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        image, annots, scale, size = sample['image'], sample['annots'], sample[
            'scale'], sample['size']

        if annots.shape[0] == 0:
            return sample

        # 裁剪除有bbox区域
        if np.random.uniform(0, 1) < self.prob:
            h, w, _ = image.shape

            # [4, ]
            max_bbox = np.concatenate([
                np.min(annots[:, 0:2], axis=0),
                np.max(annots[:, 2:4], axis=0)
            ], axis=-1)
            max_left_trans, max_up_trans = max_bbox[0], max_bbox[1]
            max_right_trans, max_down_trans = w-max_bbox[2], h-max_bbox[3]
            crop_xmin = max(
                0, int(max_bbox[0] - np.random.uniform(0, max_left_trans)))
            crop_ymin = max(
                0, int(max_bbox[1] - np.random.uniform(0, max_up_trans)))
            crop_xmax = min(
                w, int(max_bbox[2] + np.random.uniform(0, max_right_trans)))
            crop_ymax = min(
                h, int(max_bbox[3] + np.random.uniform(0, max_down_trans)))

            image = image[crop_ymin:crop_ymax, crop_xmin:crop_xmax, :]
            # x_min 和 x_max
            annots[:, [0, 2]] = annots[:, [0, 2]] - crop_xmin
            annots[:, [1, 3]] = annots[:, [1, 3]] - crop_ymin

        return {
            'image': image,
            'annots': annots,
            'scale': scale,
            'size': size,
        }


class RandomTranslate:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        image, annots, scale, size = sample['image'], sample['annots'], sample[
            'scale'], sample['size']
        if annots.shape[0] == 0:
            return sample
        if np.random.uniform(0, 1) < self.prob:
            h, w, _ = image.shape
            max_bbox = np.concatenate([
                np.min(annots[:, 0:2], axis=0),
                np.max(annots[:, 2:4], axis=0)
            ], axis=-1)
            max_left_trans, max_up_trans = max_bbox[0], max_bbox[1]
            max_right_trans, max_down_trans = w-max_bbox[2], h-max_bbox[3]
            tx = np.random.uniform(-(max_left_trans - 1),
                                   (max_right_trans - 1))
            ty = np.random.uniform(-(max_up_trans - 1), (max_down_trans - 1))
            M = np.array([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, M, (w, h))
            annots[:, [0, 2]] = annots[:, [0, 2]] + tx
            annots[:, [1, 3]] = annots[:, [1, 3]] + ty
        return {
            'image': image,
            'annots': annots,
            'scale': scale,
            'size': size,
        }


class Normalize:
    def __init__(self):
        pass

    def __call__(self, sample):
        image, annots, scale, size = sample['image'], sample['annots'], sample[
            'scale'], sample['size']

        image = image / 255.

        return {
            'image': image,
            'annots': annots,
            'scale': scale,
            'size': size,
        }


class YoloStyleResize:
    def __init__(self,
                 resize=640,
                 divisor=32,
                 stride=32,
                 multi_scale=False,
                 multi_scale_range=(0.5, 1.0)):
        self.resize = resize
        self.divisor = divisor
        self.stride = stride
        self.multi_scale = multi_scale
        self.multi_scale_range = multi_scale_range

    def __call__(self, sample):
        image, annots, scale, size = sample['image'], sample['annots'], sample[
            'scale'], sample['size']
        h, w, _ = image.shape

        if self.multi_scale:
            scale_range = [
                int(self.multi_scale_range[0] * self.resize),
                int(self.multi_scale_range[1] * self.resize)
            ]
            print(scale_range)
            resize_list = [
                i // self.stride * self.stride
                for i in range(scale_range[0], scale_range[1]+self.stride)
            ]
            resize_list = list(set(resize_list))
            random_idx = np.random.randint(0, len(resize_list))
            final_resize = resize_list[random_idx]
            print(final_resize)
        else:
            final_resize = self.resize

        factor = final_resize / max(h, w)
        resize_h, resize_w = int(round(h * factor)), int(round(w * factor))
        image = cv2.resize(image, (resize_w, resize_h))

        pad_w = 0 if resize_w % self.divisor == 0 else self.divisor - resize_w % self.divisor
        pad_h = 0 if resize_h % self.divisor == 0 else self.divisor - resize_h % self.divisor
        padded_image = np.zeros((resize_h + pad_h, resize_w + pad_w, 3),
                                dtype=np.float32)
        padded_image[:resize_h, :resize_w, :] = image

        factor = np.float(factor)
        annots[:, :4] *= factor
        scale *= factor

        return {
            'image': padded_image,
            'annots': annots,
            'scale': scale,
            'size': size,
        }


class RetinaStyleResize:
    def __init__(self,
                 resize=400,
                 divisor=32,
                 stride=32,
                 multi_scale=False,
                 multi_scale_range=(0.8, 1.0)):
        self.resize = resize
        self.divisor = divisor
        self.stride = stride
        self.multi_scale = multi_scale
        self.multi_scale_range = multi_scale_range
        self.ratio = 1333. / 800

    def __call__(self, sample):
        '''
        sample must be a dict,contains 'image'、'annots'、'scale' keys.
        '''
        image, annots, scale, size = sample['image'], sample['annots'], sample[
            'scale'], sample['size']
        h, w, _ = image.shape

        if self.multi_scale:
            scale_range = [
                int(self.multi_scale_range[0] * self.resize),
                int(self.multi_scale_range[1] * self.resize)
            ]
            resize_list = [
                i // self.stride * self.stride
                for i in range(scale_range[0], scale_range[1] + self.stride)
            ]
            resize_list = list(set(resize_list))

            random_idx = np.random.randint(0, len(resize_list))
            scales = (resize_list[random_idx],
                      int(round(self.resize * self.ratio)))
        else:
            scales = (self.resize, int(round(self.resize * self.ratio)))

        max_long_edge, max_short_edge = max(scales), min(scales)
        factor = min(max_long_edge / max(h, w), max_short_edge / min(h, w))

        resize_h, resize_w = int(round(h * factor)), int(round(w * factor))
        image = cv2.resize(image, (resize_w, resize_h))

        pad_w = 0 if resize_w % self.divisor == 0 else self.divisor - resize_w % self.divisor
        pad_h = 0 if resize_h % self.divisor == 0 else self.divisor - resize_h % self.divisor

        padded_image = np.zeros((resize_h + pad_h, resize_w + pad_w, 3),
                                dtype=np.float32)
        padded_image[:resize_h, :resize_w, :] = image

        factor = np.float32(factor)
        annots[:, :4] *= factor
        scale *= factor

        return {
            'image': padded_image,
            'annots': annots,
            'scale': scale,
            'size': size,
        }


def test():
    arr = [[12, 10, 4, 5, 0], [1, 2, 47, 56, 0],
           [2, 3, 5, 8, 1]]
    arr = np.array(arr)
    print(arr[:, 0:2])
    print(arr[:, [0, 2]])
    yolo = YoloStyleResize(multi_scale=True)
    yolo({"image": np.random.random((640, 640, 3)),
          "annots": "",
          "scale": "",
          "size": ""})


if __name__ == "__main__":
    test()

