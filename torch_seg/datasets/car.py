import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from pathlib import Path
from torch.utils.data import Dataset
import random


def load_image(filename):
    return Image.open(filename)


def unique_mask_values(idx, mask_dir: Path, mask_suffix: str):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.array(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class CarDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [f.stem for f in self.images_dir.iterdir() if f.is_file()]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        unique = list()
        for i in self.ids:
            res = unique_mask_values(i, self.mask_dir, self.mask_suffix)
            unique.append(res)
        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'

        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        # 如果两者尺寸不一致报错
        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        img = torch.as_tensor(img.copy()).float().contiguous()
        mask = torch.as_tensor(mask.copy()).long().contiguous()

        return {
            'image': img,
            'mask': mask
        }

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img: Image, scale, is_mask):
        w, h = pil_img.size
        new_w, new_h = int(scale * w), int(scale * h)
        assert new_w > 0 and new_h > 0, 'Scale is too small, resized images would have no pixel'
        # Image.NEAREST：这是最简单的重采样方法，它只考虑最近的像素点。
        pil_img = pil_img.resize((new_w, new_h),
                                 resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.array(pil_img)

        if is_mask:
            mask = np.zeros((new_h, new_w), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i
            return mask
        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            # TODO： 这里应该是已经做了标准化
            if (img > 1).any():
                img = img / 255.

            return img


if __name__ == "__main__":
    car = CarDataset(images_dir=r"D:\workspace\data\dl\car\img",
                     mask_dir=r"D:\workspace\data\dl\car\mask",
                     mask_suffix="_mask")
    data = car[10]
    image = data["image"]
    print(image.shape)
    print(image)


