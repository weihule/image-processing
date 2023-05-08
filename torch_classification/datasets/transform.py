import os
import numpy as np
import torch
from torchvision import transforms
import cv2
from PIL import Image
from torch.utils.data import Dataset

__all__ = [
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
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4, hue=0):
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


transform = {
    'train': transforms.Compose([
        OpenCV2PIL(),
        # TorchResize(resize=224),
        # TorchRandomHorizontalFlip(prob=0.5),
        PIL2OpenCV()
    ]),
    'val': transforms.Compose([
        OpenCV2PIL(),
        TorchResize(resize=400),
        PIL2OpenCV()
    ])
}
