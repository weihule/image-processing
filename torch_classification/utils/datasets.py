import os
import torch
import cv2
from PIL import Image
from torch.utils.data import Dataset


class FlowerDataset(Dataset):
    def __init__(self, img_path, label_path, transform=None):
        super(FlowerDataset, self).__init__()
        self.images_path = img_path
        self.label_path = label_path
        self.transform = transform
        if len(self.images_path) != len(self.label_path):
            raise 'images number not equal lables number'

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, index):
        simple = {}
        img = Image.open(self.images_path[index])
        if img.mode != 'RGB':
            raise ValueError(f'wrong img mode {self.images_path[index]}')
        label = self.label_path[index]
        if self.transform is not None:
            img = self.transform(img)
        simple['img'] = img
        simple['label'] = label

        return simple


def collater(simples):
    images = [p['img'] for p in simples]
    labels = [p['label'] for p in simples]

    images = torch.stack(images, dim=0)
    labels = torch.as_tensor(labels)
    return images, labels



