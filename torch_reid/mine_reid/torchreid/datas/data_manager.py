from __future__ import print_function, absolute_import
import os
import cv2
from PIL import Image
import numpy as np

from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        img_path, pid, camid = self.dataset[item]
        img = self.read_img(img_path)
        if self.transform:
            img = self.transform
        return img, pid, camid

    @staticmethod
    def read_img(img_path):
        if not os.path.exists(img_path):
            raise IOError('{} dose not exist'.format(img_path))
        got_img = False
        while not got_img:
            try:
                img = Image.open(img_path).convert('RGB')
                got_img = True
            except IOError:
                print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
                pass
        return img

