import torch
import cv2
import os
from pathlib import Path
from torch.utils.data import Dataset
import random


class CarDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale < 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if
                    isfile(join(images_dir, file)) and not file.startswith('.')]

    def __getitem__(self):
        pass

    def __len__(self):
        pass

