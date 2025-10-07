from __future__ import print_function, absolute_import
import os
import cv2
from PIL import Image
import numpy as np

from torch.utils.data import Dataset

# from torchreid.datas.datasets.images import Market1501
from .datasets.images import Market1501, DukeMTMC, MSMT17


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
            img = self.transform(img)
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


"""Create dataset"""
__img_factory = {
    'market1501': Market1501,
    'duke': DukeMTMC,
    'msmt17': MSMT17,
    'reid_debug': Market1501,
}


def get_names():
    names = []
    for k in __img_factory:
        names.append(k)

    return names


def init_img_dataset(name, **kwargs):
    if name not in __img_factory.keys():
        raise KeyError('Invalid dataset, got {}, but excepted to be one of {}'.format(name, __img_factory.keys()))
    return __img_factory[name](**kwargs)


if __name__ == "__main__":
    dataset = init_img_dataset('market1501', root='D:\\workspace\\data\\dl')

