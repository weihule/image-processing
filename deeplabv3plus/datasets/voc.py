import os
import sys
import tarfile
from pathlib import Path
import collections
from torch.utils.data import Dataset, DataLoader
import shutil
import numpy as np

from PIL import Image
from torchvision.datasets.utils import download_url

DATASET_YEAR_DICT = {
    '2012': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
        'filename': 'VOCtrainval_11-May-2012.tar',
        'md5': '6cd6e144f989b92b3379bac3b3de84fd',
        'base_dir': 'VOCdevkit/VOC2012'
    },
    '2011': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2011/VOCtrainval_25-May-2011.tar',
        'filename': 'VOCtrainval_25-May-2011.tar',
        'md5': '6c3384ef61512963050cb5d687e5bf1e',
        'base_dir': 'TrainVal/VOCdevkit/VOC2011'
    },
    '2010': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar',
        'filename': 'VOCtrainval_03-May-2010.tar',
        'md5': 'da459979d0c395079b5c75ee67908abb',
        'base_dir': 'VOCdevkit/VOC2010'
    },
    '2009': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2009/VOCtrainval_11-May-2009.tar',
        'filename': 'VOCtrainval_11-May-2009.tar',
        'md5': '59065e4b188729180974ef6572f6a212',
        'base_dir': 'VOCdevkit/VOC2009'
    },
    '2008': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2008/VOCtrainval_14-Jul-2008.tar',
        'filename': 'VOCtrainval_11-May-2012.tar',
        'md5': '2629fa636546599198acfcfbfcf1904a',
        'base_dir': 'VOCdevkit/VOC2008'
    },
    '2007': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
        'filename': 'VOCtrainval_06-Nov-2007.tar',
        'md5': 'c52e279531787c972589f7e41ab4ae64',
        'base_dir': 'VOCdevkit/VOC2007'
    }
}

def voc_map(N=256, normalized=False):
    """
    语义分割的「类别索引」（0~255）映射成「RGB 颜色值」
    """
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

class VOCSegmentation(Dataset):
    cmap = voc_map()
    def __init__(self,
                 root,
                 year='2012',
                 image_set='train',
                 download=False,
                 transform=None):
        is_aug = False
        if year == '2012_aug':
            is_aug = True
            year = '2012'

        self.root = Path(root)

        self.year = year
        self.url = DATASET_YEAR_DICT[year]['url']
        self.filename = DATASET_YEAR_DICT[year]['filename']
        self.md5 = DATASET_YEAR_DICT[year]['md5']
        self.transform = transform

        self.image_set = image_set
        base_dir = DATASET_YEAR_DICT[year]['base_dir']
        voc_root = self.root / base_dir
        image_dir = voc_root /  'JPEGImages'

        if download:
            download_extract(self.url, self.root, self.filename, self.md5)

        if not voc_root.is_dir():
            raise RuntimeError('Dataset not found or corrupted.'+
                               'You can use download=True to download it')

        if is_aug and image_set == 'train':
            mask_dir = voc_root / 'SegmentationClassAug'
            assert mask_dir.exists(), "SegmentationClassAug not found, please refer to README.md and prepare it manually"
            split_f = os.path.join(self.root, 'train_aug.txt')
        else:
            mask_dir = voc_root / 'SegmentationClass'
            splits_dir = voc_root /  'ImageSets' / 'Segmentation'
            split_f = splits_dir / (image_set.rstrip('\n') + '.txt')

        if not split_f.exists():
            raise ValueError(
                'Wrong image_set entered! Please use image_set="train" '
                'or image_set="trainval" or image_set="val"'
            )

        with open(split_f, 'r') as fr:
            file_names = [x.strip() for x in fr.readlines()]

        self.images = [image_dir/(x+'.jpg') for x in file_names]
        self.masks = [mask_dir/(x+'.png') for x in file_names]
        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])

        # 检查target是否为单通道，如果不是，则转换为单通道
        # if target.mode != "L":
        #     target = target.convert("L")

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.images)

    @classmethod
    def decode_target(cls, mask):
        return cls.cmap[mask]

def download_extract(url, root, filename, md5):
    download_url(url, root, filename, md5)
    with tarfile.open(os.path.join(root, filename), "r") as tar:
        tar.extractall(path=root)

def test():
    cmap = voc_map()
    print(cmap)


if __name__ == "__main__":
    test()






