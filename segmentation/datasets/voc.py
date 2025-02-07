import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset


VOC_CLASSES = [
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor',
]


class VOCSegmentation(Dataset):
    def __init__(self, voc_root, year="2012", transform=None, txt_name="train.txt"):
        super(VOCSegmentation, self).__init__()
        assert year in ["2007", "2012"], "year must be in ['2007', '2012']"
        root = Path(voc_root) / f"VOC{year}"
        assert root.exists(), f"path {str(root)} does not exists."

        image_dir = root / "JPEGImages"
        mask_dir = root / "SegmentationClass"
        txt_path = root / "ImageSets" / "Segmentation" / txt_name
        assert txt_path.exists(), f"path {str(txt_path)} does not exists."
        with open(str(txt_path), "r", encoding="utf-8") as fr:
            file_names = [x.strip() for x in fr.readlines() if len(x.strip()) > 0]

        self.images = [image_dir / (x+".jpg") for x in file_names]
        self.masks = [mask_dir / (x+".png") for x in file_names]
        assert len(self.images) == len(self.masks)
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB")
        target = Image.open(self.masks[index])

        sample = {
            "image": image,
            "target": target
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.images)

    @staticmethod
    def collate_fn(batch):
        images, targets = [], []
        for x in batch:
            images.append(x["image"])
            targets.append(x["target"])


def cat_lust(images, fill_value=0):
    max_size = 0


CLASSES = [
    '_background_',
    'person',
    'car',
    'motorbike',
    'dustbin',
    'chair',
    'fire_hydrant',
    'tricycle',
    'bicycle',
    'stone',
]

class ImageDataSet(Dataset):
    def __init__(self, image_paths, mask_paths, transform):
        """
        image_paths: 图片路径
        mask_paths: 标签文件
        """
        super(ImageDataSet, self).__init__()
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        mask = Image.open(self.mask_paths[index])

        sample = self.transform({'image': image, 'mask': mask})
        image, mask = sample['image'], sample['mask']

        x = np.array(image, np.float32)
        print(f"x.shape = {x.shape}")

        image = np.transpose((np.array(image, np.float32)), [2,0,1])
        mask = np.array(sample['mask'])

        return {'image': image, 'mask': mask}


def test():
    # voc = VOCSegmentation(voc_root=r"D:\workspace\data\VOCdataset")
    # sample_ = voc[10]
    # image_, target_ = sample_["image"], sample_["target"]
    # print(image_.size)
    pass



if __name__ == "__main__":
    test()

