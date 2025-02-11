import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
import torch


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
    def __init__(self, image_paths, mask_paths, input_size, num_classes, transform):
        """
        image_paths: 图片路径
        mask_paths: 标签文件
        num_classes: 类别数(包含背景这一类)
        """
        super(ImageDataSet, self).__init__()
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.input_size = input_size
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        mask = Image.open(self.mask_paths[index])

        sample = self.transform({'image': image, 'mask': mask})
        image, mask = sample['image'], sample['mask']

        image = np.transpose((np.array(image, np.float32)), [2,0,1])
        # 做归一化
        image = image / 255.

        mask = np.array(mask)
        # VOC数据集中,目标边缘的像素值是255(白色),还有大于类别数的无效像素值,都变成背景0,不做另外一个类别的处理
        mask[mask >= self.num_classes] = 0

        # [mask.reshape([-1]): [h, w] -> [h*w, ]
        # seg_labels.shape: [h*w, self.num_classes]
        seg_labels = np.eye(self.num_classes)[mask.reshape([-1])]
        seg_labels = seg_labels.reshape((int(self.input_size[0]), int(self.input_size[1]), self.num_classes))
        # print(image.shape, mask.shape, mask.reshape([-1]).shape, seg_labels.shape)

        return {'image': image, 'mask': mask, 'seg_label': seg_labels}


def dataset_collate(batch):
    images, masks, seg_labels = [], [], []
    for x in batch:
        images.append(x["image"])
        masks.append(x["mask"])
        seg_labels.append(x["seg_label"])
    images = np.stack(images, axis=0)
    masks = np.stack(masks, axis=0)
    seg_labels = np.stack(seg_labels, axis=0)
    images = torch.from_numpy(images).type(torch.FloatTensor)
    masks = torch.from_numpy(masks).long()
    seg_labels  = torch.from_numpy(seg_labels).type(torch.FloatTensor)

    return {'image': images, 'mask': masks, 'seg_label': seg_labels}

def test():
    # voc = VOCSegmentation(voc_root=r"D:\workspace\data\VOCdataset")
    # sample_ = voc[10]
    # image_, target_ = sample_["image"], sample_["target"]
    # print(image_.size)
    img = cv2.imread(r'D:\workspace\data\images\VOCdevkit\VOC2012\SegmentationClass\2007_000032.png')
    print(img)


if __name__ == "__main__":
    test()

