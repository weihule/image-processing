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


class VOCSeg(Dataset):
    def __init__(self,
                 root_dir,
                 image_sets=(('2007', 'trainval'), ('2012', 'trainval')),
                 transform=None,
                 keep_difficult=False):
        super(VOCSeg, self).__init__()

        self.cats = VOC_CLASSES
        self.num_classes = len(self.cats)

        self.cat_to_voc = {cat: i for i, cat in enumerate(self.cats)}
        self.voc_to_cat = {i: cat for i, cat in enumerate(self.cats)}

        self.keep_difficult = keep_difficult

        self.ids = []

        for (year, name) in image_sets:
            rootpath = os.path.join(root_dir, "VOC" + year)
            with open(os.path.join(rootpath, "ImageSets", "Main", name + ".txt"), "r") as fr:
                lines = fr.readlines()
            for line in lines:
                self.ids.append((rootpath, line.strip()))

        self.image_path = []
        self.annot_path = []
        for x1, x2 in self.ids:
            img_p = Path(x1) / "JPEGImages" / (x2 + ".jpg")
            ann_p = Path(x1) / "Annotations" / (x2 + ".xml")
            assert img_p.exists(), f"{img_p} not exists !"
            self.image_path.append(img_p)
            self.annot_path.append(ann_p)

        self.transform = transform

        print(f'Dataset Size:{len(self.ids)}')
        print(f'Dataset Class Num:{self.num_classes}')

    def __getitem__(self, idx):
        image = self.load_image(idx)
        annots = self.load_mask(idx)

        scale = np.array(1.).astype(np.float32)
        # [height, width]
        size = np.array([image.shape[0], image.shape[1]]).astype(np.float32)
        sample = {
            "image": image,
            "annots": annots,
            "scale": scale,
            "size": size
        }
        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.ids)

    def load_image(self, idx):
        image = cv2.imdecode(np.fromfile(str(self.image_path[idx]), dtype=np.uint8),
                             cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image.astype(np.float32)

    def load_mask(self, idx):
        annot = ET.parse(str(self.annot_path[idx])).getroot()
        targets = []
        size = annot.find("size")
        h, w = int(size.find("height").text), int(size.find("width").text)
        for obj in annot.iter("object"):
            difficult = int(obj.find("difficult").text == 1)
            if not self.keep_difficult and difficult:
                continue
            class_name = obj.find("name").text.lower().strip()
            bbox = obj.find("bndbox")

            pts = ["xmin", "ymin", "xmax", "ymax"]
            target = []
            for pt in pts:
                cur_pt = float(bbox.find(pt).text)
                target.append(cur_pt)

            # 剔除框很小的
            if target[2] - target[0] < 1 or target[3] - target[1] < 1:
                continue

            # 剔除框超过图像边界的
            if (target[0] < 0 or
                    target[1] < 0 or
                    target[2] > w or
                    target[3] > h or
                    target[2] < 0 or
                    target[3] < 0):
                continue

            if class_name not in self.cats:
                continue

            target.append(self.cat_to_voc[class_name])
            # ["xmin", "ymin", "xmax", "ymax", voc_label]
            targets.append(target)

        if len(targets) == 0:
            targets = np.zeros((0, 5))
        else:
            targets = np.array(targets)

        # [[x1, y1, x2, y2, voc_label], ...]
        return targets.astype(np.float32)