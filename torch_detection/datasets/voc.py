import os
from pathlib import Path
import sys
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset

__all__ = [
    "VOC_CLASSES",
    "VOC_CLASSES_COLOR",
    "VOCDetection"
]


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

# VOC_CLASSES_COLOR = [(128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128),
#                      (128, 0, 128), (0, 128, 128), (128, 128, 128), (64, 0, 0),
#                      (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128),
#                      (192, 0, 128), (64, 128, 128), (192, 128, 128),
#                      (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
#                      (0, 64, 128)]

VOC_CLASSES_COLOR = [
    (255, 0, 0),      # 红色
    (0, 255, 0),      # 绿色
    (0, 0, 255),      # 蓝色
    (255, 255, 0),    # 黄色
    (255, 0, 255),    # 品红
    (0, 255, 255),    # 青色
    (255, 128, 0),    # 橙色
    (128, 255, 0),    # 酸橙色
    (128, 0, 255),    # 紫色
    (255, 128, 128),  # 浅粉红
    (128, 255, 128),  # 淡绿色
    (128, 128, 255),  # 淡紫色
    (255, 128, 192),  # 粉红
    (192, 255, 128),  # 淡黄色
    (192, 128, 255),  # 深紫色
    (255, 192, 128),  # 桃色
    (128, 255, 192),  # 薄荷绿
    (192, 128, 128),  # 棕色
    (128, 192, 128),  # 橄榄绿
    (128, 128, 192),  # 银色
]


class VOCDetection(Dataset):
    def __init__(self,
                 root_dir,
                 image_sets=(('2007', 'trainval'), ('2012', 'trainval')),
                 transform=None,
                 keep_difficult=False):
        super(VOCDetection, self).__init__()

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
        annots = self.load_annot(idx)

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

    def load_annot(self, idx):
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


class VOCDetection2(Dataset):

    def __init__(self,
                 root_dir,
                 image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
                 transform=None,
                 keep_difficult=False):

        self.annotpath = os.path.join('%s', 'Annotations', '%s.xml')
        self.imagepath = os.path.join('%s', 'JPEGImages', '%s.jpg')

        self.cats = VOC_CLASSES
        self.num_classes = len(self.cats)

        self.cat_to_voc_label = {cat: i for i, cat in enumerate(self.cats)}
        self.voc_label_to_cat = {i: cat for i, cat in enumerate(self.cats)}

        self.keep_difficult = keep_difficult

        self.ids = []
        for (year, name) in image_sets:
            rootpath = os.path.join(root_dir, 'VOC' + year)
            for line in open(
                    os.path.join(rootpath, 'ImageSets', 'Main',
                                 name + '.txt')):
                self.ids.append((rootpath, line.strip()))

        self.transform = transform

        print(f'Dataset Size:{len(self.ids)}')
        print(f'Dataset Class Num:{self.num_classes}')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image = self.load_image(idx)
        annots = self.load_annots(idx)

        scale = np.array(1.).astype(np.float32)
        size = np.array([image.shape[0], image.shape[1]]).astype(np.float32)

        sample = {
            'image': image,
            'annots': annots,
            'scale': scale,
            'size': size,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, idx):
        image = cv2.imdecode(
            np.fromfile(self.imagepath % self.ids[idx], dtype=np.uint8),
            cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image.astype(np.float32)

    def load_annots(self, idx):
        annots = ET.parse(self.annotpath % self.ids[idx]).getroot()

        targets = []
        size = annots.find('size')
        h, w = int(size.find('height').text), int(size.find('width').text)
        for obj in annots.iter('object'):
            difficult = (int(obj.find('difficult').text) == 1)
            if not self.keep_difficult and difficult:
                continue

            class_name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            target = []
            for pt in pts:
                cur_pt = float(bbox.find(pt).text)
                target.append(cur_pt)

            if target[2] - target[0] < 1 or target[3] - target[1] < 1:
                continue

            if target[0] < 0 or target[1] < 0 or target[2] > w or target[3] > h:
                continue

            if class_name not in self.cats:
                continue

            target.append(self.cat_to_voc_label[class_name])
            # [xmin, ymin, xmax, ymax, voc_label]
            targets += [target]

        if len(targets) == 0:
            targets = np.zeros((0, 5))
        else:
            targets = np.array(targets)

        # format:[[x1, y1, x2, y2, voc_label], ... ]
        return targets.astype(np.float32)


def test():
    from PIL import Image, ImageFont, ImageDraw
    import matplotlib.pyplot as plt
    voc = VOCDetection(root_dir=r"D:\workspace\data\dl\VOCdataset")
    # 使用合适的字体
    font = ImageFont.truetype("./simhei.ttf", 15)
    for idx in range(15):
        s = voc[idx]
        image, annots, scale, size = s["image"], s["annots"], s["scale"], s["size"]
        # image = image[:, ::-1, :]
        image = np.flip(image, axis=1)
        image = Image.fromarray(image.astype(np.uint8))

        # 创建一个可绘制对象
        draw = ImageDraw.Draw(image)

        # 画框
        for annot in annots:
            xmin, ymin, xmax, ymax, label = annot
            # 绘制矩形框
            draw.rectangle((xmin, ymin, xmax, ymax),
                           outline=VOC_CLASSES_COLOR[int(label)],
                           width=1)

            # 获取字符串在特定字体下的长度（宽度）
            text = VOC_CLASSES[int(label)]
            # 获取文本的包围框大小
            bbox = font.getbbox(text)

            # 提取包围框的宽度和高度
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # 写入文本
            draw.text((xmin, ymin-15),
                      text=text,
                      fill=VOC_CLASSES_COLOR[int(label)],
                      font=font)

        plt.imshow(image)
        plt.show()
