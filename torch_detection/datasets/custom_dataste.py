import os
import random
import sys
import torch
import json
import cv2
import numpy as np
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

__all__ = [
    'VOCDataset',
    'COCODataset'
]


class VOCDataset(Dataset):
    def __init__(self,
                 root_dir=None,
                 image_sets=None,
                 resize=416,
                 use_mosaic=False,
                 transform=None,
                 keep_difficult=False):
        super(VOCDataset, self).__init__()
        if image_sets is None:
            self.image_sets = [('2007', 'trainval'), ('2012', 'trainval')]
        else:
            self.image_sets = image_sets
        self.resize = resize
        self.use_mosaic = use_mosaic
        self.root_dir = root_dir
        self.transform = transform
        self.categories = None

        self.category_name_to_voc_label = dict()
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pascal_voc_classes.json')

        with open(file_path, 'r', encoding='utf-8') as fr:
            self.category_name_to_voc_label = json.load(fr)['classes']

        self.voc_label_to_category_name = {v: k for k, v in self.category_name_to_voc_label.items()}
        self.keep_difficult = keep_difficult

        self.ids = list()   # 存储2007和2012中 trainval.txt 中的图片路径，有16551个元素
        for (year, name) in self.image_sets:
            rootpath = os.path.join(self.root_dir, 'VOC'+year)
            txt_file = os.path.join(rootpath, 'ImageSets', 'Main', name + '.txt')
            with open(txt_file, 'r', encoding='utf-8') as fr:
                lines = fr.readlines()
            for line in lines:
                self.ids.append((rootpath, line.strip('\n')))

    def __getitem__(self, idx):
        if self.use_mosaic and idx % 2 == 0:
            x_ctr, y_ctr = [int(random.uniform(self.resize*0.8,
                                               self.resize*1.2))
                            for _ in range(2)]
            # all 4 image indices
            img_indices = [idx] + [random.randint(0, len(self.ids)-1) for _ in range(3)]
            img_indices = [self.ids[p] for p in img_indices]

            annot = []

            # combine image by 4 images
            img = np.full((self.resize*2, self.resize*2, 3), 111, dtype=np.uint8)

            for i, img_idx in enumerate(img_indices):
                sub_img = self._load_image(img_idx)
                sub_annot = self._load_annotations(img_idx)
                origin_h, origin_w, _ = sub_img.shape

                # 这里四张子图不resize到416
                # top left
                if i == 0:
                    x1a, y1a = max(x_ctr - origin_w, 0), max(y_ctr - origin_h, 0)
                    x2a, y2a = x_ctr, y_ctr

                    x1b, y1b = max(origin_w - x_ctr, 0), max(origin_h - y_ctr, 0)
                    x2b, y2b = origin_w, origin_h

                # top right
                elif i == 1:
                    x1a, y1a = x_ctr, max(y_ctr - origin_h, 0)
                    x2a, y2a = min(self.resize * 2, x_ctr + origin_w), y_ctr

                    x1b, y1b = 0, max(origin_h - y_ctr, 0)
                    x2b, y2b = min(origin_w, self.resize * 2 - x_ctr), origin_h
                # bottom left img
                elif i == 2:
                    x1a, y1a = max(x_ctr - origin_w, 0), y_ctr
                    x2a, y2a = x_ctr, min(self.resize * 2, y_ctr + origin_h)

                    x1b, y1b = max(origin_w - x_ctr, 0), 0
                    x2b, y2b = origin_w, min(origin_h, self.resize * 2 - y_ctr)
                # bottom right img
                else:
                    x1a, y1a = x_ctr, y_ctr
                    x2a, y2a = min(self.resize * 2, x_ctr + origin_w), min(self.resize * 2, y_ctr + origin_h)

                    x1b, y1b = 0, 0
                    x2b, y2b = min(origin_w, self.resize * 2 - x_ctr), min(origin_h, self.resize * 2 - y_ctr)

                img[y1a:y2a, x1a:x2a] = sub_img[y1b:y2b, x1b:x2b]
                pad_w, pad_h = x1a - x1b, y1a - y1b
                if sub_annot.shape[0] > 0:
                    sub_annot[:, [0, 2]] += pad_w
                    sub_annot[:, [1, 3]] += pad_h
                annot.append(sub_annot)
            annot = np.concatenate(annot, axis=0)
            annot[:, :4] = np.clip(annot[:, :4], a_min=0, a_max=self.resize*2)

            annot = annot[annot[:, 2] - annot[:, 0] > 1]
            annot = annot[annot[:, 3] - annot[:, 1] > 1]
        else:
            img_id = self.ids[idx]
            img = self._load_image(img_id)
            annot = self._load_annotations(img_id)

        sample = {'img': img, 'annot': annot, 'scale': 1.}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.ids)

    def _load_image(self, img_ids):
        img_path = os.path.join(img_ids[0], 'JPEGImages', img_ids[1] + '.jpg')
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img.astype(np.float32)

    def _load_annotations(self, img_ids):
        xml_path = os.path.join(img_ids[0], 'Annotations', img_ids[1] + '.xml')
        annotations = np.zeros((0, 5))

        pts = ['xmin', 'ymin', 'xmax', 'ymax']
        targets = ET.parse(xml_path).getroot()
        for obj in targets.iter('object'):
            difficult = int(obj.find('difficult').text)
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bndbox = obj.find('bndbox')
            annotation = []
            for pt in pts:
                annotation.append(int(bndbox.find(pt).text))
            annotation.append(self.category_name_to_voc_label[name])    # [xmin, ymin, xmax, ymax, label_id]

            annotation = np.expand_dims(annotation, axis=0)    # [1, 5]
            annotations = np.append(annotations, annotation, axis=0)

        return annotations

    def _image_aspect_ratio(self, item):
        img_ids = self.ids[item]
        img = self._load_image(img_ids)
        h, w, _ = img.shape

        return float(w) / float(h)


# COCO标注中提供了类别index
# 但是原始标注的类别index不连续（1-90,但是只有80个类）
# 我们要将其转换成连续的类别index0-79
class COCODataset(Dataset):
    def __init__(self,
                 image_root_dir,
                 annotation_root_dir,
                 set_name='train2017',
                 coco_classes='coco_classes.json',
                 resize=416,
                 use_mosaic=False,
                 mosaic_center_range=None,
                 transform=None):
        super(Dataset, self).__init__()
        self.coco_label_to_category_id = None
        self.category_id_to_coco_label = None
        self.categories = None
        self.cat_ids = None
        self.image_ids = None
        self.image_root_dir = image_root_dir
        self.annotation_root_dir = annotation_root_dir
        self.set_name = set_name
        self.coco_classes = coco_classes
        self.resize = resize
        self.use_mosaic = use_mosaic
        if mosaic_center_range is None:
            self.mosaic_center_range = [0.5, 1.5]
        self.transform = transform

        self.coco = COCO(
            os.path.join(self.annotation_root_dir,
                         'instances_' + self.set_name + '.json'))

        self.load_classes()

    def load_classes(self):
        self.image_ids = self.coco.getImgIds()  # 获取图片id, 返回包含所有图片id的列表
        self.cat_ids = self.coco.getCatIds()  # 获取类别id, 返回包含所有类别id的列表, 80个元素 [1, 2, 3, ..., 90]

        # 返回包含80个元素的列表,每个元素是一个字典 {'supercategory': 'person', 'id': 1, 'name': 'person'}
        self.categories = self.coco.loadCats(self.cat_ids)
        self.categories.sort(key=lambda x: x['id'])

        # category_id is an original id, coco_id is set from 0 to 79
        self.category_id_to_coco_label = {cat['id']: idx for idx, cat in enumerate(self.categories)}
        self.coco_label_to_category_id = {idx: cat['id'] for idx, cat in enumerate(self.categories)}

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        if self.use_mosaic and index % 5 == 0:
            x_ctr, y_ctr = [
                int(random.uniform(self.resize * self.mosaic_center_range[0],
                                   self.resize * self.mosaic_center_range[1]))
                for _ in range(2)
            ]
            # all 4 image indices
            imgs_indices = [index] + [
                random.randint(0, len(self.image_ids) - 1) for _ in range(3)
            ]

            annot = []
            # combined image by 4 images
            img = np.full((self.resize * 2, self.resize * 2, 3), 111, dtype=np.uint8)

            for i, img_idx in enumerate(imgs_indices):
                sub_img = self.load_image(img_idx)
                sub_annot = self.load_annotations(img_idx)  # (N, 5)

                origin_h, origin_w, _ = sub_img.shape
                resize_factor = self.resize * 1.5 / max(origin_h, origin_w)     # 这里的 *1.5 是自己添加的
                resize_h, resize_w = int(resize_factor * origin_h), int(resize_factor * origin_w)
                sub_img = cv2.resize(sub_img, (resize_w, resize_h))
                sub_annot[:, :4] *= resize_factor

                # top left
                if i == 0:
                    # combined image coordinates
                    x1a, y1a = max(x_ctr - resize_w, 0), max(y_ctr - resize_h, 0)
                    x2a, y2a = x_ctr, y_ctr

                    # single img choose area
                    x1b, y1b = max(resize_w - x_ctr, 0), max(resize_h - y_ctr, 0)
                    x2b, y2b = resize_w, resize_h
                # top right
                elif i == 1:
                    x1a, y1a = x_ctr, max(y_ctr - resize_h, 0)
                    x2a, y2a = min(self.resize * 2, x_ctr + resize_w), y_ctr

                    x1b, y1b = 0, max(resize_h - y_ctr, 0)
                    x2b, y2b = min(resize_w, self.resize * 2 - x_ctr), resize_h
                # bottom left img
                elif i == 2:
                    x1a, y1a = max(x_ctr - resize_w, 0), y_ctr
                    x2a, y2a = x_ctr, min(self.resize * 2, y_ctr + resize_h)

                    x1b, y1b = max(resize_w - x_ctr, 0), 0
                    x2b, y2b = resize_w, min(resize_h, self.resize * 2 - y_ctr)
                # bottom right img
                else:
                    x1a, y1a = x_ctr, y_ctr
                    x2a, y2a = min(self.resize * 2, x_ctr + resize_w), min(self.resize * 2, y_ctr + resize_h)

                    x1b, y1b = 0, 0
                    x2b, y2b = min(resize_w, self.resize * 2 - x_ctr), min(resize_h, self.resize * 2 - y_ctr)

                img[y1a: y2a, x1a: x2a] = sub_img[y1b: y2b, x1b: x2b]
                pad_w, pad_h = x1a - x1b, y1a - y1b
                if sub_annot.shape[0] > 0:
                    sub_annot[:, [0, 2]] += pad_w
                    sub_annot[:, [1, 3]] += pad_h

                annot.append(sub_annot)

            annot = np.concatenate(annot, axis=0)
            annot[:, :4] = np.clip(annot[:, :4], a_min=0, a_max=self.resize * 2)

            annot = annot[annot[:, 2] - annot[:, 0] > 1]
            annot = annot[annot[:, 3] - annot[:, 1] > 1]
        else:
            img = self.load_image(index)
            annot = self.load_annotations(index)

        sample = {'img': img, 'annot': annot, 'scale': 1}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, image_index):
        # coco.loadImgs 返回一个list, 这里只有一张图片, 所以取索引为0
        img_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.image_root_dir, img_info['file_name'])
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # opencv读取的图片转成RGB
        return img.astype(np.float32)

    def load_annotations(self, image_index):
        """
        input:
            image_index: 图片的索引 (0, 1, 2, ...)
        output:
            该图片中的gt, np.ndarray类型, (N, 5)
            N 表示有多少个gt
            5 表示gt的左上右下坐标和gt所属的类别 (0, 1, ..., 79)
        """

        # coco.getAnnIds()会得到所有的annotations下的 id
        # 然后根据 ImgIdx 找到属于该ImgIdx所有的ann标注
        # 返回的即是 annotations 中的 id
        # 因为每一个annotation都有一个独一无二的id
        annotations_ids = self.coco.getAnnIds(
            imgIds=self.image_ids[image_index],
            iscrowd=None
        )
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for a in coco_annotations:
            # some annotations have basically no width/height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[:, :4] = a['bbox']
            annotation[:, 4] = self.find_coco_label_from_category_id(a['category_id'])

            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x_min, y_min, w, h] to [x_min, y_min, x_max, y_max]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations.astype(np.float32)

    def find_coco_label_from_category_id(self, category_id):
        return self.category_id_to_coco_label[category_id]

    def find_category_id_from_coco_label(self, coco_label):
        return self.coco_label_to_category_id[coco_label]

    def num_classes(self):
        return 80

    def image_aspect_ratio(self, image_idx):
        image = self.coco.loadImgs(self.image_ids[image_idx])[0]

        return float(image['width']) / float(image['height'])

    def coco_label_to_class_name(self):
        with open(self.coco_classes, 'r', encoding='utf-8') as fr:
            infos = json.load(fr)
            coco_label_to_name = {
                v: k for k, v in infos["COCO_CLASSES"].items()
            }
        return coco_label_to_name


if __name__ == "__main__":
    voc = VOCDataset(root_dir='D:\\workspace\\data\\dl\\VOCdataset')
    print(voc.resize)

