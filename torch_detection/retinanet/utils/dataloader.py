import os
import sys
import torch
import random
import numpy as np
import json

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import Sampler

from pycocotools.coco import COCO

import skimage.io
import skimage.transform
import skimage.color
import skimage
import cv2
from PIL import Image


# COCO标注中提供了类别index，
# 但是原始标注的类别index不连续（1-90，但是只有80个类）
# 我们要将其转换成连续的类别index0-79

class CocoDetection(Dataset):
    def __init__(self,
                 image_root_dir,
                 annotation_root_dir,
                 set='train2017',
                 coco_classes='coco_classes.json',
                 transform=None):
        self.image_root_dir = image_root_dir
        self.annotation_root_dir = annotation_root_dir
        self.set_name = set
        self.coco_classes = coco_classes
        self.transform = transform
 
        self.coco = COCO(
            os.path.join(self.annotation_root_dir,
                         'instances_' + self.set_name + '.json'))

        self.load_classes()

    def load_classes(self):
        self.image_ids = self.coco.getImgIds()  # 获取图片id, 返回包含所有图片id的列表
        self.cat_ids = self.coco.getCatIds()    # 获取类别id, 返回包含所有类别id的列表

        self.categories = self.coco.loadCats(self.cat_ids)
        self.categories.sort(key=lambda x: x['id'])
        
        # category_id is an original id,coco_id is set from 0 to 79
        self.category_id_to_coco_label = {cat['id']: idx for idx, cat in enumerate(self.categories)}
        self.coco_label_to_category_id = {idx: cat['id'] for idx, cat in enumerate(self.categories)}
        
    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        img = self.load_image(index)
        annot = self.load_annotations(index)

        sample = {'img': img, 'annot': annot, 'scale': 1}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, image_index):
        img_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.image_root_dir, img_info['file_name'])
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # opencv读取的图片转成RGB之后并归一化
        return img.astype(np.float32) / 255

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

        # get ground truth annotations
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

        return annotations


    def find_coco_label_from_category_id(self, category_id):
        return self.category_id_to_coco_label[category_id]

    def find_category_id_from_coco_label(self, coco_label):
        return self.coco_label_to_category_id[coco_label]

    def num_classes(self):
        return 80

    def image_aspect_ratio(self, image_idx):
        image = self.coco.loadImgs(self.image_ids[image_idx])[0]

        return float(image['width'])/float(image['height'])

    def coco_label_to_class_name(self):
        with open(self.coco_classes, 'r', encoding='utf-8') as fr:
            infos = json.load(fr)
            coco_label_to_name = {
                v: k for k, v in infos["COCO_CLASSES"].items()
            }
        return coco_label_to_name


if __name__ == "__main__":
    # main(2, 0)
    img_root = 'D:\\workspace\\data\\DL\\COCO2017\\images\\val2017'
    anno_root = 'D:\\workspace\\data\\DL\\COCO2017\\annotations'

    cd = CocoDetection(img_root, anno_root, set='val2017')
    # cd.load_classes()
    # cd.load_annotations(0)

    label_to_name = cd.coco_label_to_class_name()
    for i in range(20):
        res = cd[i]
        img = res['img']
        annot = res['annot']
        for anno in annot:
            label = anno[-1]
            cv2.putText(img, label_to_name[label], np.int32(anno[:2]), cv2.FONT_HERSHEY_PLAIN, 0.40, (255, 0, 0), 1)
            img = cv2.rectangle(img, np.int32(anno[:2]), np.int32(anno[2:4]), (0, 255, 0), 1)
        cv2.namedWindow('res', cv2.WINDOW_FREERATIO)
        cv2.imshow('res', img)
        cv2.waitKey(0)
