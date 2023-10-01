import os
import cv2
import math
import numpy as np

from pycocotools.coco import COCO
from torch.utils.data import Dataset

COCO_CLASSES = [
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'airplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'backpack',
    'umbrella',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'couch',
    'potted plant',
    'bed',
    'dining table',
    'toilet',
    'tv',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush',
]

COCO_CLASSES_COLOR = [(241, 23, 78), (63, 71, 49), (67, 79, 143),
                      (32, 250, 205), (136, 228, 157), (135, 125, 104),
                      (151, 46, 171), (129, 37, 28), (3, 248, 159),
                      (154, 129, 58), (93, 155, 200), (201, 98, 152),
                      (187, 194, 70), (122, 144, 121), (168, 31, 32),
                      (168, 68, 189), (173, 68, 45), (200, 81, 154),
                      (171, 114, 139), (216, 211, 39), (187, 119, 238),
                      (201, 120, 112), (129, 16, 164), (211, 3, 208),
                      (169, 41, 248), (100, 77, 159), (140, 104, 243),
                      (26, 165, 41), (225, 176, 197), (35, 212, 67),
                      (160, 245, 68), (7, 87, 70), (52, 107, 85),
                      (103, 64, 188), (245, 76, 17), (248, 154, 59),
                      (77, 45, 123), (210, 95, 230), (172, 188, 171),
                      (250, 44, 233), (161, 71, 46), (144, 14, 134),
                      (231, 142, 186), (34, 1, 200), (144, 42, 108),
                      (222, 70, 139), (138, 62, 77), (178, 99, 61),
                      (17, 94, 132), (93, 248, 254), (244, 116, 204),
                      (138, 165, 238), (44, 216, 225), (224, 164, 12),
                      (91, 126, 184), (116, 254, 49), (70, 250, 105),
                      (252, 237, 54), (196, 136, 21), (234, 13, 149),
                      (66, 43, 47), (2, 73, 234), (118, 181, 5),
                      (105, 99, 225), (150, 253, 92), (59, 2, 121),
                      (176, 190, 223), (91, 62, 47), (198, 124, 140),
                      (100, 135, 185), (20, 207, 98), (216, 38, 133),
                      (17, 202, 208), (216, 135, 81), (212, 203, 33),
                      (108, 135, 76), (28, 47, 170), (142, 128, 121),
                      (23, 161, 179), (33, 183, 224)]


# COCO标注中提供了类别index
# 但是原始标注的类别index不连续（1-90,但是只有80个类）
# 我们要将其转换成连续的类别index0-79
class CocoDetection(Dataset):
    def __init__(self, root_dir, set_name='train2017', transform=None):
        assert set_name in ['train2017', 'val2017'], 'Wrong set name!'

        self.image_dir = os.path.join(root_dir, 'images', set_name)
        self.annot_dir = os.path.join(root_dir, 'annotations', f'instances_{set_name}.json')
        self.coco = COCO(self.annot_dir)
        self.image_ids = self.coco.getImgIds()

        if 'train' in set_name:
            # filter image id without annotation,from 118287 ids to 117266 ids
            ids = []
            for image_id in self.image_ids:
                annot_ids = self.coco.getAnnIds(imgIds=image_id)
                annots = self.coco.loadAnns(annot_ids)
                if len(annots) == 0:
                    continue
                ids.append(image_id)
            self.image_ids = ids

        self.cat_ids = self.coco.getCatIds()
        self.cats = sorted(self.coco.loadCats(self.cat_ids), key=lambda x: x['id'])
        self.num_classes = len(self.cats)

        # cat_id is an original cat id,coco_label is set from 0 to 79
        self.cat_id_to_cat_name = {cat['id']: cat['name'] for cat in self.cats}
        self.cat_id_to_coco_label = {cat['id']: i for i, cat in enumerate(self.cats)}
        self.coco_label_to_cat_id = {
            i: cat['id']
            for i, cat in enumerate(self.cats)
        }
        self.coco_label_to_cat_name = {
            coco_label: self.cat_id_to_cat_name[cat_id]
            for coco_label, cat_id in self.coco_label_to_cat_id.items()
        }

        self.transform = transform

        print(f'Dataset Size:{len(self.image_ids)}')
        print(f'Dataset Class Num:{self.num_classes}')

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
                resize_factor = self.resize * 1.5 / max(origin_h, origin_w)  # 这里的 *1.5 是自己添加的
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

    def load_image(self, idx):
        # coco.loadImgs 返回一个list, 这里只有一张图片, 所以取索引为0
        file_name = self.coco.loadImgs(self.image_ids[idx])[0]["file_name"]
        path = os.path.join(self.image_dir, file_name)
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8),
                             cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image.astype(np.float32)

    def load_annots(self, idx):
        """
        input:
            idx: 图片的索引 (0, 1, 2, ...)
        output:
            该图片中的gt, np.ndarray类型, (N, 5)
            N 表示有多少个gt
            5 表示gt的左上右下坐标和gt所属的类别 (0, 1, ..., 79)
        """

        # coco.getAnnIds()会得到所有的annotations下的 id
        # 然后根据 ImgIdx 找到属于该ImgIdx所有的ann标注
        # 返回的即是 annotations 中的 id
        # 因为每一个annotation都有一个独一无二的id
        annot_ids = self.coco.getAnnIds(imgIds=self.image_ids[idx])
        annots = self.coco.loadAnns(annot_ids)

        image_info = self.coco.loadImgs(self.image_ids[idx])[0]
        image_h, image_w = image_info["height"], image_info["width"]

        targets = np.zeros((0, 5))

        # some images appear to miss annotations
        if len(annot_ids) == 0:
            return targets

        # parse annotations
        for annot in annots:
            if "ignore" in annot.keys():
                continue
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
