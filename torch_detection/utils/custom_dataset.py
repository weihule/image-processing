import os
import sys
import torch
import random
import numpy as np
import json
import cv2
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


# COCO标注中提供了类别index
# 但是原始标注的类别index不连续（1-90,但是只有80个类）
# 我们要将其转换成连续的类别index0-79
class CocoDetection(Dataset):
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
            img = np.full((self.resize * 2, self.resize * 2, 3), 0, dtype=np.uint8)

            for i, img_idx in enumerate(imgs_indices):
                sub_img = self.load_image(img_idx)
                sub_annot = self.load_annotations(img_idx)  # (N, 5)

                origin_h, origin_w, _ = sub_img.shape
                resize_factor = self.resize * 1.5 / max(origin_h, origin_w)
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

        # opencv读取的图片转成RGB之后并归一化
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


class VocDetection(Dataset):
    def __init__(self,
                 root_dir,
                 image_sets=None,
                 transform=None,
                 keep_difficult=False):
        super(Dataset, self).__init__()
        if image_sets is None:
            image_sets = [('2007', 'trainval'), ('2012', 'trainval')]
        self.root_dir = root_dir
        self.image_set = image_sets
        self.transform = transform
        self.categories = None

        # 这里的category_id指的就是二十个类别的名字
        self.category_name_to_voc_lable = dict()
        file_path = '/nfs/home57/weihule/code/study/torch_detection/utils/pascal_voc_classes.json'
        if not os.path.exists(file_path):
            file_path = '/workshop/weihule/code/study/torch_detection/utils/pascal_voc_classes.json'
        with open(file_path, 'r', encoding='utf-8') as fr:
            self.category_name_to_voc_lable = json.load(fr)['classes']

        self.voc_lable_to_category_id = {v: k for k, v in self.category_name_to_voc_lable.items()}

        self.keep_difficult = keep_difficult

        self.ids = list()  # 存储的是2007和2012中 trainval.txt 中的图片名, 有16551个
        for (year, name) in image_sets:
            rootpath = os.path.join(self.root_dir, 'VOC' + year)
            txt_file = os.path.join(rootpath, 'ImageSets', 'Main', name + '.txt')
            with open(txt_file, 'r', encoding='utf-8') as fr:
                lines = fr.readlines()
            for line in lines:
                self.ids.append((rootpath, line.strip('\n')))

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img = self.load_image(img_id)
        annot = self.load_annotations(img_id)

        sample = {'img': img, 'annot': annot, 'scale': 1.}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.ids)

    def load_image(self, img_id):
        img_path = os.path.join(img_id[0], 'JPEGImages', img_id[1] + '.jpg')
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img.astype(np.float32)

    def load_annotations(self, img_id):
        xml_path = os.path.join(img_id[0], 'Annotations', img_id[1] + '.xml')
        annotations = np.zeros((0, 5))

        pts = ['xmin', 'ymin', 'xmax', 'ymax']
        target = ET.parse(xml_path).getroot()
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text)
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bndbox = obj.find('bndbox')
            annotation = list()
            for p in pts:
                annotation.append(int(bndbox.find(p).text))
            annotation.append(self.category_name_to_voc_lable[name])  # [x_min, y_min, x_max, y_max, label_id]

            # 这里必须要升维
            annotation = np.expand_dims(annotation, axis=0)  # (5, ) -> (1, 5)
            annotations = np.append(annotations, annotation, axis=0)

        return annotations

    def image_aspect_ratio(self, idx):
        img_id = self.ids[idx]
        img = self.load_image(img_id)
        h, w, _ = img.shape

        return float(w) / float(h)


class Resizer:
    def __init__(self,
                 resize=600):
        self.resize = resize

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        h, w, c = image.shape
        if h >= w:
            scale = self.resize / h
            resize_w = int(round(scale * w))
            resize_h = self.resize
        else:
            scale = self.resize / w
            resize_w = self.resize
            resize_h = int(round(scale * h))

        resize_img = cv2.resize(image, (resize_w, resize_h))
        padded_img = np.zeros((self.resize, self.resize, 3), dtype=np.float32)
        padded_img[:resize_h, :resize_w, :] = resize_img
        annots[:, :4] *= scale

        return {'img': padded_img, 'annot': annots, 'scale': scale}


class RandomFlip:
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, sample):
        flip_flag = np.random.uniform(0, 1)
        if flip_flag < self.flip_prob:
            image = sample['img']
            # image = image[:, ::-1, :]   # 水平镜像
            image = cv2.flip(image, 1)
            annots = sample['annot']
            scale = sample['scale']

            height, width, channel = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            annots[:, 0] = width - x2
            annots[:, 2] = width - x1

            sample = {'img': image, 'annot': annots, 'scale': scale}

        return sample


class RandomCrop:
    def __init__(self, crop_prob=0.5):
        self.crop_prob = crop_prob

    def __call__(self, sample):
        image, annots, scale = sample['img'], sample['annot'], sample['scale']
        if annots.shape[0] == 0:
            return sample
        prob = random.uniform(0, 1)
        if prob < self.crop_prob:
            h, w, _ = image.shape
            # 找出所有gt的最小外接矩形, shape: (4, )
            max_bbox = np.concatenate((np.min(annots[:, :2], axis=0),
                                       np.max(annots[:, 2:], axis=0)), axis=-1)
            max_left_trans, max_up_trans = max_bbox[0], max_bbox[1]
            max_right_trans, max_down_trans = w - max_bbox[2], h - max_bbox[3]

            # crop_x_min = max(0, int(max_bbox[0]-np.random.uniform(0, max_left_trans)))
            crop_x_min = int(np.random.uniform(0, max_left_trans))
            crop_y_min = int(np.random.uniform(0, max_up_trans))
            crop_x_max = max(w, int(max_bbox[2] + np.random.uniform(0, max_right_trans)))
            crop_y_max = max(h, int(max_bbox[2] + np.random.uniform(0, max_down_trans)))

            image = image[crop_y_min: crop_y_max, crop_x_min: crop_x_max]
            annots[:, [0, 2]] = annots[:, [0, 2]] - crop_x_min
            annots[:, [1, 3]] = annots[:, [1, 3]] - crop_y_min

            sample = {'img': image, 'annot': annots, 'scale': scale}
        return sample


class RandomTranslate:
    def __init__(self, translate_prob=0.5):
        self.translate_prob = translate_prob

    def __call__(self, sample):
        image, annots, scale = sample['img'], sample['annot'], sample['scale']

        if annots.shape[0] == 0:
            return sample

        prob = random.uniform(0, 1)
        if prob < self.translate_prob:
            h, w, _ = image.shape
            # 找出所有annots的最小外接矩形
            max_bbox = np.concatenate((np.min(annots[:, :2], axis=0),
                                       np.max(annots[:, 2:], axis=0)), axis=-1)
            max_left_trans, max_up_trans = max_bbox[0], max_bbox[1]
            max_right_trans, max_down_trans = w - max_bbox[2], h - max_bbox[3]

            tx = random.uniform(-(max_left_trans - 1), (max_right_trans - 1))
            ty = random.uniform(-(max_up_trans - 1), (max_down_trans - 1))

            M = np.array([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, M, (w, h))

            annots[:, [0, 2]] = annots[:, [0, 2]] + tx
            annots[:, [1, 3]] = annots[:, [1, 3]] + ty

        return {'img': image, 'annot': annots, 'scale': scale}


class Normalize:
    def __init__(self):
        self.mean = torch.tensor([[[[0.471, 0.448, 0.408]]]], dtype=torch.float32)
        self.std = torch.tensor([[[[0.234, 0.239, 0.242]]]], dtype=torch.float32)

    # def __call__(self, sample):
    #     # image shape: [h, w, 3], 这里三分量的顺序已经调整成 RGB 了
    #     image, annots, scale = sample['img'], sample['annot'], sample['scale']
    #     image = (image - self.mean) / self.std
    #     return {
    #         'img': image,
    #         'annot': annots,
    #         'scale': scale,
    #     }

    def __call__(self, image):
        # image shape: [h, w, 3], 这里三分量的顺序已经调整成 RGB 了
        image = (image - self.mean) / self.std
        return image


class DataPrefetcher():
    """
    数据预读取就是模型在进行本次batch的前向计算和反向传播时
    就预先加载下一个batch的数据, 这样就节省了下加载数据的时间
    (相当于加载数据与前向计算和反向传播并行了).
    """

    def __init__(self, loader):
        # self.next_input = None
        # self.next_annot = None
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        # 当我们已经迭代完最后⼀个数据之后,
        # 再次调⽤next()函数会抛出 StopIteration的异常 ,
        # 来告诉我们所有数据都已迭代完成,
        # 不⽤再执⾏ next()函数了。
        try:
            sample = next(self.loader)
            self.next_input, self.next_annot = sample['img'], sample['annot']
        except StopIteration:
            self.next_input = None
            self.next_annot = None
            return

        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_annot = self.next_annot.cuda(non_blocking=True)
            self.next_input = self.next_input.float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        inputs = self.next_input
        annots = self.next_annot
        self.preload()

        return inputs, annots


def collater(datas):
    """
    对于一个batch的images和annotations,
    我们最后还需要用collater函数将images
    和annotations的shape全部对齐后才能输入模型进行训练。
    这里也将 numpy 转成 Tensor  参见: padded_img
    也进行了维度转换
    """
    normalizer = Normalize()
    batch_size = len(datas)
    imgs = [p['img'] for p in datas]
    # annots 里面每一个元素都是一个二维数组, (N, 5)
    # N表示当前图片中的gt数量
    annots = [torch.from_numpy(p['annot']) for p in datas]
    scales = [p['scale'] for p in datas]

    img_h_list = [p.shape[0] for p in imgs]
    img_w_list = [p.shape[1] for p in imgs]
    max_h, max_w = max(img_h_list), max(img_w_list)
    padded_img = torch.zeros((batch_size, max_h, max_w, 3), dtype=torch.float32)
    for i in range(batch_size):
        img = imgs[i]
        padded_img[i, :img.shape[0], :img.shape[1], :] = torch.from_numpy(img)
        # 这里只做标准化, dataset中已经做了归一化
    padded_img = normalizer(padded_img)
    padded_img = padded_img.permute(0, 3, 1, 2).contiguous()

    # imgs = torch.from_numpy(np.stack(imgs, axis=0))
    max_num_annots = max([annot.shape[0] for annot in annots])
    if max_num_annots > 0:
        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * (-1)
        for idx, anno in enumerate(annots):
            if anno.shape[0] > 0:
                annot_padded[idx, :anno.shape[0], :] = anno
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * (-1)

    return {'img': padded_img, 'annot': annot_padded, 'scale': scales}


class YoloStyleResize:
    def __init__(self,
                 resize=640,
                 divisor=32,
                 stride=32,
                 multi_scale=False,
                 multi_scale_range=None):
        self.resize = resize
        self.divisor = divisor
        self.stride = stride
        self.multi_scale = multi_scale
        if multi_scale_range is None:
            self.multi_scale_range = [0.5, 1.0]

    def __call__(self, sample):
        '''
        sample must be a dict,contains 'image'、'annots'、'scale' keys.
        '''
        image, annots, scale = sample['img'], sample['annot'], sample[
            'scale']
        h, w, _ = image.shape

        if self.multi_scale:
            scale_range = [
                int(self.multi_scale_range[0] * self.resize),
                int(self.multi_scale_range[1] * self.resize)
            ]
            resize_list = [
                i // self.stride * self.stride
                for i in range(scale_range[0], scale_range[1] + self.stride)
            ]
            resize_list = list(set(resize_list))

            random_idx = np.random.randint(0, len(resize_list))
            final_resize = resize_list[random_idx]
        else:
            final_resize = self.resize

        factor = final_resize / max(h, w)

        resize_h, resize_w = int(round(h * factor)), int(round(w * factor))
        image = cv2.resize(image, (resize_w, resize_h))

        pad_w = 0 if resize_w % self.divisor == 0 else self.divisor - resize_w % self.divisor
        pad_h = 0 if resize_h % self.divisor == 0 else self.divisor - resize_h % self.divisor

        padded_image = np.zeros((resize_h + pad_h, resize_w + pad_w, 3),
                                dtype=np.float32)
        padded_image[:resize_h, :resize_w, :] = image

        factor = np.float32(factor)
        annots[:, :4] *= factor
        scale *= factor

        return {
            'img': padded_image,
            'annot': annots,
            'scale': scale,
        }


class DetectionCollater:
    def __init__(self, divisor=32):
        self.divisor = divisor

    def __call__(self, sample):
        normalizer = Normalize()
        images = [s['img'] for s in sample]
        annots = [s['annot'] for s in sample]
        scales = [s['scale'] for s in sample]

        max_h = max(image.shape[0] for image in images)
        max_w = max(image.shape[1] for image in images)

        pad_h = 0 if max_h % self.divisor == 0 else self.divisor - max_h % self.divisor
        pad_w = 0 if max_w % self.divisor == 0 else self.divisor - max_w % self.divisor

        input_images = np.zeros((len(images), max_h + pad_h, max_w + pad_w, 3),
                                dtype=np.float32)
        for i, image in enumerate(images):
            input_images[i, 0:image.shape[0], 0:image.shape[1], :] = image

        # B H W 3 ->B 3 H W
        input_images = torch.from_numpy(input_images)
        input_images = normalizer(input_images)
        input_images = input_images.permute(0, 3, 1, 2)

        max_annots_num = max(annot.shape[0] for annot in annots)
        if max_annots_num > 0:
            input_annots = np.ones(
                (len(annots), max_annots_num, 5), dtype=np.float32) * (-1)
            for i, annot in enumerate(annots):
                if annot.shape[0] > 0:
                    input_annots[i, :annot.shape[0], :] = annot
        else:
            input_annots = np.ones(
                (len(annots), 1, 5), dtype=np.float32) * (-1)

        input_annots = torch.from_numpy(input_annots)

        scales = np.array(scales, dtype=np.float32)

        return {
            'img': input_images,
            'annot': input_annots,
            'scale': scales,
        }


"""
这里采用yolov3/yolov5中的multi scale方法
首先确定一个最小到最大尺寸的范围,比如没用multi scale前resize=416,
那么可以确定一个范围[0.5,1.5],乘以416之后就是最小到最大尺寸的范围。
然后,选择一个stride长度,比如yolov3/yolov5是32,
找出最小到最大尺寸中所有能被stride长度整除的尺寸。
最后,随机选择一个尺寸,将所有图片resize到该尺寸再填充成正方形图片就可以送入网络训练了。
注意用multi scale时transform里不再使用resize方法
"""


class MultiScaleCollater():
    def __init__(self,
                 mean,
                 std,
                 resize=416,
                 multi_scale_range=None,
                 stride=32,
                 use_multi_scale=False):
        self.mean = torch.tensor(mean, dtype=torch.float32).tile(1, 1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).tile(1, 1, 1, 1)
        self.resize = resize
        if multi_scale_range is None:
            self.multi_scale_range = [0.5, 1.5]
        else:
            self.multi_scale_range = multi_scale_range
        self.stride = stride
        self.use_multi_scale = use_multi_scale

    def __call__(self, samples):
        if self.use_multi_scale:
            min_resize = int(
                ((self.resize + self.stride) * self.multi_scale_range[0]) //
                self.stride * self.stride)
            max_resize = int(
                ((self.resize + self.stride) * self.multi_scale_range[1]) //
                self.stride * self.stride)
            final_resize = random.choice(range(min_resize, max_resize, self.stride))
        else:
            final_resize = self.resize

        imgs = [p['img'] for p in samples]
        annots = [p['annot'] for p in samples]
        scales = [p['scale'] for p in samples]

        batch_size = len(imgs)

        padded_img = torch.zeros((batch_size, final_resize, final_resize, 3))

        for index, img in enumerate(imgs):
            height, width, _ = img.shape
            max_image_size = max(height, width)
            resize_factor = final_resize / max_image_size
            resize_h, resize_w = int(resize_factor * height), int(resize_factor * width)
            img = cv2.resize(img, (resize_w, resize_h))
            padded_img[index, :resize_h, :resize_w] = torch.from_numpy(img)

            annots[index][:, :4] *= resize_factor
            scales[index] *= resize_factor

        # padded_img [B, H, W, 3] -> [B, 3, H, W]
        padded_img = padded_img / 255.
        padded_img = (padded_img - self.mean) / self.std
        padded_img = padded_img.permute(0, 3, 1, 2).contiguous()

        max_num_annots = max(annot.shape[0] for annot in annots)
        if max_num_annots > 0:
            padded_annots = torch.ones((batch_size, max_num_annots, 5)) * (-1)

            for annot_index, p in enumerate(annots):
                if p.shape[0] > 0:
                    padded_annots[annot_index, :p.shape[0], :] = torch.from_numpy(p)
        else:
            padded_annots = torch.ones((batch_size, 1, 5)) * (-1)

        return {'img': padded_img, 'annot': padded_annots, 'scale': scales}


if __name__ == "__main__":
    img_root = '/nfs/home57/weihule/data/dl/COCO2017/images/val2017'
    anno_root = '/nfs/home57/weihule/data/dl/COCO2017/annotations'

    root = '/nfs/home57/weihule/data/dl/VOCdataset'

    vd = VocDetection(root)
    retina_resize = Resizer()
    label_to_name = vd.voc_lable_to_category_id
    for i in range(10):
        sample_data = vd[i]
        print(sample_data['img'].shape)
        sample_data = retina_resize(sample_data)
        print(sample_data['img'].shape)
        print('=' * 10)
