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
        super(Dataset, self).__init__()
        self.coco_label_to_category_id = None
        self.category_id_to_coco_label = None
        self.categories = None
        self.cat_ids = None
        self.image_ids = None
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
        self.cat_ids = self.coco.getCatIds()  # 获取类别id, 返回包含所有类别id的列表

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
        # coco.loadImgs 返回一个list, 这里只有一张图片, 所以取索引为0
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
        self.category_id_to_voc_lable = dict()
        with open('pascal_voc_classes.json', 'r', encoding='utf-8') as fr:
            self.category_id_to_voc_lable = json.load(fr)

        self.voc_lable_to_category_id = {v: k for k, v in self.category_id_to_voc_lable.items()}

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

        return img.astype(np.float32) / 225.

    def load_annotations(self, img_id):
        xml_path = os.path.join(img_id[0], 'Annotations', img_id[1] + '.xml')
        annotations = np.zeros((0, 5))

        target = ET.parse(xml_path).getroot()
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text)
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bndbox = obj.find('bndbox')
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            annotation = list()
            for p in pts:
                annotation.append(int(bndbox.find(p).text))
            annotation.append(self.category_id_to_voc_lable[name])  # [x_min, y_min, x_max, y_max, label_id]

            # 这里必须要升维
            annotation = np.expand_dims(annotation, axis=0)  # (5, ) -> (1, 5)
            annotations = np.append(annotations, annotation, axis=0)

        return annotations

    def image_aspect_ratio(self, idx):
        img_id = self.ids[idx]
        img = self.load_image(img_id)
        h, w, _ = img.shape

        return float(w) / float(h)


class COCODataPrefetcher():
    """
    数据预读取就是模型在进行本次batch的前向计算和反向传播时
    就预先加载下一个batch的数据, 这样就节省了下加载数据的时间
    (相当于加载数据与前向计算和反向传播并行了).
    """

    def __init__(self, loader):
        self.next_input = None
        self.next_annot = None
        self.loader = iter(loader)
        self.stream = torch.cuda.stream()

    def preload(self):
        # 当我们已经迭代完最后⼀个数据之后，
        # 再次调⽤next()函数会抛出 StopIteration的异常 ，
        # 来告诉我们所有数据都已迭代完成，
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


class RetinaStyleResize:
    def __init__(self,
                 resize=400,
                 divisor=32,
                 stride=32,
                 multi_scale=False,
                 multi_scale_range=None):
        if multi_scale_range is None:
            multi_scale_range = [0.8, 1.0]
        self.resize = resize
        self.divisor = divisor
        self.stride = stride
        self.multi_scale = multi_scale
        self.multi_scale_range = multi_scale_range
        self.ratio = 1333. / 800

    def __call__(self, sample):
        """
        sample must be a dict,contains 'img'、'annot'、'scale' keys.
        """
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
            scales = (resize_list[random_idx],
                      int(round(self.resize * self.ratio)))
        else:
            scales = (self.resize, int(round(self.resize * self.ratio)))

        max_long_edge, max_short_edge = max(scales), min(scales)
        factor = min(max_long_edge / max(h, w), max_short_edge / min(h, w))

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


class Resizer:
    """Convert ndarrays in sample to Tensors."""

    def __init__(self,
                 resize=600):
        self.resize = resize
        self.ratio = 1333. / 800

    # def __call__(self, sample, min_side=608, max_side=1024):
    def __call__(self, sample):
        scales = (self.resize, int(round(self.resize * self.ratio)))
        max_side, min_side = max(scales), min(scales)
        image, annots = sample['img'], sample['annot']
        h, w, c = image.shape
        smallest_side = min(h, w)

        # rescale the image so the smallest_side is min_side
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(h, w)
        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        resized_w, resized_h = int(round(w * scale)), int(round(h * scale))
        image = cv2.resize(image, (resized_w, resized_h))
        pad_w = 0 if resized_w % 32 == 0 else 32 - resized_w % 32
        pad_h = 0 if resized_h % 32 == 0 else 32 - resized_h % 32

        padded_img = np.zeros((resized_h + pad_h, resized_w + pad_w, c), dtype=np.float32)
        padded_img[:resized_h, :resized_w, :] = image

        annots[:, :4] *= scale

        return {'img': padded_img, 'annot': annots, 'scale': scale}


class RandomFlip:
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, sample):
        flip_flag = np.random.uniform(0, 1)
        if flip_flag < self.flip_prob:
            print(flip_flag)
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


def collater(datas):
    """
    对于一个batch的images和annotations,
    我们最后还需要用collater函数将images
    和annotations的shape全部对齐后才能输入模型进行训练。
    """
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


if __name__ == "__main__":
    # main(2, 0)
    # img_root = 'D:\\workspace\\data\\DL\\COCO2017\\images\\val2017'
    # anno_root = 'D:\\workspace\\data\\DL\\COCO2017\\annotations'

    img_root = '/nfs/home57/weihule/data/dl/COCO2017/images/val2017'
    anno_root = '/nfs/home57/weihule/data/dl/COCO2017/annotations'

    cd = CocoDetection(img_root, anno_root, set='val2017')
    res = cd[0]
    for k, v in res.items():
        print(k)


    # label_to_name = cd.coco_label_to_class_name()
    # for i in range(20):
    #     res = cd[i]
    #     img = res['img']
    #     annot = res['annot']
    #     for anno in annot:
    #         label = anno[-1]
    #         cv2.putText(img, label_to_name[label], np.int32(anno[:2]), cv2.FONT_HERSHEY_PLAIN, 0.40, (255, 0, 0), 1)
    #         img = cv2.rectangle(img, np.int32(anno[:2]), np.int32(anno[2:4]), (0, 255, 0), 1)
    #     cv2.namedWindow('res', cv2.WINDOW_FREERATIO)
    #     cv2.imshow('res', img)
    #     cv2.waitKey(0)

    # root = 'D:\\workspace\\data\\DL\\VOCdataset'
    root = '/workshop/weihule/data/DL/VOCdataset'

    # vd = VocDetection(root)
    # # retina_resize = RetinaStyleResize()
    # label_to_name = vd.voc_lable_to_category_id
    # save_root = 'D:\\Desktop'
    # for i in range(0, 20, 4):
    #     batch_data = list()
    #     for j in range(i, i + 4):
    #         sample = vd[j]
    #         # res = RF(res)
    #         sample = Resizer()(sample)
    #         batch_data.append(sample)
    #     res = collater(batch_data)
    #     print(res['img'].shape)
    #     print(res['annot'].shape)
        # img = sample['img'] * 225.
        # print(i, img.shape)
        # image = Image.fromarray(np.uint8(img))
        # print(image.mode)
        # img_dra = ImageDraw.Draw(image)
        # annot = sample['annot']

        # for anno in annot:
        #     label = anno[-1]
        #     cv2.putText(img, label_to_name[int(label)], np.int32(anno[:2]), cv2.FONT_HERSHEY_PLAIN, 0.40, (255, 0, 0), 1)
        #     img = cv2.rectangle(img, np.int32(anno[:2]), np.int32(anno[2:4]), (0, 255, 0), 1)
        # cv2.namedWindow('res', cv2.WINDOW_FREERATIO)
        # cv2.imshow('res', img)
        # cv2.waitKey(0)

        # for anno in annot:
        #     label = anno[-1]
        #     img_dra.rectangle((anno[:-1].tolist()), fill=None, outline='green', width=1) 
        # plt.imshow(image)
        # plt.xticks([])
        # plt.yticks([])
        # plt.axis('off')
        # plt.show()
        # save_path = os.path.join(save_root, str(i).rjust(3, '0')+'.png')
        # image.save(save_path)

    # arr = np.zeros((0, 4))
    # arr1 = np.random.random((1, 4))
    # arr2 = np.random.random((1, 4))
    # arr = np.append(arr, arr1, axis=0)
    # arr = np.append(arr, arr2, axis=0)
    # print(arr)

    # print(np.random.uniform(0, 1, (2, 3)))

    # img_path = 'D:\\workspace\\data\\DL\\VOCdataset\\VOC2007\\JPEGImages\\000001.jpg'
    # img = cv2.imread(img_path)

    # # a = np.uint8(np.arange(36).reshape(4, 3, 3))
    # # print(a[:, ::-1])
    # img_flip = cv2.flip(img, 1)
    # img_flip = img[:, ::-1, :]
    # img_combine = np.hstack((img, img_flip))
    # cv2.imshow('res', img_combine)
    # cv2.waitKey(0)
