import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import json

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from pycocotools.coco import COCO
from custom_dataste import VOCDataset, COCODataset
from data_transfrom import RandomFlip, RandomCrop
from collater_func import MultiScaleCollater


def test_make_grid():
    data_transform = {
        'train': transforms.Compose([
            RandomFlip(flip_prob=0.5),
            RandomCrop(crop_prob=0.35),
            # RandomTranslate(translate_prob=0.2)
        ]),
        'val': transforms.Compose([
            RandomFlip(flip_prob=0.5),
        ])
    }

    # -----------------------------------
    # voc_root_dir1 = '/data/weihule/data/dl/VOCdataset'
    # voc_root_dir2 = 'D:\\workspace\\data\\dl\\VOCdataset'
    #
    # voc_root_dir = voc_root_dir2
    # dataset = VOCDataset(root_dir=voc_root_dir,
    #                      transform=data_transform['train'],
    #                      resize=640,
    #                      use_mosaic=True)
    #
    # for i, p in enumerate(dataset):
    #     print(i, p['img'].shape)
    #     if i == 10:
    #         break

    # -----------------------------------

    # -----------------------------------
    data_set_root1 = '/workshop/weihule/data/dl/COCO2017'
    data_set_root2 = 'D:\\workspace\\data\\dl\\COCO2017'
    data_set_roots = [data_set_root1, data_set_root2]
    data_set_root = data_set_root2

    dataset_path = os.path.join(data_set_root, 'images', 'val2017')
    dataset_annot_path = os.path.join(data_set_root, 'annotations')
    dataset = COCODataset(image_root_dir=dataset_path,
                          annotation_root_dir=dataset_annot_path,
                          set_name='val2017',
                          use_mosaic=True,
                          transform=data_transform['train'])
    # -----------------------------------
    # VOC
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    detec_collater = MultiScaleCollater(mean=mean,
                                        std=std,
                                        resize=560,
                                        stride=32,
                                        use_multi_scale=True,
                                        normalize=True)
    data_loader = DataLoader(dataset,
                             batch_size=4,
                             shuffle=True,
                             num_workers=4,
                             prefetch_factor=2,
                             pin_memory=True,
                             collate_fn=detec_collater)
    # COCO
    # mean = torch.tensor([[[[0.471, 0.448, 0.408]]]], dtype=torch.float32)
    # std = torch.tensor([[[[0.234, 0.239, 0.242]]]], dtype=torch.float32)

    # -----------------------------------
    # file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pascal_voc_classes.json')
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'coco_classes.json')

    with open(file_path, 'r', encoding='utf-8') as fr:
        infos = json.load(fr)
        if 'classes' in infos:
            category_name_to_lable = infos['classes']
        elif 'COCO_CLASSES' in infos:
            category_name_to_lable = infos['COCO_CLASSES']
        colors = infos['colors']
    lable_to_category_name = {v: k for k, v in category_name_to_lable.items()}
    # -----------------------------------

    save_root1 = 'images_show'
    save_root2 = 'D:\\Desktop\\infer_shows'

    save_root = save_root2
    if not os.path.exists(save_root):
        os.mkdir(save_root)

    b_mean = torch.tensor(mean, dtype=torch.float32).tile(1, 1, 1, 1)
    b_std = torch.tensor(std, dtype=torch.float32).tile(1, 1, 1, 1)
    for index, datas in enumerate(data_loader):
        print(index)
        batch_images, batch_annots = datas['img'], datas['annot']
        batch_images = batch_images.permute(0, 2, 3, 1).contiguous()
        batch_images = (batch_images * b_std + b_mean) * 255.

        c = 0
        for img, annot in zip(batch_images, batch_annots):
            c += 1
            img, annot = img.numpy(), annot.numpy()
            # img = img.transpose(1, 2, 0)  # [c, h, w] -> [h, w, c] RGB
            #
            # img = (img * std + mean) * 255.
            # img *= 255.
            img = np.uint8(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # RGB -> BGR
            image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(image)
            font_size = 16
            font = ImageFont.truetype("simhei.ttf", size=font_size)
            mask = annot[:, 4] >= 0
            annot = annot[mask]
            for point in annot:
                # point = np.int32(point[:4])
                # cv2.rectangle(img, [point[0], point[1]], [point[2], point[3]], (0, 255, 0), 1)
                label = int(point[4])
                category_name = lable_to_category_name[label]
                category_color = tuple(colors[label])
                chars_w, chars_h = font.getsize(category_name)
                draw.rectangle(point[:4], outline=category_color, width=2)  # 绘制预测框
                draw.rectangle((point[0], point[1] - chars_h, point[0] + chars_w, point[1]),
                               fill=category_color)  # 文本填充框
                draw.text((point[0], point[1] - font_size), category_name, fill=(255, 255, 255), font=font)

            save_path = os.path.join(save_root, str(index) + '_' + str(c) + '.png')
            # cv2.imwrite(save_path, img)
            image.save(save_path)
        if index == 10:
            break

    # img = make_grid(batch_images, nrow=8)
    # img = img.numpy()
    # img = img.transpose(1, 2, 0)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # cv2.imwrite('test.jpg', img)


def test_coco_api():
    val_path = '/workshop/weihule/data/dl/COCO2017/annotations/instances_val2017.json'
    coco = COCO(val_path)
    image_ids = coco.getImgIds()
    cat_ids = coco.getCatIds()
    categories = coco.loadCats(cat_ids)
    categories = sorted(categories, key=lambda x: x['id'])

    # category_id is an original id, coco_id is set from 0 to 79
    category_id_to_coco_label = {p['id']: index for index, p in enumerate(categories)}
    coco_label_to_category_id = {value: key for key, value in category_id_to_coco_label.items()}

    image_index = 0
    # coco.loadImgs 返回一个list, 这里只有一张图片, 所以取索引为0
    img_info = coco.loadImgs(image_ids[image_index])[0]

    img_infos = coco.loadImgs(image_ids)

    annotations_ids = coco.getAnnIds(
        imgIds=image_ids[image_index],
        iscrowd=None
    )
    print(len(annotations_ids))
    coco_annotations = coco.loadAnns(annotations_ids)
    print(len(coco_annotations))
    print(coco_annotations[0])


class ShowSigma:
    def __init__(self, x):
        self.x = x

    def sigma_fuc(self):
        y = 1 / (1 + np.power(np.e, -self.x))
        return y

    def sigma_variant_func(self):
        y = 2 / (1 + np.power(np.e, -self.x)) - 0.5
        return y

    def visualize(self):
        sigma_y = self.sigma_fuc()
        sigma_variant_y = self.sigma_variant_func()

        plt.figure(figsize=(10, 10))
        line1, = plt.plot(self.x, sigma_y, 'r-')
        line2, = plt.plot(self.x, sigma_variant_y, 'g--')
        plt.legend(handles=[line1, line2], labels=["sigma", "scale"], loc="best", fontsize=12)
        plt.savefig('test.png')


if __name__ == "__main__":
    test_make_grid()
    # test_coco_api()

    # x = np.arange(-10, 10, 0.5)
    # print(x)
    # ss = ShowSigma(x)
    # ss.visualize()
