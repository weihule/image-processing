import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import json
from tqdm import tqdm
import time

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from pycocotools.coco import COCO

from custom_dataset import VocDetection, CocoDetection
# from custom_dataset import DataPrefetcher, collater
from custom_dataset import MultiScaleCollater
from custom_dataset import Normalize, Resizer, RandomFlip, RandomCrop, RandomTranslate
from custom_dataset import YoloStyleResize, DetectionCollater


def test_make_grid():
    start_time = time.time()
    data_transform = {
        'train': transforms.Compose([
            RandomFlip(flip_prob=0.3),
            RandomCrop(crop_prob=0.2),
            RandomTranslate(translate_prob=0.2),
            # Resizer(resize=400),
            # YoloStyleResize(resize=416, multi_scale=True)
            # Normalize()
        ]),
        'val': transforms.Compose([
            Resizer(resize=400),
            # Normalize()
        ])
    }

    # -----------------------------------
    # voc_root_dir = '/data/weihule/data/dl/VOCdataset'
    # if not os.path.exists(voc_root_dir):
    #     voc_root_dir = '/ssd/weihule/data/dl/VOCdataset'
    # train_dataset = VocDetection(root_dir=voc_root_dir,
    #                              transform=data_transform['train'])
    # val_dataset = VocDetection(root_dir=voc_root_dir,
    #                            image_sets=[('2007', 'test')],
    #                            transform=data_transform['val'])
    # -----------------------------------

    # -----------------------------------
    data_set_root1 = '/nfs/home57/weihule/data/dl/COCO2017'
    data_set_root2 = '/workshop/weihule/data/dl/COCO2017'
    data_set_root3 = 'D:\\workspace\\data\\dl\\COCO2017'
    data_set_roots = [data_set_root1, data_set_root2, data_set_root3]
    data_set_root = ''
    for p in data_set_roots:
        if os.path.exists(p):
            data_set_root = p
            break

    # train_dataset_path = os.path.join(data_set_root, 'images', 'train2017')
    val_dataset_path = os.path.join(data_set_root, 'images', 'val2017')
    dataset_annot_path = os.path.join(data_set_root, 'annotations')
    val_dataset = CocoDetection(image_root_dir=val_dataset_path,
                                annotation_root_dir=dataset_annot_path,
                                set_name='val2017',
                                use_mosaic=True,
                                transform=data_transform['train'])
    # -----------------------------------

    # detec_collater = DetectionCollater()
    detec_collater = MultiScaleCollater(resize=416,
                                        stride=32,
                                        use_multi_scale=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=16,
                            shuffle=True,
                            num_workers=4,
                            prefetch_factor=4,
                            pin_memory=True,
                            collate_fn=detec_collater)
    mean = np.array([[[0.471, 0.448, 0.408]]])
    std = np.array([[[0.234, 0.239, 0.242]]])
    # datas是一个dict, key就是定义的三个key, 对应的value已经打了batch
    # datas = next(iter(val_loader))
    # batch_annots = datas['annot']
    # batch_images = datas['img']

    # pre_fetcher = DataPrefetcher(val_loader)
    # images, annotations = pre_fetcher.next()
    # index = 0
    # while images is not None:
    #     index += 1
    #     print(index)
    #     images, annotations = pre_fetcher.next()

    # for datas in tqdm(val_loader):
    #     batch_annots, batch_images, batch_scales = datas['annot'], datas['img'], datas['scale']
    #     print(batch_images.shape, batch_annots.shape)
    #
    # run_time = time.time() - start_time
    # print('run_time = ', run_time)

    # -----------------------------------
    # file_path = '/nfs/home57/weihule/code/study/torch_detection/utils/pascal_voc_classes.json'
    # if not os.path.exists(file_path):
    #     file_path = '/workshop/weihule/code/study/torch_detection/utils/pascal_voc_classes.json'
    # with open(file_path, 'r', encoding='utf-8') as fr:
    #     infos = json.load(fr)
    #     category_name_to_voc_lable = infos['classes']
    #     voc_colors = infos['colors']
    # voc_lable_to_category_id = {v: k for k, v in category_name_to_voc_lable.items()}
    # -----------------------------------

    file_path = 'coco_classes.json'
    if not os.path.exists(file_path):
        file_path = '/workshop/weihule/code/study/torch_detection/utils/coco_classes.json'
    with open(file_path, 'r', encoding='utf-8') as fr:
        infos = json.load(fr)
        category_name_to_coco_lable = infos['COCO_CLASSES']
        voc_colors = infos['colors']
    coco_lable_to_category_id = {v: k for k, v in category_name_to_coco_lable.items()}

    # datas = next(iter(val_loader))
    for index, datas in enumerate(val_loader):
        batch_images, batch_annots = datas['img'], datas['annot']
        save_root = 'show_images'
        c = 0
        for img, annot in zip(batch_images, batch_annots):
            c += 1
            img, annot = img.numpy(), annot.numpy()
            img = img.transpose(1, 2, 0)  # [c, h, w] -> [h, w, c] RGB

            img = (img * std + mean) * 255.
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
                category_name = coco_lable_to_category_id[label]
                category_color = tuple(voc_colors[label])
                chars_w, chars_h = font.getsize(category_name)
                draw.rectangle(point[:4], outline=category_color, width=2)  # 绘制预测框
                draw.rectangle((point[0], point[1] - chars_h, point[0] + chars_w, point[1]),
                               fill=category_color)  # 文本填充框
                draw.text((point[0], point[1] - font_size), category_name, fill=(255, 255, 255), font=font)

            save_path = os.path.join(save_root, str(index) + '_' + str(c) + '.png')
            # cv2.imwrite(save_path, img)
            image.save(save_path)
        if index == 4:
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
    print()

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


if __name__ == "__main__":
    test_make_grid()
    # test_coco_api()
