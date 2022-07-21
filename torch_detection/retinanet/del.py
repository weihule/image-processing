from ast import fix_missing_locations
import os
import numpy as np
import torch
import json
import cv2
from PIL import Image
import logging
from logging import handlers
import torch.nn as nn
import torch.nn.functional as F
import random

COCO_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]
 
colors = [
    (39, 129, 113),
    (164, 80, 133),
    (83, 122, 114),
    (99, 81, 172),
    (95, 56, 104),
    (37, 84, 86),
    (14, 89, 122),
    (80, 7, 65),
    (10, 102, 25),
    (90, 185, 109),
    (106, 110, 132),
    (169, 158, 85),
    (188, 185, 26),
    (103, 1, 17),
    (82, 144, 81),
    (92, 7, 184),
    (49, 81, 155),
    (179, 177, 69),
    (93, 187, 158),
    (13, 39, 73),
    (12, 50, 60),
    (16, 179, 33),
    (112, 69, 165),
    (15, 139, 63),
    (33, 191, 159),
    (182, 173, 32),
    (34, 113, 133),
    (90, 135, 34),
    (53, 34, 86),
    (141, 35, 190),
    (6, 171, 8),
    (118, 76, 112),
    (89, 60, 55),
    (15, 54, 88),
    (112, 75, 181),
    (42, 147, 38),
    (138, 52, 63),
    (128, 65, 149),
    (106, 103, 24),
    (168, 33, 45),
    (28, 136, 135),
    (86, 91, 108),
    (52, 11, 76),
    (142, 6, 189),
    (57, 81, 168),
    (55, 19, 148),
    (182, 101, 89),
    (44, 65, 179),
    (1, 33, 26),
    (122, 164, 26),
    (70, 63, 134),
    (137, 106, 82),
    (120, 118, 52),
    (129, 74, 42),
    (182, 147, 112),
    (22, 157, 50),
    (56, 50, 20),
    (2, 22, 177),
    (156, 100, 106),
    (21, 35, 42),
    (13, 8, 121),
    (142, 92, 28),
    (45, 118, 33),
    (105, 118, 30),
    (7, 185, 124),
    (46, 34, 146),
    (105, 184, 169),
    (22, 18, 5),
    (147, 71, 73),
    (181, 64, 91),
    (31, 39, 184),
    (164, 179, 33),
    (96, 50, 18),
    (95, 15, 106),
    (113, 68, 54),
    (136, 116, 112),
    (119, 139, 130),
    (31, 139, 34),
    (66, 6, 127),
    (62, 39, 2),
    (49, 99, 180),
    (49, 119, 155),
    (153, 50, 183),
    (125, 38, 3),
    (129, 87, 143),
    (49, 87, 40),
    (128, 62, 120),
    (73, 85, 148),
    (28, 144, 118),
    (29, 9, 24),
    (175, 45, 108),
    (81, 175, 64),
    (178, 19, 157),
    (74, 188, 190),
    (18, 114, 2),
    (62, 128, 96),
    (21, 3, 150),
    (0, 6, 95),
    (2, 20, 184),
    (122, 37, 185),
]


def gen_json():
    info_dict = dict()
    info_dict['COCO_CLASSES'] = dict()
    info_dict['colors'] = colors
    for idx, i in enumerate(COCO_CLASSES):
        info_dict['COCO_CLASSES'][i] = idx
    
    json_str = json.dumps(info_dict, indent=4, ensure_ascii=False)
    with open('../utils/coco_classes.json', 'w', encoding='utf-8') as fw:
        fw.write(json_str)


def re_json():
    old_root = 'D:\\Desktop\\COCO2017\\annotations_trainval2017\\annotations'
    new_root = 'D:\\workspace\\data\\DL\\COCO2017\\annotations'

    img_root = 'D:\\workspace\\data\\DL\\COCO2017\\images\\val2017'
    json_names = [
        # 'captions_train2017.json',
        # 'captions_val2017.json',
        # 'instances_train2017.json',
        'instances_val2017.json',
        # 'person_keypoints_train2017.json',
        # 'person_keypoints_val2017.json'
        ]

    with open('../utils/coco_classes.json', 'r', encoding='utf-8') as fr:
        all_infos = json.load(fr)
        coco_classes = all_infos["COCO_CLASSES"]
        id_transform = all_infos["id_transform"]
    reverse_coco_classes = {v:k for k, v in coco_classes.items()}

    for fn in json_names:
        fn_path = os.path.join(old_root, fn)
        with open(fn_path, 'r', encoding='utf-8') as fr:
            infos = json.load(fr)
        print(fn, len(infos['images']), len(infos['annotations']), len(infos['categories']))
        for p in infos['annotations']:
            bbox = p['bbox']
            x_min, y_min = int(bbox[0]), int(bbox[1])
            x_max, y_max = int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])

            image_id = p['image_id']
            img_name = str(image_id).rjust(12, '0') + '.jpg'
            img_path = os.path.join(img_root, img_name)
            img = cv2.imread(img_path)
            img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0,255,0), 1)

            category_id = p['category_id']
            class_name = reverse_coco_classes[id_transform[str(category_id)]]
            cv2.putText(img, class_name, np.int32((x_min, y_min)), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (255, 0, 0), 1)

            cv2.imshow(img_name, img)
            cv2.waitKey(0)
            break
        # for igs, ats in zip(infos['images'], infos['annotations']):
        #     print(igs, ats['image_id'], ats['category_id'])
            
        # new_json_str = json.dumps(infos, indent=4)
        # with open(os.path.join(new_root, fn), 'w', encoding='utf-8') as fw:
        #     fw.write(new_json_str)


def get_logger(name, log_dir):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    info_name = os.path.join(log_dir, '{}.info.log'.format(name))
    info_handler = handlers.TimedRotatingFileHandler(filename=info_name,
                                                     when='D',
                                                     encoding='utf-8')
    info_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    info_handler.setFormatter(formatter)

    logger.addHandler(info_handler)

    return logger

def custom_softmax(inputs, dim):
    exp_up = np.exp(inputs)
    nums = np.sum(exp_up, axis=dim)
    nums = np.expand_dims(nums, axis=dim)
    res = exp_up / nums
    
    return res


if __name__ == "__main__":
    arr = torch.rand(12, 5)
    bbox = arr[:, :4]
    labels = arr[:, 4]
    mask = torch.tensor([0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1])
    mask = mask > 0
    # print(bbox.shape, labels.shape)
    # print(labels)
    # print(labels[mask])

    # one_img_gt_cls = torch.tensor([21, 3, 45])
    # indices = torch.tensor([1, 0, 2, 2, 0, 0, 2, 0, 1, 0, 2, 2])
    # overlap = torch.rand(12)
    # print(overlap)
    # print(indices[overlap > 0.45])
    # anchor_gt_cls = one_img_gt_cls[indices[overlap > 0.45]] + 1
    # print(anchor_gt_cls)

    # one_img_gt_bbs = torch.rand(3, 4)
    # print(one_img_gt_bbs)
    # indices = torch.tensor([1, 0, 2, 2, 0, 0, 2, 0, 1, 0, 2, 2])
    # per_image_anchors_gt_bboxes = one_img_gt_bbs[indices]
    # print(per_image_anchors_gt_bboxes)
    # print(per_image_anchors_gt_bboxes.shape)

    # one_image_anchors_gt_cls = torch.ones(10) * (-1)
    # overlaps = torch.rand(10)
    # one_image_anchors_gt_cls[overlaps > 0.5] = 1
    # print(one_image_anchors_gt_cls)

    # a = torch.tensor([2., 5., 1., 0., 3.])
    # arr1 = a.unsqueeze(-1)
    # arr2 = torch.unsqueeze(a, dim=1)
    # print(a, a.shape)
    # print(arr1, arr1.shape)
    # print(arr2, arr2.shape)


    # sigmoid_layer = nn.Sigmoid()
    # loss_f = nn.BCELoss()
    # inputs = torch.randn((3, 4))
    # labels = torch.tensor([1, 1, 2])
    # one_hot_labels = F.one_hot(labels.long(), num_classes=inputs.shape[1])
    # one_hot_labels = one_hot_labels.float()
    # inputs = sigmoid_layer(inputs)
    # loss = loss_f(inputs, one_hot_labels)
    # print(inputs)
    # print(loss)

    # 5个预测框, 2个gt
    # confidence_score = torch.tensor([0.78, 0.23, 0.54, 0.58, 0.49])
    # ious = torch.tensor([[0.30, 0.45, 0.52, 0.76, 0.38], 
    #                     [0.56, 0.13, 0.02, 0.89, 0.21]], dtype=torch.float32)
    # b = torch.tensor(-1, dtype=torch.float32)

    # tp_lists = torch.zeros(ious.shape[1])
    # ious_mask = torch.where(ious > 0.5, confidence_score, torch.tensor(-1, dtype=torch.float32))
    # print(ious_mask)
    # iou_value, indices = torch.max(ious_mask, dim=1)
    # print(iou_value, indices)

    # tp_lists[indices] = 1
    # print(tp_lists.cpu().numpy())
    # print(iou_value, indices)
    
    # iou_values, indices = torch.max(ious, dim=1)
    # print(iou_values, indices)

    # scores = torch.tensor([0.87, 0.81, 0.78, 0.76, 0.76, 0.69, 0.67, 0.60, 0.60, 0.59, 0.54])
    # print(scores.dtype)
    # classes = torch.tensor([2, 1, 3, 4, 2, 5, 7, 6, 4, 1, 2])
    # unique_classes = torch.unique(classes)
    # for per_class in unique_classes:
    #     sub_classes = classes[classes == per_class]
    #     sub_scores = scores[classes == per_class]
    #     # scores = scores[mask]
    #     # print(classes, scores)


    arr = np.random.random(size=(3, 4))
    arr_sub1 = arr[:, :2]
    arr_sub2 = arr[:, 2:]

    min_value1 = np.min(arr_sub1, axis=0)
    min_value2 = np.min(arr_sub2, axis=0)
    min_value = np.concatenate((min_value1, min_value2), axis=-1)
    print(min_value, min_value.shape)

    a = random.uniform(0, 3)
    print(a)






    
    
    
    








