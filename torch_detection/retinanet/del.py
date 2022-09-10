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


def is_all_alpha(strings):
    """
    判断是否是纯英文数字
    """
    for ch in strings:
        if any(['0' <= ch <= '9', 'a' <= ch <= 'z', 'A' <= ch <= 'Z']) is False:
            return False
    return True

def calculate_ious(anchor, gt):
    """
    anchor: [x_min, y_min, x_max, y_max]
    gt: [x_min, y_min, x_max, y_max]
    """
    anchor_area = max(0, anchor[2]-anchor[0]) * max(0, anchor[3]-anchor[1])
    gt_area = max(0, gt[2]-anchor[0]) * max(0, gt[3]-gt[1])

    overlap_x_min, overlap_y_min = max(anchor[0], gt[0]), max(anchor[1], gt[1])
    overlap_x_max, overlap_y_max = min(anchor[2], gt[2]), min(anchor[3], gt[3])
    overlaps_w = max(0, overlap_x_max-overlap_x_min)
    overlaps_h = max(0, overlap_y_max-overlap_y_min)
    overlaps = overlaps_w * overlaps_h
    
    unions = max(anchor_area + gt_area - overlaps, 1e-4)
    iou = overlaps / unions
    
    return iou


def gen_anchors(anchor_w, anchor_h, s, t, width, height):
    anchors = []
    x = 0
    # 按列进行
    while x + anchor_w <= width:
        # 每次循环, y初始化为0
        y = 0
        while y + anchor_h <= height:
            x_min, y_min = x, y
            x_max, y_max = x+anchor_w, y+anchor_h
            anchors.append([x_min, y_min, x_max, y_max])

            y = y + t

        x = x + s
    
    return anchors

def count_num():
    """
    统计先验框与目标框相交的数目
    """
    anchor_w, anchor_h, s, t, gt_num, width, height = [1, 1, 1, 1, 1, 13, 10]
    anchors = gen_anchors(anchor_w, anchor_h, s, t, width, height)
    gt_x_min, gt_y_min, gt_w, gt_h = [9, 4, 1, 1]

    gts = [[gt_x_min, gt_y_min, gt_x_min+gt_w, gt_y_min+gt_h]]
    gt_num = 1

    count_num = 0
    # 遍历每一个先验框, 分别和三个gt做iou
    for anchor in anchors:
        for i in range(gt_num):
            iou = calculate_ious(anchor, gts[i])
            if iou > 0:
                count_num += 1
                break
    print('count_num = ', count_num)


def P_box(w, h, s, t, P, Q):
    P_boxs = {}
    x = 0
    y = 0
    while x + w <= P:
        y_t = 0
        while y_t + h <= Q:
            P_boxs[y] = [x, y_t, x + w, y_t + h]
            y += 1
            y_t += t
        x += s
    return P_boxs


def decode_arr(stack):
    if len(stack) >= 2:
        v1 = stack.pop()
        v2 = stack.pop()
        print(v1, v2)
        v = chr(int(v1+v2, 16))
        if v == '%':
            stack = decode_arr(stack)
        else:
            stack.append(v)
    return stack

def decode(stack):
    val1 = stack.pop()
    val2 = stack.pop()
    ch = chr(int(val1+val2,16))
    if ch == '%':
        stack = decode(stack)
    else:
        stack.append(ch)
    return stack


if __name__ == "__main__":
    labels = torch.tensor([0, 1, 2, 3, 4])
    print(labels.shape)

    labels = labels.expand((4, 5))
    print(labels, labels.shape)







    
    
    
    








