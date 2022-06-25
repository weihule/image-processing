import os
import numpy as np
import torch
import json
import cv2
from PIL import Image

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
    with open('utils/coco_classes.json', 'w', encoding='utf-8') as fw:
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

    with open('utils/coco_classes.json', 'r', encoding='utf-8') as fr:
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

if __name__ == "__main__":
    # gen_json()

    re_json()

    # start = time.clock()
    # arr1 = [1, 1, 4, 1, 4, 6, 1, 6]
    # arr2 = [2, 3, 7, 3, 7, 9, 2, 9]
    # area1 = Polygon(np.array(arr1).reshape(4, 2))
    # area2 = Polygon(np.array(arr2).reshape(4, 2))
    # iou = area1.intersection(area2).area / (area1.area + area2.area)



    # ar1 = [1, 1, 4, 6]
    # ar1 = [10, 10, 14, 14]
    # ar2 = [2, 3, 7, 9]
    # iou = get_iou(ar1, ar2)

    # print(f'{iou}, running time: {time.clock()-start}')


    # pred_bbox = torch.randn(7, 4)
    # print(pred_bbox, pred_bbox[:, 1])
    # a1 = (pred_bbox[:, 2] - pred_bbox[:, 0]) * (pred_bbox[:, 3] - pred_bbox[:, 1])
    # b1 = torch.tensor(10)
    # print(a1)
    # print(a1+b1)
    # temp = pred_bbox.new_zeros(10, 10)
    # print(len(temp), temp.numel())

    # features = np.array([[0, 0, 0, 0],
    #             [0, 0, 0, 1],
    #             [0, 1, 0, 1],
    #             [0, 1, 1, 0],
    #             [0, 0, 0, 0],
    #             [1, 0, 0, 0],
    #             [1, 0, 0, 1],
    #             [1, 1, 1, 1],
    #             [1, 0, 1, 2],
    #             [1, 0, 1, 2],
    #             [2, 0, 1, 2],
    #             [2, 0, 1, 1],
    #             [2, 1, 0, 1],
    #             [2, 1, 0, 2],
    #             [2, 0, 0, 0]])

    # labels = np.array(['no', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'yes', 'yes']).reshape((-1, 1))

    # # dataset = np.hstack((features, labels))
    # dataset = np.concatenate((features, labels), axis=1)
    # mask = features[:, 2] == 0		# 根据索引为2的列生成mask
    # subdataset = features[mask]		# 将mask为false的行全都去除
    # print(subdataset)
    # subdataset = np.delete(subdataset, 2, axis=1)	# 删除索引为2的列


    # a = torch.tensor([2, 4, 6, 8])
    # b = torch.tensor([4., 8., 16., 32.])

    # a = torch.rand(1, 7)
    # print(a.max(dim=1))

    # a = torch.rand(4, 3, 14, 14)
    # print(a.shape[-2:])
