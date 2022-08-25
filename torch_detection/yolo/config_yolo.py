import os
import sys
import torch
from torchvision import transforms

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from torch_detection.utils.custom_dataset import VocDetection, CocoDetection
from torch_detection.utils.custom_dataset import RandomFlip, RandomCrop, RandomTranslate
from torch_detection.utils.custom_dataset import MultiScaleCollater


class Config:
    data_set_root1 = '/workshop/weihule/data/dl/COCO2017'
    data_set_root2 = '/root/autodl-tmp/COCO2017'

    save_root1 = '/workshop/weihule/data/detection_data/yolo'
    save_root2 = '/root/autodl-tmp/detection_data/yolo'

    pre_weight1 = '/workshop/weihule/data/detection_data/yolo/checkpoints/yolov4cspdarknet53-acc77.448.pth'
    pre_weight2 = '/root/autodl-nas/yolov4cspdarknet53-acc77.448.pth'

    data_set_root = data_set_root2
    save_root = save_root2
    pre_weight = pre_weight2

    if not os.path.exists(save_root):
        os.mkdir(save_root)

    log = os.path.join(save_root, 'log')  # path to save log
    checkpoint_path = os.path.join(save_root, 'checkpoints')  # path to store checkpoint model
    resume = os.path.join(checkpoint_path, 'latest.pth')
    evaluate = None  # evaluate model path
    train_dataset_path = os.path.join(data_set_root, 'images', 'train2017')
    val_dataset_path = os.path.join(data_set_root, 'images', 'val2017')
    dataset_annot_path = os.path.join(data_set_root, 'annotations')

    pre_trained = True
    num_classes = 50
    seed = 0
    input_image_size = 416

    data_transform = {
        'train': transforms.Compose([
            RandomFlip(flip_prob=0.5),
            # RandomCrop(crop_prob=0.2),
            # RandomTranslate(translate_prob=0.2),
        ]),
        'val': None
    }

    # train_dataset = CocoDetection(image_root_dir=train_dataset_path,
    #                               annotation_root_dir=dataset_annot_path,
    #                               set_name='train2017',
    #                               use_mosaic=True,
    #                               transform=data_transform['train'])
    #
    # val_dataset = CocoDetection(image_root_dir=val_dataset_path,
    #                             annotation_root_dir=dataset_annot_path,
    #                             use_mosaic=False,
    #                             set_name='val2017')

    voc_root_dir1 = '/ssd/weihule/data/dl/VOCdataset'
    voc_root_dir2 = '/root/autodl-tmp/VOCdataset'

    voc_root_dir = voc_root_dir2
    train_dataset = VocDetection(root_dir=voc_root_dir,
                                 transform=data_transform['train'],
                                 resize=400,
                                 use_mosaic=True)
    val_dataset = VocDetection(root_dir=voc_root_dir,
                               image_sets=[('2007', 'test')],
                               transform=data_transform['val'])

    epochs = 100
    batch_size = 16
    lr = 0.0001
    lrf = 0.0001
    num_workers = 4
    print_interval = 50
    apex = True

    # COCO
    # mean = [0.471, 0.448, 0.408]
    # std = [0.234, 0.239, 0.242]

    # VOC
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    collater = MultiScaleCollater(mean=mean,
                                  std=std,
                                  resize=input_image_size,
                                  stride=32,
                                  use_multi_scale=True)
