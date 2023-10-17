import os
import sys

import torch
import torchvision.transforms as transforms

import models
import decodes
import losses
from datasets.coco import CocoDetection
from datasets.voc import VOCDetection
from datasets.transform import *
from datasets.collater import DetectionCollater
from utils.util import Logger, AverageMeter, set_seed, worker_seed_init_fn, load_state_dict


# COCO2017_path = '/root/autodl-tmp/COCO2017'
# VOCdataset_path = '/root/autodl-tmp/VOCdataset'

COCO2017_path = '/root/autodl-tmp/COCO2017'
VOCdataset_path = r'D:\workspace\data\dl\VOCdataset'


class Cfg:
    network = 'resnet50_retinanet'
    num_classes = 80
    input_image_size = [640, 640]

    # load backbone pretrained model or not
    backbone_pretrained_path = '/root/code/SimpleAICV-ImageNet-CIFAR-COCO-VOC-training/pretrained_models/classification_training/resnet/resnet50-acc76.264.pth'
    model = models.__dict__[network](**{
        'num_classes': num_classes,
    })

    # load total pretrained model or not
    trained_model_path = None
    load_state_dict(trained_model_path, model)

    train_criterion = losses.__dict__['RetinaLoss2'](
        **{
            'areas': [[32, 32], [64, 64], [128, 128], [256, 256], [512, 512]],
            'ratios': [0.5, 1, 2],
            'scales': [2**0, 2**(1.0 / 3.0), 2**(2.0 / 3.0)],
            'strides': [8, 16, 32, 64, 128],
            'alpha': 0.25,
            'gamma': 2,
            'beta': 1.0 / 9.0,
            'focal_eiou_gamma': 0.5,
            'cls_loss_weight': 1.,
            'box_loss_weight': 1.,
            'box_loss_type': 'CIoU',
        })
    test_criterion = losses.__dict__['RetinaLoss2'](
        **{
            'areas': [[32, 32], [64, 64], [128, 128], [256, 256], [512, 512]],
            'ratios': [0.5, 1, 2],
            'scales': [2**0, 2**(1.0 / 3.0), 2**(2.0 / 3.0)],
            'strides': [8, 16, 32, 64, 128],
            'alpha': 0.25,
            'gamma': 2,
            'beta': 1.0 / 9.0,
            'focal_eiou_gamma': 0.5,
            'cls_loss_weight': 1.,
            'box_loss_weight': 1.,
            'box_loss_type': 'CIoU',
        })

    decoder = decodes.__dict__['RetinaDecoder'](
        **{
            'areas': [[32, 32], [64, 64], [128, 128], [256, 256], [512, 512]],
            'ratios': [0.5, 1, 2],
            'scales': [2**0, 2**(1.0 / 3.0), 2**(2.0 / 3.0)],
            'strides': [8, 16, 32, 64, 128],
            'max_object_num': 100,
            'min_score_threshold': 0.05,
            'topn': 1000,
            'nms_type': 'python_nms',
            'nms_threshold': 0.5,
        })

    dataset_name = "VOC"
    if dataset_name == "COCO":
        train_dataset = CocoDetection(COCO2017_path,
                                      set_name='train2017',
                                      transform=transforms.Compose([
                                          RandomHorizontalFlip(prob=0.5),
                                          RandomCrop(prob=0.5),
                                          RandomTranslate(prob=0.5),
                                          YoloStyleResize(
                                              resize=input_image_size[0],
                                              divisor=32,
                                              stride=32,
                                              multi_scale=True,
                                              multi_scale_range=[0.8, 1.0]),
                                          Normalize(),
                                      ]))

        test_dataset = CocoDetection(COCO2017_path,
                                     set_name='val2017',
                                     transform=transforms.Compose([
                                         YoloStyleResize(
                                             resize=input_image_size[0],
                                             divisor=32,
                                             stride=32,
                                             multi_scale=False,
                                             multi_scale_range=[0.8, 1.0]),
                                         Normalize(),
                                     ]))
    else:
        train_dataset = VOCDetection(root_dir=VOCdataset_path,
                                     image_sets=[('2007', 'trainval'),
                                                 ('2012', 'trainval')],
                                     transform=transforms.Compose([
                                         RandomHorizontalFlip(prob=0.5),
                                         RandomCrop(prob=0.5),
                                         RandomTranslate(prob=0.5),
                                         YoloStyleResize(
                                             resize=input_image_size[0],
                                             divisor=32,
                                             stride=32,
                                             multi_scale=True,
                                             multi_scale_range=[0.5, 1.0]),
                                         Normalize(),
                                     ]),
                                     keep_difficult=False)

        test_dataset = VOCDetection(root_dir=VOCdataset_path,
                                    image_sets=[('2007', 'test')],
                                    transform=transforms.Compose([
                                        YoloStyleResize(
                                            resize=input_image_size[0],
                                            divisor=32,
                                            stride=32,
                                            multi_scale=False,
                                            multi_scale_range=[0.5, 1.0]),
                                        Normalize(),
                                    ]),
                                    keep_difficult=False)
    train_collater = DetectionCollater()
    test_collater = DetectionCollater()

    seed = 0
    # batch_size is total size
    batch_size = 8
    # num_workers is total workers
    num_workers = 16
    gpus_num = 1
    accumulation_steps = 1

    optimizer = (
        'AdamW',
        {
            'lr': 1e-4,
            'global_weight_decay': False,
            # if global_weight_decay = False
            # all bias, bn and other 1d params weight set to 0 weight decay
            'weight_decay': 1e-3,
            'no_weight_decay_layer_name_list': [],
        },
    )

    scheduler = (
        'MultiStepLR',
        {
            'warm_up_epochs': 0,
            'gamma': 0.1,
            'milestones': [8, 12],
        },
    )

    epochs = 13
    print_interval = 100

    # 'COCO' or 'VOC'
    eval_type = 'COCO'
    eval_epoch = [1, 3, 5, 8, 10, 12, 13]
    eval_voc_iou_threshold_list = [
        0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95
    ]
    save_model_metric = 'IoU=0.50:0.95,area=all,maxDets=100,mAP'
    work_dir = r"D:\workspace\data\training_data\retinanet"
    save_dir = r"D:\workspace\data\training_data\retinanet"

    sync_bn = False
    apex = True

    use_ema_model = False
    ema_model_decay = 0.9999


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    cfg = Cfg()
    train_loader = DataLoader(cfg.train_dataset,
                              batch_size=8,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True,
                              num_workers=2,
                              collate_fn=cfg.train_collater)
    for ds in train_loader:
        print(type(ds))


