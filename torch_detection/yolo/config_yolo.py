import os
import sys
from torchvision import transforms

BASE_DIR = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from torch_detection.utils.custom_dataset import VocDetection, CocoDetection
from torch_detection.utils.custom_dataset import RandomFlip, RandomCrop, RandomTranslate
from torch_detection.utils.custom_dataset import MultiScaleCollater


class Config:
    data_set_root = '/nfs/home57/weihule/data/dl/COCO2017'
    if not os.path.exists(data_set_root):
        data_set_root = '/workshop/weihule/data/dl/COCO2017'
    elif not os.path.exists(data_set_root):
        data_set_root = 'D:\\workspace\\data\\DL\\COCO2017'

    save_root = '/nfs/home57/weihule/data/detection_data/yolo'
    if not os.path.exists(save_root):
        save_root = '/workshop/weihule/data/detection_data/yolo'
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

    train_dataset = CocoDetection(image_root_dir=train_dataset_path,
                                  annotation_root_dir=dataset_annot_path,
                                  set_name='train2017',
                                  use_mosaic=True,
                                  transform=data_transform['train'])

    val_dataset = CocoDetection(image_root_dir=val_dataset_path,
                                annotation_root_dir=dataset_annot_path,
                                use_mosaic=False,
                                set_name='val2017')

    # voc_root_dir = '/data/weihule/data/dl/VOCdataset'
    # if not os.path.exists(voc_root_dir):
    #     voc_root_dir = '/ssd/weihule/data/dl/VOCdataset'
    # train_dataset = VocDetection(root_dir=voc_root_dir,
    #                              transform=data_transform['train'])
    # val_dataset = VocDetection(root_dir=voc_root_dir,
    #                            image_sets=[('2007', 'test')],
    #                            transform=data_transform['val'])

    epochs = 50
    batch_size = 64
    lr = 0.0001
    lrf = 0.0001
    num_workers = 4
    print_interval = 50
    apex = True

    collater = MultiScaleCollater(resize=input_image_size,
                                  use_multi_scale=True)

