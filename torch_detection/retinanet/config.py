import os
import sys
from torchvision import transforms

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from torch_detection.utils.custom_dataset import Resizer, RandomFlip, RandomCrop, RandomTranslate
from torch_detection.utils.custom_dataset import VocDetection, CocoDetection
from torch_detection.utils.custom_dataset import MultiScaleCollater


class Config:
    data_set_root1 = '/workshop/weihule/data/dl/COCO2017'
    data_set_root2 = 'D:\\workspace\\data\\dl\\COCO2017'
    data_set_root = data_set_root1

    save_root1 = '/workshop/weihule/data/detection_data/retinanet'
    save_root2 = '/root/autodl-nas/detection_data/retinanet'
    save_root = save_root2

    log = os.path.join(save_root, 'log')  # path to save log
    checkpoint_path = os.path.join(save_root, 'checkpoints')  # path to store checkpoint model
    resume = os.path.join(checkpoint_path, 'latest.pth')
    evaluate = None  # evaluate model path
    train_dataset_path = os.path.join(data_set_root, 'images', 'train2017')
    val_dataset_path = os.path.join(data_set_root, 'images', 'val2017')
    dataset_annot_path = os.path.join(data_set_root, 'annotations')

    pre_trained = True
    num_classes = 20
    seed = 0
    input_image_size = 640

    data_transform = {
        'train': transforms.Compose([
            RandomFlip(flip_prob=0.5)
        ]),
        'val': None
    }

    # train_dataset = CocoDetection(image_root_dir=train_dataset_path,
    #                               annotation_root_dir=dataset_annot_path,
    #                               set='train2017',
    #                               transform=data_transform['train'])
    #
    # val_dataset = CocoDetection(image_root_dir=val_dataset_path,
    #                             annotation_root_dir=dataset_annot_path,
    #                             set='val2017',
    #                             transform=data_transform['val'])

    voc_root_dir1 = '/ssd/weihule/data/dl/VOCdataset'
    voc_root_dir2 = '/root/autodl-tmp/VOCdataset'

    voc_root_dir = voc_root_dir2
    # 这里的 resize 参数是mosaic时使用的参数
    train_dataset = VocDetection(root_dir=voc_root_dir,
                                 transform=data_transform['train'],
                                 resize=400,
                                 use_mosaic=True)

    val_dataset = VocDetection(root_dir=voc_root_dir,
                               image_sets=[('2007', 'test')],
                               transform=data_transform['val'])

    epochs = 250
    batch_size = 40
    lr = 0.0001
    lrf = 0.001
    num_workers = 4
    print_interval = 50
    apex = True

    # VOC
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    collater = MultiScaleCollater(mean=mean,
                                  std=std,
                                  resize=input_image_size,
                                  stride=32,
                                  use_multi_scale=False,
                                  normalize=True)

    pre_train_path1 = '/workshop/weihule/data/detection_data/retinanet/checkpoints/resnet50-acc76.322.pth'
    pre_train_path2 = '/root/autodl-nas/detection_data/retinanet/checkpoints/resnet50-acc76.322.pth'
    pre_train_path = pre_train_path2


if __name__ == "__main__":
    d1 = os.path.abspath(__file__)
    print(d1)

    print(len(Config.train_dataset))
    print(len(Config.val_dataset))
