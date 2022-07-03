import os
import sys
import cv2

from utils.custom_dataset import CocoDetection, Resizer, RandomFlip
from torchvision import transforms, datasets

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)


class Config:
    save_root = '/nfs/home57/weihule/data/detection_data/retinanet'
    data_set_root = '/nfs/home57/weihule/data/dl/COCO2017'

    if not os.path.exists(save_root):
        save_root = '/workshop/weihule/data/detection_data/retinanet'

    if not os.path.exists(data_set_root):
        data_set_root = '/workshop/weihule/data/dl/COCO2017'
    elif not os.path.exists(data_set_root):
        data_set_root = 'D:\\workspace\\data\\DL\\COCO2017'

    log = os.path.join(save_root, 'log')  # path to save log
    checkpoint_path = os.path.join(save_root, 'checkpoints')  # path to store checkpoint model
    resume = os.path.join(checkpoint_path, 'latest.pth')
    evaluate = None  # evaluate model path
    train_dataset_path = os.path.join(data_set_root, 'images', 'train2017')
    val_dataset_path = os.path.join(data_set_root, 'images', 'val2017')
    dataset_annot_path = os.path.join(data_set_root, 'annotations')

    pre_trained = False
    num_classes = 80
    seed = 0
    input_image_size = 600

    data_transform = {
        'train': transforms.Compose([
            RandomFlip(flip_prob=0.5),
            Resizer(resize=600)
        ]),
        'val': transforms.Compose([
            Resizer(resize=600)
        ])
    }

    train_dataset = CocoDetection(image_root_dir=train_dataset_path,
                                  annotation_root_dir=dataset_annot_path,
                                  set='train2017',
                                  transform=data_transform['train'])

    val_dataset = CocoDetection(image_root_dir=val_dataset_path,
                                annotation_root_dir=dataset_annot_path,
                                set='val2017',
                                transform=data_transform['val'])

    epochs = 12
    batch_size = 32
    lr = 1e-4
    num_workers = 2
    print_interval = 100
    apex = True


if __name__ == "__main__":
    d1 = os.path.abspath(__file__)
    print(d1)

    # img_path = '/nfs/home57/weihule/data/dl/flower/test/tulips05.jpg'
    # img = cv2.imread(img_path)
    # print(img.shape)
