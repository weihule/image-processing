from utils.datasets import FlowerDataset
from utils.util import get_paths
from torchvision import transforms
import torch
# from backbones import yolov4backbone
import backbones


class Config:
    mode = 'company'      # company   autodl

    if mode == 'local':
        dataset_path = 'D:\\workspace\\data\\dl\\flower'
        pre_weight_path = 'D:\\workspace\\data\\weights\\yolov4backbone\\yolov4cspdarknet53-acc77.448.pth'
    elif mode == 'company':
        dataset_path = '/workshop/weihule/data/dl/flower'
        pre_weight_path = '/workshop/weihule/data/weights/yolov4backbone/darknet530.835.pth'
        # pre_weight_path = None
        save_path = '/workshop/weihule/data/weights/yolov4backbone/darknet53'
    else:
        dataset_path = '/workshop/weihule/data/dl/flower'
        pre_weight_path = '/workshop/weihule/data/weights/mobilenet/mobilenet_v2_pre.pth'

    epochs = 70
    batch_size = 32
    lr = 0.001
    lrf = 0.001
    num_workers = 4
    freeze_layer = False
    apex = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transforms = {
        'train': transforms.Compose([transforms.RandomResizedCrop(256),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        'val': transforms.Compose([transforms.Resize((256, 256)),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }

    train_images_path, train_images_label = get_paths(dataset_path, "train", 'utils/flower_indices.json')
    val_images_path, val_images_label = get_paths(dataset_path, "val", 'utils/flower_indices.json')

    train_dataset = FlowerDataset(train_images_path, train_images_label, transforms['train'])
    val_dataset = FlowerDataset(val_images_path, val_images_label, transforms['val'])

    model = yolov4backbone.__dict__['yolov4_csp_darknet53'](
        **{'num_classes': 5}
    )


if __name__ == "__main__":
    dataset_path = 'D:\\workspace\\data\\dl\\flower'
    train_images_path, train_images_label = get_paths(dataset_path, "train", 'utils/flower_indices.json')
    for i, j in zip(train_images_path, train_images_label):
        print(i, j)


