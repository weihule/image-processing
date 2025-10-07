from .imagenet100 import ImageNet100Dataset, ImageNet100
from .flower5 import Flower5
from .eye_multi_label import EyeDataset
from .kitchen import KitchenDataset
from .multi_label import MultiDataset


def init_dataset(name, root_dir, set_name, transform, **kwargs):
    """
    初始化数据集dataset
    Args:
        name: 数据集名称
        root_dir: 数据集根目录
        set_name: 设置训练集或验证集
        transform: 图像变换
    Returns:
    """
    class_file = kwargs.get('class_file', None)
    if name == "imagenet100":
        dataset = ImageNet100(root_dir=root_dir,
                              set_name=set_name,
                              class_file=class_file,
                              transform=transform)
    elif name == "flower5":
        dataset = Flower5(root_dir=root_dir,
                          set_name=set_name,
                          class_file=class_file,
                          transform=transform)
    elif name == "eye":
        dataset = EyeDataset(root=root_dir,
                             train_csv=kwargs["train_csv"],
                             transform=transform)
    elif name == "kitchen":
        dataset = KitchenDataset(root_dir=root_dir,
                                 set_name=set_name,
                                 transform=transform)
    elif name == "multilabel":
        dataset = MultiDataset(root=root_dir,
                               set_name=set_name,
                               transform=transform)
    else:
        dataset = KitchenDataset(root_dir=root_dir,
                                 set_name=set_name,
                                 transform=transform)

    return dataset
