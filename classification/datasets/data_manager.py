from .imagenet100 import ImageNet100Dataset, ImageNet100
from .flower5 import Flower5
from .eye_multi_label import EyeDataset
from .kitchen import KitchenDataset


def init_dataset(name, root_dir, set_name, class_file, transform, **kwargs):
    """
    初始化数据集dataset
    Args:
        name: 数据集名称
        root_dir: 数据集根目录
        set_name: 设置训练集或验证集
        class_file: 数据集的类别文件
        transform: 图像变换
    Returns:

    """
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
                                 transform=transform)
    else:
        dataset = None

    return dataset
