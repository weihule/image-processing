from .imagenet100 import ImageNet100Dataset, ImageNet100
from .flower5 import Flower5


def init_dataset(name, root_dir, set_name, class_file, transform, **kwargs):
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
    else:
        dataset = None

    return dataset
