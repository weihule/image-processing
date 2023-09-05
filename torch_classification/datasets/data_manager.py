from .imagenet100 import ImageNet100Dataset, ImageNet100
from .flower5 import Flower5


def init_dataset(name, root_dir, class_file, transform_dict, **kwargs):
    if name == "imagenet100":
        train_dataset = ImageNet100(root_dir=root_dir,
                                    set_name="imagenet100_train",
                                    class_file=class_file,
                                    transform=transform_dict["train"])
        val_dataset = ImageNet100(root_dir=root_dir,
                                  set_name="imagenet100_val",
                                  class_file=class_file,
                                  transform=transform_dict["val"])
    elif name == "flower5":
        train_dataset = Flower5(root_dir=root_dir,
                                set_name="train",
                                class_file=class_file,
                                transform=transform_dict["train"])
        val_dataset = Flower5(root_dir=root_dir,
                              set_name="val",
                              class_file=class_file,
                              transform=transform_dict["val"])
    else:
        train_dataset = None
        val_dataset = None

    return train_dataset, val_dataset
