from .imagenet100 import ImageNet100Dataset


def init_dataset(name, root_dir, transform_dict, **kwargs):
    if name == "imagenet100":
        train_dataset = ImageNet100Dataset(root_dir=root_dir,
                                           set_name="imagenet100_train",
                                           transform=transform_dict["train"])
        val_dataset = ImageNet100Dataset(root_dir=root_dir,
                                         set_name="imagenet100_val",
                                         transform=transform_dict["val"])
    else:
        train_dataset = None
        val_dataset = None

    return train_dataset, val_dataset


