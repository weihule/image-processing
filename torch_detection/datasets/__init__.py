from .custom_dataste import VOCDataset, COCODataset


def init_dataset(dataset_name,
                 root_dir=None,
                 image_root_dir=None,
                 annotation_root_dir=None,
                 resize=416,
                 use_mosaic=False,
                 transform=None,
                 **kwargs
                 ):
    if dataset_name == 'voc':
        dataset = VOCDataset(
            root_dir=root_dir,
            resize=resize,
            use_mosaic=use_mosaic,
            transform=transform,
            **kwargs
        )
    elif dataset_name == 'coco2017':
        dataset = COCODataset(
            image_root_dir=image_root_dir,
            annotation_root_dir=annotation_root_dir,
            resize=resize,
            use_mosaic=use_mosaic,
            transform=transform,
            **kwargs
        )
    else:
        raise KeyError(f'Unsupported {dataset_name} type')

    return dataset
