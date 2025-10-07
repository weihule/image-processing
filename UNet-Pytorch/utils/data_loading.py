import logging
from loguru import logger
import numpy as np
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import Dataset

def load_image(filename):
    if isinstance(filename, Path):
        filename = Path(filename)
    suffix = Path(filename).suffix
    if suffix == '.npy':
        return Image.fromarray(np.load(filename))
    elif suffix in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)
    
def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + ".*"))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        # axis=0,表示行作为一个基础单位,按行找出不同值
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f"Loaded masks should have 2 or 3 dimensions, found {mask.ndim}")
    

class BasicDataset(Dataset):
    def __init__(self, images_dir, mask_dir, scale=1.0, mask_suffix=''):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [file.stem for file in self.images_dir.glob('*') if file.is_file() and not file.name.startswith('.')]
        if not self.ids:
            raise RuntimeError(f"No input file found in {images_dir}, make sure you put your images there")
        
        logger.info(f"Creating dataset with {len(self.ids)} examples")
        logger.info(f"Scanning mask files to determine unique values")
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, 
                               mask_dir=self.mask_dir, 
                               mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))

        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logger.info(f'Unique mask values: {self.mask_values}')
    
    def __len__(self):
        return len(self.ids)
    
    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    # img == v 的shape是[H, W, C],在最后一个维度做逻辑与操作 -> [H, W]
                    mask[(img == v).all(axis = -1)] = i
            return mask
        else:
            if img.ndim == 2:
                # 扩展到三维
                img = np.expand_dims(img, axis=0)
            else:
                # [H, W, C] -> [C, H, W]
                img = img.transpose((2, 0, 1))
            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name+self.mask_suffix+".*"))
        img_file = list(self.images_dir.glob(name+".*"))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'

        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'
        
        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        img = torch.from_numpy(img.copy()).float().contiguous()
        mask = torch.from_numpy(mask.copy()).long().contiguous()

        return {
            'image': img,
            'mask': mask
        }
    

class CarvanDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask')


def test():
    images_dir = r"D:\workspace\data\carvana-image-masking-challenge\train"
    mask_dir = r"D:\workspace\data\carvana-image-masking-challenge\train_masks"
    carvan = CarvanDataset(images_dir, mask_dir)
    datas = carvan[11]
    img, mask = datas['image'], datas['mask']
    print(img.shape, mask.shape)



if __name__ == "__main__":
    test()
    