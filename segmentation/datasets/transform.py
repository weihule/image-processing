import numpy as np
from PIL import Image


def pad_if_smaller(img, size, fill=0):
    # 如果图像最小边长小于给定size，则用数值fill进行padding
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        # 创建一个新的空白图像，用0值填充
        new_img = Image.new('RGB', (ow+padw, oh+padh),
                            color=(fill, fill, fill))

        # 将原始图像粘贴到新图像上
        new_img.paste(img, (0, 0))
    return img


class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        if isinstance(image, Image.Image) and isinstance(mask, Image.Image):
            raise f"type expected Image.Image, but get {type(image)} and {type(mask)}"

        if np.random.uniform(0, 1) < self.prob:
            image = image[:, ::-1, :]
            _, w, _ = image.shape
            # 水平翻转图像
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {
            'image': image,
            'mask': mask,
        }


class RandomVerticalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        if isinstance(image, Image.Image) and isinstance(mask, Image.Image):
            raise f"type expected Image.Image, but get {type(image)} and {type(mask)}"

        if np.random.uniform(0, 1) < self.prob:
            image = image[:, ::-1, :]
            _, w, _ = image.shape
            # 水平翻转图像
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        return {
            'image': image,
            'mask': mask,
        }


if __name__ == "__main__":
    # 创建一个三维张量
    tensor = np.random.rand(2, 3, 4)

    # 使用 transpose 对张量进行转置，交换前两个维度和后两个维度
    transposed_tensor = np.transpose(tensor, (1, 2, 0))

    # 输出转置后的张量
    print(transposed_tensor.shape)


