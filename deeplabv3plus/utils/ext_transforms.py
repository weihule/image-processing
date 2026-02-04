import collections
import torchvision
import matplotlib.pyplot as plt
import math
import torch
import torchvision.transforms.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import random
import numbers
import numpy as np
from PIL import Image

class SegmentationTransform(object):
    def __init__(self, img_transform=None, lbl_transform=None):
        self.img_transform = img_transform
        self.lbl_transform = lbl_transform

    def __call__(self, img, lbl):
        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.lbl_transform is not None:
            lbl = self.lbl_transform(lbl)
        return img, lbl

class ExtCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, lbl):
        for t in self.transforms:
            img, lbl = t(img, lbl)
        return img, lbl


class ExtRandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, lbl):
        if random.random() < self.p:
            return F.hflip(img), F.hflip(lbl)
        return img, lbl

    def __repr__(self):
        return self.__class__.__name__ + f'(p={self.p})'


class ExtRandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, lbl):
        if random.random() < self.p:
            return (
                F.vflip(img), F.vflip(lbl)
            )
        return (img, lbl)

class ExtCenterCrop(object):
    def __init__(self, size: int | float | tuple[int, int]):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, lbl):
        return F.center_crop(img, self.size), F.center_crop(lbl, self.size)

    def __repr__(self):
        return self.__class__.__name__ + f'(size={self.size})'


class ExtRandomScale(object):
    def __init__(self, scale_range,
                       interpolation=InterpolationMode.BILINEAR):
        self.scale_range = scale_range
        self.interpolation = interpolation

    def __call__(self, img, lbl):
        assert img.size == lbl.size
        width, height = img.size
        scale_factor = random.uniform(self.scale_range[0], self.scale_range[1])
        target_height = int(height * scale_factor)
        target_width = int(width * scale_factor)
        target_size = [target_height, target_width]
        return (
            F.resize(img, target_size, interpolation=self.interpolation),
            F.resize(lbl, target_size, interpolation=InterpolationMode.NEAREST)
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__} scale_range={self.scale_range} interpolation={self.interpolation}"
        )


class ExtScale(object):
    def __init__(self, scale, interpolation=InterpolationMode.BILINEAR):
        self.scale = scale
        self.interpolation = interpolation

    def __call__(self, img, lbl):
        assert img.size == lbl.size
        target_size = [ int(img.size[1]*self.scale), int(img.size[0]*self.scale) ]
        return (
            F.resize(img, target_size, interpolation=self.interpolation),
            F.resize(lbl, target_size, interpolation=InterpolationMode.NEAREST)
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__} scale_range={self.scale} interpolation={self.interpolation}"
        )


class ExtRandomRotation(object):
    def __init__(self, degrees: int | float | tuple[int, int],
                 interpolation: InterpolationMode = InterpolationMode.BILINEAR,
                 expand=False,
                 center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.interpolation = interpolation
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params(degrees):
        angle = random.uniform(degrees[0], degrees[1])
        return angle

    def __call__(self, img, lbl):
        angle = self.get_params(self.degrees)
        return (
            F.rotate(img, angle, self.interpolation, self.expand, self.center),
            F.rotate(img, angle, InterpolationMode.NEAREST, self.expand, self.center)
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__} expand={self.expand} center={self.center}"
        )


class ExtPad(object):
    def __init__(self, diviser=32):
        self.diviser = diviser

    def __call__(self, img, lbl):
        assert img.size == lbl.size, f"尺寸不匹配: {img.size} vs {lbl.size}"
        width, height = img.size

        pad_width = (
                ((width // self.diviser) + 1) * self.diviser - width
                if width % self.diviser != 0 else 0
        )
        pad_height = (
                ((height // self.diviser) + 1) * self.diviser - height
                if height % self.diviser != 0 else 0
        )

        padding = [
            pad_width // 2,
            pad_height // 2,
            pad_width - pad_width//2,
            pad_height - pad_height // 2
        ]

        img_padded = F.pad(img, padding=padding, fill=0)
        lbl_padded = F.pad(lbl, padding=padding, fill=0)

        return img_padded, lbl_padded

    def __repr__(self):
        return f"{self.__class__.__name__} diviser={self.diviser}"


class ExtToTensor(object):
    def __init__(self, normalize=True, target_type='uint8'):
        self.normalize = normalize
        self.target_type = target_type

    def __call__(self, img, lbl):
        if self.normalize:
            return F.to_tensor(img), torch.from_numpy(np.array(lbl, dtype=self.target_type))
        else:
            return (
                torch.from_numpy(np.array(img, dtype=np.float32).transpose(2, 0, 1)),
                torch.from_numpy(np.array(lbl, dtype=self.target_type))
            )
    def __repr__(self):
        return f"{self.__class__.__name__}"

class ExtNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img: torch.Tensor, lbl: torch.Tensor):
        """
        img: [C, H, W]
        lbl:
        """
        return F.normalize(img, mean=self.mean, std=self.std)

    def __repr__(self):
        return f"{self.__class__.__name__} mean={self.mean} std={self.std}"

class ExtRandomCrop(object):
    def __init__(self, size: int | float | tuple[int, int], padding=0, pad_if_needed=False):
        """
        size: 期望的裁剪输出大小
        padding: 图像每个边界的可选填充。默认为0（无填充）若为长度4的序列，则分别填充左、上、右、下边界。
        pad_if_needed: 若为True，当图像小于期望大小时自动填充避免异常
        """
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed

    @staticmethod
    def get_params(img, output_size):
        """
        img: PIL.Image
        output_size: 期望的裁剪输出大小 (height, width)
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h-th)
        j = random.randint(0, w-tw)
        return i, j, th, tw

    def __call__(self, img: Image, lbl: Image):
        assert img.size == lbl.size, (
            f'图像和标签大小应一致 img:{img.size} lbl: {lbl.size}'
        )
        if self.padding > 0:
            img = F.pad(img, [self.padding])
            lbl = F.pad(lbl, [self.padding])

        # 需要时填充宽度
        if self.pad_if_needed and img.size[0] < self.size[1]:
            pad_width = math.ceil((self.size[1] - img.size[0]) / 2)
            img = F.pad(img, [pad_width, 0])    # (左右, 上下)
            lbl = F.pad(lbl, [pad_width, 0])

        # 需要时填充高度
        if self.pad_if_needed and img.size[1] < self.size[0]:
            pad_height = math.ceil((self.size[0] - img.size[1]) / 2)
            img = F.pad(img, [0, pad_height])    # (左右, 上下)
            lbl = F.pad(lbl, [0, pad_height])

        i, j, h, w = self.get_params(img, self.size)
        return F.crop(img, i, j, h, w), F.crop(lbl, i, j, h, w)

    def __repr__(self):
        return f"{self.__class__.__name__} size={self.size} padding={self.padding}"


class ExtResize(object):
    def __init__(self, size, interpolation=InterpolationMode.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img, lbl):
        assert img.size == lbl.size, (
            f'图像和标签大小应一致 img:{img.size} lbl: {lbl.size}'
        )
        img = F.resize(img, size=self.size, interpolation=self.interpolation)
        lbl = F.resize(lbl, size=self.size, interpolation=InterpolationMode.NEAREST)
        return img, lbl

    def __repr__(self):
        return f"{self.__class__.__name__} size={self.size} interpolation={self.interpolation}"


class ExtColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness_range = (0.8, 1.2)
        self.contrast_range = (0.8, 1.2)
        self.saturation_range = (0.8, 1.2)
        self.hue_range = (-0.1, 0.1)

    def __call__(self, img, lbl):
        """
        Args:
            img (PIL Image): Input image.
        Returns:
            PIL Image: Color jittered image.
        """
        assert img.size == lbl.size, (
            f'图像和标签大小应一致 img:{img.size} lbl: {lbl.size}'
        )
        brightness_factor = random.uniform(*self.brightness_range)
        contrast_factor = random.uniform(*self.contrast_range)
        saturation_factor = random.uniform(*self.saturation_range)
        hue_factor = random.uniform(*self.hue_range)

        # 3. 按顺序调整色彩（无需 Compose/Lambda，直接调用 F 函数）
        img = F.adjust_brightness(img, brightness_factor)
        img = F.adjust_contrast(img, contrast_factor)
        img = F.adjust_saturation(img, saturation_factor)
        img = F.adjust_hue(img, hue_factor)

        return img, lbl

    def __repr__(self):
        return (f"{self.__class__.__name__} brightness_range={self.brightness_range}"
                f"contrast_range={self.contrast_range}"
                f"saturation_range={self.saturation_range}"
                f"hue_range={self.hue_range}"
                )

class Lambda(object):
    def __init__(self, lambd):
        assert callable(lambd), repr(type(lambd).__name__) + " object is not callable"
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'



def visualize_transform(img_path, lbl_path, transform=None,
                        title="Original vs Transformed"):
    img = Image.open(img_path).convert("RGB")
    lbl = Image.open(lbl_path).convert("L")  # 强制转为灰度
    print(f"img.size={img.size} lbl.size={lbl.size}")

    # 如果有转换函数，则应用之
    if transform is not None:
        img_t, lbl_t = transform(img, lbl)
    else:
        img_t, lbl_t = img, lbl
    print(f"img_t.size={img_t.size} lbl_t.size={lbl_t.size}")

    # 转换为numpy数组以便于显示
    def to_numpy(im):
        return np.array(im)

    img_np = to_numpy(img)
    lbl_np = to_numpy(lbl)
    img_t_np = to_numpy(img_t)
    lbl_t_np = to_numpy(lbl_t)

    # 创建画布和子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle(f"{title} - Original", fontsize=16)

    ax1.imshow(img_np)
    ax1.set_title("Original Image")
    ax1.axis('off')

    ax2.imshow(lbl_np, cmap='tab20')
    ax2.set_title("Original Label")
    ax2.axis('off')

    plt.tight_layout()
    plt.show()

    # 展示变换后的图像和标签
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle(f"{title} - Transformed", fontsize=16)

    ax1.imshow(img_t_np)
    ax1.set_title("Transformed Image")
    ax1.axis('off')

    ax2.imshow(lbl_t_np, cmap='tab20')
    ax2.set_title("Transformed Label")
    ax2.axis('off')

    plt.tight_layout()
    plt.show()

def test():
    img_path = r'D:\workspace\data\images\VOCdevkit\VOC2012\JPEGImages\2007_000323.jpg'
    lbl_path = r'D:\workspace\data\images\VOCdevkit\VOC2012\SegmentationClass\2007_000323.png'
    # trans = ExtResize(size=(640, 640))
    # trans = ExtColorJitter()
    trans = ExtRandomCrop(size=(360, 360))
    print(trans)
    visualize_transform(img_path, lbl_path, trans)


if __name__ == "__main__":
    test()











