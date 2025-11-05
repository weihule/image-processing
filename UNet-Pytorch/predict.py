import os
from PIL import Image
from pathlib import Path
from loguru import logger
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet


def preprocess(pil_img, scale):
    w, h = pil_img.size
    newW, newH = int(scale * w), int(scale * h)
    assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
    pil_img = pil_img.resize((newW, newH), resample=Image.BICUBIC)
    img = np.asarray(pil_img)
    img = img.transpose((2, 0, 1))
    if (img > 1).any():
        img = img / 255.0
    
    return img

def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros(shape=(mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)
    
    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


def plot_img_and_mask2(img, mask, alpha=0.5, contour_color=(0, 0, 255)):
    """
    显示原图 + 半透明mask
    """
    if isinstance(img, Image.Image):
        img_array = np.array(img)
    else:
        img_array = img.copy()

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    h, w = mask.shape

    colors = [
        [0, 0, 0],           
        [255, 0, 0],         
        [0, 255, 0],        
        [0, 0, 255],        
        [255, 255, 0],      
        [255, 0, 255],       
        [0, 255, 255],       
    ]

    # 生成彩色mask
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(len(colors)):
        colored_mask[mask == i] = colors[i % len(colors)]

    # 原图与彩色mask融合
    blended = cv2.addWeighted(src1=img_array,
                              alpha=1 - alpha,
                              src2=colored_mask,
                              beta=alpha,
                              gamma=0)
    
    # 提取轮廓
    mask_uint8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # 在融合图上绘制轮廓
    cv2.drawContours(blended, contours, -1, contour_color, 2)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = preprocess(full_img, scale_factor)
    img = np.expand_dims(img, axis=0)
    img = torch.from_numpy(img)
    img = img.to(device=device, dtype=torch.float32)

    # size: [W, H]
    with torch.no_grad():
        output = net(img).cpu()
        print(output.shape)
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            # [batch_size, n_classes, H, W]  沿着 类别维度 找最大值的索引
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold
    
    return mask[0].long().squeeze().numpy()


def mask2image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)


def plot_img_and_mask(img, mask):
    print(f"mask = {mask}")
    classes = mask.max() + 1
    fig, ax = plt.subplots(1, classes+1, figsize=(5 * (classes + 1), 5))
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    ax[0].axis('off')  # 隐藏坐标轴框
    for i in range(classes):
        ax[i + 1].set_title(f'Mask (class {i + 1})')
        ax[i + 1].imshow(mask == i)
        ax[i + 1].axis('off')
    plt.xticks([]), plt.yticks([])
    plt.show()


def main():
    model_path = r"D:\workspace\weight_data\pre_weight\pytorch_unet\unet_carvana_scale1.0_epoch2.pth"
    file_path = r"D:\workspace\data\images\carvana-image-masking-challenge\train\1aba91a601c6_12.jpg"
    output_dir = r"D:\Desktop\delete"
    classes = 2
    cfgs = {
        "model_path": model_path,
        "classes": classes,
        "bilinear": False,
        "scale": 0.5,
        "mask_threshold": 0.5,
        "visualize": True,
        "output_dir": output_dir,
        "viz": True,
        "save_img": True
    }

    net = UNet(n_channels=3, n_classes=cfgs['classes'], bilinear=cfgs['bilinear'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(net)

    net.to(device=device)
    state_dict = torch.load(cfgs['model_path'], map_location=device)
    # print(state_dict.keys())
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    img = Image.open(file_path)

    mask = predict_img(net=net,
                        full_img=img,
                        scale_factor=cfgs['scale'],
                        out_threshold=cfgs['mask_threshold'],
                        device=device)
    
    if cfgs["save_img"]:
        file_stem = Path(file_path).stem
        file_suffix = Path(file_path).suffix
        save_path = Path(cfgs["output_dir"]) / (file_stem+"_out"+file_suffix)
        result = mask_to_image(mask, mask_values)
        result.save(str(save_path))
        print(f"mask 保存到 {save_path}")

    if cfgs["viz"]:
        plot_img_and_mask2(img, mask)


def test():
    img_path = r'D:\workspace\data\images\carvana-image-masking-challenge\train\0cdf5b5d0ce1_01.jpg'
    mask_path = r'D:\workspace\data\images\carvana-image-masking-challenge\train_masks\0cdf5b5d0ce1_01_mask.gif'
    plot_img_and_mask(Image.open(img_path), np.array(Image.open(mask_path)))


if __name__ == "__main__":
    # test()
    main()







    