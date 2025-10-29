import numpy as np
import torch
import torch.nn as nn
import matplotlib.pylab as plt

from unet.unet_parts import DoubleDonv, Up

def test1():
    # 模拟RGB mask图像 (3D: H×W×C)
    img = np.array([
        [[0, 0, 0], [255, 0, 0], [0, 255, 0], [255, 0, 0]],      # 黑、红、绿
        [[255, 0, 0], [0, 0, 255], [0, 0, 0], [255, 0, 0]],      # 红、蓝、黑  
        [[0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 0, 0]],     # 绿、蓝、红
        [[0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 0, 0]]     # 绿、蓝、红
    ])
    print("RGB mask形状:", img.shape)  
    print(f"img.ndim = {img.ndim}")   

    mask_values = [
        [0, 0, 0],       # i=0: 背景
        [255, 0, 0],     # i=1: 类别1  
        [0, 255, 0],     # i=2: 类别2
        [0, 0, 255]      # i=3: 类别3
    ]

    mask = np.zeros((4, 4), dtype=np.int64)

    for i, v in enumerate(mask_values):
        print(f"\n处理颜色 {v}, 索引 {i}")
        
        # 关键：比较每个像素的RGB值
        condition = (img == v)  # 形状: (3, 3, 3)
        print(f"img == {v} 的形状: {condition.shape}")
        print(f"逐通道比较结果:")
        print(condition)
        
        # .all(-1) 在最后一个维度(通道维)上求AND
        final_condition = condition.all(-1)  # 形状: (3, 3)
        print(f"all(-1) 结果:")
        print(final_condition)
        
        mask[final_condition] = i
        print(f"更新后的mask:")
        print(mask)


def test2():
    # Dilation的影响
    x = torch.randn(1, 3, 64, 64)
    print("Dilation的影响（感受野）：")
    print("=" * 70)
    for dilation in [2, 4]:
        conv = nn.Conv2d(3, 64, kernel_size=3, padding=2, stride=1, dilation=dilation)
        output = conv(x)
        
        # 完整公式：(Input - Dilation*(Kernel-1) - 1 + 2*Padding) / Stride + 1
        H_out = (64 - dilation*(3-1) - 1 + 2*dilation) // 1 + 1
        
        effective_kernel = dilation * (3 - 1) + 1  # 有效感受野
        
        print(f"Dilation={dilation}, Padding={dilation} → "
            f"输出: {output.shape} {output.shape[2]}x{output.shape[3]} "
            f"(有效卷积核: {effective_kernel}x{effective_kernel})")
        

def test3():
    x = torch.randn(1, 64, 32, 32)
    up1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
    out1 = up1(x)
    print(f"out1.shape: {out1.shape}")

    # up2 = nn.Upsample(scale_factor=2)
    up2 = nn.Upsample(size=(64, 64))
    out2 = up2(x)
    print(f"out2.shape: {out2.shape}")


def test4():
    dc = DoubleDonv(3, 32, 16)
    x = np.random.random(size=(1, 3, 640, 640)).astype(np.float32)
    x = torch.from_numpy(x)
    x_out = dc(x)
    print(x.shape, x_out.shape)


def test5():
    x1 = np.random.random(size=(1, 128, 16, 16)).astype(np.float32)
    x2 = np.random.random(size=(1, 128, 32, 32)).astype(np.float32)
    
    x1, x2 = torch.from_numpy(x1), torch.from_numpy(x2)
    up = Up(in_channels=256, out_channels=128)
    out = up(x1, x2)
    print(out.shape)

import cv2
import numpy as np
from PIL import Image

def mask_to_image(mask: np.ndarray, mask_values=None):
    """
    将mask转换为RGB图像（带颜色）
    mask: [H, W] 的整数数组，每个像素值代表类别
    """
    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)
    
    h, w = mask.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 定义颜色列表（可自定义）
    colors = [
        [0, 0, 0],           # 类别0: 黑色
        [255, 0, 0],         # 类别1: 红色
        [0, 255, 0],         # 类别2: 绿色
        [0, 0, 255],         # 类别3: 蓝色
        [255, 255, 0],       # 类别4: 黄色
        [255, 0, 255],       # 类别5: 紫色
        [0, 255, 255],       # 类别6: 青色
    ]
    
    # 根据mask值给对应区域着色
    for i in range(len(colors)):
        colored_mask[mask == i] = colors[i % len(colors)]
    
    return Image.fromarray(colored_mask)


def plot_img_and_mask_with_overlay(img, mask, alpha=0.5, contour_color=(0, 0, 255)):
    """
    显示原图 + 半透明mask + 轮廓线的分割效果
    
    Args:
        img: PIL Image 或 numpy array 格式的原图
        mask: [H, W] 的整数数组，每个像素值代表类别
        alpha: mask的透明度 (0-1)
        contour_color: 轮廓线颜色 (B, G, R) 格式
    """
    # 转换图像格式
    if isinstance(img, Image.Image):
        img_array = np.array(img)
    else:
        img_array = img.copy()
    
    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)
    
    h, w = mask.shape
    
    # 定义颜色列表（BGR格式）
    colors = [
        [0, 0, 0],           # 类别0
        [255, 0, 0],         # 类别1: 蓝色
        [0, 255, 0],         # 类别2: 绿色
        [0, 0, 255],         # 类别3: 红色
        [255, 255, 0],       # 类别4: 青色
        [255, 0, 255],       # 类别5: 紫色
        [0, 255, 255],       # 类别6: 黄色
    ]
    
    # 生成彩色mask
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(len(colors)):
        colored_mask[mask == i] = colors[i % len(colors)]
    
    # 原图与彩色mask融合
    blended = cv2.addWeighted(img_array, 1 - alpha, colored_mask, alpha, 0)
    
    # 提取轮廓
    mask_uint8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # 在融合图上绘制轮廓
    cv2.drawContours(blended, contours, -1, contour_color, 2)
    
    # 显示结果
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return blended


if __name__ == "__main__":
    test5()

