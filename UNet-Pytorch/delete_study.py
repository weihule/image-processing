import numpy as np
import torch
import torch.nn as nn

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

def 

if __name__ == "__main__":
    test5()

