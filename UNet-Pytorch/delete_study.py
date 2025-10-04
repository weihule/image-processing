import numpy as np

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
