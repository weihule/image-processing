import numpy as np
from PIL import Image
import json


def main():
    mask_path = r"D:\workspace\data\VOCdataset\VOC2012\SegmentationClass\2007_001288.png"
    # 读取mask标签
    target = Image.open(mask_path)
    print(target.size, target.mode)
    # 获取调色板
    # 返回一个默认的调色板，长度为 768。这个默认调色板包含了 RGB 颜色空间中 256
    # 种颜色的调色板信息（每种颜色有红、绿、蓝三个通道），长度为 768。
    palette = target.getpalette()
    print(len(palette))
    palette = np.reshape(palette, (-1, 3)).tolist()
    # 转换成字典子形式
    pd = dict((i, color) for i, color in enumerate(palette))
    json_str = json.dumps(pd, indent=4)
    with open("palette.json", "w") as f:
        f.write(json_str)


if __name__ == "__main__":
    main()



