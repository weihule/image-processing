import os
import random
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm


def main():
    random.seed(1)
    seg_path = r"D:\workspace\data\images\Seg\people\JPEGImages"
    mask_path = r"D:\workspace\data\images\Seg\people\SegmentationClass"
    save_path = r"D:\workspace\data\images\Seg\people\ImageSets\Segmentation"

    images = list(Path(seg_path).glob("*.jpg"))
    image_names = []
    # 检查数据集的完整性和mask图
    for image in tqdm(images):
        file_name = image.stem
        image_names.append(file_name)
        mask_file = Path(mask_path) / (file_name+".png")
        if not mask_file.exists():
            assert f"{mask_file} not exists!"
        img = Image.open(mask_file)
        if img.mode != "P":
            assert f"{mask_file} 不是八位彩图"
    print(f"数据集检查完成")

    random.shuffle(image_names)
    train_percent = 0.9
    train_datas = image_names[:int(train_percent * len(image_names))]
    val_datas = image_names[int(train_percent * len(image_names)):]

    with open(Path(save_path) / "train.txt", "w") as fw:
        for item in train_datas:
            # 将元素写入文件，并添加换行符
            fw.write(item + '\n')

    with open(Path(save_path) / "val.txt", "w") as fw:
        for item in val_datas:
            # 将元素写入文件，并添加换行符
            fw.write(item + '\n')

    print(f"写入完成！")

if __name__ == "__main__":
    main()

    
