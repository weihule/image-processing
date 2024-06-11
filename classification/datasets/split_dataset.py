from pathlib import Path
from random import sample, shuffle
import shutil


def main():
    root = r"D:\Desktop\dataset"
    train_root = Path(root) / "train"
    val_root = Path(root) / "val"
    test_root = Path(root) / "test"
    for image_dir in Path(root).iterdir():
        dir_name = image_dir.parts[-1]
        images = [i for i in image_dir.glob("*.jpg")]
        shuffle(images)
        train_num = int(len(images) * 0.8)
        val_num = (len(images) - train_num) // 2
        test_num = len(images) - train_num - val_num

        assert train_num + test_num + val_num == len(images)

        train_part = images[:train_num]
        val_part = images[train_num:train_num+val_num]
        test_part = images[train_num+val_num:]

        if not (train_root / dir_name).exists():
            (train_root / dir_name).mkdir(parents=True)
        if not (val_root / dir_name).exists():
            (val_root / dir_name).mkdir(parents=True)
        if not (test_root / dir_name).exists():
            (test_root / dir_name).mkdir(parents=True)

        for i in train_part:
            shutil.copy(src=i, dst=train_root/dir_name/i.name)

        for i in val_part:
            shutil.copy(src=i, dst=val_root/dir_name/i.name)

        for i in test_part:
            shutil.copy(src=i, dst=test_root/dir_name/i.name)


if __name__ == "__main__":
    main()


