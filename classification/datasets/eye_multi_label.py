from pathlib import Path
import pandas as pd
import numpy as np
import cv2
from torch.utils.data import Dataset


impression = ['uveitis',
              'central serous chorioretinopathy',
              'other',
              'chorioretinal scar',
              'unremarkable changes',
              'choroidal mass',
              'polypoidal choroidal vasculopathy',
              'macular neovascularization',
              'central retinal vein occlusion',
              'myopia',
              'chorioretinal atrophy',
              'diabetic retinopathy',
              'branch retinal vein occlusion',
              'cystoid macular edema',
              'dry age-related macular degeneration',
              'central retinal artery occlusion',
              'retinal vein occlusion',
              'retinal dystrophy',
              'retinal arterial macroaneurysm',
              'retinal pigment epithelial detachment',
              'proliferative diabetic retinopathy',
              'epiretinal membrane',
              'pachychoroid pigment epitheliopathy']


class EyeDataset(Dataset):
    def __init__(self, root, train_csv):
        self.root = root
        self.train_csv = train_csv
        self.imglabels = self.get_imglabel()

    def __getitem__(self, item):
        image = cv2.imread(str(self.imglabels[item][0]))
        label = self.imglabels[item][1]

        sample = {
            "image": image,
            "label": label
        }
        
        return sample
    
    def __len__(self):
        return len(self.imglabels)

    def get_folder2label(self):
        """
        获取每个文件夹对应的多标签label
        {'1737_R': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
        """
        infos = pd.read_csv(self.train_csv)
        infos = infos[["Impression", "Folder"]]
        label_dict = {}
        for _, row in infos.iterrows():
            labels, folder = row["Impression"], row["Folder"]
            labels = [impression.index(i) for i in labels.split(",") if len(i) > 0]
            mask_label = [0] * len(impression)
            for i in labels:
                mask_label[i] = 1
            label_dict[folder] = mask_label
            # break
        return label_dict
    
    def get_imglabel(self):
        label_dict = self.get_folder2label()
        img_path = Path(self.root) / "Train" / "Train"
        imglabels = []
        for folder in img_path.iterdir():
            imgs = list(folder.glob("*.jpg"))
            folder_name = folder.parts[-1]
            for img_path in imgs:
                imglabels.append([img_path, label_dict[folder_name]])
        return imglabels


def main():
    root = r"D:\workspace\data\dl\eye_com"
    train_csv = r"D:\workspace\data\dl\eye_com\Train\Train.csv"
    ed = EyeDataset(root=root,
                    train_csv=train_csv)
    ed[10]


if __name__ == "__main__":
    main()
