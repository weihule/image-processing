from pathlib import Path
import pandas as pd
import numpy as np
import cv2
from torch.utils.data import Dataset

__all__ = [
    "impression",
    "HyperF_Type",
    "HyperF_Area_DA",
    "HyperF_Fovea",
    "HyperF_ExtraFovea",
    "HyperF_Y",
    "HypoF_Type",
    "HypoF_Area_DA",
    "HypoF_Fovea",
    "HypoF_ExtraFovea",
    "HypoF_Y",
    "CNV",
    "Vascular_abnormality_DR",
    "Pattern",
    "EyeDataset"
]

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

HyperF_Type = ['leakage', 'no', 'staining', 'pooling', 'window defect']

HyperF_Area_DA = ['4', '5', 'no']

HyperF_Fovea = ['yes', 'no']

HyperF_ExtraFovea = ['no',
                     'inferior nasal',
                     'disc',
                     'diffuse',
                     'inferior temporal',
                     'temporal',
                     'superior nasal',
                     'superior temporal',
                     'inferior',
                     'superior',
                     'inferior to disc',
                     'venular',
                     'superotemporal',
                     'periphery',
                     'temporal to disc',
                     'superior to disc',
                     'nasal',
                     'nasal to disc']

HyperF_Y = ['no', 'intraretinal', 'preretinal', 'subretinal']

# 单分类
HypoF_Type = ['no', 'blockage', 'capillary non-perfusion']

# 单分类
HypoF_Area_DA = ['no', '4', '5']

# 单分类
HypoF_Fovea = ['yes', 'no']

# 多分类
HypoF_ExtraFovea = ['no',
                    'inferior nasal',
                    'disc',
                    'diffuse',
                    'inferior temporal',
                    'inferior to disc',
                    'temporal',
                    'superior nasal',
                    'superior temporal',
                    'inferior',
                    'superior',
                    'periphery',
                    'temporal to disc',
                    'superior to disc',
                    'nasal',
                    'nasal to disc']

# 多分类
HypoF_Y = ['no', 'intraretinal', 'preretinal', 'subretinal']

# 单分类
CNV = ['yes', 'no']

# 多分类
Vascular_abnormality_DR = ['retinal neovascularization elsewhere',
                           'collateral vessel',
                           'tortuous dilate',
                           'optociliary shunt',
                           'no',
                           'vasculitis',
                           'mild tortuous vessel',
                           'microaneurysm',
                           'telangiectasia',
                           'intraretinal microvascular abnormalities',
                           'macroaneurysm',
                           'vessel dilation',
                           'venous beading',
                           'tortuous',
                           'retinal neovascularization of the disc']

# 多分类
Pattern = ['polyp',
           'retinal pigment epithelial tear',
           'pcv',
           'smoke stack',
           'branching neovascular network',
           'laser scar',
           'no',
           'drusen',
           'starry sky',
           'ink blot',
           'segmental panretinal photocoagulation',
           'petaloid',
           'panretinal photocoagulation',
           'light bulb']


# impression_score:3.7990
# hyperf_type_score:4.0460
# hyperf_extrafovea_score:3.6760
# hypof_extrafovea_score:4.7400
# vascular_abnormality_dr_score:5.3070
# pattern_score:5.2070


class EyeDataset(Dataset):
    def __init__(self, root, train_csv, name, sick_name, transform=None):
        self.root = root
        self.train_csv = train_csv
        self.name = name
        self.sick_name = sick_name
        self.imglabels = self.get_imglabel()
        if transform is not None:
            self.transform = transform

    def __getitem__(self, item):
        image = cv2.imread(str(self.imglabels[item][0]))
        label = self.imglabels[item][1]

        sample = {
            "image": image,
            "label": label
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.imglabels)

    def get_folder2label(self):
        """
        获取每个文件夹对应的多标签label
        {'1737_R': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
        """
        infos = pd.read_csv(self.train_csv)
        infos = infos[[f"{self.sick_name}", "Folder"]]
        infos[f"{self.sick_name}"] = infos[f"{self.sick_name}"].fillna('no')
        label_dict = {}
        for _, row in infos.iterrows():
            labels, folder = row[f"{self.sick_name}"], row["Folder"]
            labels = [eval(f"{self.name}").index(i) for i in labels.split(",") if len(i) > 0]
            mask_label = [0] * len(eval(f"{self.name}"))
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
