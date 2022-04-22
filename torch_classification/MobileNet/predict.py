import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import json

from modelV2 import MobileNetV2

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    # load images
    root = "../predPic"
    for fn in os.listdir(root):
        path = os.path.join(root, fn)
        img = Image.open(path)
        # plt.imshow(img)
        # plt.show()
        img = data_transform(img)
        img = torch.unsqueeze(img, dim=0)

        with open("./demo/class_indices.json", "r", encoding="utf-8") as j:
            class_dict = json.load(j)
        
        # create model
        model = MobileNetV2(num_classes=5)
        model.load_state_dict(torch.load("./weights/mobileNetV2_01.pth", map_location=device))

        model.eval()
        with torch.no_grad():
            output = model(img)
            predict = torch.softmax(output, dim=-1)
            prob, index = torch.max(predict, dim=-1)
            print(fn, prob.item(), class_dict[str(index.item())])
        # break
        

if __name__ == "__main__":
    main()