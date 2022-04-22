import os
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import glob
import random
from mydataset import MyDataSet
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import time
from model import resnet50

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
 
    # load image
    img_root = "../predPic"
    assert os.path.exists(img_root), "file: {} not exists".format(img_root)
    img_path = [os.path.join(img_root, p) for p in os.listdir(img_root)]

    # read class_indict
    json_path = './demo/class_indices.json'
    assert os.path.exists(json_path), f"file: '{json_path}' dose not exist."

    json_file = open(json_path, "r", encoding="utf-8")
    class_indict = json.load(json_file)  

    # create model
    model = resnet50(num_classes=5).to(device)

    # load weights
    weights_path = "./weights/resNet50_01.pth"
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # prediction
    img_list = []
    model.eval()
    with torch.no_grad():
        # start = time.time()
        # for i in img_path:
        #     img = Image.open(i)
        #     img = data_transform(img)
        #     # img_list.append(img)
        #     img = torch.unsqueeze(img, dim=0)
        #     output = model(img.to(device))
        #     predict = torch.softmax(output, dim=1)
        #     print(predict)
        #     prob, classes = torch.max(predict, dim=1)
        #     print(prob, classes)
        # print("running time: ", time.time() - start)

        #     # print(i.split(os.sep)[-1], class_indict[str(classes.item())], prob.item())
            
        start = time.time()
        for i in img_path:
            img = Image.open(i)
            img = data_transform(img)
            img_list.append(img)
        batch_img = torch.stack(img_list, dim=0)
        output = model(batch_img.to(device)).cpu()
        print(output)
        print("running time: ", time.time() - start)
            
if __name__ == "__main__":
    main()