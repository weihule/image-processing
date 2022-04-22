import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import json
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision.models.vgg import vgg16
from tqdm import tqdm
# from model import vgg

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])    
        plt.axis("off")
        plt.show()

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} for training".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    }

    img_path = "../dataset"
    train_dataset = datasets.ImageFolder(os.path.join(img_path, "train"),
                                        transform=data_transform["train"])
    
    json_str = json.dumps(train_dataset.class_to_idx, indent=4)
    if not os.path.exists("./class_indices.json"):
        with open("./class_indices.json", "w") as f:
            f.write(json_str)

    BATCHSIZE = 4
    train_loader = DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True)

    val_dataset = datasets.ImageFolder(os.path.join(img_path, "val"),
                                    transform=data_transform["val"])
    val_loader = DataLoader(val_dataset,  batch_size=BATCHSIZE, shuffle=False)

    train_num = len(train_dataset)
    val_num = len(val_dataset)
    print("using {} images for training, {} images for validation".format(train_num, val_num))

    # # show images
    # for images, labels in train_loader:
    #     grid = make_grid(images, nrow=4, padding=1)
    #     print(grid.size())
    #     show(grid)
    #     break

    # net = vgg(model_name="vgg16", num_classes=5, init_weights=True)
    net = vgg16()
    net.to(device)

    pre_weights_path = "./weights/vgg16_pre.pth"
    pre_dict = torch.load(pre_weights_path, map_location=device)
    pre_dict = {k:v for k, v in pre_dict.items() if "classifier" not in k}
    net.load_state_dict(pre_dict, strict=False)

    loss_function = nn.CrossEntropyLoss()
    # freeze features weights
    for param in net.features.parameters():
        param.requires_grad = False
    params = [p for p in net.parameters() if p.requires_grad==True]
    optimizer = torch.optim.Adam(params, lr=0.0001)

    best_acc = 0.
    save_path = "./weights/vgg16_01.pth"
    train_steps = len(train_loader)
    EPOCHS = 30
    for epoch in range(EPOCHS):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for data in train_bar:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = net(images)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss: {:.3f}".format(epoch+1, EPOCHS, loss.item())
            
        # validata
        net.eval()
        acc = 0.
        with torch.no_grad():
            val_bar = tqdm(val_loader)
            for data in train_bar:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                predict = torch.argmax(outputs)
                print(predict)
        break

if __name__ == "__main__":
    main()