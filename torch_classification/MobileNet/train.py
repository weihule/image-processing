import os
import sys
import json

import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from torchvision.models.mobilenetv3 import mobilenet_v3_small

from modelV2 import MobileNetV2

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device for training".format(device))

    data_transform = {
            "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
            "val": transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    
    image_path = "../dataset"
    BATCHSIZE = 4
    nw = min([os.cpu_count(), BATCHSIZE if BATCHSIZE > 1 else 0, 8])
    train_dataset = datasets.ImageFolder(os.path.join(image_path, "train"),
                                            transform=data_transform["train"])
    train_num = len(train_dataset)

    # {'daisy': 0, 'dandelion': 1, 'roses': 2, 'sunflowers': 3, 'tulips': 4}
    flower_list = train_dataset.class_to_idx

    invert_dict = {v:k for k,v in flower_list.items()}

    train_loader = DataLoader(dataset=train_dataset,
                        batch_size=BATCHSIZE,
                        shuffle=True,
                        num_workers=nw)

    val_dataset = datasets.ImageFolder(os.path.join(image_path, "val"), 
                                transform=data_transform["val"])
    val_num = len(val_dataset)
    val_loader = DataLoader(val_dataset, 
                            batch_size=BATCHSIZE,
                            shuffle=False,
                            num_workers=nw)
    print("using {} images for training, {} images for validation".format(train_num, val_num))

    # create model
    net = mobilenet_v3_small(num_classes=5)
    net.to(device)

    # pre_weights_path = "./weights/mobilenet_v2_pre.pth"
    pre_weights_path = "./weights/mobilenet_v3_small_pre.pth"
    # 字典形式
    pre_dict = torch.load(pre_weights_path, map_location=device)
    # for k, v in pre_dict.items():
    #     print(k)
    # 只载入特征提取层的权重
    pre_dict = {k:v for k,v in pre_dict.items() if "classifier" not in k}
    net.load_state_dict(pre_dict, strict=False)

    # freeze features weights
    for param in net.features.parameters():
        param.requires_grad = False

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.0001)

    best_acc = 0.0
    save_path = "./weights/mobileNetV3_small_01.pth"
    train_steps = len(train_loader)
    EPOCHS = 50
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
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                        EPOCHS, loss.item())

        # validate
        net.eval()
        acc = 0.
        with torch.no_grad():
            val_bar = tqdm(val_loader)
            for data in val_bar:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                predict = torch.max(outputs, dim=-1)[1]
                acc += torch.eq(predict, labels).sum().item()

                val_bar.desc = "valid epoch {}/{}".format(epoch+1,
                                                        EPOCHS)

        val_acc = acc / val_num
        print("epoch {}/{} loss:{:.3f} acc:{:.3f}".format(epoch+1, EPOCHS,
                                    running_loss / train_steps, val_acc))

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), save_path)

    print("Finished")

if __name__ == "__main__":
    main()

    