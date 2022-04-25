import os
from pydoc import doc
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets
import cv2
from PIL import Image
from torch.utils.data import DataLoader
from model import Net
from tqdm import tqdm
from torch.nn.functional import softmax

def train():
    MINI_BATCH = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])
    root = "/workshop/cyclone_data/process/data/DL"
    trainset = datasets.CIFAR10(root=root, train=True, transform=transform, download=True)
    train_loader = DataLoader(trainset, batch_size=MINI_BATCH, shuffle=True, num_workers=4)
    train_num = len(trainset)
    train_steps = len(train_loader)
    print("train_num :{}  train_loader: {}".format(train_num, train_steps))
    testset = datasets.CIFAR10(root=root, train=False, transform=transform, download=True)
    test_loader = DataLoader(testset, batch_size=MINI_BATCH, shuffle=False, num_workers=4)
    test_num = len(testset)

    net = Net(num_classes=10)
    net = net.to(device)
    net.load_state_dict(torch.load("/workshop/cyclone_data/process/data/weights/cifar10/6499.pth"), strict=False)
    
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(params=net.parameters(), lr=0.001, momentum=0.9)
    optimizer = torch.optim.Adam(params=net.parameters(), lr=0.0001)

    best_acc = 0.0
    save_root = "/workshop/cyclone_data/process/data/weights/cifar10"
    EPOCHES = 15
    for epoch in range(EPOCHES):
        net.train()
        running_loss = 0
        train_bar = tqdm(train_loader)
        for data in train_bar:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_bar.desc = "train epoch[{} / {}] loss:{:.3f}".format(epoch+1, EPOCHES, loss.item())
            
        net.eval()
        test_bar = tqdm(test_loader)
        correct = 0
        acc = 0
        with torch.no_grad():
            for data in test_bar:
                images, labels = data
                images, labels = images.to(device), labels.to(device)

                outputs = net(images)      # (64, 10)
                prob = softmax(outputs, dim=1)
                probablity, index = torch.max(prob, dim=1)
                correct += torch.eq(labels, index).sum().item()
                test_bar.desc = "valid epoch[{} / {}]".format(epoch+1, EPOCHES)
        acc = correct / test_num

        print("epoch:{}  training loss:{}  val acc:{}".format(
            epoch, running_loss / train_steps, acc
        ))
        if acc > best_acc:
            best_acc = acc
            save_path = os.path.join(save_root, str(best_acc)[2:6]+".pth")
            torch.save(net.state_dict(), save_path)

if __name__ == "__main__":
    train()



