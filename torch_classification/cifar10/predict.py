import os
import sys
sys.path.append(os.getcwd())
import torch
import cv2
from torchvision import transforms, datasets
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from model import Net
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.functional import softmax

def imshow(image):
    # 输入数据: torch.tensor[c, h, w]
    # img = img * 0.5 + 0.5
    image = np.transpose(image.cpu().numpy(), (1, 2, 0))
    cv2.imwrite("test.png", image)

def main():
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    MINI_BATCH = 8
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # weight_path = "/workshop/cyclone_data/process/data/weights/cifar10/6499.pth"
    # root = "/workshop/cyclone_data/process/data/DL"
    weight_path = "/nfs/home57/weihule/data/weights/cifar10/6499.pth"
    root = "/nfs/home57/weihule/data/DL"
    net = Net()
    net = net.to(device)
    net.load_state_dict(torch.load(weight_path))

    testset = datasets.CIFAR10(root=root, train=False, transform=transform, download=True)
    test_loader = DataLoader(testset, batch_size=MINI_BATCH, shuffle=False, num_workers=4)

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            # image = make_grid(images)
            # imshow(image)
            outputs = net(images)
            probs = softmax(outputs, dim=1)
            _, pred_y = torch.max(probs, dim=1)
            print(labels)
            print(pred_y)
            # print(pred_y.cpu().numpy(), pred_y.shape)
            # print(labels, labels.shape)

            break

if __name__ == "__main__":
    main()

