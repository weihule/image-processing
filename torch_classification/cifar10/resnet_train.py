import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn.functional import softmax
import torchvision
from torchvision import transforms, datasets
from torchvision.models import resnet18
from torchvision.models.resnet import resnet18, resnet50
from tqdm import tqdm


def train():
    MINI_BATCH = 128
    num_classes = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])
    root = "/workshop/cyclone_data/process/data/DL"
    trainset = datasets.CIFAR10(root=root, train=True, transform=transform, download=False)
    trainloader = DataLoader(trainset, batch_size=MINI_BATCH, shuffle=True, num_workers=4)
    testset = datasets.CIFAR10(root=root, train=False, transform=transform, download=False)
    test_loader = DataLoader(testset, batch_size=MINI_BATCH, shuffle=False)
    train_num = len(trainset)
    train_steps = len(trainloader)
    test_num = len(testset)

    net = resnet18(pretrained=False)
    net.fc = nn.Linear(512, num_classes)
    net = net.to(device)

    # load feature weights
    pre_weight_path = "/nfs/home57/weihule/data/weights/resnet/resnet18_offical_pre.pth"
    pre_dict = torch.load(pre_weight_path, map_location=device)
    pre_dict = {k:v for k,v in pre_dict.items() if "fc" not in k}
    pre_dict = net.load_state_dict(pre_dict, strict=False)

    print(net)

    # freeze features weights
    # features = ["avgpool", "layer4", "layer3", "layer2", "layer1", "maxpool", "relu", "bn1", "conv1"]
    # for param in net.avgpool.parameters():
    #     param.requires_grad = False
    # for param in net.layer4.parameters():
    #     param.requires_grad = False
    # for param in net.layer3.parameters():
    #     param.requires_grad = False
    # for param in net.layer2.parameters():
    #     param.requires_grad = False
    # for param in net.layer1.parameters():
    #     param.requires_grad = False
    # for param in net.maxpool.parameters():
    #     param.requires_grad = False
    # for param in net.relu.parameters():
    #     param.requires_grad = False
    # for param in net.bn1.parameters():
    #     param.requires_grad = False
    # for param in net.conv1.parameters():
    #     param.requires_grad = False

    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.0001)

    save_root = "/workshop/cyclone_data/process/data/weights/cifar10"
    best_acc = 0.00
    EPOCHES = 10
    for epoch in range(EPOCHES):
        net.train()
        running_loss = 0.00
        train_bar = tqdm(trainloader)
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)
            # print(images.shape)
            outputs = net(images)   # (128, 10)
            print(outputs, outputs.shape)
            # print(labels, labels.shape)
        #
        #     optimizer.zero_grad()
        #     loss = loss_function(outputs, labels)
        #     running_loss += loss.item()
        #     loss.backward()
        #     optimizer.step()
        #     train_bar.desc = "epoch{}/{}, loss {:.3f}".format(epoch+1, EPOCHES, loss.item())
        #
        # net.eval()
        # correct = 0.
        # test_bar = tqdm(test_loader)
        # with torch.no_grad():
        #     for images, labels in test_bar:
        #         images, labels = images.to(device), labels.to(device)
        #         outputs = net(images)
        #         prob = softmax(outputs, dim=1)
        #         _, pred_y = torch.max(prob, dim=1)
        #         correct += torch.eq(pred_y, labels).sum().item()
        #         test_bar.desc = "validation epoch {}/{}".format(epoch+1, EPOCHES)
        #     acc = correct / test_num
        #     print("epoch {}, acc {}, running_loss {}".format(
        #         epoch+1, acc, running_loss/train_steps
        #     ))






if __name__ == "__main__":
    train()
    # arr = torch.tensor([3, 1, -3], dtype=torch.float32)
    # res = torch.softmax(arr, dim=0)
    #
    # res2 = torch.log_softmax(arr, dim=0)
    # print(res)
    # print(res2)