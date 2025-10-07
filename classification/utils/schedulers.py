from __future__ import print_function, absolute_import
from functools import partial
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
import torch
import math

__all__ = [
    'init_scheduler',
    'adjust_learning_rate'
]


def adjust_learning_rate(optimizer, current_epoch, max_epoch, lr_min=0., lr_max=0.1, warm_up_epoch=5, warm_up=True):
    if warm_up:
        warm_up_epoch = warm_up_epoch
    else:
        warm_up_epoch = 0

    if current_epoch < warm_up_epoch:
        # 如果当前epoch为0, current_epoch就赋值为0.1
        current_epoch = 0.1 if current_epoch == 0 else current_epoch
        lr = lr_max * current_epoch / warm_up_epoch
        print('-- lr=', lr, current_epoch, warm_up_epoch)
    else:
        lr = lr_min + (lr_max - lr_min) * (
                    1 + math.cos(math.pi * (current_epoch - warm_up_epoch) / (max_epoch - warm_up_epoch))) / 2
        print('***** end_lr = ', lr_max, current_epoch, current_epoch - warm_up_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def init_scheduler(scheduler, optimizer, **kwargs):
    if scheduler == 'step_lr':
        # 每 step 个epoch之后, lr 衰减为 lr * gamma (gamma一般为0.1)
        return torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                               step_size=kwargs["step_size"],
                                               gamma=kwargs["gamma"])
    elif scheduler == 'cosine_annealing_lr':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                          T_max=kwargs["T_max"])
    elif scheduler == 'cosine_annealing_warm_restarts':
        return CosineAnnealingWarmRestarts(optimizer,
                                           T_0=kwargs["T_0"],
                                           T_mult=kwargs["T_mult"],
                                           eta_min=kwargs["eta_min"])
    else:
        raise KeyError('Unsupported scheduler: {}'.format(scheduler))


import torch.nn as nn


class DeBugModel(nn.Module):
    def __init__(self, num_classes=3):
        super(DeBugModel, self).__init__()
        self.conv = nn.Conv2d(3, 10, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(10)
        self.relu = nn.ReLU(inplace=True)
        self.global_avg = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(10, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.global_avg(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)

        return x


import matplotlib.pyplot as plt
import math


def de_bug_main():
    datasets = [torch.randn(4, 3, 16, 16) for _ in range(10)]
    epochs = 300
    model = DeBugModel(num_classes=3)
    model = model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0005, weight_decay=5e-04, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=350)

    lr_list = []
    model.train()
    for epoch in range(epochs):
        adjust_learning_rate(optimizer,
                             current_epoch=epoch,
                             max_epoch=epochs,
                             lr_min=1e-12,
                             lr_max=0.0005,
                             warm_up_epoch=10,
                             warm_up=True)
        for p in datasets:
        #     p = p.cuda()
        #     outputs = model(p)
        #     optimizer.zero_grad()
        #     loss = torch.tensor(0.8, requires_grad=True)
        #     loss.backward()
            optimizer.step()
            # print('[', epoch, ']', optimizer.state_dict()['param_groups'][0]['lr'], '***', scheduler.get_last_lr())
        scheduler.step()
        print('[', epoch, ']', optimizer.state_dict()['param_groups'][0]['lr'], '***', scheduler.get_last_lr())
        lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
    plt.plot(range(epochs), lr_list, color='b')
    plt.grid(True)
    plt.show()


class CosineAnnealingWarmUpLR(LambdaLR):
    def __init__(self,
                 optimizer,
                 lr_min=1e-12,
                 lr_max=0.0005,
                 warm_up=True,
                 warm_up_epoch=10,
                 epochs=20):
        super(CosineAnnealingWarmUpLR, self).__init__(optimizer, self.lr_lambda, -1)
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.warm_up = warm_up
        self.warm_up_epoch = warm_up_epoch
        self.epochs = epochs
        print('**', self.epochs)

    def lr_lambda(self, current_epoch):
        print(f"current_epoch = {current_epoch}")
        print("调用lr_lambda", current_epoch, self.epochs, self.lr_min, self.lr_max, self.warm_up, self.warm_up_epoch)

        if current_epoch < self.warm_up_epoch:
            # 如果当前epoch为0, current_epoch就赋值为0.1
            current_epoch = 0.1 if current_epoch == 0 else current_epoch
            lr = self.lr_max * current_epoch / self.warm_up_epoch
            print('-- lr=', lr, current_epoch, self.warm_up_epoch)
        else:
            lr = self.lr_min + (self.lr_max - self.lr_min) * (
                        1 + math.cos(math.pi * (current_epoch - self.warm_up_epoch) / (self.epochs - self.warm_up_epoch))) / 2
            print('***** end_lr = ', self.lr_max, current_epoch, current_epoch - self.warm_up_epoch)

        return lr


def de_bug_main2():
    datasets = [torch.randn(4, 3, 16, 16) for _ in range(10)]
    epochs = 20
    model = DeBugModel(num_classes=3)
    model = model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=5e-05, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    scheduler = CosineAnnealingWarmRestarts(optimizer, 10, 1, 1e-12)

    lr_list = []
    model.train()
    for epoch in range(epochs):
        for p in datasets:
            optimizer.step()
            # print('[', epoch, ']', optimizer.state_dict()['param_groups'][0]['lr'], '***', scheduler.get_last_lr())
        scheduler.step()
        print('[', epoch, ']', optimizer.state_dict()['param_groups'][0]['lr'], '***', scheduler.get_last_lr())
        lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
    plt.plot(range(epochs), lr_list, color='b')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    de_bug_main2()
