from __future__ import print_function, absolute_import
import torch
import torch.nn as nn
import math

__all__ = [
    'Scheduler'
]

"""
scheduler = (
    'MultiStepLR',
    {
        'warm_up_epochs': 0,
        'gamma': 0.1,
        'milestones': [8, 12],
    },
)

optimizer = (
    'AdamW',
    {
        'lr': 1e-4,
        'global_weight_decay': False,
        # if global_weight_decay = False
        # all bias, bn and other 1d params weight set to 0 weight decay
        'weight_decay': 1e-3,
        'no_weight_decay_layer_name_list': [],
    },
)
"""


class Scheduler:
    def __init__(self, cfgs):
        self.scheduler_name = cfgs.scheduler[0]
        self.scheduler_parameters = cfgs.scheduler[1]
        self.warm_up_epochs = self.scheduler_parameters['warm_up_epochs']

        self.epochs = cfgs.epochs

        self.optimizer_parameters = cfgs.optimizer[1]
        self.lr = self.optimizer_parameters['lr']
        self.current_lr = self.lr

        assert self.scheduler_name in ['MultiStepLR', 'CosineLR',
                                       'PolyLR'], 'Unsupported scheduler!'
        assert self.warm_up_epochs >= 0, 'Illegal warm_up_epochs!'
        assert self.epochs > 0, 'Illegal epochs!'

    def step(self, optimizer, epoch):
        if self.scheduler_name == 'MultiStepLR':
            gamma = self.scheduler_parameters['gamma']
            milestones = self.scheduler_parameters['milestones']
            if epoch < self.warm_up_epochs:
                self.current_lr = epoch / self.warm_up_epochs * self.lr
            else:
                self.current_lr = gamma**len([m for m in milestones if m <= epoch]) * self.lr

        elif self.scheduler_name == 'CosineLR':
            min_lr = 0. if 'min_lr' not in self.scheduler_parameters.keys() else self.scheduler_parameters['min_lr']
            if epoch < self.warm_up_epochs:
                self.current_lr = epoch / self.warm_up_epochs * self.lr
            else:
                self.current_lr = 0.5 * (math.cos((epoch - self.warm_up_epochs) / (
                        self.epochs - self.warm_up_epochs) * math.pi) + 1) * (
                        self.lr - min_lr) + min_lr

        for param_group in optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = self.current_lr * param_group["lr_scale"]
            else:
                param_group["lr"] = self.current_lr

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items()}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)


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
    epochs = 100
    model = DeBugModel(num_classes=3)
    model = model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, weight_decay=5e-04, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    lr_list = []

    for epoch in range(epochs):
        model.train()
        adjust_learning_rate(optimizer,
                             current_epoch=epoch,
                             max_epoch=epochs,
                             lr_min=1e-12,
                             lr_max=0.0001,
                             warmup_epoch=10,
                             warmup=True)
        for p in datasets:
            p = p.cuda()
            outputs = model(p)
            optimizer.zero_grad()
            loss = torch.tensor(0.8, requires_grad=True)
            loss.backward()
            optimizer.step()
        print(len(optimizer.param_groups))
        print('[', epoch, ']', optimizer.state_dict()['param_groups'][0]['lr'], '***', scheduler.get_lr())
        # print(optimizer.state_dict())
        # print(scheduler.state_dict())
        scheduler.step()
        lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
    plt.plot(range(epochs), lr_list, color='r')
    plt.show()


class MyClass:
    def __init__(self):
        self.my_attribute = 42


def test():
    dd = MyClass()
    print(dd.__dict__)


if __name__ == "__main__":
    # de_bug_main()
    test()

