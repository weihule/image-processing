from torch import optim
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
import torch
import math


def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)


def init_optimizer(optim, params, lr, weight_decay, **kwargs):
    if optim == 'adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optim == 'sgd':
        return torch.optim.SGD(params, lr=lr, momentum=kwargs["momentum"], weight_decay=weight_decay)
    else:
        raise KeyError('Unsupported optim: {}'.format(optim))


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

def test1():
    # 定义一个简单的模型
    model = torch.nn.Linear(10, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    # 创建 CosineAnnealingWarmRestarts 调度器
    # T_0 = 10  # 第一个周期的长度
    # T_mult = 2  # 每个周期长度的乘数因子
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=0.001)

    T_max = 20  # 总 epoch 数
    scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=0.001)
    # 模拟训练过程
    for epoch in range(30):
        # 训练步骤
        optimizer.step()
        # 更新学习率
        scheduler.step()
        print(f'Epoch {epoch}, Learning Rate: {optimizer.param_groups[0]["lr"]}')


if __name__ == '__main__':
    test1()

