import torch.optim as optim
import torch
from torch.optim.lr_scheduler import LambdaLR
from torchvision.models.resnet import resnet50

model = resnet50(weights=None)

# 创建一个优化器
optimizer = optim.SGD(model.parameters(), lr=0.1)


# 自定义的 lr_lambda 函数，接受一个 epoch 参数
def custom_lr_lambda(epoch):
    # 在示例中，我们简单地将学习率减半
    if epoch < 50:
        return 0.1
    else:
        return 0.05


# 创建 LambdaLR 调度器，传递 lr_lambda 函数
scheduler = LambdaLR(optimizer, lr_lambda=custom_lr_lambda)

datasets = [torch.randn(4, 3, 16, 16) for _ in range(10)]

# 训练循环
for epoch in range(100):  # 假设进行 100 个训练周期
    for p in datasets:
        outputs = model(p)
        optimizer.zero_grad()
        loss = torch.tensor(0.8, requires_grad=True)
        loss.backward()
        optimizer.step()

    # 在每个 epoch 开始前使用 scheduler.step(epoch) 更新学习率
    scheduler.step(epoch)

    # 获取当前步骤的学习率
    current_lr = optimizer.param_groups[0]['lr']

    # 打印当前步骤的学习率
    print(f"Epoch {epoch + 1}, Learning Rate: {current_lr} -- {scheduler.get_last_lr()[0]}")

    # 训练模型的代码...
