from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn


class Visualizer(object):
    def __init__(self, logdir="./runs", env="main", exp_id=None):
        logdir = Path(logdir) / env
        logdir.mkdir(parents=True, exist_ok=True)

        if exp_id is not None:
            logdir = logdir / f"[{exp_id}]_{env}"
            logdir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(str(logdir))
        self.exp_id = exp_id
        self.env = env
        self.global_step = 0

        print(f"✅ TensorBoard 启动成功")
        print(f"📂 日志路径: {logdir}")
        print(f"🚀 查看日志: tensorboard --logdir=./runs")

    def set_step(self, step):
        """设置全局步数"""
        self.global_step = step

    def vis_scalar(self, name, value, step=None):
        """绘制标量图"""
        if step is None:
            step = self.global_step

        # if self.exp_id is not None:
        #     name = f"[{self.exp_id}]{name}"

        self.writer.add_scalar(name, value, global_step=int(step))
        self.writer.flush()

    def vis_scalars(self, tag_scalar_dict, step=None):
        """一次记录多个标量"""
        if step is None:
            step = self.global_step

        for name, value in tag_scalar_dict.items():
            # if self.exp_id is not None:
            #     name = f"[{self.exp_id}]{name}"
            self.writer.add_scalar(name, value, global_step=int(step))

    def vis_image(self, name, image, step=None):
        """
        绘制图像（支持多张）
        img: torch.Tensor (C, H, W) 或 (B, C, H, W)
        """
        if step is None:
            step = self.global_step

        # if self.exp_id is not None:
        #     name = f"[{self.exp_id}]{name}"

        self.writer.add_image(name, image, global_step=int(step))
        self.writer.flush()

    def vis_images(self, name, imgs, step=None):
        """绘制多张图像网格"""
        if step is None:
            step = self.global_step

        # if self.exp_id is not None:
        #     name = f"[{self.exp_id}]{name}"

        # imgs: (B, C, H, W)
        self.writer.add_images(name, imgs, global_step=int(step))
        self.writer.flush()

    def vis_histogram(self, name, values, step=None):
        """直方图 - 查看权重/梯度分布"""
        if step is None:
            step = self.global_step

        # if self.exp_id is not None:
        #     name = f"[{self.exp_id}]{name}"

        self.writer.add_histogram(name, values, global_step=int(step))

    def vis_text(self, name, text, step=None):
        if step is None:
            step = self.global_step

        # if self.exp_id is not None:
        #     name = f"[{self.exp_id}]{name}"

        self.writer.add_text(name, text, global_step=int(step))
        self.writer.flush()

    def vis_hparams(self, hparams, metrics):
        """超参数+指标对比"""
        self.writer.add_hparams(hparams, metrics)
        self.writer.flush()

    def close(self):
        """关闭writer"""
        self.writer.close()


def evaluate(model, val_loader):
    """验证模型，返回平均loss"""
    model.eval()
    total_loss = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data, target in val_loader:
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()

    model.train()
    return total_loss / len(val_loader)


def create_dummy_dataset(num_samples=1000, input_size=784, num_classes=10):
    X = torch.randn(num_samples, input_size)
    y = torch.randint(0, num_classes, (num_samples,))

    dataset = TensorDataset(X, y)
    return dataset

def test():
    vis = Visualizer(logdir="./runs", env="test_exp", exp_id="v2")

    train_dataset = create_dummy_dataset(num_samples=1000, input_size=784, num_classes=10)
    val_dataset = create_dummy_dataset(num_samples=200, input_size=784, num_classes=10)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    print(f"✅ 训练集: {len(train_loader)} batches")
    print(f"✅ 验证集: {len(val_loader)} batches")

    # for (x, y) in train_loader:
    #     print(x.shape, y.shape)

    model = nn.Linear(784, 10)
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    hparams = {
        'lr': 0.001,
        'batch_size': 32,
        'optimizer': 'Adam'
    }

    for epoch in range(100):
        vis.set_step(epoch)  # 设置全局步数

        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # 可选：每N个batch记录一次
            if batch_idx % 100 == 0:
                vis.vis_scalar('Loss/batch', loss.item(),
                               step=epoch * len(train_loader) + batch_idx)

        # 4️⃣ 每个epoch记录一次
        avg_loss = total_loss / len(train_loader)
        val_loss = evaluate(model, val_loader)

        vis.vis_scalars({
            'Loss/train': avg_loss,
            'Loss/val': val_loss,
            'LR': optimizer.param_groups[0]['lr'],
        }, step=epoch)

        # 5️⃣ 记录权重分布
        for name, param in model.named_parameters():
            if 'weight' in name:
                vis.vis_histogram(f'Weights/{name}', param, step=epoch)
    # 6️⃣ 最后记录超参数对比
    vis.vis_hparams(
        hparams,
        {'accuracy': 0.95, 'loss': 0.05}
    )
    vis.close()


if __name__ == "__main__":
    test()




