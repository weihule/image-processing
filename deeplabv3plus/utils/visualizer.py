from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


class Visualizer(object):
    def __init__(self, logdir="./runs", env="main", id=None):
        # 创建日志目录
        if id is not None:
            logdir = Path(logdir) / f"[{id}]_{env}"
        else:
            logdir = Path(logdir) / env

        self.writer = SummaryWriter(str(logdir))
        self.id = id
        self.env = env
        print(f"✅ TensorBoard已启动，日志保存在: {logdir}")

    def vis_scalar(self, name, x, y, opts=None):
        """绘制标量图"""
        if not isinstance(x, list):
            x = [x]
        if not isinstance(y, list):
            y = [y]

        if self.id is not None:
            name = f"[{self.id}]{name}"

        # Tensorboard中x是step, y是value
        for step, value in zip(x, y):
            self.writer.add_scalar(name, value, global_step=int(step))

        self.writer.flush()

    def vis_image(self, name, img, env=None, opts=None):
        if env is None:
            env = self.env
        if self.id is not None:
            name = f"[{self.id}]{name}"
        # img需要是 (3, H, W) 或 (1, H, W) 格式
        self.writer.add_image(name, img)
        self.writer.flush()

    def vis_histogram(self, name, values, opts=None):
        """直方图 - 查看权重分布"""
        if self.id is not None:
            name = f"[{self.id}]{name}"

        self.writer.add_histogram(name, values)
        self.writer.flush()

    def vis_hparams(self, hparams, metrics):
        """超参数+指标（适合做对比实验）"""
        self.writer.add_hparams(hparams, metrics)
        self.writer.flush()

    def close(self):
        """关闭writer"""
        self.writer.close()





