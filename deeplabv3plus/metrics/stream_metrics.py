import numpy as np
import torch
from typing import Dict, Union
from sklearn.metrics import confusion_matrix

class _StreamMetrics(object):
    def __init__(self):
        raise NotImplementedError()

    def update(self, gt, pred):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def get_results(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def to_str(self, metrics):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def reset(self):
        """ Overridden by subclasses """
        raise NotImplementedError()


class StreamSegMetrics(_StreamMetrics):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_mat = np.zeros((n_classes, n_classes))

    def update(self, label_trues, label_preds):
        if isinstance(label_trues, torch.Tensor):
            label_trues = label_trues.cpu().numpy()
        if isinstance(label_preds, torch.Tensor):
            label_preds = label_preds.cpu().numpy()

        for lt, lp in zip(label_trues, label_preds):
            self.confusion_mat += self._fast_hist(lt.flatten(), lp.flatten())

    @staticmethod
    def to_str(results: Dict) -> str:
        """打印结果"""
        string = '\n'
        for k, v in results.items():
            if k != "Class IoU":
                string += f"{k}: {v:.4f}\n"
        return string

    def _fast_hist(self, label_true: np.ndarray, label_preds: np.ndarray) -> np.ndarray:
        """计算混淆矩阵"""
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(self.n_classes*label_true[mask]+label_preds[mask],
                           minlength=self.n_classes ** 2).reshape(self.n_classes, self.n_classes)
        return hist

    def get_results(self) -> Dict:
        hist = self.confusion_mat

        # 避免除以0
        sum_per_class = hist.sum(axis=1)        # 真实标签的每类总数
        sum_pred_per_class = hist.sum(axis=0)   # 预测标签的每类总数

        # 总体准确率
        total = hist.sum()
        acc = np.diag(hist).sum() / hist.sum() if hist.sum() > 0 else 0.0

        # 没有样本的类设置为nan，不计入平均值
        acc_cls = np.divide(
            np.diag(hist),
            sum_per_class,
            where=sum_per_class > 0,
            out=np.full_like(np.diag(hist), np.nan, dtype=float)
        )
        acc_cls_mean = np.nanmean(acc_cls)    # nanmean 会忽略 nan 值

        # IoU
        denominator = sum_per_class + sum_pred_per_class - np.diag(hist)
        iu = np.divide(
            np.diag(hist),
            denominator,
            where=denominator > 0,
            out=np.full_like(np.diag(hist), np.nan, dtype=float)
        )
        mean_iu = np.nanmean(iu)

        freq = sum_per_class / total if total > 0 else np.zeros(self.n_classes)
        # 只计算有效的 IoU（非nan）
        valid_mask = ~np.isnan(iu)
        fwavacc = (freq[valid_mask] * iu[valid_mask]).sum()

        # ===== Class IoU =====
        cls_iu = dict(zip(range(self.n_classes), iu))
        return {
            "Overall Acc": float(acc),
            "Mean Acc": float(acc_cls_mean),
            "FreqW Acc": float(fwavacc),
            "Mean IoU": float(mean_iu),
            "Class IoU": cls_iu,
        }

    def reset(self):
        """重置混淆矩阵"""
        self.confusion_mat = np.zeros((self.n_classes, self.n_classes), dtype=np.int64)

class AverageMeter(object):
    """
    计算各类平均值统计量
    """
    def __init__(self):
        self.book = dict()

    def reset_all(self):
        """重置所有统计"""
        self.book.clear()

    def reset(self, key:str):
        """重置单个指标"""
        for key in self.book:
            self.book[key] = {'sum': 0, 'count': 0, 'value': []}

    def update(self, key: str, val: float, n: int=1):
        """更新指标"""
        if key not in self.book:
            self.book[key] = {'sum': 0, 'count': 0, 'value': []}

        self.book[key]['sum'] += val * n
        self.book[key]['count'] += n
        self.book[key]['value'].append(val)

    def get_results(self, key: str, default: float = 0.0) -> float:
        """获取平均值"""
        if key not in self.book or self.book[key]['count'] == 0:
            return default
        ret = self.book[key]['sum'] / self.book[key]['count']

    def get_std(self, key: str) -> float:
        """获取标准差"""
        if key not in self.book or len(self.book[key]['values']) < 2:
            return 0.0
        return np.std(self.book[key]['sum'])

    def get_min(self, key: str) -> float:
        """获取最小值"""
        if key not in self.book or self.book[key]['count'] == 0:
            return float('inf')
        return min(self.book[key]['values'])

    def get_max(self, key: str) -> float:
        """获取最大值"""
        if key not in self.book or self.book[key]['count'] == 0:
            return float('-inf')
        return max(self.book[key]['values'])

    def get_all_stats(self, key: str) -> Dict[str, float]:
        if key not in self.book or self.book[key]['count'] == 0:
            return {}

        return {
            'mean': self.get_results(key),
            'std': self.get_std(key),
            'min': self.get_min(key),
            'max': self.get_max(key),
            'count': self.book[key]['count'],
        }

    def __repr__(self) -> str:
        """打印所有指标"""
        result = "\n" + "=" * 50 + "\n"
        for key, stats in self.book.items():
            if stats['count'] > 0:
                mean = stats['sum'] / stats['count']
                result += f"{key:20s}: {mean:10.6f}\n"
        result += "=" * 50
        return result

    def __str__(self) -> str:
        """字符串表示"""
        return self.__repr__()


def test():
    y_true = np.array([0, 1, 2, 0, 1, 2, 1, 1, 2, 0, 1, 1, 2, 0, 0, 1])
    y_pred = np.array([0, 1, 2, 0, 1, 1, 0, 1, 1, 0, 1, 1, 2, 2, 1, 1])
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    n_classes = 3
    y_true2 = y_true.reshape((4, 4))
    y_pred2 = y_pred.reshape((4, 4))
    # 构造一维索引
    indices = n_classes * y_true + y_pred
    cm2 = np.bincount(indices, minlength=n_classes**2).reshape(n_classes, n_classes)
    sum_per_class = cm2.sum(axis=1)
    print(indices)
    print(cm2)
    print(sum_per_class, sum_per_class.shape)
    print(np.diag(sum_per_class))


def test2():
    # 创建指标计算器
    metrics = StreamSegMetrics(n_classes=5)
    metrics2 = StreamSegMetrics2(n_classes=5)

    # 模拟一个 batch 的预测结果
    batch_size = 4
    height, width = 32, 32

    for i in range(10):  # 10个batch
        # 随机生成标签和预测
        label_true = np.random.randint(0, 5, (batch_size, height, width))
        label_pred = np.random.randint(0, 5, (batch_size, height, width))

        # 更新指标
        metrics.update(label_true, label_pred)
        metrics2.update(label_true, label_pred)

    # 获取结果
    results = metrics.get_results()
    print(metrics.to_str(results))
    print("Class IoU:", {k: f"{v:.4f}" for k, v in results['Class IoU'].items()})

    results2 = metrics2.get_results()
    print(metrics2.to_str(results2))
    print("Class IoU:", {k: f"{v:.4f}" for k, v in results2['Class IoU'].items()})


if __name__ == "__main__":
    # test()
    test2()



