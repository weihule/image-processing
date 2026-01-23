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
        super(StreamSegMetrics, self).__init__()
        self.n_classes = n_classes
        self.confusion_mat = np.zeros((n_classes, n_classes))

    def update(self, label_trues, label_preds):
        if isinstance(label_trues, torch.Tensor):
            label_trues = label_trues.cpu().numpy()
        if isinstance(label_preds, torch.Tensor):
            label_trues = label_preds.cpu().numpy()

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
        sum_per_class = hist.sum(axis=1)
        sum_pred_per_class = hist.sum(axis=0)

        # 总体准确率
        acc = np.diag(hist).sum() / hist.sum() if hist.sum() > 0 else 0.0

        acc_cls = np.divide(
            np.diag(hist),
            sum_per_class,
            where=sum_per_class > 0,
            out=np.full_like(np.diag(hist), np.nan, dtype=float)
        )

        acc_cls = np.nanmean(acc_cls)

        # IoU



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


if __name__ == "__main__":
    test()




