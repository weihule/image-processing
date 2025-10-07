import torch
import torch.nn as nn
import torch.nn.functional as F


class CELoss(nn.Module):
    """
    Cross Entropy Loss
    """
    def __init__(self, use_custom=False):
        super(CELoss, self).__init__()
        self.use_custom = use_custom

    def forward(self, pred, label):
        """
        Args:
            pred: [B, num_classes]
            label: [B]

        Returns:
        """
        if self.use_custom:
            one_hot_label = F.one_hot(label, num_classes=pred.shape[1]).float()  # [N, num_class]
            print(one_hot_label)
            softmax_pred = F.softmax(pred, dim=1)

            # mean
            loss = -torch.sum(one_hot_label * torch.log(softmax_pred)) / pred.shape[0]
        else:
            loss_func = nn.CrossEntropyLoss(reduction='mean')
            loss = loss_func(pred, label)

        return loss


if __name__ == "__main__":
    outs = torch.randn(1, 2, 20, 20)
    labels = torch.randint(low=0, high=1,
                           size=(1, 20, 20))
    ce = CELoss(use_custom=True)
    lo = ce(outs, labels)
    print(f"loss = {lo}")

