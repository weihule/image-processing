import torch
import torch.nn as nn
import torch.nn.functional as F


def custom_cross_entropy(input_data, target, num_class, use_custom=True):
    """
    :param use_custom: bool
    :param input_data: [N, num_class]
    :param target: [N]
    :param num_class: int
    :return:
    """
    if use_custom:
        one_hot = F.one_hot(target, num_classes=num_class).float()  # [N, num_class]
        custom_softmax = torch.exp(input_data) / torch.sum(torch.exp(input_data), dim=1).reshape((-1, 1))
        losses = -torch.sum(one_hot * torch.log(custom_softmax)) / input_data.shape[0]
    else:
        # 1
        # log_soft = F.log_softmax(input_data, dim=1)
        # losses = F.nll_loss(log_soft, target)

        # 2
        losses = F.cross_entropy(input_data, target)

    return losses


def custom_bce(input_data, target, num_class, use_custom=True):
    one_hot_target = F.one_hot(target, num_classes=num_class).float()

    if use_custom:
        print(input_data)
        print(one_hot_target)
        losses = -one_hot_target * torch.log(torch.sigmoid(input_data)) \
                 - (1 - one_hot_target) * (torch.log(1 - torch.sigmoid(input_data)))
        losses = losses.mean()
    else:
        losses = F.binary_cross_entropy_with_logits(input_data, one_hot_target)

    return losses


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
            softmax_pred = F.softmax(pred, dim=1)

            # 求均值
            loss = -torch.sum(one_hot_label * torch.log(softmax_pred)) / pred.shape[0]
        else:
            loss_func = nn.CrossEntropyLoss(reduction='mean')
            loss = loss_func(pred, label)

        return loss


class FocalCELoss(nn.Module):
    def __init__(self, gamma=2.0):
        super(FocalCELoss, self).__init__()
        self.gamma = gamma

    def forward(self, pred, label):
        pred = F.softmax(pred, dim=-1)
        one_hot_label = F.one_hot(label, num_classes=pred.shape[1]).float()

        pt = torch.where(torch.eq(one_hot_label, 1.), pred, 1.-pred)
        focal_weight = torch.pow((1. - pt), self.gamma)

        loss = -one_hot_label * torch.log(pred)
        loss = focal_weight * loss
        loss = torch.sum(loss) / pred.shape[0]

        return loss


if __name__ == "__main__":
    # inputs = torch.tensor([[0.0235, 0.4266, 0.7232, 0.5329, 0.4321],
    #                       [0.8937, 0.0001, 0.8116, 0.4274, 0.7896],
    #                       [0.6525, 0.1494, 0.6610, 0.0002, 0.2387],
    #                       [0.0987, 0.1023, 0.0777, 0.9893, 0.2198]])

    inputs = torch.tensor([[0.0235, 0.0266, 0.7232, 0.0329, 0.0321],
                          [0.8937, 0.0001, 0.0116, 0.0274, 0.0896],
                          [0.6525, 0.0494, 0.0610, 0.0002, 0.0387],
                          [0.0987, 0.1023, 0.0777, 0.9893, 0.2198]])

    # [2, 0, 0, 3]
    labels = torch.tensor([2, 0, 0, 3])

    ce_loss_func1 = CELoss(use_custom=False)
    ce_loss_func2 = CELoss(use_custom=True)

    focal_loss_func = FocalCELoss()

    loss1 = ce_loss_func1(inputs, labels)
    loss2 = ce_loss_func2(inputs, labels)
    # loss3 = focal_loss_func(inputs, labels)

    print(loss1, loss2)


