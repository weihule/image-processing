import torch
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
        # losses = F.binary_cross_entropy(input_data, one_hot_target)
        losses = F.binary_cross_entropy_with_logits(input_data, one_hot_target)

    return losses


if __name__ == "__main__":
    inputs = torch.rand((3, 4))
    labels = torch.tensor([0, 2, 3])
    custom_ce_loss = custom_cross_entropy(input_data=inputs,
                                          target=labels,
                                          num_class=4,
                                          use_custom=True)
    offical_ce_loss = custom_cross_entropy(input_data=inputs,
                                           target=labels,
                                           num_class=4,
                                           use_custom=False)

    custom_bce_loss = custom_bce(input_data=inputs,
                                 target=labels,
                                 num_class=4,
                                 use_custom=True)
    offical_bce_loss = custom_bce(input_data=inputs,
                                 target=labels,
                                 num_class=4,
                                 use_custom=False)
    print(custom_bce_loss, offical_bce_loss)