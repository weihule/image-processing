import torch
from torch import Tensor

__all__ = [
    "dice_loss",
    "dice_coeff",
    "multiclass_dice_coeff"
]


def dice_coeff(inputs: Tensor, targets: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    """
    dice = 2|X ∩ Y| / |X| + |Y|
    Args:
        inputs:
        targets:
        reduce_batch_first:
        epsilon:

    Returns:

    """
    assert inputs.size() == targets.size()
    assert inputs.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if inputs.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    # 计算2|X ∩ Y|, 因为要计算所有通道的相加和, 所以这里的sum_dim是所有通道
    inter = 2 * (inputs * targets).sum(dim=sum_dim)
    sets_sum = inputs.sum(dim=sum_dim) + targets.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(inputs: Tensor, targets: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    return dice_coeff(inputs.flatten(0, 1),
                      targets.flatten(0, 1),
                      reduce_batch_first, epsilon)


def dice_loss(inputs: Tensor, targets: Tensor, multiclass: bool = False):
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(inputs, targets, reduce_batch_first=True)


if __name__ == "__main__":
    arr = torch.randn(4, 3, 640, 640)
    # print(arr.flatten(0, 1).shape)
    print(arr.squeeze().shape)


