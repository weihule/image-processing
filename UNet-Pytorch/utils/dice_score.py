import torch
from torch import Tensor


def dice_coeff(input: Tensor, target: Tensor,  reduce_batch_first: bool = False, epsilon: float = 1e-6):
    """
    reduce_batch_first: 
        False, 逐样本计算, 先对每个样本单独计算Dice系数, 再求平均, 每个样本的权重都是相同的
        True, 批计算, 可以保证数值的稳定性
        假设batch中有：
        样本1: 大肿瘤，1000个像素，Dice=0.9
        样本2: 小肿瘤，10个像素，Dice=0.5
        False: (0.9 + 0.5) / 2 = 0.7                ✓ 两个样本同等重要
        True:  2*(900+5)/(1000+10+1000+10) ≈ 0.896  ✗ 被大肿瘤主导
    """
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)
    print(f"sum_dim = {sum_dim}")

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)

    dice = (inter + epsilon) / (sets_sum + epsilon)

    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # 计算所有类别的平均Dice系数
    # TODO: 检查
    # 这里的input的shape应该是[N, n_classes, H, W]
    ret = dice_coeff(input.flatten(0, 1), 
                     target.flatten(0, 1),
                     reduce_batch_first, 
                     epsilon)
    return ret


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    loss = 1 - fn(input, target, reduce_batch_first=True)
    return loss


def multiclass_dice_coeff_per_class(input: Tensor, target: Tensor, epsilon: float = 1e-6):
    """
    返回每个类别的Dice系数
    """
    assert input.size() == target.size()
    B, n_classes = input.shape[:2]

    dice_per_class = []
    for c in range(n_classes):
        # 对每个类别单独计算Dice, [N, n_classes, H, W]  -> [N, H, W]
        dice_per_class = dice_coeff(input[:, n_classes])
        dice_per_class.append(dice_per_class)
    
    return torch.stack(dice_per_class)  # [C, ]


def test():
    x1 = torch.randint(0, 4, size=(2, 4, 5, 5))
    x2 = torch.randint(0, 4, size=(2, 4, 5, 5))

    print(x1)
    print(x1[:, 0], x1[:, 0].shape)
    print(x1[:, 1], x1[:, 1].shape)

    # x1 = x1.flatten(0, 1)
    # print(x1.shape)
    # print(x1, x1.size(), x1.dim())
    # print(x2, x2.size())
    # out = multiclass_dice_coeff(x1, x2)
    # print(f"out = {out}")


if __name__ == "__main__":
    test()



