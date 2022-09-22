import torch
import torch.nn as nn


class HorizontalPooling(nn.Module):
    def __init__(self):
        super(HorizontalPooling, self).__init__()

    def forward(self, x):
        x_width = x.shape[3]

        return torch.nn.functional.max_pool2d(x, kernel_size=(1, x_width))


if __name__ == "__main__":
    hp = HorizontalPooling()
    inputs = torch.randn(4, 10, 8, 4)
    outputs = hp(inputs)
    print(outputs.shape)


