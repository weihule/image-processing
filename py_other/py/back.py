import re
import torch
import torch.nn as nn

def test():
    x = torch.tensor([1., 2.], requires_grad=True)
    y = (x + 2) ** 2
    z = torch.sum(y)
    z.backward()
    print(x.grad)
    


if __name__ == "__main__":
    test()

    a = torch.tensor(0.999)
    print(a)
    print(a.item())