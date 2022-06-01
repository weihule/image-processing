import torch
import torch.nn as nn
# hyper parameters
in_dim = 1
n_hidden_1 = 2
n_hidden_2 = 2
out_dim = 1

class Net(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Net, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1),
            nn.ReLU(inplace=True)
        )