from __future__ import print_function, absolute_import
from collections import defaultdict
import numpy as np

import torch
from torch.utils.data.sampler import Sampler


class RandomIdentitySampler(Sampler):
    def __init__(self, data_source: list, num_instances=4):
        super(Sampler, self).__init__()
        self.data_source = data_source
        self.num_instance = num_instances
        self.index_dict = defaultdict(list)

        for index, (_, pid, _) in enumerate(self.data_source):
            self.index_dict[pid].append(pid)

        self.pids = list(self.index_dict.keys())
        self.num_identities = len(self.pids)

    def __iter__(self):
        # 将0~n-1 (包括0和n-1)随机打乱获得数字序列
        indices = torch.randperm(self.num_identities)
        res = []
        for i in indices:
            pid = self.pids[i]
            ts = self.index_dict[pid]
            replace = False if len(ts) >= self.num_instance else True
            # 从ts中随机抽取size个元素, replace为True表示可以抽取相同元素, 为False表示不可以抽取相同元素
            ts = np.random.choice(ts, size=self.num_instance, replace=replace)
            res.extend(ts)

        return iter(res)

    def __len__(self):
        return self.num_identities * self.num_instance

