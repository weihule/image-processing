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

# class RandomIdentitySampler(Sampler):
#     """
#     Randomly sample N identities, then for each identity,
#     randomly sample K instances, therefore batch size is N*K.
#
#     Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/data/sampler.py.
#
#     Args:
#         data_source (Dataset): dataset to sample from.
#         num_instances (int): number of instances per identity.
#     """
#     def __init__(self, data_source, num_instances=4):
#         self.data_source = data_source
#         self.num_instances = num_instances
#         self.index_dic = defaultdict(list)
#         for index, (_, pid, _) in enumerate(data_source):
#             self.index_dic[pid].append(index)
#         self.pids = list(self.index_dic.keys())
#         self.num_identities = len(self.pids)
#
#     def __iter__(self):
#         indices = torch.randperm(self.num_identities)
#         ret = []
#         for i in indices:
#             pid = self.pids[i]
#             t = self.index_dic[pid]
#             replace = False if len(t) >= self.num_instances else True
#             t = np.random.choice(t, size=self.num_instances, replace=replace)
#             ret.extend(t)
#         return iter(ret)
#
#     def __len__(self):
#         return self.num_identities * self.num_instances

