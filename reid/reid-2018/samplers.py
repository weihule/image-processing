from __future__ import absolute_import
from collections import defaultdict
import numpy as np

import torch
from torch.utils.data.sampler import Sampler


class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/data/sampler.py.

    Args:
        data_source (Dataset): dataset to sample from.
        num_instances (int): number of instances per identity.
    """
    def __init__(self, data_source, num_instances=4):
        super(Sampler, self).__init__()
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)

        # 以market1501为例, 训练集有 12936 张图片, 但是只有751个行人, 即不同的图片可能是同一个行人, 即pid一样
        # 所以self.index_dic存储键是0-750, 对应属于同一pid的索引data_source的值
        for index, (_, pid, _) in enumerate(data_source):
            self.index_dic[pid].append(index)

        self.pids = list(self.index_dic.keys())

        self.num_identities = len(self.pids)

    def __iter__(self):
        # 将0~n-1（包括0和n-1）随机打乱后获得的数字序列
        indices = torch.randperm(self.num_identities)
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            replace = False if len(t) >= self.num_instances else True
            # 从t中随机抽取 size 个元素, replace参数为True表示可以抽取相同元素, 为False表示不可以抽取相同元素
            # 即每个pid随机抽取num_instances张图片
            t = np.random.choice(t, size=self.num_instances, replace=replace)
            ret.extend(t)

        return iter(ret)

    def __len__(self):
        return self.num_identities * self.num_instances

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


if __name__ == "__main__":
    import data_manager

    dataset = data_manager.init_img_dataset(root='D:\\workspace\\data\\dl',
                                            name='market1501')
    train_dataset = dataset.train

    random_sampler = RandomIdentitySampler(data_source=train_dataset,
                                           num_instances=4)

