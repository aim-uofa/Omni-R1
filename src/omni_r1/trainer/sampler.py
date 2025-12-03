from typing import Iterator, List, Optional

import torch
from torch.utils.data import Sampler


class GroupSampler(Sampler):
    """
    分组采样器，确保同一训练步骤中的样本来自同一数据类别组。

    参数:
        data_sources: 每个样本对应的数据源/类别索引列表
        group_ids: 每个数据源属于哪个组（0: RefAVS组, 1: 其他组）
        batch_size: 批量大小
        shuffle: 是否打乱数据
        seed: 随机种子
        distributed: 是否在分布式环境中使用
        num_replicas: 分布式训练的副本数
        rank: 当前进程的rank
    """

    def __init__(
        self,
        data_sources: List[int],
        group_ids: List[int],
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
        distributed: bool = False,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
    ):
        self.data_sources = data_sources
        self.group_ids = group_ids
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        # 分布式训练参数
        self.distributed = distributed
        if distributed:
            if num_replicas is None or rank is None:
                raise ValueError(
                    "For distributed sampling, num_replicas and rank must be provided"
                )
            self.num_replicas = num_replicas
            self.rank = rank
        else:
            self.num_replicas = 1
            self.rank = 0

        # 将数据索引按组分类
        self.group_indices = {}
        for i, ds_idx in enumerate(self.data_sources):
            group = self.group_ids[ds_idx]
            if group not in self.group_indices:
                self.group_indices[group] = []
            self.group_indices[group].append(i)

        # 计算每个组需要的样本数量，确保能够被 batch_size * num_replicas 整除
        self.total_size = 0
        self.num_samples = 0

        # 调整每个组的样本数以适应分布式训练和批处理
        for group in self.group_indices:
            group_size = len(self.group_indices[group])
            # 计算需要的样本数量，确保能被 batch_size * num_replicas 整除
            group_total_size = (
                (
                    (group_size + batch_size * num_replicas - 1)
                    // (batch_size * num_replicas)
                )
                * batch_size
                * num_replicas
            )

            # 如果不够，则通过复制一些样本来补足
            if group_total_size > group_size:
                # 计算需要的额外样本数
                extra_needed = group_total_size - group_size
                # 循环复制已有样本直到达到所需数量
                indices_to_add = self.group_indices[group][:extra_needed]
                self.group_indices[group] = self.group_indices[group] + indices_to_add

            self.total_size += group_total_size

        # 计算当前进程的样本数
        self.num_samples = self.total_size // self.num_replicas

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # 打乱每个组内的索引
        shuffled_indices = {}
        for group in self.group_indices:
            if self.shuffle:
                indices = torch.randperm(
                    len(self.group_indices[group]), generator=g
                ).tolist()
                shuffled_indices[group] = [
                    self.group_indices[group][i] for i in indices
                ]
            else:
                shuffled_indices[group] = self.group_indices[group]

        # 创建批次，确保每个批次只来自一个组
        batches = []
        for group in shuffled_indices:
            group_indices = shuffled_indices[group]
            for i in range(0, len(group_indices), self.batch_size * self.num_replicas):
                # 获取一批数据，即使最后一批可能不足
                batch = group_indices[i : i + self.batch_size * self.num_replicas]
                # 如果最后一批数据不足，我们之前已经通过复制扩展了数据集，所以这里不需要额外处理
                batches.append(batch)

        # 打乱批次顺序
        if self.shuffle:
            indices = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in indices]

        # 展平批次并进行分布式采样
        flattened_indices = []
        for batch in batches:
            # 为每个进程分配相应的数据
            for i in range(self.rank, len(batch), self.num_replicas):
                if i < len(batch):  # 确保索引不越界
                    flattened_indices.append(batch[i])

        # 确保每个进程获取相同数量的样本
        # 如果不够，则重复使用一些样本
        if len(flattened_indices) < self.num_samples:
            extra_needed = self.num_samples - len(flattened_indices)
            flattened_indices.extend(flattened_indices[:extra_needed])

        return iter(flattened_indices[: self.num_samples])

    def __len__(self) -> int:
        # 返回当前进程应处理的样本数
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """设置当前 epoch，用于确保每个 epoch 洗牌不同"""
        self.epoch = epoch


# import torch
# from torch.utils.data import Sampler
# import numpy as np
# from typing import List, Iterator, Optional

# class GroupedBatchSampler(Sampler):
#     """
#     采样器确保每个批次只包含一类数据
#     """
#     def __init__(
#         self,
#         dataset_sizes: List[int],
#         batch_size: int,
#         group_ids: List[int],
#         shuffle: bool = True,
#         seed: int = 42,
#         drop_last: bool = False,
#         distributed: bool = False,
#         rank: int = 0,
#         world_size: int = 1
#     ):
#         """
#         Args:
#             dataset_sizes: 每个数据集的大小列表
#             batch_size: 批次大小
#             group_ids: 每个数据集对应的组ID列表 (RefAVS为0，其他为1)
#             shuffle: 是否打乱数据
#             seed: 随机种子
#             drop_last: 是否丢弃不足一个批次的数据
#             distributed: 是否使用分布式训练
#             rank: 当前进程在分布式训练中的rank
#             world_size: 分布式训练的总进程数
#         """
#         self.dataset_sizes = dataset_sizes
#         self.batch_size = batch_size
#         self.group_ids = group_ids
#         self.shuffle = shuffle
#         self.seed = seed
#         self.drop_last = drop_last
#         self.distributed = distributed
#         self.rank = rank
#         self.world_size = world_size

#         # 计算累积大小，用于索引映射
#         self.cumulative_sizes = [0]
#         for size in dataset_sizes:
#             self.cumulative_sizes.append(self.cumulative_sizes[-1] + size)

#         # 按组分类索引
#         self.grouped_indices = {group_id: [] for group_id in set(group_ids)}
#         for dataset_idx, group_id in enumerate(group_ids):
#             start_idx = self.cumulative_sizes[dataset_idx]
#             end_idx = self.cumulative_sizes[dataset_idx + 1]
#             self.grouped_indices[group_id].extend(list(range(start_idx, end_idx)))

#         # 计算每组的批次数量
#         self.batches_per_group = {}
#         for group_id, indices in self.grouped_indices.items():
#             num_samples = len(indices)
#             if self.drop_last:
#                 self.batches_per_group[group_id] = num_samples // self.batch_size
#             else:
#                 self.batches_per_group[group_id] = (num_samples + self.batch_size - 1) // self.batch_size

#         self.epoch = 0

#     def __iter__(self) -> Iterator[List[int]]:
#         if self.shuffle:
#             # 每个epoch设置不同的种子
#             g = torch.Generator()
#             g.manual_seed(self.seed + self.epoch)

#             # 为每个组单独打乱
#             for group_id in self.grouped_indices:
#                 indices = self.grouped_indices[group_id]
#                 perm = torch.randperm(len(indices), generator=g).tolist()
#                 self.grouped_indices[group_id] = [indices[idx] for idx in perm]

#         # 创建批次
#         all_batches = []
#         for group_id, indices in self.grouped_indices.items():
#             for i in range(0, len(indices), self.batch_size):
#                 batch = indices[i:i + self.batch_size]
#                 if len(batch) < self.batch_size and self.drop_last:
#                     continue
#                 all_batches.append(batch)

#         # 打乱批次顺序，但保持每个批次内的数据类型一致
#         if self.shuffle:
#             g = torch.Generator()
#             g.manual_seed(self.seed + self.epoch + 1)  # 不同的种子避免和索引打乱重复
#             perm = torch.randperm(len(all_batches), generator=g).tolist()
#             all_batches = [all_batches[idx] for idx in perm]

#         # 分布式训练支持
#         if self.distributed:
#             num_samples = len(all_batches)
#             total_size = num_samples - (num_samples % self.world_size)
#             # 确保每个进程拿到均等的批次数量
#             if num_samples != total_size:
#                 all_batches = all_batches[:total_size]

#             # 将批次按rank分配
#             rank_batches = []
#             for i in range(self.rank, len(all_batches), self.world_size):
#                 rank_batches.append(all_batches[i])
#             all_batches = rank_batches

#         # 将批次展平为索引列表
#         for batch in all_batches:
#             yield batch

#     def __len__(self) -> int:
#         if self.distributed:
#             total_batches = sum(self.batches_per_group.values())
#             return (total_batches + self.world_size - 1) // self.world_size
#         else:
#             return sum(self.batches_per_group.values())

#     def set_epoch(self, epoch: int):
#         """设置当前epoch，用于分布式训练"""
#         self.epoch = epoch
