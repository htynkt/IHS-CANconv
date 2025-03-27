import torch
import torch.nn as nn

from .cache import kmeans_batched

new_module_id = 0

class KMeans(nn.Module):
    def __init__(self, cluster_num, *args, **kwargs) -> None:
        # 调用父类 nn.Module 的构造函数
        super().__init__(*args, **kwargs)

        # 为当前模块分配一个唯一的 ID
        global new_module_id
        self.module_id = new_module_id
        new_module_id += 1

        # 保存聚类的数量
        self.cluster_num = cluster_num

    def forward(self, x, cache_indice=None, cluster_num=None):
        # 如果模型处于非训练模式，将缓存的聚类索引置为 None
        if not self.training:
            cache_indice = None

        # 如果未指定聚类数量，使用初始化时设置的聚类数量
        if cluster_num is None:
            cluster_num = self.cluster_num

        # 调用 kmeans_batched 函数对输入数据进行聚类
        return kmeans_batched(x, cluster_num, cache_indice, self.module_id)