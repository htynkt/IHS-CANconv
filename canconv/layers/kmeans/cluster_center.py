import torch
from einops import rearrange, repeat, reduce


def get_cluster_centers_mask(samples: torch.Tensor, cluster_indice: torch.Tensor, cluster_num: int) -> torch.Tensor:
    """
    Args:
        samples: (batch_size, sample_num, feature_dim)
        cluster_indice: (batch_size, sample_num)
        cluster_num: int
    Returns:
        cluster_centers: (batch_size, cluster_num, feature_dim)
    """
    dev = samples.device
    batch_size = samples.shape[0]
    feature_dim = samples.shape[2]
    cluster_centers = torch.zeros(
        batch_size, cluster_num, feature_dim, device=dev, dtype=samples.dtype)
    for i in range(cluster_num):
        cluster_centers[:, i, :] = torch.mean(
            samples[cluster_indice == i, :], dim=0)
    return cluster_centers


def get_cluster_centers_scatter(samples: torch.Tensor, cluster_indice: torch.Tensor, cluster_num: int) -> torch.Tensor:
    """
    Args:
        samples: (batch_size, sample_num, feature_dim)
        cluster_indice: (batch_size, sample_num)
        cluster_num: int
    Returns:
        cluster_centers: (batch_size, cluster_num, feature_dim)
    """
    # 获取样本数据所在的设备（如 CPU 或 GPU）
    dev = samples.device
    # 获取样本数据的批量大小
    batch_size = samples.shape[0]
    # 获取每个批量中的样本数量
    sample_num = samples.shape[1]
    # 获取每个样本的特征维度
    feature_dim = samples.shape[2]

    # 初始化一个全零的张量 cluster_centers，用于存储每个聚类的质心之和
    # 形状为 (batch_size, cluster_num, feature_dim)
    cluster_centers = torch.zeros(batch_size, cluster_num, feature_dim, device=dev).scatter_add_(
        # 在维度 1 上进行 scatter_add_ 操作
        dim=1,
        # 重复 cluster_indice 张量，使其形状变为 (batch_size, sample_num, feature_dim)
        # 这样可以在每个特征维度上使用相同的聚类索引
        index=repeat(cluster_indice, 'b p -> b p s', s=feature_dim),
        # 源张量，即样本数据
        src=samples
    )

    # 初始化一个全零的张量 cluster_size，用于存储每个聚类中的样本数量
    # 形状为 (batch_size, cluster_num)
    cluster_size = torch.zeros(batch_size, cluster_num, device=dev).scatter_add_(
        # 在维度 1 上进行 scatter_add_ 操作
        dim=1,
        # 聚类索引张量
        index=cluster_indice,
        # 源张量，全为 1 的张量，用于统计每个聚类中的样本数量
        src=torch.ones(batch_size, sample_num, device=dev)
    ).unsqueeze_(dim=2)  # 在维度 2 上增加一个维度，使其形状变为 (batch_size, cluster_num, 1)

    # 避免除数为零的情况，将聚类大小小于 1 的值设为 1
    cluster_size[cluster_size < 1] = 1

    # 计算每个聚类的质心，即质心之和除以聚类中的样本数量
    cluster_centers /= cluster_size

    # 返回计算得到的聚类质心
    return cluster_centers
