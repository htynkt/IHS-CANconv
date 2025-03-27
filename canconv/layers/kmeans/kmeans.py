import logging
import torch
import numpy as np
from torch.profiler import record_function
from einops import rearrange, repeat, reduce

logger = logging.getLogger(__name__)
logger.info("Begin to load kmeans operator...")
try:
    from .libKMCUDA import kmeans_cuda  # type: ignore
except ImportError as e:
    logger.error("Fail to load kmeans operator from local path.")
    logger.exception(e)
    print("Please use libKMCUDA built from https://github.com/duanyll/kmcuda. The built libKMCUDA.so file should be placed in the same directory as this file. Do not use the official libKMCUDA from pip.")
    raise e
logger.info("Finish loading kmeans operator.")

seed = 42


def kmeans(samples: torch.Tensor, cluster_num: int, cached_center=None) -> torch.Tensor:
    """
    Run kmeans on samples. Result is on the same device as samples. If cached_center is not None, it will be used as the initial cluster center.
    Args:
        samples: (sample_num, feature_dim)
        cluster_num: int
        cached_center: (cluster_num, feature_dim)
    Returns:
        cluster_idx: (sample_num)
    """
    # 如果聚类数量小于等于 1，直接返回全零张量
    if cluster_num <= 1:
        return torch.zeros(samples.shape[0])

    # 如果聚类数量大于样本数量，记录警告信息并将聚类数量设置为样本数量
    if cluster_num > samples.shape[0]:
        logger.warning(
            f"cluster_num ({cluster_num}) > sample_num ({samples.shape[0]}).")
        cluster_num = samples.shape[0]

    # 使用 torch.profiler 记录函数执行时间
    with record_function("kmeans"):
        if cached_center is None:
            # 如果没有提供缓存的聚类中心，调用 kmeans_cuda 函数进行聚类
            idx, _ = kmeans_cuda(samples, cluster_num, seed=seed)
        else:
            # 如果提供了缓存的聚类中心，将其作为初始聚类中心调用 kmeans_cuda 函数进行聚类
            idx, _ = kmeans_cuda(samples, cluster_num,
                                 cached_center, seed=seed)

    # 将聚类索引转换为长整型并返回
    return idx.long()
