import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
from .kmeans import KMeans, get_cluster_centers
from .pwac import filter_indice, dispatch_indice, permute, inverse_permute, batched_matmul_conv
import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity


import torch
import torch.nn as nn
from sklearn.cluster import KMeans

class CANConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 cluster_num=32,
                 kernel_size=3,
                 mlp_inner_dims=16,
                 bias="cluster",  # or "global_param" or "global_adaptive" or "none"
                 detach_centroid=False,
                 cluster_source="channel",  # "spatial" or "pixel"
                 kernel_generator="low_rank",  # or "weighted_sum" or "low_rank"
                 kernel_count=8,  # required when kernel_generator is "weighted_sum"
                 cluster_ablation="none",  # or "global" or "pixelwise"
                 filter_threshold=0,
                 ) -> None:
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            cluster_num: Number of clusters
            kernel_size: Kernel size
            mlp_inner_dims: Number of hidden units in for the MLP that generates the kernel
            bias: "none" for no bias, "cluster" for bias for each cluster, "global_param" use a uniform bias like nn.Conv2d,
                  "global_adaptive" generates global bias like LAGConv
        """
        # 调用父类 nn.Module 的构造函数
        super().__init__()
        # 保存输入通道数
        self.in_channels = in_channels
        # 保存输出通道数
        self.out_channels = out_channels
        # 定义一个 Unfold 层，用于从输入特征图中提取局部块
        self.unfold = nn.Unfold(kernel_size=kernel_size,
                                padding=(kernel_size - 1) // 2)
        # 计算卷积核的面积（即卷积核元素的总数）
        self.kernel_area = kernel_size ** 2
        # 保存聚类的数量
        self.cluster_num = cluster_num
        # 保存偏置模式
        self.bias_mode = bias
        # 保存是否分离质心的标志
        self.detatch_centroid = detach_centroid
        # 保存聚类的源（即聚类的依据）
        self.cluster_source = cluster_source
        # 保存卷积核生成的方式
        self.kernel_generator = kernel_generator
        # 保存聚类消融的模式
        self.cluster_ablation = cluster_ablation
        # 保存滤波器的阈值
        self.filter_threshold = filter_threshold

        # 初始化 KMeans 聚类模型
        self.kmeans = KMeans(cluster_num)

        # 根据卷积核生成方式进行不同的初始化
        if self.kernel_generator == "spatial":
            # 定义一个 MLP 网络，用于将聚类质心转换为卷积核
            self.centroid_to_kernel = nn.Sequential(
                # 第一个全连接层，输入维度为输入通道数乘以卷积核面积，输出维度为 MLP 隐藏单元数
                nn.Linear(in_features=self.in_channels * self.kernel_area,
                          out_features=mlp_inner_dims),
                # ReLU 激活函数
                nn.ReLU(),
                # 第二个全连接层，输入和输出维度均为 MLP 隐藏单元数
                nn.Linear(in_features=mlp_inner_dims,
                          out_features=mlp_inner_dims),
                # ReLU 激活函数
                nn.ReLU(),
                # 第三个全连接层，输入维度为 MLP 隐藏单元数，输出维度为卷积核面积
                nn.Linear(in_features=mlp_inner_dims,
                          out_features=self.kernel_area),
                # Sigmoid 激活函数
                nn.Sigmoid()
            )
            # 定义可学习的卷积核参数
            self.kernels = nn.parameter.Parameter(
                torch.randn(self.in_channels, self.kernel_area, self.out_channels))
            # 使用 Kaiming 初始化方法对卷积核进行初始化
            nn.init.kaiming_normal_(self.kernels, nonlinearity="relu")
        elif self.kernel_generator == "weighted_sum":
            # 定义一个 MLP 网络，用于将聚类质心转换为权重
            self.centroid_to_kernel = nn.Sequential(
                # 第一个全连接层，输入维度为输入通道数乘以卷积核面积，输出维度为 MLP 隐藏单元数
                nn.Linear(in_features=self.in_channels * self.kernel_area,
                          out_features=mlp_inner_dims),
                # ReLU 激活函数
                nn.ReLU(),
                # 第二个全连接层，输入和输出维度均为 MLP 隐藏单元数
                nn.Linear(in_features=mlp_inner_dims,
                          out_features=mlp_inner_dims),
                # ReLU 激活函数
                nn.ReLU(),
                # 第三个全连接层，输入维度为 MLP 隐藏单元数，输出维度为卷积核数量
                nn.Linear(in_features=mlp_inner_dims,
                          out_features=kernel_count),
                # Softmax 激活函数，用于将输出转换为权重
                nn.Softmax()
            )
            # 定义可学习的卷积核参数
            self.kernels = nn.parameter.Parameter(
                torch.randn(kernel_count, self.in_channels * self.kernel_area, self.out_channels))
            # 使用 Kaiming 初始化方法对卷积核进行初始化
            nn.init.kaiming_normal_(self.kernels, nonlinearity="relu")
        elif self.kernel_generator == "low_rank":
            # 定义一个 MLP 网络，作为卷积核生成的头部
            self.kernel_head = nn.Sequential(
                # 第一个全连接层，输入维度为输入通道数乘以卷积核面积，输出维度为 MLP 隐藏单元数
                nn.Linear(self.in_channels * self.kernel_area, mlp_inner_dims),
                # ReLU 激活函数
                nn.ReLU(),
                # 第二个全连接层，输入和输出维度均为 MLP 隐藏单元数
                nn.Linear(mlp_inner_dims, mlp_inner_dims),
                # ReLU 激活函数
                nn.ReLU(),
            )
            # 定义一个全连接层，用于将 MLP 输出转换为卷积核面积维度
            self.to_area = nn.Linear(mlp_inner_dims, self.kernel_area)
            # 定义一个全连接层，用于将 MLP 输出转换为输入通道维度
            self.to_cin = nn.Linear(mlp_inner_dims, self.in_channels)
            # 定义一个全连接层，用于将 MLP 输出转换为输出通道维度
            self.to_cout = nn.Linear(mlp_inner_dims, self.out_channels)
            # 定义可学习的卷积核参数
            self.kernels = nn.parameter.Parameter(
                torch.randn(self.in_channels, self.kernel_area, self.out_channels))
            # 使用 Kaiming 初始化方法对卷积核进行初始化
            nn.init.kaiming_normal_(self.kernels, nonlinearity="relu")
        else:
            # 如果卷积核生成方式不是指定的三种之一，抛出 ValueError 异常
            raise ValueError(
                "kernel_generator must be either 'spatial' or 'weighted_sum' or 'low_rank'")

        # 根据不同的偏置模式进行不同的初始化操作
        if bias == "cluster":
            # 当偏置模式为 "cluster" 时，需要为每个聚类生成偏置
            # 定义一个 MLP 网络，用于将聚类质心转换为偏置
            self.centroid_to_bias = nn.Sequential(
                # 第一个全连接层，输入维度为输入通道数乘以卷积核面积，输出维度为 MLP 隐藏单元数
                nn.Linear(in_features=self.in_channels * self.kernel_area,
                          out_features=mlp_inner_dims),
                # ReLU 激活函数，引入非线性
                nn.ReLU(),
                # 第二个全连接层，输入和输出维度均为 MLP 隐藏单元数
                nn.Linear(in_features=mlp_inner_dims,
                          out_features=mlp_inner_dims),
                # ReLU 激活函数，引入非线性
                nn.ReLU(),
                # 第三个全连接层，输入维度为 MLP 隐藏单元数，输出维度为输出通道数
                nn.Linear(in_features=mlp_inner_dims,
                          out_features=self.out_channels),
            )
        elif bias == "global_param":
            # 当偏置模式为 "global_param" 时，使用一个统一的偏置，类似于 nn.Conv2d 中的偏置
            # 定义可学习的全局偏置参数，形状为 (输出通道数,)
            self.bias = nn.parameter.Parameter(
                torch.randn(self.out_channels))
        elif bias == "global_adaptive":
            # 当偏置模式为 "global_adaptive" 时，生成全局自适应偏置，类似于 LAGConv 中的做法
            # 定义一个包含自适应平均池化、卷积层和激活函数的序列
            self.global_bias = nn.Sequential(
                # 自适应平均池化层，将输入特征图池化为 1x1 的大小
                nn.AdaptiveAvgPool2d(1),
                # 1x1 卷积层，输入通道数为输入通道数，输出通道数为输出通道数
                nn.Conv2d(in_channels, out_channels, 1),
                # ReLU 激活函数，引入非线性，inplace=True 表示直接在原张量上进行操作以节省内存
                nn.ReLU(inplace=True),
                # 1x1 卷积层，输入和输出通道数均为输出通道数
                nn.Conv2d(out_channels, out_channels, 1)
            )
        elif bias == "none":
            # 当偏置模式为 "none" 时，不使用偏置
            self.bias = None

    def generate_kernel(self, centroids: torch.Tensor):
        """
        Args:
            centroids: (batch_size, cluster_num, patch_dims)
        Returns:
            kernel_by_cluster: (batch_size, cluster_num, in_channels * kernel_area, out_channels)
        """
        # 根据不同的卷积核生成方式生成卷积核
        if self.kernel_generator == "spatial":
            # 使用 centroid_to_kernel 网络将聚类质心转换为空间权重
            # 并将输出形状从 (batch_size, cluster_num, area) 重排为 (batch_size, cluster_num, 1, area, 1)
            spatial_weights = rearrange(
                self.centroid_to_kernel(centroids), 'b k area -> b k 1 area 1')
            # 将空间权重与预定义的卷积核相乘
            # 然后将结果形状从 (batch_size, cluster_num, in_channels, area, out_channels)
            # 重排为 (batch_size, cluster_num, in_channels * area, out_channels)
            kernel_by_cluster = rearrange(
                spatial_weights * self.kernels, 'b k cin area cout -> b k (cin area) cout')
        elif self.kernel_generator == "weighted_sum":
            # 使用 centroid_to_kernel 网络将聚类质心转换为核权重
            # 并将输出形状从 (batch_size, cluster_num, n) 重排为 (batch_size, cluster_num, n, 1, 1)
            kernel_weights = rearrange(
                self.centroid_to_kernel(centroids), 'b k n -> b k n 1 1')
            # 将核权重与预定义的卷积核相乘，并对 n 维度求和
            # 得到形状为 (batch_size, cluster_num, in_channels * area, out_channels) 的卷积核
            kernel_by_cluster = reduce(
                kernel_weights * self.kernels, 'b k n cinarea cout -> b k cinarea cout', 'sum')
        else:
            # 当卷积核生成方式为 "low_rank" 时
            # 先将聚类质心输入 kernel_head 网络得到特征表示
            kf = self.kernel_head(centroids)
            # 将 kernel_head 的输出输入 to_cin 网络，经过 Sigmoid 激活函数后
            # 并将输出形状从 (batch_size, cluster_num, in_channels) 重排为 (batch_size, cluster_num, in_channels, 1, 1)
            w_cin = rearrange(F.sigmoid(self.to_cin(kf)),
                              'b k cin -> b k cin 1 1')
            # 将 kernel_head 的输出输入 to_area 网络，经过 Sigmoid 激活函数后
            # 并将输出形状从 (batch_size, cluster_num, area) 重排为 (batch_size, cluster_num, 1, area, 1)
            w_area = rearrange(F.sigmoid(self.to_area(kf)),
                               'b k area -> b k 1 area 1')
            # 将 kernel_head 的输出输入 to_cout 网络，经过 Sigmoid 激活函数后
            # 并将输出形状从 (batch_size, cluster_num, out_channels) 重排为 (batch_size, cluster_num, 1, 1, out_channels)
            w_cout = rearrange(F.sigmoid(self.to_cout(kf)),
                               'b k cout -> b k 1 1 cout')
            # 将三个权重与预定义的卷积核相乘
            kernel_by_cluster = (w_cin * w_area * w_cout) * self.kernels
            # 将结果形状从 (batch_size, cluster_num, in_channels, area, out_channels)
            # 重排为 (batch_size, cluster_num, in_channels * area, out_channels)
            kernel_by_cluster = rearrange(
                kernel_by_cluster, 'b k cin area cout -> b k (cin area) cout')

        return kernel_by_cluster

    def generate_bias(self, centroids: torch.Tensor, x):
        """
        Args:
            centroids: (batch_size, cluster_num, patch_dims)
        Returns:
            bias_by_cluster: (batch_size, cluster_num, out_channels)
        """
        # 根据不同的偏置模式生成偏置
        if self.bias_mode == "cluster":
            # 当偏置模式为 "cluster" 时，使用 centroid_to_bias 网络将聚类质心转换为偏置
            return self.centroid_to_bias(centroids)
        elif self.bias_mode == "global_param":
            # 当偏置模式为 "global_param" 时，直接返回全局偏置参数
            return self.bias
        elif self.bias_mode == "global_adaptive":
            # 当偏置模式为 "global_adaptive" 时，将输入 x 通过 global_bias 网络处理
            # 并将输出形状从 (batch_size, out_channels, 1, 1) 重排为 (batch_size, 1, out_channels)
            return rearrange(self.global_bias(x), 'b cout 1 1 -> b 1 cout')
        elif self.bias_mode == "none":
            # 当偏置模式为 "none" 时，不使用偏置，返回 None
            return None

    def downsample_to_cluster_feature(self, x: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patches: (batch_size, patch_num, patch_dims)
        Returns:
            res: (batch_size, patch_num, cluster_feature_dims)
        """
        # 根据不同的聚类源对输入的 patch 进行下采样以得到聚类特征
        if self.cluster_source == "channel":
            # 当聚类源为 "channel" 时，对每个 patch 在通道维度上求均值
            # 假设输入通道数为 cin，卷积核面积为 area，将 patch 从 (batch_size, patch_num, cin * area)
            # 下采样为 (batch_size, patch_num, cin)
            return reduce(patches, 'b s (cin area) -> b s cin', 'mean', cin=self.in_channels, area=self.kernel_area)
        elif self.cluster_source == "spatial":
            # 当聚类源为 "spatial" 时，对每个 patch 在空间维度上求均值
            # 假设卷积核面积为 area，将 patch 从 (batch_size, patch_num, cin * area)
            # 下采样为 (batch_size, patch_num, area)
            return reduce(patches, 'b s (cin area) -> b s area', 'mean', area=self.kernel_area)
        elif self.cluster_source == "pixel":
            # 当聚类源为 "pixel" 时，将输入 x 从 (batch_size, cin, h, w) 重排为 (batch_size, h * w, cin)
            return rearrange(x, 'b cin h w -> b (h w) cin')
        else:
            # 如果聚类源不是 "channel"、"spatial" 或 "pixel"，抛出 ValueError 异常
            raise ValueError(
                "cluster_source must be either 'channel', 'spatial', or 'pixel'")

    def convolution_by_cluster(self, patches: torch.Tensor, indice: torch.Tensor, weight: torch.Tensor, bias=None):
        """
        Args:
            patches: (batch_size, patch_num, patch_dims)
            indice: (batch_size, patch_num)
            weight: (batch_size, cluster_num, in_channels * kernel_area, out_channels)
            bias: (batch_size, cluster_num, out_channels)
        Returns:
            res: (batch_size, patch_num, out_channels)
        """
        # 获取输入 patches 的 batch_size
        b = patches.shape[0]
        # 获取权重的聚类数量
        k = weight.shape[1]

        # 将 patches 的形状从 (batch_size, patch_num, patch_dims) 重排为 (batch_size * patch_num, patch_dims)
        patches = rearrange(patches, "b s f -> (b s) f")
        # 将权重的形状从 (batch_size, cluster_num, in_channels * kernel_area, out_channels)
        # 重排为 (batch_size * cluster_num, in_channels * kernel_area, out_channels)
        weight = rearrange(weight, "b k f cout -> (b k) f cout")
        # 对 indice 进行处理，为每个 batch 内的元素加上偏移量，以确保每个元素的索引唯一
        indice = indice + torch.arange(b, device=indice.device).view(-1, 1) * k
        # 将 indice 的形状从 (batch_size, patch_num) 重排为 (batch_size * patch_num)
        indice = rearrange(indice, "b hw -> (b hw)")
        if bias is not None:
            # 如果存在偏置，将偏置的形状从 (batch_size, cluster_num, out_channels)
            # 重排为 (batch_size * cluster_num, out_channels)
            bias = rearrange(bias, "b k cout -> (b k) cout")

        # 调用 dispatch_indice 函数对 indice 进行处理，得到一些中间结果
        indice_perm, padded_patch_num, cluster_size_sorted, permuted_offset, cluster_perm, batch_height = dispatch_indice(
            indice, b * k)
        # 调用 permute 函数对 patches 进行重排
        input_permuted = permute(patches, indice_perm, padded_patch_num)
        # 调用 batched_matmul_conv 函数进行批量矩阵乘法卷积操作
        output_permuted = batched_matmul_conv(
            input_permuted, weight, permuted_offset, cluster_perm, batch_height, bias)
        # 调用 inverse_permute 函数对输出进行逆重排
        output = inverse_permute(output_permuted, indice_perm)

        # 将输出的形状从 (batch_size * patch_num, out_channels) 重排为 (batch_size, patch_num, out_channels)
        return rearrange(output, "(b hw) cout -> b hw cout", b=b)

    def cluster_ablation_global_forward(self, x: torch.Tensor):
        # 获取输入 x 的形状信息
        b, cin, h, w = x.shape
        # 使用 unfold 操作从输入 x 中提取局部块
        patches = self.unfold(x)
        # 将提取的局部块的形状从 (batch_size, in_channels * kernel_area, h * w)
        # 重排为 (batch_size, h * w, in_channels * kernel_area)
        patches = rearrange(
            patches, 'b (cin area) (h w) -> b (h w) (cin area)', area=self.kernel_area, h=h, w=w)
        # 对所有 patch 求均值，得到全局的质心，形状为 (batch_size, 1, patch_dims)
        centroids = reduce(patches, 'b s f -> b 1 f', 'mean')
        # 根据全局质心生成卷积核
        kernel = self.generate_kernel(centroids)
        # 根据全局质心生成偏置
        bias = self.generate_bias(centroids, x)
        # 进行矩阵乘法并加上偏置
        result = torch.matmul(patches, rearrange(
            kernel, 'b 1 f cout -> b f cout')) + bias
        # 将结果的形状从 (batch_size, h * w, out_channels) 重排为 (batch_size, out_channels, h, w)
        return rearrange(result, 'b (h w) cout -> b cout h w', h=h, w=w), torch.zeros(b, h * w, dtype=torch.long,
                                                                                      device=x.device)

    def cluster_ablation_pixelwise_forward(self, x: torch.Tensor):
        # 获取输入 x 的形状信息
        b, cin, h, w = x.shape
        # 使用 unfold 操作从输入 x 中提取局部块
        patches = self.unfold(x)
        # 将提取的局部块的形状从 (batch_size, in_channels * kernel_area, h * w)
        # 重排为 (batch_size, h * w, in_channels * kernel_area)
        patches = rearrange(
            patches, 'b (cin area) (h w) -> b (h w) (cin area)', area=self.kernel_area, h=h, w=w)
        # 根据每个 patch 生成卷积核
        kernel = self.generate_kernel(patches)
        # 根据每个 patch 生成偏置
        bias = self.generate_bias(patches, x)
        # 进行矩阵乘法并加上偏置
        result = torch.matmul(rearrange(patches, 'b s f -> b s 1 f'),
                              kernel) + rearrange(bias, 'b s cout -> b s 1 cout')
        # 将结果的形状从 (batch_size, h * w, 1, out_channels) 重排为 (batch_size, out_channels, h, w)
        return rearrange(result, 'b (h w) 1 cout -> b cout h w', h=h, w=w), repeat(torch.arange(h * w, device=x.device),
                                                                                   's -> b s', b=b)

    # When cluster_override is given, the module will not perform clustering and use the given indice instead
    # When cluster_override is not persent and cache_indice is given, the module will try to use the cached indice
    # When both cluster_override and cache_indice are not present, the module will always perform clustering
    # The second return value is the indice used for clustering
    def forward(self, x: torch.Tensor, cache_indice=None, cluster_override=None):
        # 若聚类消融模式为 "global"，调用全局聚类消融前向传播方法
        if self.cluster_ablation == "global":
            return self.cluster_ablation_global_forward(x)
        # 若聚类消融模式为 "pixelwise"，调用逐像素聚类消融前向传播方法
        elif self.cluster_ablation == "pixelwise":
            return self.cluster_ablation_pixelwise_forward(x)

        # 获取输入张量 x 的批量大小
        batch_size = x.shape[0]
        # 获取输入张量 x 的通道数
        in_channels = x.shape[1]
        # 获取输入张量 x 的高度
        height = x.shape[2]
        # 获取输入张量 x 的宽度
        width = x.shape[3]

        # Step 1: 将输入 x 展开为 patch 并进行聚类
        # 使用 unfold 操作从输入 x 中提取局部块
        patches = self.unfold(x)
        # 将提取的局部块的形状从 (batch_size, in_channels * kernel_area, height * width)
        # 重排为 (batch_size, height * width, in_channels * kernel_area)
        patches = rearrange(
            patches, 'b (cin area) (h w) -> b (h w) (cin area)', area=self.kernel_area, h=height, w=width)

        # 若传入了自定义的聚类索引，则使用该索引
        if cluster_override is not None:
            cluster_indice = cluster_override
        else:
            # 否则，使用 KMeans 算法对降采样后的特征进行聚类，得到聚类索引
            cluster_indice = self.kmeans(self.downsample_to_cluster_feature(
                x, patches), cache_indice=cache_indice)
            # 若滤波器阈值大于 0，对聚类索引进行过滤
            if self.filter_threshold > 0:
                cluster_indice = filter_indice(
                    cluster_indice, self.cluster_num, self.filter_threshold).to(x.device)

        # Step 2: 计算每个聚类的质心
        # 调用 get_cluster_centers 函数计算每个聚类的质心
        centroids = get_cluster_centers(
            patches, cluster_indice, self.cluster_num + 1 if self.filter_threshold > 0 else self.cluster_num)
        # 若滤波器阈值大于 0，计算全局质心并赋值给最后一个聚类的质心
        if self.filter_threshold > 0:
            global_center = reduce(patches, 'b s f -> b f', 'mean')
            centroids[:, self.cluster_num, :] = global_center

        # 若需要分离质心，将质心从计算图中分离
        if self.detatch_centroid:
            centroids = centroids.detach()
        # 注释掉的代码，可能用于对质心进行进一步处理
        # centroids = reduce(centroids, 'b k (cin area) -> b k cin',
        #                    'mean', cin=self.in_channels, area=self.kernel_area)

        # Step 3: 从每个质心生成卷积核
        # 调用 generate_kernel 方法根据质心生成卷积核
        kernel_by_cluster = self.generate_kernel(centroids)
        # 调用 generate_bias 方法根据质心生成偏置
        bias = self.generate_bias(centroids, x)

        # Step 4: 应用卷积操作
        # 若偏置模式为 "cluster"，在卷积时传入偏置
        if self.bias_mode == "cluster":
            result = self.convolution_by_cluster(
                patches, cluster_indice, kernel_by_cluster, bias)
        else:
            # 否则，不传入偏置进行卷积
            result = self.convolution_by_cluster(
                patches, cluster_indice, kernel_by_cluster)
            # 若偏置模式为 "global_param" 或 "global_adaptive"，在卷积结果上加上偏置
            if self.bias_mode == "global_param" or self.bias_mode == "global_adaptive":
                result += bias
        # 将卷积结果的形状从 (batch_size, height * width, out_channels) 重排为 (batch_size, out_channels, height, width)
        return rearrange(result, 'b (h w) cout -> b cout h w', h=height, w=width), cluster_indice

def test_kmconv_layer():
    dev = torch.device("cuda:0")
    module = CANConv(32, 32).to(dev)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(
            wait=2,
            warmup=2,
            active=6,
            repeat=1),
        record_shapes=True,
        with_stack=True
    ) as prof:
        for step in range(10):
            with record_function('single_run'):
                x = torch.randn(1, 32, 64, 64, device=dev)
                y = module(x)
                # print(y.shape)
            prof.step()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    prof.export_chrome_trace("trace.json")


if __name__ == "__main__":
    test_kmconv_layer()
