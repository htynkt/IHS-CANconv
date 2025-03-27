import torch
import torch.nn as nn

# 从当前目录下的 model.py 文件中导入 CANNet 类
from .model import MultiStageIHS

# 从 canconv 包的 util 子包中的 trainer.py 文件里导入 SimplePanTrainer 类
from canconv.util.trainer import SimplePanTrainer
# 从 canconv 包的 layers 子包中的 kmeans.py 文件里导入 reset_cache 函数和 KMeansCacheScheduler 类
from canconv.layers.kmeans import reset_cache, KMeansCacheScheduler


class CANNetTrainer(SimplePanTrainer):
    # 类的构造函数，接收配置参数 cfg
    def __init__(self, cfg) -> None:
        # 调用父类 SimplePanTrainer 的构造函数
        super().__init__(cfg)

    # 创建模型、损失函数、优化器和调度器的方法
    def _create_model(self, cfg):
        # 根据配置文件中的 loss 参数选择损失函数
        if cfg["loss"] == "l1":
            # 如果选择 l1 损失，创建 L1 损失函数并将其移动到指定设备上
            self.criterion = nn.L1Loss(reduction='mean').to(self.dev)
        elif cfg["loss"] == "l2":
            # 如果选择 l2 损失，创建均方误差损失函数并将其移动到指定设备上
            self.criterion = nn.MSELoss(reduction='mean').to(self.dev)
        else:
            # 如果配置文件中的 loss 参数不是 l1 或 l2，抛出未实现的错误
            raise NotImplementedError(f"Loss {cfg['loss']} not implemented")
        # 创建 CANNet 模型，并将其移动到指定设备上
        self.model = MultiStageIHS(cfg['spectral_num'], cfg['channels'],
                            cfg['cluster_num'], cfg["filter_threshold"]).to(self.dev)
        # 创建 Adam 优化器，用于更新模型的参数
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=cfg["learning_rate"], weight_decay=0)
        # 创建学习率调度器，根据指定的步长更新学习率
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=cfg["lr_step_size"])

        # 创建 KMeans 缓存调度器，用于管理 KMeans 缓存的更新
        self.km_scheduler = KMeansCacheScheduler(cfg['kmeans_cache_update'])

    # 在训练开始前执行的方法
    def _on_train_start(self):
        # 重置 KMeans 缓存，传入训练数据集的长度
        reset_cache(len(self.train_dataset))

    # 在每个 epoch 开始前执行的方法
    def _on_epoch_start(self, epoch):
        # 调用 KMeans 缓存调度器的 step 方法，更新 KMeans 缓存
        self.km_scheduler.step()

    # 前向传播方法，定义模型的前向计算过程
    def forward(self, data):
        # 检查输入数据中是否包含 'index' 键
        if "index" in data:
            # 如果包含 'index' 键，将数据移动到指定设备上并传入模型进行前向计算
            return self.model(data['pan'].to(self.dev), data['lms'].to(self.dev), data['index'].to(self.dev))
        else:
            # 如果不包含 'index' 键，只将 'pan' 和 'lms' 数据移动到指定设备上并传入模型进行前向计算
            return self.model(data['pan'].to(self.dev), data['lms'].to(self.dev))