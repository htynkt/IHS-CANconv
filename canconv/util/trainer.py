from abc import ABCMeta, abstractmethod
import os
from glob import glob
import time
import shutil
import logging
from datetime import datetime
import json
import inspect
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import torch
import torch.utils.data
from tqdm import tqdm

from .seed import seed_everything
from .git import git, get_git_commit
from .log import BufferedReporter, to_rgb
from ..dataset.h5pan import H5PanDataset

class SimplePanTrainer(metaclass=ABCMeta):
    cfg: dict
    
    model: torch.nn.Module
    criterion: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    
    train_dataset: H5PanDataset
    val_dataset: H5PanDataset
    test_dataset: H5PanDataset
    
    train_loader: DataLoader
    val_loader: DataLoader
    
    out_dir: str
    
    disable_alloc_cache: bool
    
    @abstractmethod
    def forward(self, batch):
        raise NotImplementedError
    
    @abstractmethod
    def _create_model(self, cfg):
        raise NotImplementedError
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.logger = logging.getLogger(f"canconv.{cfg['exp_name']}")
        self.logger.setLevel(logging.INFO)
        seed_everything(cfg["seed"])
        self.logger.info(f"Seed set to {cfg['seed']}")
        
        self.dev = torch.device(cfg['device'])
        if self.dev.type != "cuda":
            raise ValueError(f"Only cuda device is supported, got {self.dev.type}")
        if self.dev.index != 0:
            self.logger.warning("Warning: Multi-GPU is not supported, the code may not work properly with GPU other than cuda:0. Please use CUDA_VISIBLE_DEVICES to select the device.")
            torch.cuda.set_device(self.dev)
            
        self.logger.info(f"Using device: {self.dev}")
        self._create_model(cfg)
        self.forward({
            'gt': torch.randn(cfg['batch_size'], cfg['spectral_num'], 64, 64),
            'ms': torch.randn(cfg['batch_size'], cfg['spectral_num'], 16, 16),
            'lms': torch.randn(cfg['batch_size'], cfg['spectral_num'], 64, 64),
            'pan': torch.randn(cfg['batch_size'], 1, 64, 64)
        })
        self.disable_alloc_cache = cfg.get("disable_alloc_cache", False)
        self.logger.info(f"Model loaded.")
        
    def _load_dataset(self):
        self.train_dataset = H5PanDataset(self.cfg["train_data"])
        self.val_dataset = H5PanDataset(self.cfg["val_data"])
        self.test_dataset = H5PanDataset(self.cfg["test_reduced_data"])
        
    def _create_output_dir(self):
        self.out_dir = os.path.join('runs', self.cfg["exp_name"])
        os.makedirs(os.path.join(self.out_dir, 'weights'), exist_ok=True)
        logging.info(f"Output dir: {self.out_dir}")
    def _dump_config(self):
        with open(os.path.join(self.out_dir, "cfg.json"), "w") as file:
            self.cfg["git_commit"] = get_git_commit()
            self.cfg["run_time"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S%z")
            json.dump(self.cfg, file, indent=4)
            
        try:
            source_path = inspect.getsourcefile(self.__class__)
            assert source_path is not None
            source_path = os.path.dirname(source_path)
            shutil.copytree(source_path, os.path.join(self.out_dir, "source"), ignore=shutil.ignore_patterns('*.pyc', '__pycache__'), dirs_exist_ok=True)
        except Exception as e:
            self.logger.warning(f"Failed to copy source code: ")
            self.logger.exception(e)
            
    def _on_train_start(self):
        pass
    
    def _on_val_start(self):
        pass
    
    def _on_epoch_start(self, epoch):
        pass
    
    @torch.no_grad()
    def run_test(self, dataset: H5PanDataset):
        # 将模型设置为评估模式，关闭 Dropout 和 BatchNorm 等训练特有的操作
        self.model.eval()

        # 创建一个全零张量 sr，用于存储模型的输出结果
        # sr 的形状与数据集中的低分辨率多光谱图像 (lms) 的形状一致
        # 但高度和宽度与全色图像 (pan) 的高度和宽度一致
        sr = torch.zeros(
            dataset.lms.shape[0],  # 数据集的样本数量
            dataset.lms.shape[1],  # 低分辨率多光谱图像的通道数
            dataset.pan.shape[2],  # 全色图像的高度
            dataset.pan.shape[3],  # 全色图像的宽度
            device=self.dev  # 将张量放置在指定的设备上（如 GPU）
        )

        # 遍历数据集中的每个样本
        for i in range(len(dataset)):
            # 对当前样本进行前向传播，获取模型的输出
            # dataset[i:i+1] 获取第 i 个样本，并保持批次维度
            sr[i:i + 1] = self.forward(dataset[i:i + 1])

        # 返回模型的输出结果
        return sr

    @torch.no_grad()
    def run_test_for_selected_image(self, dataset, image_ids):
        self.model.eval()
        sr = torch.zeros(
            len(image_ids), dataset.lms.shape[1], dataset.pan.shape[2], dataset.pan.shape[3], device=self.dev)
        for i, image_id in enumerate(image_ids):
            sr[i:i+1] = self.forward(dataset[image_id:image_id+1])
        return sr

    def train(self):
        # 加载数据集
        self._load_dataset()
        # 创建训练数据加载器
        train_loader = DataLoader(
            dataset=self.train_dataset,  # 使用训练数据集
            batch_size=self.cfg['batch_size'],  # 每个批次的大小
            shuffle=True,  # 每个 epoch 打乱数据
            drop_last=False,  # 不丢弃最后一个不完整的批次
            pin_memory=True  # 使用 pin_memory 提高数据传输效率
        )
        # 创建验证数据加载器
        val_loader = DataLoader(
            dataset=self.val_dataset,  # 使用验证数据集
            batch_size=self.cfg['batch_size'],  # 每个批次的大小
            shuffle=True,  # 每个 epoch 打乱数据
            drop_last=False,  # 不丢弃最后一个不完整的批次
            pin_memory=True  # 使用 pin_memory 提高数据传输效率
        )
        # 记录日志，表示数据集已加载
        self.logger.info(f"Dataset loaded.")

        # 创建输出目录
        self._create_output_dir()
        # 将配置信息保存到文件
        self._dump_config()
        # 调用训练开始前的钩子函数
        self._on_train_start()

        # 创建 TensorBoard 的 SummaryWriter，用于记录训练过程中的日志
        writer = SummaryWriter(log_dir=self.out_dir)
        # 创建用于记录训练损失、验证损失、训练时间和验证时间的 BufferedReporter
        train_loss = BufferedReporter(f'train/{self.criterion.__class__.__name__}', writer)
        val_loss = BufferedReporter(f'val/{self.criterion.__class__.__name__}', writer)
        train_time = BufferedReporter('train/time', writer)
        val_time = BufferedReporter('val/time', writer)

        # 记录日志，表示开始训练
        self.logger.info(f"Begin Training.")

        # 遍历每个 epoch
        for epoch in tqdm(range(1, self.cfg['epochs'] + 1, 1)):
            # 调用每个 epoch 开始前的钩子函数
            self._on_epoch_start(epoch)

            # 将模型设置为训练模式
            self.model.train()
            # 遍历训练数据加载器中的每个批次
            for batch in tqdm(train_loader):
                # 记录当前批次的开始时间
                start_time = time.time()

                # 清空模型的梯度
                self.model.zero_grad()
                # 前向传播，获取模型的输出
                sr = self.forward(batch)
                # 计算损失函数
                loss = self.criterion(sr, batch['gt'].to(self.dev))
                # 将损失值添加到 BufferedReporter 中
                train_loss.add_scalar(loss.item())
                # 反向传播
                loss.backward()
                # 更新模型参数
                self.optimizer.step()

                # 如果启用了内存分配缓存清理，则清空 CUDA 缓存
                if self.disable_alloc_cache:
                    torch.cuda.empty_cache()

                # 记录当前批次的训练时间
                train_time.add_scalar(time.time() - start_time)
            # 将训练损失和训练时间的记录刷新到 TensorBoard
            train_loss.flush(epoch)
            train_time.flush(epoch)
            # 调用学习率调度器
            self.scheduler.step()
            # 记录日志，表示当前 epoch 的训练已完成
            self.logger.debug(f"Epoch {epoch} train done")

            # 如果当前 epoch 是验证间隔的整数倍
            if epoch % self.cfg['val_interval'] == 0:
                # 调用验证开始前的钩子函数
                self._on_val_start()
                # 将模型设置为评估模式
                with torch.no_grad():
                    self.model.eval()
                    # 遍历验证数据加载器中的每个批次
                    for batch in val_loader:
                        # 记录当前批次的开始时间
                        start_time = time.time()
                        # 前向传播，获取模型的输出
                        sr = self.forward(batch)
                        # 计算损失函数
                        loss = self.criterion(sr, batch['gt'].to(self.dev))
                        # 将损失值添加到 BufferedReporter 中
                        val_loss.add_scalar(loss.item())
                        # 记录当前批次的验证时间
                        val_time.add_scalar(time.time() - start_time)
                    # 将验证损失和验证时间的记录刷新到 TensorBoard
                    val_loss.flush(epoch)
                    val_time.flush(epoch)
                # 记录日志，表示当前 epoch 的验证已完成
                self.logger.debug(f"Epoch {epoch} val done")

            # 如果当前 epoch 是检查点间隔的整数倍，或者需要保存第一个 epoch 的权重
            if epoch % self.cfg['checkpoint'] == 0 or (
                    "save_first_epoch" in self.cfg and epoch <= self.cfg["save_first_epoch"]):
                # 保存模型权重到指定路径
                torch.save(self.model.state_dict(), os.path.join(
                    self.out_dir, f'weights/{epoch}.pth'))
                # 记录日志，表示当前 epoch 的检查点已保存
                self.logger.info(f"Epoch {epoch} checkpoint saved")

        # 保存最终的模型权重
        torch.save(self.model.state_dict(), os.path.join(self.out_dir, "weights/final.pth"))
        # 记录日志，表示训练已完成
        self.logger.info(f"Training finished.")