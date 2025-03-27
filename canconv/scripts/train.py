# 导入 logging 模块，用于日志记录
import logging
# 导入 os 模块，用于操作系统相关的功能
import os
# 导入 importlib 模块，用于动态导入模块
import importlib
# 导入 argparse 模块，用于解析命令行参数
import argparse
# 导入 json 模块，用于处理 JSON 数据
import json
# 导入 sys 模块，用于访问与 Python 解释器强相关的变量和函数
import sys
# 导入 time 模块，用于获取当前时间
import time
# 从 canconv.util.log 模块中导入 save_mat_data 函数
import torch

from canconv.util.log import save_mat_data


# 定义主函数 main，接收模型名称、配置、是否保存矩阵、预设配置和覆盖配置等参数
def main(model_name, resume_from,cfg=None, save_mat=True, preset=None, override=None):
    # 动态导入指定模型名称对应的模块，模块路径为 canconv.models 下的子模块
    module = importlib.import_module(f"canconv.models.{model_name}")

    # 如果没有传入配置文件，则使用模块中默认的 cfg 配置
    if cfg is None:
        cfg = module.cfg
    # 如果指定了预设配置名称
    if preset is not None:
        # 打开预设配置文件 presets.json 并加载其内容
        with open("presets.json", 'r') as f:
            presets = json.load(f)
        # 将预设配置与当前配置合并，并更新实验名称以包含预设名称
        cfg = cfg | presets[preset]
        cfg["exp_name"] += f'_{preset}'
    # 如果指定了覆盖配置
    if override is not None:
        # 将覆盖配置（以 JSON 格式传入）加载为字典，并与当前配置合并
        cfg = cfg | json.loads(override)

    # 使用配置创建模型对应的 Trainer 对象
    trainer = module.Trainer(cfg)

    # 如果指定了预训练模型权重文件路径，则加载权重
    if resume_from is not None:
        logging.info(f"Loading weights from {resume_from}")
        trainer.model.load_state_dict(torch.load(resume_from, map_location=torch.device('cuda'), weights_only=True))

    # 调用 Trainer 的 train 方法开始训练
    trainer.train()

    # 如果需要保存矩阵数据
    if save_mat:
        # 在测试数据集上运行模型并获取结果
        sr = trainer.run_test(trainer.test_dataset)
        # 调用 save_mat_data 函数保存矩阵数据，传入结果、数据集缩放比例和输出目录
        save_mat_data(sr, trainer.test_dataset.scale, trainer.out_dir)


if __name__ == "__main__":
    # 创建一个名为 "logs" 的目录，如果目录已存在，则不会报错（exist_ok=True）
    os.makedirs("logs", exist_ok=True)

    # 配置日志系统，设置日志的基本格式和输出方式
    logging.basicConfig(
        # 设置日志记录的最低级别为 INFO
        level=logging.INFO,
        # 定义日志的格式，包括时间、日志级别和日志消息
        format="%(asctime)s %(levelname)s %(message)s",
        # 定义日志的输出处理器
        handlers=[
            # 将日志写入到文件中，文件名包含当前时间戳和进程 ID
            logging.FileHandler(
                f"logs/train_{int(time.time())}_{os.getpid()}.log"),
            # 同时将日志输出到标准输出（即控制台）
            logging.StreamHandler(sys.stdout)
        ]
    )

    # 记录一条日志，内容为脚本被调用时的命令行参数
    logging.info(f"Train script invoked with args: {sys.argv}")

    # 创建一个命令行参数解析器
    parser = argparse.ArgumentParser()
    # 添加一个位置参数 model_name，表示模型名称，必须提供
    parser.add_argument("model_name", type=str)
    # 添加一个可选的位置参数 preset，默认值为 None
    parser.add_argument("preset", nargs='?', type=str, default=None)
    # 添加一个可选参数 --cfg，用于指定配置文件路径，默认值为 None
    parser.add_argument("--cfg", type=str, default=None)
    # 添加一个可选参数 --save_mat，默认值为 True，表示是否保存矩阵
    parser.add_argument("--save_mat", type=bool, default=True)
    # 添加一个可选参数 --override，用于指定覆盖选项，默认值为 None
    parser.add_argument("--override", type=str, default=None)
    parser.add_argument("--resume_from", type=str, default='runs/cannet_wv3/weights/300.pth')  # 添加新的参数

    # 解析命令行参数
    args = parser.parse_args()

    # 初始化配置变量 cfg 为 None
    cfg = None
    # 如果指定了配置文件路径
    if args.cfg is not None:
        # 打开配置文件并加载 JSON 数据到 cfg 变量中
        with open(args.cfg, 'r') as f:
            cfg = json.load(f)

    # 尝试调用 main 函数，传入解析后的参数
    try:
        main(args.model_name, args.resume_from, cfg, args.save_mat, args.preset, args.override)
    # 如果发生异常，记录异常信息并重新抛出异常
    except Exception as e:
        logging.exception(e)
        raise e