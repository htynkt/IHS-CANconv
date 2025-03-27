import numpy as np
import torch


def objective_function(alpha, multispectral_image, panchromatic_image, gamma):
    multispectral_image = multispectral_image[0].detach().cpu().numpy()
    panchromatic_image = panchromatic_image[0].detach().cpu().numpy()
    multispectral_image=np.transpose(multispectral_image, (1, 2, 0))
    panchromatic_image = np.squeeze(panchromatic_image, axis=0)
    I = np.dot(multispectral_image[..., :3], alpha[:3]) + alpha[3] * multispectral_image[..., 3]  # 计算亮度分量
    residual = np.sum((I - panchromatic_image) ** 2)  # 计算残差
    penalty = np.sum(np.maximum(0, -alpha) ** 2)  # 计算惩罚项
    return residual + gamma * penalty  # 返回目标函数值

def match_histograms(panchromatic_image, intensity_image):
    if isinstance(panchromatic_image, torch.Tensor):
        panchromatic_image = panchromatic_image.detach().cpu().numpy()
    if isinstance(intensity_image, torch.Tensor):
        intensity_image = intensity_image.detach().cpu().numpy()
        
    mean_I = np.mean(intensity_image)  # 计算亮度图像的均值
    std_I = np.std(intensity_image)  # 计算亮度图像的标准差
    mean_P = np.mean(panchromatic_image)  # 计算全色图像的均值
    std_P = np.std(panchromatic_image)  # 计算全色图像的标准差
    matched_P = (std_I / std_P) * (panchromatic_image - mean_P) + mean_I  # 匹配直方图
    return matched_P

