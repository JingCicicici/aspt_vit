# datasets/cifar10.py

from typing import Tuple
import math

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_cifar10_dataloaders(
    data_dir: str = "../data",
    batch_size: int = 128,
    num_workers: int = 4,
    val_ratio: float = 0.1,
    train_ratio: float = 1.0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    返回 CIFAR-10 的 train / val / test 三个 DataLoader，
    严格保证测试集只在训练完成后使用，不参与任何超参选择。

    参数说明:
        data_dir   : 数据存放路径
        batch_size : batch 大小
        num_workers: DataLoader 的 num_workers
        val_ratio  : 从 (训练部分) 中划出多少比例作为验证集 (例如 0.1 -> 训练:验证 = 9:1)
        train_ratio: 使用多少比例的训练数据 (1.0 表示全部 50k，0.1 表示只用 10% 做小样本实验)
        seed       : 用于划分 train/val 的随机种子，保证可复现

    返回:
        train_loader, val_loader, test_loader
    """

    # ----------------------
    # 1. 定义数据增强 & 预处理
    # ----------------------
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ])

    # ----------------------
    # 2. 加载原始 train / test 数据集
    # ----------------------
    full_train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,       # 第一次会自动下载，之后会直接复用
        transform=transform_train,
    )

    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transform_test,
    )

    num_total = len(full_train_dataset)  # CIFAR-10 训练集是 50000

    # ----------------------
    # 3. 按 train_ratio 抽取一个子集，用于后续 train+val
    #    例如 train_ratio=0.1 -> 只用 5000 张图做 train+val
    # ----------------------
    if not (0.0 < train_ratio <= 1.0):
        raise ValueError(f"train_ratio 应该在 (0, 1] 之间，当前为 {train_ratio}")

    num_used = int(math.floor(num_total * train_ratio))
    if num_used <= 0:
        raise ValueError("train_ratio 太小，导致没有样本可用")

    # 用固定随机种子打乱索引，保证可复现
    g = torch.Generator()
    g.manual_seed(seed)
    indices = torch.randperm(num_total, generator=g).tolist()
    used_indices = indices[:num_used]

    # ----------------------
    # 4. 在 used_indices 中划分 train / val
    # ----------------------
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio 应该在 (0, 1) 之间，当前为 {val_ratio}")

    num_val = int(math.floor(num_used * val_ratio))
    num_train = num_used - num_val
    if num_train <= 0 or num_val <= 0:
        raise ValueError(
            f"train_ratio={train_ratio}, val_ratio={val_ratio} 导致 train/val 数量非法："
            f"num_train={num_train}, num_val={num_val}"
        )

    train_indices = used_indices[:num_train]
    val_indices = used_indices[num_train:num_used]

    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_train_dataset, val_indices)

    # ----------------------
    # 5. 构造 DataLoader
    # ----------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,        # 训练集需要打乱
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,       # 验证/测试一般不 shuffle
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
