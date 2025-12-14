# datasets/cifar10.py

from typing import Tuple
import math

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets as tv_datasets, transforms


def _build_transforms(
    strong_aug: bool = False,
    randaug_n: int = 2,
    randaug_m: int = 9,
    erasing_p: float = 0.25,
):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    if not strong_aug:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        return transform_train, transform_test

    aug_list = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]

    # 尽量兼容不同 torchvision 版本
    if hasattr(transforms, "RandAugment"):
        aug_list.append(transforms.RandAugment(num_ops=randaug_n, magnitude=randaug_m))
    else:
        if hasattr(transforms, "AutoAugment") and hasattr(transforms, "AutoAugmentPolicy"):
            aug_list.append(transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10))
        elif hasattr(transforms, "TrivialAugmentWide"):
            aug_list.append(transforms.TrivialAugmentWide())
        # 否则退化为仅 crop+flip

    aug_list += [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]

    if hasattr(transforms, "RandomErasing") and erasing_p > 0:
        aug_list.append(
            transforms.RandomErasing(
                p=erasing_p,
                scale=(0.02, 0.2),
                ratio=(0.3, 3.3),
                value="random",
            )
        )

    transform_train = transforms.Compose(aug_list)
    return transform_train, transform_test


def get_cifar10_dataloaders(
    data_dir: str = "../data",
    batch_size: int = 128,
    num_workers: int = 4,
    val_ratio: float = 0.1,
    train_ratio: float = 1.0,
    seed: int = 42,
    strong_aug: bool = False,
    randaug_n: int = 2,
    randaug_m: int = 9,
    erasing_p: float = 0.25,
    download: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    transform_train, transform_test = _build_transforms(
        strong_aug=strong_aug,
        randaug_n=randaug_n,
        randaug_m=randaug_m,
        erasing_p=erasing_p,
    )

    # train/val 都来自 train=True，但 transform 不同（val 不含随机增强）
    full_train_dataset = tv_datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=download,
        transform=transform_train,
    )

    full_val_dataset = tv_datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=False,
        transform=transform_test,
    )

    # test 严格独立：train=False
    test_dataset = tv_datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=download,
        transform=transform_test,
    )

    num_total = len(full_train_dataset)  # 50000

    if not (0.0 < train_ratio <= 1.0):
        raise ValueError(f"train_ratio 应该在 (0, 1] 之间，当前为 {train_ratio}")
    num_used = int(math.floor(num_total * train_ratio))
    if num_used <= 0:
        raise ValueError("train_ratio 太小，导致没有样本可用")

    g = torch.Generator()
    g.manual_seed(seed)
    indices = torch.randperm(num_total, generator=g).tolist()
    used_indices = indices[:num_used]

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
    val_dataset = Subset(full_val_dataset, val_indices)

    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    return train_loader, val_loader, test_loader
