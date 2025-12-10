# datasets/cifar10.py
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_cifar10_dataloaders(
    data_dir: str = "../data",
    batch_size: int = 128,
    num_workers: int = 4,
):
    """
    返回 CIFAR-10 的训练集和测试集 DataLoader
    """
    # 训练集数据增强：随机裁剪 + 随机水平翻转 + 归一化
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ])

    # 测试集：不做随机增强，只做 ToTensor + 归一化
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ])

    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=False,           # 第一次会自动下载
        transform=transform_train,
    )

    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=False,
        transform=transform_test,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
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

    return train_loader, test_loader
