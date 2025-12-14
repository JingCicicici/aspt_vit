# 兼容旧导入名：models.baselines.resnet18_baseline
from .resnet18_cifar10 import create_resnet18_baseline, create_resnet18_cifar10

__all__ = ["create_resnet18_baseline", "create_resnet18_cifar10"]
