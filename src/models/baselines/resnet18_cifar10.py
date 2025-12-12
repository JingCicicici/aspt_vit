# models/resnet18_baseline.py
import torch.nn as nn
from torchvision.models import resnet18


def create_resnet18_baseline(num_classes: int = 10):
    """
    构建一个用于 CIFAR-10 的 ResNet-18 分类模型
    不使用预训练权重，从头训练。
    """
    try:
        # 适配 torchvision 新版本的接口
        model = resnet18(weights=None)
    except TypeError:
        # 旧版本接口
        model = resnet18(pretrained=False)

    # 替换最后一层全连接，使输出类别数等于 num_classes
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
# 兼容旧名字（如果你不想改 train_resnet18.py 也可以用这个）
def create_resnet18_baseline(num_classes: int = 10):
    return create_resnet18_cifar10(num_classes)