# train_resnet18.py

import torch
import torch.nn as nn
import torch.optim as optim

from datasets.cifar10 import get_cifar10_dataloaders
from models.resnet18_baseline import create_resnet18_baseline
from engine.trainer import train_classification_model


def train_resnet18():
    # ----------------- 基本配置 -----------------
    # 如果脚本在 src/ 下运行，数据在项目根目录的 data/ 下，写成 "../data"
    data_dir = "../data"
    batch_size = 128
    num_workers = 4
    num_epochs = 60        # ResNet 可以适当长一点
    lr = 0.1
    weight_decay = 5e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ----------------- 数据加载 -----------------
    # 注意：新版 get_cifar10_dataloaders 返回 train/val/test 三个 loader
    train_loader, val_loader, test_loader = get_cifar10_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        val_ratio=0.1,      # 例如 10% 训练数据做验证
        train_ratio=1.0,    # 用满 100% 训练集
    )

    # 这里的 test_loader 不在训练过程中使用，只留给单独 test 脚本用
    del test_loader

    # ----------------- 模型、损失、优化器 -----------------
    model = create_resnet18_baseline(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay,
    )

    # 学习率调度器（每 30 个 epoch 衰减一次）
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # ----------------- 调用通用训练引擎 -----------------
    history = train_classification_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs,
        work_dir="checkpoints",
        exp_name="resnet18_cifar10",
        save_best=True,
    )

    # 如果你以后想画 train/val 曲线，可以把 history 保存成 json / npy / pth 等
    return history


if __name__ == "__main__":
    train_resnet18()
