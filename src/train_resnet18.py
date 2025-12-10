# train_resnet18.py
import os
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from datasets.cifar10 import get_cifar10_dataloaders
from models.resnet18_baseline import create_resnet18_baseline


def evaluate(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> Tuple[float, float]:
    """
    在验证/测试集上评估模型，返回 (平均 loss, 准确率)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc


def train_resnet18():
    # ----------------- 基本配置 -----------------
    data_dir = "../data"
    batch_size = 128
    num_workers = 4
    num_epochs = 2         # 先设 10，看情况可以缩短 / 延长
    lr = 0.1
    weight_decay = 5e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ----------------- 数据加载 -----------------
    train_loader, test_loader = get_cifar10_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # ----------------- 模型、损失、优化器 -----------------
    model = create_resnet18_baseline(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay,
    )

    # 学习率调度器（可选，先简单用一个）
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    best_acc = 0.0

    # ----------------- 训练循环 -----------------
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch}/{num_epochs}]", ncols=100)

        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device)

            # 前向
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向 & 更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计
            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

            loop.set_postfix({
                "train_loss": running_loss / total if total > 0 else 0.0,
                "train_acc": correct / total if total > 0 else 0.0,
            })

        scheduler.step()

        # -------- 每个 epoch 之后在测试集上评估 ----------
        test_loss, test_acc = evaluate(model, test_loader, device, criterion)
        print(
            f"[Epoch {epoch}] "
            f"Train Loss: {running_loss / total:.4f}, Train Acc: {correct / total:.4f}, "
            f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}"
        )

        # 保存最好模型（可选）
        if test_acc > best_acc:
            best_acc = test_acc
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/resnet18_cifar10_best.pth")
            print(f"  >> New best accuracy: {best_acc:.4f}, model saved.")


if __name__ == "__main__":
    train_resnet18()
