# engine/trainer.py

from typing import Tuple, Dict, Optional

import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
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


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    num_epochs: int,
) -> Tuple[float, float]:
    """
    单个 epoch 的训练过程，返回 (平均训练 loss, 训练集准确率)
    """
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

    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc


def train_classification_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    num_epochs: int = 50,
    work_dir: str = "checkpoints",
    exp_name: str = "model",
    save_best: bool = True,
) -> Dict[str, list]:
    """
    通用的分类模型训练入口，只在 train+val 上训练和调参，不碰测试集。

    返回一个 history 字典，可以用来画 loss/acc 曲线：
      {
        "train_loss": [...],
        "train_acc": [...],
        "val_loss": [...],
        "val_acc": [...],
      }
    """
    os.makedirs(work_dir, exist_ok=True)
    best_acc = 0.0
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(1, num_epochs + 1):
        # 1) 训练一个 epoch
        train_loss, train_acc = train_one_epoch(
            model=model,
            train_loader=train_loader,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            num_epochs=num_epochs,
        )

        # 2) 在验证集上评估
        val_loss, val_acc = evaluate(
            model=model,
            data_loader=val_loader,
            device=device,
            criterion=criterion,
        )

        # 3) 学习率调度器（如果有的话）
        if scheduler is not None:
            scheduler.step()

        print(
            f"[Epoch {epoch}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        # 4) 记录历史
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # 5) 根据验证集性能保存 best model
        if save_best and val_acc > best_acc:
            best_acc = val_acc
            ckpt_path = os.path.join(work_dir, f"{exp_name}_best.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  >> New best Val Acc: {best_acc:.4f}, saved to {ckpt_path}")

    return history
