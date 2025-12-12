# train_drspt_vit.py

import os

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from datasets.cifar10 import get_cifar10_dataloaders
from models.methods.ours_drspt_vit_main import create_drspt_vit_cifar10
from models.components.drspt_modules import (
    view_entropy_loss,
    reliability_smoothness_loss,
)


@torch.no_grad()
def evaluate(model, data_loader, device, criterion):
    """
    只用主任务交叉熵在 val / test 上评估，不加正则项。
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        # 验证/测试阶段不需要 return_aux
        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        _, preds = logits.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    avg_acc = correct / total
    return avg_loss, avg_acc


def train_drspt_vit():
    # ----------------- 基本配置 -----------------
    data_dir = "../data"
    batch_size = 128
    num_workers = 4

    num_epochs = 100
    lr = 3e-4
    weight_decay = 5e-4

    # 正则系数（可以后面做消融）
    lambda_view = 1e-2
    lambda_smooth = 1e-2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ----------------- 数据加载 -----------------
    # 严格三划分：train / val / test
    train_loader, val_loader, test_loader = get_cifar10_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        val_ratio=0.1,   # 10% 训练数据作为验证集
        train_ratio=1.0, # 先用满数据做 baseline，小样本可以改成 0.1 之类
    )

    # ----------------- 模型、损失、优化器 -----------------
    model = create_drspt_vit_cifar10().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs
    )

    # ----------------- 训练循环（只看 train / val） -----------------
    best_val_acc = 0.0
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = "checkpoints/drspt_vit_cifar10_best.pth"

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        running_ce_loss = 0.0
        running_reg_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch}/{num_epochs}]", ncols=100)

        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device)

            # -------- 前向：这里一定要 return_aux=True 拿到 a, r --------
            logits, a, r = model(images, return_aux=True)

            # 主任务交叉熵
            ce = criterion(logits, labels)

            # 视图熵正则项（鼓励 router 有主见）
            L_view = view_entropy_loss(a)

            # 可靠性平滑正则项（鼓励 r 在空间上平滑）
            # CIFAR-10: img_size=32, patch_size=4 => 8 x 8 patch 网格
            L_smooth = reliability_smoothness_loss(r, grid_size=(8, 8))

            reg = lambda_view * L_view + lambda_smooth * L_smooth
            loss = ce + reg

            # -------- 反向 & 更新 --------
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # -------- 统计训练指标 --------
            running_loss += loss.item() * images.size(0)
            running_ce_loss += ce.item() * images.size(0)
            running_reg_loss += reg.item() * images.size(0)

            _, preds = logits.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

            loop.set_postfix({
                "loss": running_loss / total if total > 0 else 0.0,
                "ce": running_ce_loss / total if total > 0 else 0.0,
                "reg": running_reg_loss / total if total > 0 else 0.0,
                "acc": correct / total if total > 0 else 0.0,
            })

        scheduler.step()

        # -------- 每个 epoch 结束后，在验证集上评估 --------
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        print(
            f"[Epoch {epoch}] "
            f"Train Loss: {running_loss / total:.4f}, "
            f"Train Acc: {correct / total:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        # -------- 按验证集精度选 best model，保存权重 --------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)
            print(f"  >> New best val acc: {best_val_acc:.4f}, saved to {ckpt_path}")

    print(f"Training finished. Best val acc = {best_val_acc:.4f}")
    print("后面在单独的 test 脚本里，用这个 best checkpoint 在测试集上评估。")

    # 方便 test 脚本复用 evaluate，可以返回 test_loader
    return test_loader


if __name__ == "__main__":
    train_drspt_vit()
