# train_plain_vit.py

import torch
import torch.nn as nn
import torch.optim as optim

from datasets.cifar10 import get_cifar10_dataloaders
from models.components.vit_core import create_plain_vit_cifar10
from engine.trainer import train_classification_model


def train_plain_vit():
    data_dir = "../data"
    batch_size = 128
    num_workers = 4
    num_epochs = 50
    lr = 3e-4
    weight_decay = 5e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = get_cifar10_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        val_ratio=0.1,
        train_ratio=1.0,
    )
    del test_loader  # 训练阶段不用测试集

    model = create_plain_vit_cifar10().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs
    )

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
        exp_name="plain_vit_cifar10",
        save_best=True,
    )

    return history


if __name__ == "__main__":
    train_plain_vit()
