# test_drspt_vit.py

import torch
import torch.nn as nn

from datasets.cifar10 import get_cifar10_dataloaders
from models.methods.ours_drspt_vit_main import create_drspt_vit_cifar10
from train_drspt_vit import evaluate


def test_drspt_vit():
    data_dir = "../data"
    batch_size = 128
    num_workers = 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 这里重新拿到 test_loader，仍然不会动 train/val
    _, _, test_loader = get_cifar10_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        val_ratio=0.1,
        train_ratio=1.0,
    )

    # 创建模型并加载最优验证集 checkpoint
    model = create_drspt_vit_cifar10().to(device)
    ckpt_path = "checkpoints/drspt_vit_cifar10_best.pth"
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Loaded checkpoint from {ckpt_path}")

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, test_loader, device, criterion)
    print(f"[TEST] Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")


if __name__ == "__main__":
    test_drspt_vit()
