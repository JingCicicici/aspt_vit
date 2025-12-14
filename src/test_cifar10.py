# test_cifar10.py
import argparse
import os
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_model(model_name: str):
    """
    构建模型实例（不加载权重）。
    为了适配你当前项目里“文件位置可能变动”的情况，这里做了 import fallback。
    """
    if model_name == "resnet18":
        # 兼容两种位置：models/resnet18_baseline.py 或 models/baselines/resnet18_baseline.py
        try:
            from models.resnet18_baseline import create_resnet18_baseline
        except Exception:
            from models.baselines.resnet18_baseline import create_resnet18_baseline
        return create_resnet18_baseline(num_classes=10)

    if model_name == "plain_vit":
        # 兼容两种位置：models/vit_plain_cifar10.py 或 models/baselines/vit_plain_cifar10.py
        try:
            from models.vit_plain_cifar10 import create_vit_plain_cifar10
        except Exception:
            from models.baselines.vit_plain_cifar10 import create_vit_plain_cifar10
        return create_vit_plain_cifar10()

    if model_name == "ours_full":
        from models.methods.ours_drspt_vit_ablation import create_ours_full_cifar10
        return create_ours_full_cifar10()

    if model_name == "ours_no_shift":
        from models.methods.ours_drspt_vit_ablation import create_ours_no_shift_cifar10
        return create_ours_no_shift_cifar10()

    if model_name == "ours_no_drspt":
        from models.methods.ours_drspt_vit_ablation import create_ours_no_drspt_cifar10
        return create_ours_no_drspt_cifar10()

    if model_name == "ours_no_head":
        from models.methods.ours_drspt_vit_ablation import create_ours_no_reliability_head_cifar10
        return create_ours_no_reliability_head_cifar10()

    raise ValueError(f"Unknown model: {model_name}")


def load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> None:
    """
    兼容多种保存格式：
    1) 直接 torch.save(model.state_dict(), path)
    2) torch.save({"model": state_dict, ...}, path)
    3) torch.save({"state_dict": state_dict, ...}, path)
    """
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict):
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            state = ckpt["model"]
        elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state = ckpt["state_dict"]
        else:
            state = ckpt
    else:
        state = ckpt

    model.load_state_dict(state, strict=True)


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader, device: torch.device, criterion) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_num = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        pred = logits.argmax(dim=1)
        total_correct += (pred == labels).sum().item()
        total_num += images.size(0)

    avg_loss = total_loss / max(total_num, 1)
    avg_acc = total_correct / max(total_num, 1)
    return avg_loss, avg_acc


def _resolve_cifar10_root(data_dir: str) -> str:
    """
    解析 CIFAR-10 的 root 目录。
    torchvision 会在 root 下寻找 cifar-10-batches-py/。
    """
    data_dir = os.path.abspath(data_dir)
    marker = os.path.join(data_dir, "cifar-10-batches-py")
    if os.path.isdir(marker):
        return data_dir

    # 如果用户给的目录不对，给出明确报错，避免误触发下载
    raise FileNotFoundError(
        f"在 data_dir={data_dir} 下未找到 cifar-10-batches-py/。\n"
        f"请把 --data_dir 指到包含该文件夹的目录。\n"
        f"例如你在 src/ 下运行，通常应当是：--data_dir ../data"
    )


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.abspath(os.path.join(here, "..", "data"))  # = /.../aspt_vit/data

    parser = argparse.ArgumentParser("CIFAR-10 Test-Only Evaluation (no leakage)")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["resnet18", "plain_vit", "ours_full", "ours_no_shift", "ours_no_drspt", "ours_no_head"],
    )
    parser.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint .pth")

    # 关键改动：默认指向项目根目录 data（src/ 下运行时就是 ../data）
    parser.add_argument("--data_dir", type=str, default=default_data_dir)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    # 默认 ckpt 命名（你也可以手动 --ckpt 指定）
    default_ckpt = {
        "resnet18": "checkpoints/resnet18_cifar10_best.pth",
        "plain_vit": "checkpoints/plain_vit_cifar10_best.pth",
    }.get(args.model, None)

    ckpt_path = args.ckpt or default_ckpt
    if not ckpt_path:
        raise ValueError("请通过 --ckpt 指定要评估的权重文件路径（该模型没有默认 ckpt 名称）。")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model).to(device)
    load_checkpoint(model, ckpt_path, device)

    # 只加载 TEST split（train=False），不加载 train/val，避免任何误触发下载/泄露
    data_root = _resolve_cifar10_root(args.data_dir)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ])

    test_dataset = datasets.CIFAR10(
        root=data_root,
        train=False,
        download=False,  # 测试脚本不下载，目录不对就直接报错
        transform=transform_test,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, test_loader, device, criterion)

    print(f"[TEST-ONLY] model={args.model} ckpt={ckpt_path}")
    print(f"[TEST-ONLY] data_dir={data_root}")
    print(f"[TEST-ONLY] Loss={test_loss:.4f}  Acc={test_acc:.4f}")


if __name__ == "__main__":
    main()
