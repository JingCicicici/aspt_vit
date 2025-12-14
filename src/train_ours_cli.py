# train_ours_cli.py

import os
import argparse
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from datasets.cifar10 import get_cifar10_dataloaders
from models.components.drspt_modules import (
    view_entropy_loss,
    reliability_smoothness_loss,
)


def build_model(args):
    if args.model == "ours_full":
        from models.methods.ours_drspt_vit_main import create_drspt_vit_cifar10
        model = create_drspt_vit_cifar10()

    elif args.model == "ours_no_shift":
        from models.methods.ours_drspt_vit_ablation import create_ours_no_shift_cifar10
        model = create_ours_no_shift_cifar10()

    elif args.model == "ours_no_drspt":
        from models.methods.ours_drspt_vit_ablation import create_ours_no_drspt_cifar10
        model = create_ours_no_drspt_cifar10()

    elif args.model == "ours_no_head":
        from models.methods.ours_drspt_vit_ablation import create_ours_no_reliability_head_cifar10
        model = create_ours_no_reliability_head_cifar10()

    elif args.model == "plain_vit":
        from models.baselines.vit_plain_cifar10 import create_vit_plain_cifar10
        model = create_vit_plain_cifar10()

    elif args.model == "resnet18":
        from models.baselines.resnet18_baseline import create_resnet18_baseline
        model = create_resnet18_baseline(num_classes=10)

    else:
        raise ValueError(f"Unknown model type: {args.model}")

    return model


@torch.no_grad()
def evaluate(model, data_loader, device, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def mixup_batch(x, y, alpha: float):
    """
    返回 mixed_x, y_a, y_b, lam
    """
    if alpha <= 0:
        return x, y, y, 1.0

    # 用 torch 的 Beta，避免依赖 numpy
    beta = torch.distributions.Beta(alpha, alpha)
    lam = float(beta.sample().item())

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1.0 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_loss(criterion, pred, y_a, y_b, lam: float):
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="ours_full",
        choices=["ours_full", "ours_no_shift", "ours_no_drspt", "ours_no_head", "plain_vit", "resnet18"],
    )

    # 数据相关
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--train_ratio", type=float, default=1.0)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)

    # 强增强（对抗 ViT 过拟合的关键）
    parser.add_argument("--strong_aug", action="store_true", help="启用强增强（RandAugment/AutoAugment + RandomErasing）")
    parser.add_argument("--randaug_n", type=int, default=2)
    parser.add_argument("--randaug_m", type=int, default=9)
    parser.add_argument("--erasing_p", type=float, default=0.25)

    # 训练超参
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)

    # Label smoothing（抗过拟合，低成本高收益）
    parser.add_argument("--label_smoothing", type=float, default=0.0, help="CrossEntropy label_smoothing，建议 0.1")

    # Mixup（强力抗过拟合）
    parser.add_argument("--mixup_alpha", type=float, default=0.0, help=">0 启用 mixup，建议 0.2~0.4")
    parser.add_argument("--mixup_prob", type=float, default=1.0, help="mixup 的使用概率")

    # 可选：梯度裁剪（稳定训练）
    parser.add_argument("--grad_clip_norm", type=float, default=0.0, help=">0 启用 clip_grad_norm_，建议 1.0")

    # 正则项（你的 view/smooth）
    parser.add_argument("--no_reg", action="store_true", help="不使用 view/smooth 正则")
    parser.add_argument("--lambda_view", type=float, default=1e-2)
    parser.add_argument("--lambda_smooth", type=float, default=1e-2)

    # 早停（你观测到 30-40 epoch 后过拟合，这个非常必要）
    parser.add_argument("--early_stop_patience", type=int, default=0, help=">0 启用早停，建议 10~20")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(
        f"Model={args.model} train_ratio={args.train_ratio} val_ratio={args.val_ratio} "
        f"no_reg={args.no_reg} strong_aug={args.strong_aug} data_dir={args.data_dir}"
    )

    # 数据加载
    train_loader, val_loader, test_loader = get_cifar10_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        train_ratio=args.train_ratio,
        seed=args.seed,
        strong_aug=args.strong_aug,
        randaug_n=args.randaug_n,
        randaug_m=args.randaug_m,
        erasing_p=args.erasing_p,
        download=True,  # 若你确定数据一定存在，也可改成 False
    )

    # 模型
    model = build_model(args).to(device)

    # Loss：加入 label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 你的正则系数：baseline 或 no_reg 时强制为 0
    lambda_view = args.lambda_view
    lambda_smooth = args.lambda_smooth
    if args.no_reg or args.model in ["plain_vit", "resnet18"]:
        lambda_view = 0.0
        lambda_smooth = 0.0

    exp_name = (
        f"{args.model}_ratio{args.train_ratio}_"
        f"aug{'strong' if args.strong_aug else 'weak'}_"
        f"ls{args.label_smoothing}_mix{args.mixup_alpha}_"
        f"noreg{args.no_reg}"
    )
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = f"checkpoints/{exp_name}_best.pth"

    best_val_acc = -1.0
    best_epoch = -1
    no_improve = 0

    # 训练
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = running_ce = running_reg = 0.0
        correct = total = 0

        loop = tqdm(train_loader, desc=f"[{exp_name}] Epoch {epoch}/{args.epochs}", ncols=110)

        for images, labels in loop:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Mixup（按概率启用）
            use_mix = (args.mixup_alpha > 0) and (torch.rand(1).item() < args.mixup_prob)
            if use_mix:
                images, y_a, y_b, lam = mixup_batch(images, labels, alpha=args.mixup_alpha)

            # 前向：ours 支持 return_aux
            if args.model.startswith("ours"):
                outputs = model(images, return_aux=True)
            else:
                outputs = model(images)

            if isinstance(outputs, tuple):
                logits, a, r = outputs
            else:
                logits = outputs
                a, r = None, None

            # CE / Mixup CE
            if use_mix:
                ce = mixup_loss(criterion, logits, y_a, y_b, lam)
                # mixup 下训练 acc 不再严格可解释，这里用 y_a 仅用于观察趋势
                labels_for_acc = y_a
            else:
                ce = criterion(logits, labels)
                labels_for_acc = labels

            # view/smooth 正则（可选）
            reg = 0.0
            if (lambda_view > 0 or lambda_smooth > 0) and (a is not None and r is not None):
                L_view = view_entropy_loss(a)
                L_smooth = reliability_smoothness_loss(r, grid_size=(8, 8))
                reg = lambda_view * L_view + lambda_smooth * L_smooth

            loss = ce + reg

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if args.grad_clip_norm and args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm)

            optimizer.step()

            bs = images.size(0)
            running_loss += loss.item() * bs
            running_ce += ce.item() * bs
            running_reg += float(reg) * bs

            _, preds = logits.max(1)
            correct += preds.eq(labels_for_acc).sum().item()
            total += bs

            loop.set_postfix({
                "loss": running_loss / max(total, 1),
                "ce": running_ce / max(total, 1),
                "reg": running_reg / max(total, 1),
                "acc(~)": correct / max(total, 1) if args.mixup_alpha > 0 else correct / max(total, 1),
            })

        scheduler.step()

        # 验证
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        print(
            f"[{exp_name}] Epoch {epoch}: "
            f"Train Loss={running_loss/max(total,1):.4f}, Train Acc={correct/max(total,1):.4f}, "
            f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}"
        )

        # 保存 best
        if val_acc > best_val_acc + 1e-6:
            best_val_acc = val_acc
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"  >> New best val acc={best_val_acc:.4f} (epoch {best_epoch}), saved to {ckpt_path}")
        else:
            no_improve += 1

        # Early stopping
        if args.early_stop_patience and args.early_stop_patience > 0:
            if no_improve >= args.early_stop_patience:
                print(
                    f"[{exp_name}] Early stopping triggered at epoch {epoch}. "
                    f"Best Val Acc={best_val_acc:.4f} at epoch {best_epoch}."
                )
                break

    print(f"[{exp_name}] Training done. Best Val Acc={best_val_acc:.4f} at epoch {best_epoch}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Best checkpoint not found: {ckpt_path}")

    # TEST（加载 best）
    best_model = build_model(args).to(device)
    best_model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_loss, test_acc = evaluate(best_model, test_loader, device, criterion)
    print(f"[{exp_name}] TEST Loss={test_loss:.4f}, Acc={test_acc:.4f}")


if __name__ == "__main__":
    main()
