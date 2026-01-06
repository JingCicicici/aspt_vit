# models/vit_core.py
"""
ViT 基础模块：
- PatchEmbed: 图像 -> patch token
- MultiHeadSelfAttention: 标准多头自注意力
- MLP: 前馈网络
- ViTEncoderBlock / ViTEncoder: Transformer 编码器堆叠
- LearnablePositionalEmbedding: 可学习位置编码
- PlainViTClassifier: 简单的 ViT 分类器 (CIFAR-10 baseline)
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    """Stochastic Depth per sample"""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # (B,1,1,...)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = random_tensor.floor()
        return x.div(keep_prob) * random_tensor


# ---------------------------
# 1. Patch Embedding
# ---------------------------


class PatchEmbed(nn.Module):
    """
    图像切 patch 并映射到 embed 向量空间。

    输入:
        x: [B, C, H, W]

    输出:
        tokens: [B, N, D]
            其中 N = (H / patch_size) * (W / patch_size)
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 384,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.embed_dim = embed_dim

        # 用一个 Conv2d 实现“切 patch + 线性映射”
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"输入图像尺寸 {H}x{W} 与预设 img_size={self.img_size} 不一致"

        x = self.proj(x)              # [B, D, H', W']
        x = x.flatten(2)              # [B, D, N]
        x = x.transpose(1, 2)         # [B, N, D]
        return x


# ---------------------------
# 2. Multi-Head Self-Attention
# ---------------------------

class MultiHeadSelfAttention(nn.Module):
    """
    标准多头自注意力，输入输出形状相同 [B, N, D]
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, "embed dim 必须能被 head 数整除"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        # qkv: [B, N, 3 * C]
        qkv = self.qkv(x)
        # reshape -> [3, B, num_heads, N, head_dim]
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # 注意这里使用 (q, k, v) 的顺序

        # 注意力权重: [B, num_heads, N, N]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # 注意力输出: [B, num_heads, N, head_dim]
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, C)  # [B, N, C]

        out = self.proj(out)
        out = self.proj_drop(out)
        return out


# ---------------------------
# 3. MLP / FFN
# ---------------------------

class MLP(nn.Module):
    """
    Transformer Block 里的前馈网络 FFN
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ---------------------------
# 4. Transformer Encoder Block
# ---------------------------

class ViTEncoderBlock(nn.Module):
    """
    标准 ViT 编码器 Block:
        x -> LN -> MSA -> 残差
        x -> LN -> MLP -> 残差
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mlp_drop: float = 0.0,
        drop_path: float = 0.0,              # ✅ 新增
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MultiHeadSelfAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.drop_path1 = DropPath(drop_path)  # ✅ 新增

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim=dim, hidden_dim=mlp_hidden_dim, dropout=mlp_drop)
        self.drop_path2 = DropPath(drop_path)  # ✅ 新增

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.attn(self.norm1(x)))  # ✅ 包起来
        x = x + self.drop_path2(self.mlp(self.norm2(x)))   # ✅ 包起来
        return x


class ViTEncoder(nn.Module):
    """
    多层堆叠的 ViT 编码器，不包含 patch embedding 和分类头。
    """

    def __init__(
        self,
        depth: int,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mlp_drop: float = 0.0,
        drop_path: float = 0.0,              # ✅ 新增：最大 drop_path
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()

        # ✅ 每层从 0 线性增加到 drop_path（DeiT/ConvNeXt 常用做法）
        dpr = torch.linspace(0.0, float(drop_path), steps=depth).tolist()

        self.layers = nn.ModuleList([
            ViTEncoderBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                mlp_drop=mlp_drop,
                drop_path=dpr[i],              # ✅ 每层不同
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])
        self.norm = norm_layer(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.layers:
            x = blk(x)
        x = self.norm(x)
        return x


# ---------------------------
# 5. Learnable Positional Embedding
# ---------------------------

class LearnablePositionalEmbedding(nn.Module):
    """
    简单的可学习 1D 位置编码，不使用 CLS token。
    """

    def __init__(self, num_patches: int, embed_dim: int):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, D], 其中 N 应等于 num_patches
        """
        return x + self.pos_embed


# ---------------------------
# 6. Plain ViT Classifier (baseline)
# ---------------------------

class PlainViTClassifier(nn.Module):
    """
    一个简单的 ViT 分类器作为 baseline:
      - PatchEmbed
      - LearnablePositionalEmbedding
      - ViTEncoder
      - Global Average Pooling + FC

    主要用于:
      1) 与 ResNet18 做 baseline 对比
      2) 与 DRSPT-ViT 做对比，体现你的改进效果
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_chans: int = 3,
        num_classes: int = 10,
        embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mlp_drop: float = 0.0,
        drop_path: float = 0.0,  # ✅ 新增
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.pos_embed = LearnablePositionalEmbedding(
            num_patches=num_patches,
            embed_dim=embed_dim,
        )

        self.encoder = ViTEncoder(
            depth=depth,
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            mlp_drop=mlp_drop,
            drop_path=drop_path,  # ✅ 新增
            norm_layer=nn.LayerNorm,
        )

        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        tokens = self.patch_embed(x)       # [B, N, D]
        tokens = self.pos_embed(tokens)    # [B, N, D]
        tokens = self.encoder(tokens)      # [B, N, D]

        # 使用 GAP 聚合所有 patch token
        feat = tokens.mean(dim=1)          # [B, D]
        logits = self.head(feat)           # [B, num_classes]
        return logits


def create_plain_vit_cifar10() -> PlainViTClassifier:
    """
    工厂函数：创建一个适用于 CIFAR-10 的 Plain ViT baseline。
    你可以在 train_xxx.py 里直接调用这个函数。
    """
    model = PlainViTClassifier(
        img_size=32,
        patch_size=4,
        in_chans=3,
        num_classes=10,
        embed_dim=384,
        depth=6,
        num_heads=6,
        mlp_ratio=4.0,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        mlp_drop=0.0,
    )
    return model
