# models/ours_drspt_vit_main.py
"""
Ours-DRSPT-ViT 主模型：
- 使用 vit_core.py 里的 PatchEmbed / ViTEncoder / LearnablePositionalEmbedding
- 使用 drspt_modules.py 里的 LearnableShiftViews / DynamicRoutedSPT / ReliabilityTopKHead
不在这里重复实现任何基础模块。
"""

from typing import Tuple

import torch
import torch.nn as nn

from models.components.vit_core import (
    PatchEmbed,
    ViTEncoder,
    LearnablePositionalEmbedding,
)
from models.components.drspt_modules import (
    LearnableShiftViews,
    DynamicRoutedSPT,
    ReliabilityTopKHead,
)


class DRSPTViT(nn.Module):
    """
    DRSPT-ViT 主体模型（你的 Ours）

    结构：
      x (B,3,H,W)
        -> LearnableShiftViews: 生成 V 个平移视图
        -> PatchEmbed (共享): 每个视图 -> patch tokens
        -> DynamicRoutedSPT: 多视图动态路由聚合，得到
             c: [B,N,D]   聚合后的 patch tokens
             a: [B,V,N]   视图权重
             r: [B,N]     patch 可靠性
        -> LearnablePositionalEmbedding
        -> ViTEncoder: 标准 Transformer encoder（无 RAG 改动）
        -> ReliabilityTopKHead: 利用 r 做 Top-K 可靠 token 聚合，输出 logits
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
        num_views: int = 5,
        drspt_iters: int = 1,
        topk_ratio: float = 0.25,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.num_views = num_views
        self.drspt_iters = drspt_iters
        self.topk_ratio = topk_ratio

        # 1) 可学习视图平移
        self.shift_module = LearnableShiftViews(num_views=num_views)

        # 2) 共享 Patch Embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.num_patches = self.patch_embed.num_patches
        self.grid_size = self.patch_embed.grid_size  # 记录下来，后面正则会用到

        # 3) DR-SPT 动态路由聚合
        self.drspt = DynamicRoutedSPT(
            embed_dim=embed_dim,
            num_views=num_views,
            num_iters=drspt_iters,
        )

        # 4) 位置编码 + ViT Encoder（标准版）
        self.pos_embed = LearnablePositionalEmbedding(
            num_patches=self.num_patches,
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
            norm_layer=nn.LayerNorm,
        )

        # 5) 可靠性引导 Top-K 分类头
        self.head = ReliabilityTopKHead(
            embed_dim=embed_dim,
            num_classes=num_classes,
            topk_ratio=topk_ratio,
            min_topk=1,
        )

    # ---------------- 公共特征提取 ----------------
    def forward_features(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: [B, 3, H, W]

        返回:
            tokens_enc: [B, N, D]  Encoder 输出 tokens
            r         : [B, N]     patch 可靠性
            a         : [B, V, N]  视图权重
        """
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"输入尺寸 {H}x{W} 与 img_size={self.img_size} 不一致"

        # (1) 生成多视图图像
        views = self.shift_module(x)  # list of V tensors, each [B, C, H, W]
        assert len(views) == self.num_views

        # (2) 每个视图做 Patch Embedding
        tokens_list = [self.patch_embed(v) for v in views]  # V 个 [B,N,D]
        x_mv = torch.stack(tokens_list, dim=1)              # [B, V, N, D]

        # (3) DR-SPT 聚合
        c, a, r = self.drspt(x_mv)                          # c:[B,N,D], a:[B,V,N], r:[B,N]

        # (4) 位置编码 + Encoder
        tokens = self.pos_embed(c)                          # [B, N, D]
        tokens_enc = self.encoder(tokens)                   # [B, N, D]

        return tokens_enc, r, a

    # ---------------- 主 forward ----------------
    def forward(self, x: torch.Tensor, return_aux: bool = False):
        """
        x: [B, 3, H, W]
        return_aux:
            False -> 只返回 logits
            True  -> 返回 (logits, a, r) 以便在 loss 里加正则或做可视化
        """
        tokens_enc, r, a = self.forward_features(x)
        logits = self.head(tokens_enc, reliability=r)

        if return_aux:
            return logits, a, r
        return logits


def create_drspt_vit_cifar10() -> DRSPTViT:
    """
    工厂函数：创建 CIFAR-10 用的 Ours-DRSPT-ViT
    """
    model = DRSPTViT(
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
        num_views=5,
        drspt_iters=1,
        topk_ratio=0.25,
    )
    return model
