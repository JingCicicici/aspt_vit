# models/methods/ours_drspt_vit_ablation.py
"""
DR-SPT-ViT（带开关的消融版）:

这个文件的目标：
    - 提供一个带开关的 DRSPT-ViT 主干模型，用于做各种消融实验；
    - 不去污染你已有的主方法文件 `ours_drspt_vit_main.py`；
    - 通过不同的工厂函数创建：
        * 完整体 ours_full
        * 去掉可学习视图 ours_no_shift
        * 去掉动态路由 ours_no_drspt
        * 去掉可靠性头 ours_no_reliability_head

依赖模块：
    - PatchEmbed / ViTEncoder / LearnablePositionalEmbedding 来自 vit_core
    - LearnableShiftViews / FixedShiftViews / DynamicRoutedSPT / SimpleViewAggregator
      / ReliabilityTopKHead 来自 drspt_modules
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
    FixedShiftViews,
    DynamicRoutedSPT,
    SimpleViewAggregator,
    ReliabilityTopKHead,
)


class DRSPTViT_Ablation(nn.Module):
    """
    DR-SPT-ViT（可配置版）:

    结构（概念上）：
        图像 x
          -> 多视图模块（可学习平移 or 固定平移）
          -> PatchEmbed（逐视图）
          -> 多视图聚合模块（DR-SPT or 简单均值聚合）
          -> 1D 位置编码
          -> ViT Encoder
          -> 分类头（可靠性 Top-K 头 or 普通 GAP+FC）

    通过若干 boolean 开关实现消融：
        - use_learnable_shift:    True  -> LearnableShiftViews
                                  False -> FixedShiftViews
        - use_drspt:              True  -> DynamicRoutedSPT
                                  False -> SimpleViewAggregator
        - use_reliability_head:   True  -> ReliabilityTopKHead
                                  False -> 普通 GAP + Linear

    注意：
        - forward(x, return_aux=False)   -> 只返回 logits
        - forward(x, return_aux=True)    -> 返回 (logits, a, r)
          其中 a 为视图权重 [B, V, N]，r 为 patch 可靠性 [B, N]，
          方便在训练代码中添加正则项、可视化等。
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_chans: int = 3,
        num_classes: int = 10,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        num_views: int = 5,
        num_iters: int = 3,
        topk_ratio: float = 0.25,
        use_learnable_shift: bool = True,
        use_drspt: bool = True,
        use_reliability_head: bool = True,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mlp_drop: float = 0.0,
        drop_path: float = 0.0,  # ✅ 新增：stochastic depth 最大值

    ):
        super().__init__()

        self.img_size = img_size
        self.num_views = num_views
        self.use_reliability_head = use_reliability_head
        self.num_patches = num_patches

        # -----------------------------
        # 1) 多视图模块: Learnable vs Fixed
        # -----------------------------
        if use_learnable_shift:
            self.shift_module = LearnableShiftViews(num_views=num_views)
        else:
            # 简单非可学习平移视图（与你原始 SPT 行为一致）
            self.shift_module = FixedShiftViews(num_views=num_views)

        # -----------------------------
        # 2) Patch Embedding
        # -----------------------------
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        # ✅ 给 smoothness 正则用：训练脚本会 getattr(model, "grid_size", None)
        gs = self.patch_embed.grid_size
        self.grid_size = (gs, gs)
        # -----------------------------
        # 3) 多视图聚合: DR-SPT vs 简单平均
        # -----------------------------
        if use_drspt:
            self.drspt = DynamicRoutedSPT(
                embed_dim=embed_dim,
                num_views=num_views,
                num_iters=num_iters,
            )
        else:
            self.drspt = SimpleViewAggregator(num_views=num_views)

        # -----------------------------
        # 4) 1D 可学习位置编码 + ViT Encoder
        # -----------------------------
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

        # -----------------------------
        # 5) 分类头: Reliability-TopK vs GAP+FC
        # -----------------------------
        if use_reliability_head:
            self.head = ReliabilityTopKHead(
                embed_dim=embed_dim,
                num_classes=num_classes,
                topk_ratio=topk_ratio,
                min_topk=1,
            )
        else:
            # 退化版：普通 GAP + Linear
            self.head = nn.Linear(embed_dim, num_classes)

    # -----------------------------------------------------
    # feature 提取：返回 encoder 输出 + 可靠性 r + 视图权重 a
    # -----------------------------------------------------
    def forward_features(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: [B, 3, H, W]

        返回:
            tokens_enc: [B, N, D]  ViT Encoder 输出 tokens
            r         : [B, N]     patch 可靠性
            a         : [B, V, N]  视图权重
        """
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"输入尺寸 {H}x{W} 与 img_size={self.img_size} 不一致"

        # (1) 生成多视图图像
        views = self.shift_module(x)  # list of V tensors, each [B, C, H, W]
        assert len(views) == self.num_views, \
            f"视图数量不匹配: 得到 {len(views)}, 预期 {self.num_views}"

        # (2) 每个视图做 Patch Embedding
        tokens_list = [self.patch_embed(v) for v in views]  # V 个 [B, N, D]
        x_mv = torch.stack(tokens_list, dim=1)              # [B, V, N, D]

        # (3) 视图聚合: DR-SPT 或 简单平均
        c, a, r = self.drspt(x_mv)                          # c:[B,N,D], a:[B,V,N], r:[B,N]

        # (4) 位置编码 + ViT Encoder
        tokens = self.pos_embed(c)                          # [B, N, D]
        tokens_enc = self.encoder(tokens)                   # [B, N, D]

        return tokens_enc, r, a

    # -----------------------------------------------------
    # 主 forward：支持 return_aux 控制是否返回 (logits, a, r)
    # -----------------------------------------------------
    def forward(self, x: torch.Tensor, return_aux: bool = False):
        """
        x: [B, 3, H, W]
        return_aux:
            False -> 只返回 logits
            True  -> 返回 (logits, a, r)
        """
        tokens_enc, r, a = self.forward_features(x)

        # 使用哪种分类头
        if isinstance(self.head, ReliabilityTopKHead):
            logits = self.head(tokens_enc, reliability=r)
        else:
            # 普通 GAP 头
            z = tokens_enc.mean(dim=1)   # [B, D]
            logits = self.head(z)        # [B, num_classes]

        if return_aux:
            return logits, a, r
        return logits


# =========================================================
# 若干工厂函数：方便在训练脚本中直接调用不同配置
# =========================================================

def create_ours_full_cifar10() -> nn.Module:
    """
    完整体 DR-SPT-ViT：
        - LearnableShiftViews
        - DynamicRoutedSPT
        - ReliabilityTopKHead
    """
    model = DRSPTViT_Ablation(
        img_size=32,
        patch_size=4,
        in_chans=3,
        num_classes=10,

        # ✅ 更稳的容量配置（建议从这里起步）
        embed_dim=256,
        depth=10,
        num_heads=8,
        mlp_ratio=3.0,

        num_views=5,
        num_iters=3,
        topk_ratio=0.25,

        use_learnable_shift=True,
        use_drspt=True,

        # ✅ 主模型不用 topk（你自己也说这是正确方向）
        use_reliability_head=False,

        # ✅ 结构正则（不属于数据增强）
        attn_drop=0.0,
        proj_drop=0.0,
        mlp_drop=0.0,
        drop_path=0.1,  # ✅ 关键
    )
    return model


def create_ours_no_shift_cifar10() -> nn.Module:
    """
    消融1：去掉可学习视图（改用固定平移视图）。
    """
    model = DRSPTViT_Ablation(
        img_size=32,
        patch_size=4,
        in_chans=3,
        num_classes=10,
        embed_dim=384,
        depth=6,
        num_heads=6,
        mlp_ratio=4.0,
        num_views=5,
        num_iters=3,
        topk_ratio=0.25,
        use_learnable_shift=False,   # 关键
        use_drspt=True,
        use_reliability_head=False,
    )
    return model


def create_ours_no_drspt_cifar10() -> nn.Module:
    """
    消融2：去掉动态路由（改用简单视图均值聚合）。
    """
    model = DRSPTViT_Ablation(
        img_size=32,
        patch_size=4,
        in_chans=3,
        num_classes=10,
        embed_dim=384,
        depth=6,
        num_heads=6,
        mlp_ratio=4.0,
        num_views=5,
        num_iters=1,           # 无 DR-SPT 时迭代次数无意义
        topk_ratio=0.25,
        use_learnable_shift=True,
        use_drspt=False,       # 关键
        use_reliability_head=False,
    )
    return model


def create_ours_no_reliability_head_cifar10() -> nn.Module:
    """
    消融3：去掉可靠性 Top-K 头（改用普通 GAP + FC）。
    """
    model = DRSPTViT_Ablation(
        img_size=32,
        patch_size=4,
        in_chans=3,
        num_classes=10,
        embed_dim=384,
        depth=6,
        num_heads=6,
        mlp_ratio=4.0,
        num_views=5,
        num_iters=3,
        topk_ratio=0.25,
        use_learnable_shift=True,
        use_drspt=True,
        use_reliability_head=False,   # 关键
        attn_drop=0.05,
        proj_drop=0.05,
        mlp_drop=0.05,
    )
    return model
