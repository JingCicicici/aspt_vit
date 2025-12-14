# models/drspt_modules.py
"""
DR-SPT & 可靠性模块集合：
- LearnableShiftViews: 可学习移位视图 (learnable shifted views)
- DynamicRoutedSPT    : 多视图动态路由分词，输出 c / a / r
- ReliabilityTopKHead : 利用 r 做 Top-K token 聚合的分类头
- view_entropy_loss   : 视图熵正则项
- reliability_smoothness_loss : 可靠性图空间平滑正则项
"""

from typing import Tuple, Optional, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "LearnableShiftViews",
    "FixedShiftViews",              # 新增
    "DynamicRoutedSPT",
    "SimpleViewAggregator",         # 新增
    "ReliabilityTopKHead",
    "view_entropy_loss",
    "reliability_smoothness_loss",
]


# ----------------------------------------------------------------------
# 1. LearnableShiftViews: 可学习移位视图模块
# ----------------------------------------------------------------------

class LearnableShiftViews(nn.Module):
    """
    Learnable Shift Views Module

    功能：
        对输入图像 x 构造 V 个“轻微平移”的视图，每个视图的平移偏移 (dy, dx)
        作为可学习参数，通过 grid_sample 实现可导的平移采样。

    输入:
        x: [B, C, H, W]

    输出:
        views: list[Tensor]，长度 = num_views，每个元素形状 [B, C, H, W]

    说明：
        - raw_offsets 存的是“像素单位”的偏移 (dy, dx)；
        - 通过线性映射转换到 [-1, 1] 坐标系下的偏移量，给 grid_sample 使用；
        - 初始化时可以设置为中心 + 上下左右 1 像素，之后在训练中自动微调。
    """

    def __init__(self, num_views: int = 5, init_offsets: Optional[torch.Tensor] = None):
        super().__init__()
        self.num_views = num_views

        if init_offsets is None:
            # 以像素为单位的初始偏移: [dy, dx]
            # center, up, down, left, right
            init_offsets = torch.tensor([
                [0.0,  0.0],   # center
                [-1.0, 0.0],   # up
                [1.0,  0.0],   # down
                [0.0, -1.0],   # left
                [0.0,  1.0],   # right
            ])

        init_offsets = init_offsets[:num_views].float()  # [V, 2]
        # raw_offsets: 像素偏移，可学习参数
        self.raw_offsets = nn.Parameter(init_offsets)     # [V, 2]

    def forward(self, x: torch.Tensor):
        """
        x: [B, C, H, W]
        return: list[Tensor], len = num_views, 每个视图 [B, C, H, W]
        """
        B, C, H, W = x.shape
        device = x.device

        # 构建基础坐标网格 [-1, 1] x [-1, 1]
        ys = torch.linspace(-1.0, 1.0, H, device=device)
        xs = torch.linspace(-1.0, 1.0, W, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")  # [H, W]
        base_grid = torch.stack([xx, yy], dim=-1).unsqueeze(0)  # [1, H, W, 2]

        # 像素偏移 -> 归一化偏移 (grid_sample 的坐标系中 [-1,1] 对应图像边界)
        scale_x = 2.0 / max(W - 1, 1)
        scale_y = 2.0 / max(H - 1, 1)

        views = []
        for v in range(self.num_views):
            dy_px, dx_px = self.raw_offsets[v]  # 标量（像素偏移）

            # 如有需要，可以使用 tanh 对 raw_offsets 限幅:
            # dy_px = torch.tanh(dy_px) * 2.0
            # dx_px = torch.tanh(dx_px) * 2.0

            dy_norm = dy_px * scale_y
            dx_norm = dx_px * scale_x

            grid_v = base_grid.clone()
            # grid[..., 0] = x, grid[..., 1] = y
            grid_v[..., 0] = grid_v[..., 0] + dx_norm
            grid_v[..., 1] = grid_v[..., 1] + dy_norm

            # 扩展到 batch 维度
            grid_v = grid_v.expand(B, -1, -1, -1)  # [B, H, W, 2]

            # 双线性插值，超出区域用 0 填充
            v_img = F.grid_sample(
                x,
                grid_v,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            views.append(v_img)

        return views


# ----------------------------------------------------------------------
# 1.b FixedShiftViews: 非可学习的多视图平移模块（消融用）
# ----------------------------------------------------------------------

class FixedShiftViews(nn.Module):
    """
    Fixed Shift Views Module（非可学习）

    功能：
        使用固定的（dy, dx）像素偏移，对输入图像构造 V 个平移视图。
        用于和 LearnableShiftViews 做消融比较。

    输入:
        x: [B, C, H, W]

    输出:
        views: list[Tensor]，长度 = num_views，每个 [B, C, H, W]
    """

    def __init__(self, num_views: int = 5, offsets: Optional[torch.Tensor] = None):
        super().__init__()
        self.num_views = num_views

        if offsets is None:
            # 与 LearnableShiftViews 的初始化保持一致：
            # center, up, down, left, right
            offsets = torch.tensor([
                [0,  0],   # center
                [-1, 0],   # up
                [1,  0],   # down
                [0, -1],   # left
                [0,  1],   # right
            ], dtype=torch.long)

        # 只取前 num_views 个偏移
        offsets = offsets[:num_views]
        # 注册为 buffer（不是可学习参数），在 forward 时直接使用
        self.register_buffer("offsets", offsets)   # [V, 2]

    @staticmethod
    def _shift_one(x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
        """
        对一个 batch 的图像做整数像素平移，空缺用 0 填充，不循环。
        dy: + 向下，- 向上
        dx: + 向右，- 向左
        """
        B, C, H, W = x.shape
        out = torch.zeros_like(x)

        # 源区域
        y0_src = max(0, dy)
        y1_src = H + min(0, dy)
        x0_src = max(0, dx)
        x1_src = W + min(0, dx)

        # 目标区域
        y0_dst = max(0, -dy)
        y1_dst = y0_dst + (y1_src - y0_src)
        x0_dst = max(0, -dx)
        x1_dst = x0_dst + (x1_src - x0_src)

        if y1_src > y0_src and x1_src > x0_src:
            out[:, :, y0_dst:y1_dst, x0_dst:x1_dst] = x[:, :, y0_src:y1_src, x0_src:x1_src]

        return out

    def forward(self, x: torch.Tensor):
        """
        x: [B, C, H, W]
        return: list[Tensor], len = num_views, 每个视图 [B, C, H, W]
        """
        views = []
        B, C, H, W = x.shape

        for v in range(self.num_views):
            dy, dx = self.offsets[v].tolist()   # 整数像素偏移
            v_img = self._shift_one(x, int(dy), int(dx))
            views.append(v_img)

        return views






# ----------------------------------------------------------------------
# 2. DynamicRoutedSPT: 多视图动态路由分词
# ----------------------------------------------------------------------

class DynamicRoutedSPT(nn.Module):
    """
    Dynamic Routed Selective Patch Tokens (DR-SPT)

    功能：
        给定多视图 patch tokens x_mv，使用一个 routing-by-agreement 风格的
        动态路由机制，在 patch 级别上对视图进行加权聚合，得到：
          - 融合后的 patch tokens c
          - 视图权重 a
          - patch 可靠性 r

    输入:
        x_mv: [B, V, N, C]
            - B: batch size
            - V: num_views
            - N: patch 数量
            - C: embed_dim

    输出:
        c: [B, N, C]   聚合后的 patch tokens
        a: [B, V, N]   每个 patch 的视图权重分布 (softmax 结果)
        r: [B, N]      每个 patch 的可靠性，依据视图权重熵归一化到 [0,1]

    参数:
        embed_dim: token 的维度 C
        num_views: 视图数量 V
        num_iters: 路由迭代次数 (>=1)，默认为 1 即可
    """

    def __init__(
        self,
        embed_dim: int,
        num_views: int,
        num_iters: int = 1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_views = num_views
        self.num_iters = num_iters

        # view-specific projection: 每个视图一个线性变换
        self.view_proj = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(num_views)
        ])

        # routing MLP：输入 concat([x_mv^k, c_prev])，输出一个标量得分
        self.routing_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, x_mv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x_mv: [B, V, N, C]
        """
        B, V, N, C = x_mv.shape
        assert V == self.num_views, f"num_views 不匹配: 输入 V={V}, 模块定义={self.num_views}"
        assert C == self.embed_dim, f"embed_dim 不匹配: 输入 C={C}, 模块定义={self.embed_dim}"

        # 初始化聚合 token：简单用视图平均作为 c^{(0)}
        c = x_mv.mean(dim=1)  # [B, N, C]

        for _ in range(self.num_iters):
            # 将当前聚合 token c 扩展到视图维度，便于拼接
            c_exp = c.unsqueeze(1).expand(B, V, N, C)  # [B, V, N, C]
            concat = torch.cat([x_mv, c_exp], dim=-1)  # [B, V, N, 2C]

            # routing scores: [B, V, N]
            scores = self.routing_mlp(concat.view(B * V * N, -1))  # [B*V*N, 1]
            scores = scores.view(B, V, N)
            a = torch.softmax(scores, dim=1)  # 在视图维度 softmax

            # view-specific projection 后加权求和
            proj_tokens = []
            for k in range(V):
                # x_mv[:, k]: [B, N, C]
                proj_tokens.append(self.view_proj[k](x_mv[:, k]))  # [B, N, C]
            proj = torch.stack(proj_tokens, dim=1)  # [B, V, N, C]

            c = (a.unsqueeze(-1) * proj).sum(dim=1)  # [B, N, C]

        # 基于视图权重熵计算 patch 可靠性 r（熵越小，可靠性越高）
        eps = 1e-8
        entropy = -(a * (a + eps).log()).sum(dim=1)  # [B, N]
        max_entropy = math.log(float(V))
        r = 1.0 - entropy / max_entropy            # 归一化到 [0,1]

        return c, a, r



# ----------------------------------------------------------------------
# 2.b SimpleViewAggregator: 简单视图均值聚合（消融用）
# ----------------------------------------------------------------------

class SimpleViewAggregator(nn.Module):
    """
    Simple View Aggregator（无动态路由版）

    功能：
        给定多视图 patch tokens x_mv，直接在视图维度做均值：
            c = mean_v x_mv
        并返回：
            - c: [B, N, C] 聚合后的 tokens
            - a: [B, V, N] 视图权重（这里是均匀分布 1/V）
            - r: [B, N]    可靠性（这里简单设为全 1）

    用途：
        - 作为 DynamicRoutedSPT 的消融对照：仅仅“多视图 + 简单平均”，
          不进行 routing-by-agreement。
    """

    def __init__(self, num_views: Optional[int] = None):
        super().__init__()
        self.num_views = num_views

    def forward(self, x_mv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x_mv: [B, V, N, C]
        返回:
            c: [B, N, C]
            a: [B, V, N]
            r: [B, N]
        """
        B, V, N, C = x_mv.shape
        if self.num_views is not None:
            assert V == self.num_views, f"SimpleViewAggregator 期望 V={self.num_views}, 但输入 V={V}"

        # 直接在视图维度做均值
        c = x_mv.mean(dim=1)  # [B, N, C]

        # 构造均匀视图权重 a，以及全 1 的可靠性 r，方便与 DR-SPT 接口对齐
        a = x_mv.new_full((B, V, N), 1.0 / V)  # [B, V, N]
        r = x_mv.new_ones(B, N)               # [B, N]

        return c, a, r





# ----------------------------------------------------------------------
# 3. ReliabilityTopKHead: 可靠性引导 Top-K 分类头
# ----------------------------------------------------------------------

class ReliabilityTopKHead(nn.Module):
    """
    Reliability-guided Top-K Classification Head

    功能：
        给定编码后的 patch tokens 和 patch 可靠性 r：
          - 利用 r 选择每张图中 Top-K 个最可靠的 patch；
          - 使用 r 作为权重，对这些 token 做加权聚合得到图级特征；
          - 再通过一个线性层得到最终分类 logits。

    输入:
        tokens: [B, N, C]  来自 ViTEncoder 的输出
        reliability: [B, N]  来自 DynamicRoutedSPT 的 r（在 [0,1]）

    输出:
        logits: [B, num_classes]

    参数:
        embed_dim : token 维度 C
        num_classes: 类别数
        topk_ratio: 选取的 token 比例 (0,1]，例如 0.25 表示选取 25% 的 patch
        min_topk  : 最少选取的 token 数，避免 N 很小时 K=0 的情况

    特殊情况：
        - 如果 reliability 为 None 或 topk_ratio >= 1.0，则退化为普通 GAP(head)。
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        topk_ratio: float = 0.25,
        min_topk: int = 1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.topk_ratio = topk_ratio
        self.min_topk = min_topk

        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(
        self,
        tokens: torch.Tensor,
        reliability: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        tokens: [B, N, C]
        reliability: [B, N] or None
        """
        B, N, C = tokens.shape

        # 如果没有 r，或者 topk_ratio 无效，则退化为普通 GAP Head
        if (
            reliability is None
            or self.topk_ratio is None
            or self.topk_ratio >= 1.0
        ):
            feat = tokens.mean(dim=1)  # [B, C]
            logits = self.fc(feat)
            return logits

        # 计算 Top-K
        k = max(int(N * self.topk_ratio), self.min_topk)
        k = min(k, N)  # 不超过 N

        # 在 patch 维度上按 r 取 Top-K
        topk_vals, topk_idx = torch.topk(reliability, k, dim=1)  # [B, k]

        # 收集对应的 tokens: [B, k, C]
        batch_idx = torch.arange(B, device=tokens.device).unsqueeze(-1)  # [B, 1]
        topk_tokens = tokens[batch_idx, topk_idx]  # [B, k, C]

        # 用 r (topk_vals) 做 softmax 权重
        weights = torch.softmax(topk_vals, dim=1).unsqueeze(-1)  # [B, k, 1]
        feat = (weights * topk_tokens).sum(dim=1)  # [B, C]

        logits = self.fc(feat)  # [B, num_classes]
        return logits


# ----------------------------------------------------------------------
# 4. 正则项: 视图熵正则 & 可靠性空间平滑正则
# ----------------------------------------------------------------------

def view_entropy_loss(a: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    视图熵正则项 (用于约束 DynamicRoutedSPT 中的视图权重 a)

    输入:
        a: [B, V, N]  每个 patch 的视图权重分布（softmax 输出）

    输出:
        标量 loss（越大表示视图更“平均”，越小表示视图更“有主见”）

    一般用法：
        L_view = view_entropy_loss(a)
        loss = ce_loss + lambda_view * L_view
    """
    # 在视图维度上计算熵
    entropy = -(a * (a + eps).log()).sum(dim=1)  # [B, N]
    return entropy.mean()


def reliability_smoothness_loss(
    r: torch.Tensor,
    grid_size: Optional[Union[int, Tuple[int, int]]] = None,
) -> torch.Tensor:
    """
    可靠性空间平滑正则 (约束 r 在空间上的变化不要过于剧烈)

    输入:
        r: [B, N]  patch 可靠性 (0~1)
        grid_size:
            - 如果为 int，表示是 H_p = W_p = grid_size 的方形网格；
            - 如果为 (H_p, W_p)，则 N 应等于 H_p * W_p；
            - 如果为 None，则尝试用 sqrt(N) 作为方形网格边长。

    输出:
        标量 loss = E[ (dx)^2 + (dy)^2 ]，越小表示 r 在空间上越平滑。

    一般用法：
        L_smooth = reliability_smoothness_loss(r, grid_size=(H_p, W_p))
        loss = ce_loss + lambda_view * L_view + lambda_smooth * L_smooth
    """
    B, N = r.shape

    if isinstance(grid_size, int):
        H_p = W_p = grid_size
    elif isinstance(grid_size, (tuple, list)) and len(grid_size) == 2:
        H_p, W_p = grid_size
    else:
        # 默认尝试方形网格
        s = int(math.sqrt(N))
        H_p = W_p = s
        assert H_p * W_p == N, (
            f"无法自动推断 grid_size，N={N} 不是完全平方数，请显式传入 grid_size"
        )

    r_map = r.view(B, H_p, W_p)  # [B, H_p, W_p]

    # 水平 & 垂直方向的差分
    dx = r_map[:, :, 1:] - r_map[:, :, :-1]   # [B, H_p, W_p-1]
    dy = r_map[:, 1:, :] - r_map[:, :-1, :]   # [B, H_p-1, W_p]

    loss = dx.pow(2).mean() + dy.pow(2).mean()
    return loss
