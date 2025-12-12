import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# 基础模块：Patch Embedding
# ----------------------------
class PatchEmbed(nn.Module):
    """
    标准 ViT patch embedding，用 Conv2d 实现
    """
    def __init__(self, img_size: int = 32, patch_size: int = 4,
                 in_chans: int = 3, embed_dim: int = 192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        x = self.proj(x)  # [B, embed_dim, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]
        return x


# ----------------------------
# 多视图平移构造（SPT 思想）
# ----------------------------
def shift_tensor(x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """
    对图像进行平移，空缺用 0 填充，不做循环。
    dy: 在高度方向的位移（+ 向下，- 向上）
    dx: 在宽度方向的位移（+ 向右，- 向左）
    """
    B, C, H, W = x.shape
    out = torch.zeros_like(x)

    y0_src = max(0, dy)
    y1_src = H + min(0, dy)
    x0_src = max(0, dx)
    x1_src = W + min(0, dx)

    y0_dst = max(0, -dy)
    y1_dst = y0_dst + (y1_src - y0_src)
    x0_dst = max(0, -dx)
    x1_dst = x0_dst + (x1_src - x0_src)

    if y1_src > y0_src and x1_src > x0_src:
        out[:, :, y0_dst:y1_dst, x0_dst:x1_dst] = x[:, :, y0_src:y1_src, x0_src:x1_src]
    return out


def make_shifted_views(x: torch.Tensor, num_views: int = 5) -> List[torch.Tensor]:
    """
    构造多视图平移版本，目前固定为 5 视图：
    原图 + 上移 + 下移 + 左移 + 右移
    """
    views = []
    # 原图
    views.append(x)
    # 上下左右平移 1 像素
    views.append(shift_tensor(x, dy=-1, dx=0))  # 上移
    views.append(shift_tensor(x, dy=1, dx=0))   # 下移
    views.append(shift_tensor(x, dy=0, dx=-1))  # 左移
    views.append(shift_tensor(x, dy=0, dx=1))   # 右移

    if num_views != 5:
        # 如果你以后想试更多视图，可以在这里扩展
        views = views[:num_views]

    return views


# ----------------------------
# DR-SPT：动态路由多视图分词
# ----------------------------
class DynamicRoutedSPT(nn.Module):
    """
    x: [B, V, N, C] -> 输出：
      - c: [B, N, C]   聚合后的 patch token
      - r: [B, N]      由路由权重熵计算出的可靠性 (0~1)
    """
    def __init__(self, dim: int, num_views: int = 5,
                 num_iters: int = 3, hidden_dim: int = None):
        super().__init__()
        self.num_views = num_views
        self.num_iters = num_iters
        if hidden_dim is None:
            hidden_dim = dim

        # score MLP: 输入 [p_ij^k, c_ij^{t-1}] -> 标量分数
        self.score_mlp = nn.Sequential(
            nn.Linear(2 * dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # 每个视图一个线性映射 W_k
        self.view_proj = nn.ModuleList([
            nn.Linear(dim, dim, bias=False)
            for _ in range(num_views)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, V, N, C = x.shape
        assert V == self.num_views, f"num_views mismatch: {V} vs {self.num_views}"

        # 初始化 c^{(0)} 为各视图平均
        c = x.mean(dim=1)  # [B, N, C]

        a = None  # 最后一轮的权重
        for _ in range(self.num_iters):
            # 扩展 c 到 [B, V, N, C]
            c_exp = c.unsqueeze(1).expand(-1, V, -1, -1)
            score_input = torch.cat([x, c_exp], dim=-1)  # [B, V, N, 2C]
            s = self.score_mlp(score_input).squeeze(-1)   # [B, V, N]
            # 在视图维度做 softmax
            a = torch.softmax(s, dim=1)                   # [B, V, N]

            # 视图专属投影
            proj_x_list = []
            for k in range(V):
                proj_x_list.append(self.view_proj[k](x[:, k]))  # [B, N, C]
            proj_x = torch.stack(proj_x_list, dim=1)            # [B, V, N, C]

            a_exp = a.unsqueeze(-1)                             # [B, V, N, 1]
            c = (a_exp * proj_x).sum(dim=1)                     # [B, N, C]

        # 由最后一轮权重 a 计算视角熵 -> 可靠性 r_ij
        # a: [B, V, N]
        eps = 1e-8
        log_a = torch.log(a + eps)
        entropy = -(a * log_a).sum(dim=1)        # [B, N]
        max_entropy = math.log(V)
        r = 1.0 - entropy / max_entropy          # [B, N] in [0,1]

        return c, r


# ----------------------------
# 坐标驱动的位置编码
# ----------------------------
class CoordPosEmbed(nn.Module):
    def __init__(self, num_patches: int, dim: int, grid_size: Tuple[int, int]):
        super().__init__()
        self.H, self.W = grid_size
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(2, dim),
            nn.Tanh(),
            nn.Linear(dim, dim)
        )

        assert self.H * self.W == num_patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, C]
        B, N, C = x.shape
        device = x.device
        h = torch.arange(self.H, device=device)
        w = torch.arange(self.W, device=device)
        yy, xx = torch.meshgrid(h, w, indexing="ij")  # [H, W]
        coords = torch.stack([yy, xx], dim=-1).float()  # [H, W, 2]
        coords = coords / torch.tensor(
            [self.H - 1, self.W - 1], device=device
        )  # [0,1]
        coords = coords * 2 - 1  # [-1,1]
        coords = coords.view(1, N, 2).expand(B, -1, -1)  # [B, N, 2]
        pos = self.mlp(coords)  # [B, N, C]
        return x + pos


# ----------------------------
# 可靠性感知的 Transformer Block
# ----------------------------
class ReliabilityTransformerBlock(nn.Module):
    """
    RAG (Reliability-aware Attention Gating) + 可选 R-FFN
    """
    def __init__(self, dim: int, num_heads: int = 3,
                 mlp_ratio: float = 4.0,
                 attn_alpha: float = 0.5,
                 ffn_alpha: float = 0.0,
                 drop: float = 0.0):
        super().__init__()
        self.attn_alpha = attn_alpha
        self.ffn_alpha = ffn_alpha

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )

        hidden_dim = int(dim * mlp_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, C]
        r: [B, N]   每个 token 的可靠性 in [0,1]
        """
        B, N, C = x.shape
        # ---- Attention 残差 + 可靠性门控 ----
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        # r: [B, N] -> [B, N, 1]
        r_exp = r.unsqueeze(-1)
        gamma = 1.0 + self.attn_alpha * (2.0 * r_exp - 1.0)  # [B, N, 1]
        x = x + gamma * attn_out

        # ---- FFN 残差（可选可靠性门控）----
        z_norm = self.norm2(x)
        ffn_out = self.mlp(z_norm)  # [B, N, C]
        if self.ffn_alpha != 0.0:
            beta = 1.0 + self.ffn_alpha * (2.0 * r_exp - 1.0)
            x = x + beta * ffn_out
        else:
            x = x + ffn_out

        return x


# ----------------------------
# 可靠性引导的 Top-K Pooling 分类头
# ----------------------------
class ReliabilityTopKHead(nn.Module):
    def __init__(self, dim: int, num_classes: int,
                 topk: int = 16, pool_hidden_dim: int = None):
        super().__init__()
        self.topk = topk
        if pool_hidden_dim is None:
            pool_hidden_dim = dim

        self.pool_W = nn.Linear(dim, pool_hidden_dim)
        self.pool_v = nn.Linear(pool_hidden_dim, 1)
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, C]  encoder 输出 token
        r: [B, N]     对应 token 可靠性
        """
        B, N, C = x.shape
        K = min(self.topk, N)

        # 1) 选 Top-K 可靠 token
        vals, idx = torch.topk(r, K, dim=1)  # idx: [B, K]
        idx_expand = idx.unsqueeze(-1).expand(-1, -1, C)  # [B, K, C]
        x_topk = torch.gather(x, 1, idx_expand)  # [B, K, C]

        # 2) attention pooling
        h = torch.tanh(self.pool_W(x_topk))          # [B, K, H]
        score = self.pool_v(h).squeeze(-1)           # [B, K]
        alpha = torch.softmax(score, dim=1)          # [B, K]
        alpha = alpha.unsqueeze(-1)                  # [B, K, 1]
        z = (alpha * x_topk).sum(dim=1)              # [B, C]

        logits = self.fc(z)                          # [B, num_classes]
        return logits


# ----------------------------
# 整体 DR-SPT-ViT 模型
# ----------------------------
class DRSPTViT(nn.Module):
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
        topk: int = 16,
    ):
        super().__init__()
        self.num_views = num_views

        # 1) patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        grid_size = self.patch_embed.grid_size

        # 2) DR-SPT 模块
        self.dr_spt = DynamicRoutedSPT(
            dim=embed_dim,
            num_views=num_views,
            num_iters=num_iters,
        )

        # 3) 坐标位置编码
        self.pos_embed = CoordPosEmbed(
            num_patches=num_patches,
            dim=embed_dim,
            grid_size=grid_size,
        )

        # 4) ViT 编码器（带可靠性门控）
        self.blocks = nn.ModuleList([
            ReliabilityTransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                attn_alpha=0.5,
                ffn_alpha=0.0,
                drop=0.0,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # 5) 可靠性引导的 Top-K 分类头
        self.head = ReliabilityTopKHead(
            dim=embed_dim,
            num_classes=num_classes,
            topk=topk,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, H, W]
        B = x.size(0)

        # 多视图平移
        views = make_shifted_views(x, num_views=self.num_views)  # list of [B,3,H,W]

        # 对每个视图做 patch embedding
        patch_tokens = []
        for v in views:
            p = self.patch_embed(v)  # [B, N, C]
            patch_tokens.append(p)

        x_mv = torch.stack(patch_tokens, dim=1)   # [B, V, N, C]

        # DR-SPT: 得到 tokens 和可靠性 r
        tokens, r = self.dr_spt(x_mv)             # tokens: [B,N,C], r:[B,N]

        # 坐标位置编码
        tokens = self.pos_embed(tokens)           # [B, N, C]

        # 逐层 encoder（r 在所有层公用）
        h = tokens
        for blk in self.blocks:
            h = blk(h, r)
        h = self.norm(h)                          # [B, N, C]

        # 分类头：可靠性引导的 Top-K pooling
        logits = self.head(h, r)                  # [B, num_classes]
        return logits


def create_drspt_vit_cifar10() -> nn.Module:
    """
    工厂函数：构建一个适用于 CIFAR-10 的 DR-SPT-ViT 模型
    """
    model = DRSPTViT(
        img_size=32,
        patch_size=4,
        in_chans=3,
        num_classes=10,
        embed_dim=192,
        depth=6,
        num_heads=3,
        mlp_ratio=4.0,
        num_views=5,
        num_iters=3,
        topk=16,
    )
    return model
