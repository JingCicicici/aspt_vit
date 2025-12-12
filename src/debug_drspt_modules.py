import torch
from models.drspt_modules import LearnableShiftViews, DynamicRoutedSPT, ReliabilityTopKHead

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) 测试 LearnableShiftViews
    imgs = torch.randn(2, 3, 32, 32).to(device)
    shift_module = LearnableShiftViews(num_views=5).to(device)
    views = shift_module(imgs)
    print("num views:", len(views), "view shape:", views[0].shape)

    # 2) 测试 DynamicRoutedSPT
    B, V, N, C = 2, 5, 16, 32
    x_mv = torch.randn(B, V, N, C).to(device)
    drspt = DynamicRoutedSPT(embed_dim=C, num_views=V, num_iters=2).to(device)
    c, a, r = drspt(x_mv)
    print("c:", c.shape, "a:", a.shape, "r:", r.shape)

    # 3) 测试 ReliabilityTopKHead
    head = ReliabilityTopKHead(embed_dim=C, num_classes=10, topk_ratio=0.25).to(device)
    logits = head(c, r)
    print("logits:", logits.shape)

if __name__ == "__main__":
    main()
