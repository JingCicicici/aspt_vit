# auto_tune_ours_full.py
import subprocess

def main():
    # 固定一些参数
    model = "ours_full"
    train_ratio = 1.0
    epochs = 40

    # 要搜索的超参数组合（先用一个很小的网格）
    lrs = [1e-3, 5e-4, 3e-4]          # 学习率候选
    lambda_views = [0.0, 1e-2]        # 视图熵正则系数候选
    lambda_smooths = [0.0, 1e-2]      # 平滑正则系数候选

    for lr in lrs:
        for lv in lambda_views:
            for ls in lambda_smooths:
                # 如果两个正则都是 0，就等价于 no_reg 情况
                use_no_reg = (lv == 0.0 and ls == 0.0)

                cmd = [
                    "python", "train_ours_cli.py",
                    "--model", model,
                    "--train_ratio", str(train_ratio),
                    "--epochs", str(epochs),
                    "--lr", str(lr),
                    "--lambda_view", str(lv),
                    "--lambda_smooth", str(ls),
                ]
                if use_no_reg:
                    cmd.append("--no_reg")

                print("\n>>> Running:", " ".join(cmd))
                # check=True 遇到报错会直接中断，方便你发现问题
                subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
