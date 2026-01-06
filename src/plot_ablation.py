import os
import json
import argparse
import matplotlib.pyplot as plt

def load_summary(run_dir: str):
    p = os.path.join(run_dir, "summary.json")
    with open(p, "r") as f:
        return json.load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", help="多个 runs/<exp_name...>/ 目录")
    ap.add_argument("--out", default="ablation.png")
    args = ap.parse_args()

    labels, accs = [], []
    for rd in args.run_dirs:
        s = load_summary(rd)
        labels.append(s.get("exp_name", os.path.basename(rd)))
        accs.append(float(s.get("test_acc", 0.0)))

    plt.figure()
    plt.bar(range(len(accs)), accs)
    plt.xticks(range(len(labels)), labels, rotation=30, ha="right")
    plt.ylabel("test accuracy")
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    plt.close()

if __name__ == "__main__":
    main()
