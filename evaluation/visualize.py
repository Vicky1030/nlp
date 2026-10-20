"""读取 run_all 生成的 comparison_results.csv，绘制指标对比图。"""
import argparse
import csv
from pathlib import Path


def visualize(metrics_csv, output_dir):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("绘图需要 matplotlib，请运行：pip install matplotlib") from exc

    metrics_csv, output_dir = Path(metrics_csv), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with metrics_csv.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{metrics_csv} 中没有指标记录")
    models = [r["model"] for r in rows]
    metric_specs = [
        ("accuracy", "Accuracy", "accuracy_comparison.png"),
        ("macro_f1", "Macro-F1", "macro_f1_comparison.png"),
        ("f1_positive", "F1 (positive label=1)", "positive_f1_comparison.png"),
    ]
    for key, title, filename in metric_specs:
        values = [float(r[key]) for r in rows]
        fig, ax = plt.subplots(figsize=(max(7, len(models) * 1.2), 4.5))
        bars = ax.bar(models, values)
        ax.set_title(title)
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", labelrotation=25)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, value + 0.01, f"{value:.3f}",
                    ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=200)
        plt.close(fig)

    # 延迟图只展示具备延迟数据的模型
    latency_rows = [(r["model"], r.get("latency_mean_ms", "")) for r in rows if r.get("latency_mean_ms", "").strip()]
    if latency_rows:
        fig, ax = plt.subplots(figsize=(max(7, len(latency_rows) * 1.2), 4.5))
        models_l = [x[0] for x in latency_rows]
        values_l = [float(x[1]) for x in latency_rows]
        bars = ax.bar(models_l, values_l)
        ax.set_title("Mean inference latency")
        ax.set_ylabel("Milliseconds per pair")
        ax.tick_params(axis="x", labelrotation=25)
        for bar, value in zip(bars, values_l):
            ax.text(bar.get_x() + bar.get_width()/2, value, f"{value:.3f}",
                    ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / "latency_comparison.png", dpi=200)
        plt.close(fig)
    return sorted(p.name for p in output_dir.glob("*.png"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="outputs/comparison_results.csv")
    parser.add_argument("--output-dir", default="outputs/figures")
    args = parser.parse_args()
    print("Generated:", visualize(args.metrics, args.output_dir))


if __name__ == "__main__":
    main()
