"""Create metric comparison and confusion-matrix plots."""
import csv
from pathlib import Path


def visualize(metrics_csv, output_dir, split_name):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return ["SKIPPED: matplotlib not installed"]
    metrics_csv, output_dir = Path(metrics_csv), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with metrics_csv.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return ["SKIPPED: no model rows"]
    models = [r["model"] for r in rows]
    generated = []
    for key, title, filename in [
        ("accuracy", "Accuracy", "accuracy"),
        ("macro_f1", "Macro-F1", "macro_f1"),
        ("f1_positive", "Positive-class F1 (label=1)", "positive_f1"),
    ]:
        vals = [float(r[key]) for r in rows]
        fig, ax = plt.subplots(figsize=(max(7, len(models)*1.15), 4.6))
        bars = ax.bar(models, vals)
        ax.set_title(f"{title} — {split_name}")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        ax.tick_params(axis="x", labelrotation=25)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, min(0.99, val+0.012), f"{val:.3f}",
                    ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        p = output_dir / f"{split_name}_{filename}.png"
        fig.savefig(p, dpi=180)
        plt.close(fig)
        generated.append(p.name)

    latency_rows = [r for r in rows if r.get("latency_mean_ms", "").strip()]
    if latency_rows:
        models_l = [r["model"] for r in latency_rows]
        vals = [float(r["latency_mean_ms"]) for r in latency_rows]
        fig, ax = plt.subplots(figsize=(max(7, len(models_l)*1.15), 4.6))
        bars = ax.bar(models_l, vals)
        ax.set_title(f"Mean recorded latency — {split_name} (not necessarily comparable)")
        ax.set_ylabel("Milliseconds per pair")
        ax.tick_params(axis="x", labelrotation=25)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, val, f"{val:.1f}",
                    ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        p = output_dir / f"{split_name}_latency.png"
        fig.savefig(p, dpi=180)
        plt.close(fig)
        generated.append(p.name)
    return generated
