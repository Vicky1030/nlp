"""一键评价 results/ 下所有模型结果，并生成指标表、错例表和图。"""
import argparse
import csv
import json
import sys
from pathlib import Path

try:
    from .config import GOLD_FILE, RESULTS_DIR, OUTPUT_DIR
    from .evaluate import evaluate_file
    from .error_analysis import analyze
    from .visualize import visualize
except ImportError:
    from config import GOLD_FILE, RESULTS_DIR, OUTPUT_DIR
    from evaluate import evaluate_file
    from error_analysis import analyze
    from visualize import visualize


def run(results_dir=RESULTS_DIR, gold_path=GOLD_FILE, output_dir=OUTPUT_DIR, make_plots=True):
    results_dir, output_dir = Path(results_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_files = sorted(p for p in results_dir.glob("*.jsonl") if p.name != Path(gold_path).name)
    if not pred_files:
        raise FileNotFoundError(
            f"没有发现模型预测文件：{results_dir}/*.jsonl。请将各模型结果放入 results/。"
        )

    metric_rows = []
    detailed = {}
    for pred_path in pred_files:
        result = evaluate_file(pred_path, gold_path)
        model = result.get("model") or pred_path.stem
        # CSV 中保留简明核心指标；完整指标分别写 JSON
        row = {
            "model": model,
            "n_samples": result["n_samples"],
            "accuracy": result["accuracy"],
            "precision_positive": result["precision_positive"],
            "recall_positive": result["recall_positive"],
            "f1_positive": result["f1_positive"],
            "macro_f1": result["macro_f1"],
            "correct": result["correct"],
            "errors": result["errors"],
            "error_rate": result["error_rate"],
            "latency_mean_ms": result.get("latency_mean_ms", ""),
            "latency_median_ms": result.get("latency_median_ms", ""),
            "latency_p95_ms": result.get("latency_p95_ms", ""),
        }
        metric_rows.append(row)
        detailed[pred_path.stem] = result
        (output_dir / f"{pred_path.stem}_metrics.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    csv_path = output_dir / "comparison_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(metric_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metric_rows)

    (output_dir / "all_metrics.json").write_text(
        json.dumps(detailed, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    error_result = analyze(results_dir, gold_path, output_dir)
    charts = []
    if make_plots:
        try:
            charts = visualize(csv_path, output_dir / "figures")
        except RuntimeError as exc:
            print(f"[WARN] {exc}", file=sys.stderr)
    return {"models_evaluated": len(pred_files), "metrics_csv": str(csv_path),
            "error_analysis": error_result, "charts": charts}


def main():
    parser = argparse.ArgumentParser(description="AFQMC 多模型统一评价一键入口")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--gold", default=str(GOLD_FILE))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--no-plots", action="store_true", help="跳过绘图")
    args = parser.parse_args()
    try:
        result = run(args.results_dir, args.gold, args.output_dir, not args.no_plots)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
