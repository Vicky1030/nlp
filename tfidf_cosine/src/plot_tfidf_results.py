"""Generate tables, plots, and notes from existing TF-IDF results only."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_loader import load_data

OUTPUT_FILES = ("validation_best_comparison.csv", "evaluation_metrics.csv", "confusion_matrix.csv", "macro_f1_by_threshold.png", "macro_f1_by_threshold.svg", "evaluation_metrics.png", "evaluation_metrics.svg", "experiment_results.md")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train", "threshold-table", "summary", "stage3-report", "predictions", "error-report", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    return parser.parse_args()


def refuse_existing_outputs(output_dir):
    output_dir = output_dir.resolve()
    conflicts = [str(output_dir / name) for name in OUTPUT_FILES if (output_dir / name).exists()]
    if conflicts:
        raise SystemExit("Refusing to overwrite existing output files; choose a new --output-dir:\n" + "\n".join(conflicts))
    output_dir.mkdir(parents=True, exist_ok=True)


def read_inputs(args):
    summary = list(csv.DictReader(args.summary.open(encoding="utf-8")))
    threshold_table = list(csv.DictReader(args.threshold_table.open(encoding="utf-8")))
    report = json.loads(args.stage3_report.read_text(encoding="utf-8"))
    errors = json.loads(args.error_report.read_text(encoding="utf-8"))
    predictions = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
    return summary, threshold_table, report, errors, predictions


def fixed_all_zero_baseline(predictions):
    """The reference is metric(labels, [0] * len(labels)); it is never selected."""
    labels = [row["label"] for row in predictions]
    true_negative = sum(label == 0 for label in labels)
    false_negative = sum(label == 1 for label in labels)
    denominator = len(labels)
    f1_zero = 2 * true_negative / (2 * true_negative + false_negative) if denominator else 0.0
    return {"prediction": 0, "definition": "fixed all-zero predictions", "evaluation_accuracy": true_negative / denominator if denominator else 0.0, "evaluation_macro_f1": f1_zero / 2, "TN": true_negative, "FP": 0, "FN": false_negative, "TP": 0}


def write_tables(output_dir, summary, report):
    fields = ["scheme", "threshold_index", "threshold", "vocabulary_size", "accuracy", "precision_1", "recall_1", "f1_1", "macro_f1", "TN", "FP", "FN", "TP"]
    with (output_dir / "validation_best_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in summary)
    metrics = report["metrics"]
    latency = report["latency_ms"]
    fields = ["n_samples", "accuracy", "precision_1", "recall_1", "f1_1", "macro_f1", "TN", "FP", "FN", "TP", "predicted_0", "predicted_1", "latency_min_ms", "latency_mean_ms", "latency_median_ms", "latency_p95_ms", "latency_max_ms"]
    row = {key: metrics.get(key, "") for key in fields}
    row.update(latency_min_ms=latency["min"], latency_mean_ms=latency["mean"], latency_median_ms=latency["median"], latency_p95_ms=latency["p95"], latency_max_ms=latency["max"])
    with (output_dir / "evaluation_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)
    with (output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", "0", "1"])
        writer.writerow(["0", metrics["TN"], metrics["FP"]])
        writer.writerow(["1", metrics["FN"], metrics["TP"]])


def write_plots(output_dir, threshold_table, report, predictions):
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    figure, axis = plt.subplots(figsize=(8, 5))
    for scheme in ("char_1", "char_1_2", "word_1"):
        rows = [row for row in threshold_table if row["scheme"] == scheme]
        axis.plot([float(row["threshold"]) for row in rows], [float(row["macro_f1"]) for row in rows], label=scheme)
    axis.axvline(float(report["config"]["threshold"]), color="black", linestyle="--", label="最终选择 0.28")
    axis.set_xlabel("threshold")
    axis.set_ylabel("Validation Macro-F1")
    axis.set_title("验证集阈值与 Macro-F1")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "macro_f1_by_threshold.png", dpi=160)
    figure.savefig(output_dir / "macro_f1_by_threshold.svg")
    plt.close(figure)
    baseline = fixed_all_zero_baseline(predictions)
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.bar(["TF-IDF cosine", "全部预测0"], [report["metrics"]["macro_f1"], baseline["evaluation_macro_f1"]], color=["#4c78a8", "#f58518"])
    axis.set_ylabel("Evaluation Macro-F1")
    axis.set_title("最终评估 Macro-F1 对照")
    figure.tight_layout()
    figure.savefig(output_dir / "evaluation_metrics.png", dpi=160)
    figure.savefig(output_dir / "evaluation_metrics.svg")
    plt.close(figure)
    return baseline


def write_note(output_dir, threshold_table, report, errors, baseline):
    metrics = report["metrics"]
    lines = ["# TF-IDF＋余弦相似度结果说明", "", "## 数据分工与防泄漏", "TF-IDF 词表和 IDF 只从内部 train 的两列句子拟合；validation 只用于三种表示和 102 个阈值的比较；evaluation 在配置冻结后才用于预测、评分和错误分析。", "", "## 验证选优", f"共比较 3 种表示 × 102 个阈值 = {len(threshold_table)} 组。先按 Macro-F1，再 Accuracy，再较大阈值；跨表示仍按 Macro-F1、Accuracy 和固定方案顺序处理。获选配置为 `{report['config']['representation']}`，阈值 `{report['config']['threshold']}`，规则 `>=`。", "", "## 最终评估", f"Accuracy={metrics['accuracy']:.6f}，正类 Precision={metrics['precision_1']:.6f}，Recall={metrics['recall_1']:.6f}，F1={metrics['f1_1']:.6f}，Macro-F1={metrics['macro_f1']:.6f}。混淆矩阵行是真实标签、列是预测标签，顺序为 [0,1]：`[[{metrics['TN']},{metrics['FP']}],[{metrics['FN']},{metrics['TP']}]]`。", "", f"固定全部预测0参照：Accuracy={baseline['evaluation_accuracy']:.6f}，Macro-F1={baseline['evaluation_macro_f1']:.6f}。该参照不参与配置选择。", "", "## 错误分析", f"错误样本={errors['errors']}，FP={errors['false_positive']}，FN={errors['false_negative']}。A/B/C 只是规则筛选的人工复核候选，类别可重叠，不能将候选数量相加当作错误总数。", "", "## 局限", "TF-IDF 余弦相似度依赖表面词/字符重合，可能忽略语序、否定、数字和语义替换；候选原因需要人工复核。Accuracy 受类别比例影响，应与 Macro-F1 一起观察。图表只使用 validation 比较和固定配置后的 evaluation 结果，不涉及 N-gram 语言模型或神经网络语言模型实验。"]
    (output_dir / "experiment_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    refuse_existing_outputs(output_dir)
    summary, threshold_table, report, errors, predictions = read_inputs(args)
    write_tables(output_dir, summary, report)
    baseline = write_plots(output_dir, threshold_table, report, predictions)
    write_note(output_dir, threshold_table, report, errors, baseline)
    print(json.dumps({"status": "passed", "table_rows": len(threshold_table), "all_zero_baseline": baseline, "evaluation_metrics": report["metrics"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
