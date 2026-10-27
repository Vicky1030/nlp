"""Unified evaluation for frozen and newly generated predictions."""
import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from .config import GOLD_FILES, PREDICTION_DIRS, OUTPUT_DIR
from .io_utils import load_jsonl, validate_gold, align_predictions
from .metrics import calculate_metrics
from .error_analysis import analyze_group, write_csv
from .visualize import visualize


EXPECTED_METHODS = [
    {"method": "tfidf_cosine", "split": "evaluation", "owner": "A", "status": "available"},
    {"method": "tfidf_lr", "split": "evaluation", "owner": "A", "status": "available"},
    {"method": "P0.predictions", "split": "evaluation", "owner": "B", "status": "available"},
    {"method": "near_conditions.predictions", "split": "evaluation", "owner": "B", "status": "available"},
    {"method": "fixed2_last_exact.predictions", "split": "evaluation", "owner": "B", "status": "available"},
    {"method": "lora_r16_a32_30899_final", "split": "evaluation", "owner": "C", "status": "available"},
    {"method": "lora_r8_a16_3000", "split": "evaluation", "owner": "C", "status": "not_provided_on_evaluation"},
    {"method": "lora_r16_a32_3000", "split": "evaluation", "owner": "C", "status": "not_provided_on_evaluation"},
]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def discover_models(split_name):
    pred_dir = PREDICTION_DIRS[split_name]
    pred_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(pred_dir.glob("*.jsonl"))
    if not paths:
        return {}
    gold_path = GOLD_FILES[split_name]
    gold = validate_gold(load_jsonl(gold_path), gold_path)
    aligned = {}
    for path in paths:
        records = load_jsonl(path)
        aligned[path.stem] = align_predictions(records, gold, path)
    return aligned


def build_metrics(split_name, aligned_models, out_dir, make_plots=True):
    rows, detailed, confusion_rows = [], {}, []
    for model, aligned in aligned_models.items():
        m = calculate_metrics(aligned)
        methods = sorted({str(r.get("method", model)) for r in aligned})
        m.update({
            "model": model,
            "method": methods[0] if len(methods) == 1 else ";".join(methods),
            "split": split_name,
            "prediction_file": str(PREDICTION_DIRS[split_name] / f"{model}.jsonl"),
            "gold_file": str(GOLD_FILES[split_name]),
            "gold_sha256": sha256(GOLD_FILES[split_name]),
        })
        detailed[model] = m
        rows.append({
            "split": split_name, "model": model, "method": m["method"],
            "n_samples": m["n_samples"], "accuracy": m["accuracy"],
            "precision_positive": m["precision_positive"],
            "recall_positive": m["recall_positive"], "f1_positive": m["f1_positive"],
            "macro_f1": m["macro_f1"], "correct": m["correct"], "errors": m["errors"],
            "error_rate": m["error_rate"], "invalid_predictions": m["invalid_predictions"],
            "latency_count": m["latency_count"],
            "latency_mean_ms": "" if m["latency_mean_ms"] is None else m["latency_mean_ms"],
            "latency_median_ms": "" if m["latency_median_ms"] is None else m["latency_median_ms"],
            "latency_p95_ms": "" if m["latency_p95_ms"] is None else m["latency_p95_ms"],
        })
        cm = m["confusion_matrix"]["rows_true_cols_pred"]
        confusion_rows.append({"split": split_name, "model": model,
                               "tn": cm[0][0], "fp": cm[0][1], "fn": cm[1][0], "tp": cm[1][1],
                               "invalid_predictions": m["invalid_predictions"]})
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"comparison_results_{split_name}.csv"
    fields = list(rows[0].keys()) if rows else [
        "split","model","method","n_samples","accuracy","precision_positive","recall_positive",
        "f1_positive","macro_f1","correct","errors","error_rate","invalid_predictions",
        "latency_count","latency_mean_ms","latency_median_ms","latency_p95_ms"]
    write_csv(csv_path, rows, fields)
    write_csv(out_dir / f"confusion_matrices_{split_name}.csv", confusion_rows)
    (out_dir / f"metrics_{split_name}.json").write_text(
        json.dumps(detailed, ensure_ascii=False, indent=2), encoding="utf-8")
    error_summary = analyze_group(aligned_models, out_dir, split_name) if aligned_models else {"models": [], "error_rows": 0}
    charts = visualize(csv_path, out_dir / "figures", split_name) if (aligned_models and make_plots) else []
    return {"models": len(aligned_models), "comparison_csv": str(csv_path),
            "error_analysis": error_summary, "charts": charts}



def write_summary_report(out_dir, results):
    """Human-readable report generated from the same machine-readable metrics."""
    out_dir = Path(out_dir)
    lines = [
        "# AFQMC 统一评估结果摘要",
        "",
        f"- 运行时间（UTC）：{datetime.now(timezone.utc).isoformat()}",
        "- 评价规则：以 label=1 为正类；Macro-F1 对 label 0/1 的 F1 求平均。",
        "- 混淆矩阵行是真实标签、列是预测标签；无效预测计为错误。",
        "",
        "## 最终 evaluation",
        "",
    ]
    eval_csv = out_dir / "comparison_results_evaluation.csv"
    if eval_csv.exists():
        with eval_csv.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        if rows:
            lines += [
                "| 模型 | 样本数 | Accuracy | Precision(1) | Recall(1) | F1(1) | Macro-F1 | 错误数 | 平均延迟(ms) |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
            for r in sorted(rows, key=lambda x: float(x["macro_f1"]), reverse=True):
                latency = r["latency_mean_ms"]
                latency_text = f"{float(latency):.3f}" if latency else "N/A"
                lines.append(
                    f"| {r['model']} | {r['n_samples']} | {float(r['accuracy']):.4f} | "
                    f"{float(r['precision_positive']):.4f} | {float(r['recall_positive']):.4f} | "
                    f"{float(r['f1_positive']):.4f} | {float(r['macro_f1']):.4f} | "
                    f"{r['errors']} | {latency_text} |"
                )
        else:
            lines.append("当前没有 evaluation 预测文件。")
    else:
        lines.append("当前没有 evaluation 预测文件。")

    lines += ["", "## Validation 与 LoRA 秩对比", ""]
    val_csv = out_dir / "comparison_results_validation.csv"
    if val_csv.exists():
        with val_csv.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        if rows:
            lines += [
                "| 模型 | 样本数 | Accuracy | F1(1) | Macro-F1 | 错误数 |",
                "|---|---:|---:|---:|---:|---:|",
            ]
            for r in sorted(rows, key=lambda x: float(x["macro_f1"]), reverse=True):
                lines.append(
                    f"| {r['model']} | {r['n_samples']} | {float(r['accuracy']):.4f} | "
                    f"{float(r['f1_positive']):.4f} | {float(r['macro_f1']):.4f} | {r['errors']} |"
                )
            by_name = {r["model"]: r for r in rows}
            r8, r16 = by_name.get("lora_r8_a16_3000"), by_name.get("lora_r16_a32_3000")
            if r8 and r16:
                lines += [
                    "",
                    "### Rank=8 与 Rank=16（相同 3,000 条训练样本，validation）",
                    "",
                    f"- Rank=8 / alpha=16：Accuracy={float(r8['accuracy']):.4f}，"
                    f"F1(1)={float(r8['f1_positive']):.4f}，Macro-F1={float(r8['macro_f1']):.4f}。",
                    f"- Rank=16 / alpha=32：Accuracy={float(r16['accuracy']):.4f}，"
                    f"F1(1)={float(r16['f1_positive']):.4f}，Macro-F1={float(r16['macro_f1']):.4f}。",
                    f"- Rank=16 相对 Rank=8 的 Macro-F1 差值："
                    f"{float(r16['macro_f1']) - float(r8['macro_f1']):+.4f}；"
                    f"Accuracy 差值：{float(r16['accuracy']) - float(r8['accuracy']):+.4f}。",
                    "- 该秩对比仅使用 validation，不能写成最终 evaluation 成绩。",
                ]
        else:
            lines.append("当前没有 validation 预测文件。")
    else:
        lines.append("当前没有 validation 预测文件。")

    lines += [
        "",
        "## 解释边界与缺失项",
        "",
        "- A 的 Cosine 与 LR 预测可在统一 evaluation 上比较；C 的全量训练 r16 模型也有 evaluation 预测。",
        "- B 的 P0、最终 Zero-shot 和最终 Few-shot prediction 均已包含并按统一 evaluation 评分。",
        "- C 的 3,000 条训练 r8/r16 只有 validation 结果；目前没有它们的最终 evaluation 预测。",
        "- 延迟来自不同硬件、依赖版本与计时口径，不能直接据此做严格速度排名。",
        "- 错误标签为启发式筛选，报告中应人工阅读典型句对后再解释语义错误原因。",
        "",
    ]
    (out_dir / "summary_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_frozen_evaluation(output_dir=OUTPUT_DIR, make_plots=True, emit_json=True):
    """Re-evaluate every committed prediction file without loading model weights."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    availability = []
    for item in EXPECTED_METHODS:
        split = item["split"]
        found = (PREDICTION_DIRS[split] / f"{item['method']}.jsonl").exists()
        status = "available" if found else item["status"]
        availability.append({**item, "actual_file_present": found, "final_status": status})
    write_csv(out / "model_availability.csv", availability)

    results = {}
    failures = []
    for split_name in ("evaluation", "validation"):
        try:
            aligned = discover_models(split_name)
            if not aligned:
                results[split_name] = {"models": 0, "note": "No prediction files"}
                continue
            results[split_name] = build_metrics(split_name, aligned, out, make_plots=make_plots)
        except Exception as exc:
            failures.append({"split": split_name, "error": str(exc)})
            print(f"[ERROR] {split_name}: {exc}", file=sys.stderr)

    write_summary_report(out, results)
    run_record = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(Path(__file__).resolve().parents[1]),
        "results": results,
        "failures": failures,
        "notes": [
            "Evaluation and validation are reported separately; their scores must not be mixed.",
            "latency_ms is preserved from each producer and may not be comparable across hardware or timing scopes.",
            "Semantic error tags are heuristic candidates for manual review, not automatic ground-truth explanations.",
            "B Zero-shot and Few-shot evaluation predictions are included and scored.",
        ],
    }
    (out / "run_report.json").write_text(json.dumps(run_record, ensure_ascii=False, indent=2), encoding="utf-8")
    if emit_json:
        print(json.dumps(run_record, ensure_ascii=False, indent=2))
    if failures:
        raise RuntimeError(f"Evaluation failed for {len(failures)} split(s)")
    return run_record


def main():
    parser = argparse.ArgumentParser(description="AFQMC one-command evaluation")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    run_frozen_evaluation(args.output_dir, make_plots=not args.no_plots)


if __name__ == "__main__":
    main()
