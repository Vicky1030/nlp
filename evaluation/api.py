"""Reusable D-member evaluation API for fresh prediction directories."""
import csv
import json
from datetime import datetime, timezone
from html import escape
from pathlib import Path

from .error_analysis import analyze_group, analyze_three_method_cases, write_csv
from .io_utils import align_predictions, load_jsonl, validate_gold
from .metrics import calculate_metrics
from .run_all import sha256
from .visualize import visualize


DISPLAY_NAMES = {
    "tfidf_cosine": "A · TF-IDF + Cosine",
    "tfidf_lr": "A · TF-IDF + Logistic Regression",
    "qwen_zero_shot": "B · Qwen Zero-shot",
    "qwen_few_shot": "B · Qwen Few-shot",
    "qwen_lora": "C · Qwen + LoRA",
}

ROLE_NAMES = {
    "A": "TF-IDF + LR",
    "B": "Qwen Few-shot",
    "C": "Qwen + LoRA",
}


def label_text(value):
    return "1（相似）" if value == 1 else "0（不相似）" if value == 0 else f"{value}（无效）"


def evaluate_prediction_directory(
    *, split_name, gold_path, prediction_dir, output_dir, make_plots=True
):
    """Validate, align, score and analyze every prediction JSONL in a directory."""
    gold_path = Path(gold_path).resolve()
    prediction_dir = Path(prediction_dir).resolve()
    output_dir = Path(output_dir).resolve()
    gold = validate_gold(load_jsonl(gold_path), gold_path)
    prediction_paths = sorted(prediction_dir.glob("*.jsonl"))
    if not prediction_paths:
        raise ValueError(f"No prediction JSONL files found in {prediction_dir}")

    aligned_models = {
        path.stem: align_predictions(load_jsonl(path), gold, path)
        for path in prediction_paths
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, details, confusion_rows = [], {}, []
    gold_digest = sha256(gold_path)
    for model, aligned in aligned_models.items():
        metrics = calculate_metrics(aligned)
        methods = sorted({str(row.get("method", model)) for row in aligned})
        metrics.update({
            "model": model,
            "display_name": DISPLAY_NAMES.get(model, model),
            "method": methods[0] if len(methods) == 1 else ";".join(methods),
            "split": split_name,
            "prediction_file": str(prediction_dir / f"{model}.jsonl"),
            "gold_file": str(gold_path),
            "gold_sha256": gold_digest,
        })
        details[model] = metrics
        rows.append({
            "split": split_name,
            "model": model,
            "display_name": metrics["display_name"],
            "method": metrics["method"],
            "n_samples": metrics["n_samples"],
            "accuracy": metrics["accuracy"],
            "precision_positive": metrics["precision_positive"],
            "recall_positive": metrics["recall_positive"],
            "f1_positive": metrics["f1_positive"],
            "macro_f1": metrics["macro_f1"],
            "correct": metrics["correct"],
            "errors": metrics["errors"],
            "error_rate": metrics["error_rate"],
            "invalid_predictions": metrics["invalid_predictions"],
            "latency_count": metrics["latency_count"],
            "latency_mean_ms": "" if metrics["latency_mean_ms"] is None else metrics["latency_mean_ms"],
            "latency_median_ms": "" if metrics["latency_median_ms"] is None else metrics["latency_median_ms"],
            "latency_p95_ms": "" if metrics["latency_p95_ms"] is None else metrics["latency_p95_ms"],
        })
        matrix = metrics["confusion_matrix"]["rows_true_cols_pred"]
        confusion_rows.append({
            "split": split_name,
            "model": model,
            "tn": matrix[0][0],
            "fp": matrix[0][1],
            "fn": matrix[1][0],
            "tp": matrix[1][1],
            "invalid_predictions": metrics["invalid_predictions"],
        })

    fields = list(rows[0])
    csv_path = output_dir / f"comparison_results_{split_name}.csv"
    write_csv(csv_path, rows, fields)
    write_csv(output_dir / f"confusion_matrices_{split_name}.csv", confusion_rows)
    metrics_path = output_dir / f"metrics_{split_name}.json"
    metrics_path.write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
    error_summary = analyze_group(aligned_models, output_dir, split_name)
    hard_cases = analyze_three_method_cases(aligned_models, output_dir)
    charts = visualize(csv_path, output_dir / "figures", split_name) if make_plots else []
    summary_md, summary_html = write_reports(
        output_dir, split_name, rows, charts, hard_cases
    )
    result = {
        "split": split_name,
        "models": len(aligned_models),
        "comparison_csv": str(csv_path),
        "metrics_json": str(metrics_path),
        "summary_markdown": str(summary_md),
        "summary_html": str(summary_html),
        "error_analysis": error_summary,
        "hard_cases": hard_cases,
        "charts": charts,
    }
    (output_dir / "run_report.json").write_text(
        json.dumps({
            "run_at_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "fresh_predictions",
            "gold_file": str(gold_path),
            "prediction_dir": str(prediction_dir),
            "result": result,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def write_reports(output_dir, split_name, rows, charts, hard_cases):
    """Write a human-readable Markdown report and a browser-friendly HTML report."""
    output_dir = Path(output_dir)
    ordered = sorted(rows, key=lambda row: float(row["macro_f1"]), reverse=True)
    quick_n = rows[0]["n_samples"] if rows else 0
    md = [
        "# AFQMC 实时运行统一评测结果",
        "",
        f"- 数据划分：{split_name}",
        "- 指标口径：label=1 为正类，无效预测保留在分母中并计为错误。",
        "- 所有方法使用同一份 gold 子集，并经过 ID、原句与标签严格对齐。",
        "",
    ]
    if split_name == "quick":
        md += [
            f"> **QUICK 录屏模式：** 本次是从完整 evaluation 按 seed=42 均匀随机抽取的 {quick_n} 条，"
            "用于快速验证端到端流程；其指标是抽样估计，正式成绩以 4,316 条全量 evaluation 为准。",
            "",
        ]
    md += [
        "| 方法 | 样本数 | Accuracy | Precision(1) | Recall(1) | F1(1) | Macro-F1 | 平均延迟(ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    body_rows = []
    for row in ordered:
        latency = row["latency_mean_ms"]
        latency_text = f"{float(latency):.3f}" if latency != "" else "N/A"
        md.append(
            f"| {row['display_name']} | {row['n_samples']} | {float(row['accuracy']):.4f} | "
            f"{float(row['precision_positive']):.4f} | {float(row['recall_positive']):.4f} | "
            f"{float(row['f1_positive']):.4f} | {float(row['macro_f1']):.4f} | {latency_text} |"
        )
        body_rows.append(
            "<tr>"
            f"<td>{escape(row['display_name'])}</td><td>{row['n_samples']}</td>"
            f"<td>{float(row['accuracy']):.4f}</td><td>{float(row['precision_positive']):.4f}</td>"
            f"<td>{float(row['recall_positive']):.4f}</td><td>{float(row['f1_positive']):.4f}</td>"
            f"<td>{float(row['macro_f1']):.4f}</td><td>{latency_text}</td></tr>"
        )
    md += [
        "",
        "详细混淆矩阵、逐模型错例、跨模型分歧样本与启发式错误标签位于同目录。",
        "延迟包含的步骤以各方法运行记录为准，不同硬件或计时范围下不能做严格速度排名。",
        "",
        "## 三类典型样例",
        "",
        "由于 NF4 量化推理存在微小数值噪声，复现运行时满足上述模式的具体样本 ID "
        "可能变化；`error_analysis.py` 会根据本次预测自动重新筛选。",
        "",
    ]
    hard_case_html = []
    for case in hard_cases.get("cases", []):
        selected = case.get("selected_case")
        if selected:
            predictions = selected["predictions"]
            md += [
                f"### {case['title']}",
                "",
                f"- 匹配数量：{case['match_count']}；展示样本 ID：{selected['id']}",
                f"- 句子1：{selected['sentence1']}",
                f"- 句子2：{selected['sentence2']}",
                f"- 真实标签：{label_text(selected['label'])}",
                "",
                "| 角色 | 方法 | 预测结果 | 是否正确 |",
                "|---|---|---:|---|",
            ]
            for role in ("A", "B", "C"):
                result = predictions[role]
                md.append(
                    f"| {role} | {ROLE_NAMES[role]} | {label_text(result['prediction'])} | "
                    f"{'正确' if result['correct'] else '错误'} |"
                )
            md += ["", case["explanation"], ""]
            result_rows = "".join(
                "<tr>"
                f"<td>{role}</td><td>{escape(ROLE_NAMES[role])}</td>"
                f"<td>{escape(label_text(predictions[role]['prediction']))}</td>"
                f"<td>{'正确' if predictions[role]['correct'] else '错误'}</td></tr>"
                for role in ("A", "B", "C")
            )
            hard_case_html.append(
                f'<article class="case"><h3>{escape(case["title"])}</h3>'
                f'<p class="note">匹配数量：{case["match_count"]}；展示样本 ID：{selected["id"]}</p>'
                f'<p><strong>句子1：</strong>{escape(selected["sentence1"])}</p>'
                f'<p><strong>句子2：</strong>{escape(selected["sentence2"])}</p>'
                f'<p><strong>真实标签：</strong>{escape(label_text(selected["label"]))}</p>'
                '<table><thead><tr><th>角色</th><th>方法</th><th>预测结果</th><th>是否正确</th></tr></thead>'
                f'<tbody>{result_rows}</tbody></table><p>{escape(case["explanation"])}</p></article>'
            )
        else:
            md.append(f"- **{case['title']}**：本次子集中未找到满足该模式的样例。")
            hard_case_html.append(
                f'<article class="case"><h3>{escape(case["title"])}</h3>'
                "<p>本次子集中未找到满足该模式的样例。</p></article>"
            )
    if not hard_cases.get("cases"):
        message = "本次没有同时包含 A/B/C 三个代表方法，未执行三方法难例筛选。"
        md.append(f"- {message}")
        hard_case_html.append(f'<article class="case"><p>{message}</p></article>')
    md_path = output_dir / "summary_report.md"
    md_path.write_text("\n".join(md), encoding="utf-8")

    chart_html = "".join(
        f'<figure><img src="figures/{escape(name)}" alt="{escape(name)}"></figure>'
        for name in charts if not str(name).startswith("SKIPPED")
    )
    scope_html = (
        '<p class="warning"><strong>QUICK 录屏模式：</strong>本次从完整 evaluation 按 seed=42 '
        f"均匀随机抽取 {quick_n} 条，仅用于快速验证流程；正式成绩以 4,316 条全量 evaluation 为准。</p>"
        if split_name == "quick" else ""
    )
    html_text = f"""<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">
<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>AFQMC 统一评测结果</title>
<style>body{{font-family:system-ui,"Microsoft YaHei",sans-serif;max-width:1180px;margin:32px auto;padding:0 20px;color:#1f2937}}
table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{border:1px solid #d1d5db;padding:9px;text-align:right}}
th:first-child,td:first-child{{text-align:left}}th{{background:#1f4e78;color:white}}img{{max-width:100%;height:auto}}
.note{{color:#4b5563;line-height:1.7}}.warning{{background:#fff7d6;border:1px solid #e5b700;border-radius:8px;padding:12px 15px;line-height:1.6}}.case{{border:1px solid #dbe3ec;border-radius:10px;padding:14px 18px;margin:18px 0;background:#f8fafc}}
.case table{{background:white;margin:10px 0}}</style></head><body><h1>AFQMC 统一评测结果</h1>
<p class=\"note\">数据划分：{escape(split_name)}。同一 gold 子集、同一 D 评测实现；无效预测计为错误。</p>
{scope_html}
<table><thead><tr><th>方法</th><th>样本数</th><th>Accuracy</th><th>Precision(1)</th><th>Recall(1)</th><th>F1(1)</th><th>Macro-F1</th><th>平均延迟 ms</th></tr></thead>
<tbody>{''.join(body_rows)}</tbody></table>
<h2>三类典型样例</h2>
<p class=\"note\">由于 NF4 量化推理存在微小数值噪声，复现运行时满足上述模式的具体样本 ID 可能变化；error_analysis.py 会根据本次预测自动重新筛选。</p>
{''.join(hard_case_html)}{chart_html}
<p class=\"note\">完整结构化结果见 hard_case_output.json；详细错例见 errors 目录与 cross_model_errors CSV。</p></body></html>"""
    html_path = output_dir / "summary_report.html"
    html_path.write_text(html_text, encoding="utf-8")
    return md_path, html_path
