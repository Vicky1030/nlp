import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "Zero-shot_NF4与BF16对比.md"
METRICS = ROOT / "outputs/zero_shot_nf4_validation.metrics.json"
PREDICTIONS = {
    "BF16": ROOT / "outputs/zero_shot_validation.jsonl",
    "NF4": ROOT / "outputs/zero_shot_nf4_validation.jsonl",
}


def read_records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def divide(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def evaluate(records):
    valid = lambda prediction: type(prediction) is int and prediction in (0, 1)
    correct = sum(valid(record["prediction"]) and record["prediction"] == record["label"] for record in records)
    invalid = sum(not valid(record["prediction"]) for record in records)
    confusion = [[sum(record["label"] == label and valid(record["prediction"]) and record["prediction"] == predicted for record in records) for predicted in (0, 1)] for label in (0, 1)]
    scores = {}
    for label in (0, 1):
        true_positive = confusion[label][label]
        predicted_count = sum(valid(record["prediction"]) and record["prediction"] == label for record in records)
        support = sum(record["label"] == label for record in records)
        precision = divide(true_positive, predicted_count)
        recall = divide(true_positive, support)
        scores[str(label)] = {
            "precision": precision, "recall": recall,
            "f1": divide(2 * precision * recall, precision + recall), "support": support,
        }
    return {
        "total": len(records), "correct": correct, "accuracy": divide(correct, len(records)),
        "invalid_count": invalid, "invalid_rate": divide(invalid, len(records)),
        "confusion_matrix_valid_0_1": confusion, "per_class": scores,
        "macro_f1": (scores["0"]["f1"] + scores["1"]["f1"]) / 2,
        "positive_predictions": sum(valid(record["prediction"]) and record["prediction"] == 1 for record in records),
        "mean_generation_latency_ms": sum(record["latency_ms"] for record in records) / len(records),
    }


def main():
    for destination in (REPORT, METRICS):
        if destination.exists():
            raise FileExistsError(destination)
    records = {name: read_records(path) for name, path in PREDICTIONS.items()}
    summaries = {name: json.loads(path.with_suffix(".summary.json").read_text(encoding="utf-8")) for name, path in PREDICTIONS.items()}
    baseline = summaries["BF16"]
    fields = ("id", "sentence1", "sentence2", "label")
    validation = read_records(Path(baseline["input"]))
    expected = [tuple(record[key] for key in fields) for record in validation]
    assert hashlib.sha256(Path(baseline["input"]).read_bytes()).hexdigest() == baseline["input_sha256"]
    metrics = {}
    for name, rows in records.items():
        summary = summaries[name]
        assert summary["status"] == "complete"
        assert len(rows) == len({record["id"] for record in rows}) == summary["completed"] == summary["requested_count"] == 3433
        assert [tuple(record[key] for key in fields) for record in rows] == expected
        assert [record["id"] for record in rows] == summary["selected_ids"]
        assert all(set(record) == {"id", "sentence1", "sentence2", "label", "prediction", "method", "latency_ms"} for record in rows)
        assert all(record["method"] == "zero_shot" for record in rows)
        for key in ("input_sha256", "prompt_template", "enable_thinking", "generation_settings", "invalid_policy"):
            assert summary[key] == baseline[key], (name, key)
        for key in ("model_fingerprint", "dtype", "torch", "transformers", "gpu"):
            assert summary["model_info"][key] == baseline["model_info"][key], (name, key)
        metrics[name] = evaluate(rows)
        assert metrics[name]["correct"] == summary["correct"]
        assert metrics[name]["invalid_count"] == summary["invalid_count"]
        assert abs(metrics[name]["accuracy"] - summary["accuracy"]) < 1e-12
        assert abs(metrics[name]["mean_generation_latency_ms"] - summary["mean_latency_ms"]) < 0.001
    assert baseline["model_info"]["quantization"] is None
    quantization = summaries["NF4"]["model_info"]["quantization"]
    assert quantization["load_in_4bit"] and quantization["bnb_4bit_quant_type"] == "nf4"
    assert quantization["bnb_4bit_use_double_quant"] and quantization["bnb_4bit_compute_dtype"] == "bfloat16"
    assert quantization["linear4bit_modules"] > 0
    prompt = (ROOT / "zero_shot.txt").read_text(encoding="utf-8")
    assert prompt == baseline["prompt_template"]
    assert hashlib.sha256(prompt.encode("utf-8")).hexdigest() == summaries["NF4"]["prompt_sha256"]
    paired = {
        "prediction_changed": sum(first["prediction"] != second["prediction"] for first, second in zip(records["BF16"], records["NF4"])),
        "corrected": sum(first["prediction"] != first["label"] and second["prediction"] == second["label"] for first, second in zip(records["BF16"], records["NF4"])),
        "regressed": sum(first["prediction"] == first["label"] and second["prediction"] != second["label"] for first, second in zip(records["BF16"], records["NF4"])),
    }
    assert metrics["NF4"]["correct"] - metrics["BF16"]["correct"] == paired["corrected"] - paired["regressed"]
    output = {
        "verified": True, "metrics": metrics, "paired_nf4_vs_bf16": paired,
        "prediction_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in PREDICTIONS.items()},
        "input_sha256": baseline["input_sha256"],
        "model_fingerprint": baseline["model_info"]["model_fingerprint"],
        "prompt_sha256": summaries["NF4"]["prompt_sha256"],
        "quantization": quantization,
        "invalid_policy": "Keep invalid predictions as wrong; include them in true-class recall/F1 denominators",
    }
    lines = [
        "# 最初版 Zero-shot：NF4 与 BF16 完整验证集对比", "",
        "本次使用最初版 `zero_shot.txt`，在本地 Qwen3-1.7B 上完成 NF4 推理；未加载 LoRA Adapter，未进行微调。BF16 一列取此前保存的完整验证集结果，本次没有重跑 BF16。", "",
        "## 设置与核对", "",
        "- 数据：A 提供的完整 validation，3,433 条；两份结果的 ID、句子、标签和顺序均与原始验证集一致。",
        "- 提示词：与最初 BF16 实验保存的提示词逐字一致；无 system prompt，非思考模式。",
        "- 生成：do_sample=False、max_new_tokens=8、seed=42；严格解析最终 0/1，无效输出计错，不从分母删除。",
        "- NF4：bitsandbytes 4-bit NF4、double quantization、BF16 计算，固定单张 RTX 4060 Laptop GPU。",
        f"- 依赖：bitsandbytes {quantization['bitsandbytes']}、accelerate {quantization['accelerate']}；实际转换 {quantization['linear4bit_modules']} 个 Linear4bit 模块。",
        "- 两次实验的本地基础模型文件指纹、GPU、计算 dtype、PyTorch 与 Transformers 版本相同。相同权重文件不代表运行中的量化权重与 BF16 权重数值完全一致。",
        f"- 验证集 SHA256：`{baseline['input_sha256']}`。",
        f"- 基础模型文件指纹：`{baseline['model_info']['model_fingerprint']}`。",
        "- 仅使用 validation，本次未使用 evaluation。单条冒烟测试单独保存，不计入以下完整验证集指标。", "",
        "## 完整结果", "",
        "| 指标 | 原版 BF16 | 原版 NF4 |", "|---|---:|---:|",
    ]
    for title, values, specifier in (
        ("样本数", [metrics[name]["total"] for name in ("BF16", "NF4")], "d"),
        ("正确数", [metrics[name]["correct"] for name in ("BF16", "NF4")], "d"),
        ("Accuracy", [metrics[name]["accuracy"] for name in ("BF16", "NF4")], ".2%"),
        ("Macro-F1", [metrics[name]["macro_f1"] for name in ("BF16", "NF4")], ".4f"),
        ("正类 Precision", [metrics[name]["per_class"]["1"]["precision"] for name in ("BF16", "NF4")], ".2%"),
        ("正类 Recall", [metrics[name]["per_class"]["1"]["recall"] for name in ("BF16", "NF4")], ".2%"),
        ("正类 F1", [metrics[name]["per_class"]["1"]["f1"] for name in ("BF16", "NF4")], ".4f"),
        ("预测为 1 的数量", [metrics[name]["positive_predictions"] for name in ("BF16", "NF4")], "d"),
        ("无效输出", [metrics[name]["invalid_count"] for name in ("BF16", "NF4")], "d"),
        ("平均生成耗时（ms）", [metrics[name]["mean_generation_latency_ms"] for name in ("BF16", "NF4")], ".2f"),
    ):
        lines.append(f"| {title} | {format(values[0], specifier)} | {format(values[1], specifier)} |")
    lines.extend([
        "", "混淆矩阵使用 `[[TN, FP], [FN, TP]]`；若有无效输出，矩阵只包含有效预测，无效数另列，Accuracy 与 F1 仍计入这些错误。", "",
        f"- BF16：`{metrics['BF16']['confusion_matrix_valid_0_1']}`。",
        f"- NF4：`{metrics['NF4']['confusion_matrix_valid_0_1']}`。", "",
        "## 配对变化与解释", "",
        f"- NF4 相比 BF16，{paired['prediction_changed']} 条预测发生变化：改对 {paired['corrected']} 条，改错 {paired['regressed']} 条，净增加正确 {metrics['NF4']['correct'] - metrics['BF16']['correct']:+d} 条。",
        f"- Accuracy 变化 {(metrics['NF4']['accuracy'] - metrics['BF16']['accuracy']) * 100:+.2f} 个百分点；Macro-F1 变化 {metrics['NF4']['macro_f1'] - metrics['BF16']['macro_f1']:+.4f}；正类召回率变化 {(metrics['NF4']['per_class']['1']['recall'] - metrics['BF16']['per_class']['1']['recall']) * 100:+.2f} 个百分点。",
        "- 这是同一最初版提示词在两种权重表示下的开发集结果，不把 NF4 表现当作微调收益，也不据此保证 C 的 LoRA 会达到相同结果。与 C 比较还需核对完整提示词、模型文件、生成设置、解析口径及依赖环境。",
        "- 原版、v2、v3 的历史提示词对比采用 BF16；本次只补跑原版 NF4，不能把量化和提示词变化混在一起归因。若后续统一改用 NF4，应在同一精度下重新比较候选方案。",
        "- 耗时均为预热后、同步 GPU 的 model.generate，排除加载、分词、解码和写文件。两次运行时间及电脑负载不同，延迟数据仅作记录，不能作为严格量化加速实验。", "",
        "## 文件与复现", "",
        "- 预测：`outputs/zero_shot_nf4_validation.jsonl`（D 的七字段，method=zero_shot）。",
        "- 配置与汇总：`outputs/zero_shot_nf4_validation.summary.json`。",
        "- 原始调用：`outputs/zero_shot_nf4_validation.calls.jsonl`。",
        "- 独立指标与配对变化：`outputs/zero_shot_nf4_validation.metrics.json`。",
        "- 计算脚本：`src/work/compare_nf4_baseline.py`。", "",
        "```powershell",
        ".\\.venv\\Scripts\\python.exe zero_shot.py --quantization nf4 --prompt-file zero_shot.txt --limit 0 --output outputs\\zero_shot_nf4_validation.jsonl",
        "```", "",
        "结果已存在，脚本会拒绝覆盖；复跑时使用新的输出文件名。", "",
    ])
    with METRICS.open("x", encoding="utf-8") as destination:
        destination.write(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    with REPORT.open("x", encoding="utf-8") as destination:
        destination.write("\n".join(lines))
    print(json.dumps(output, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
