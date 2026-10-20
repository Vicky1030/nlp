"""单模型错例与跨模型错误模式分析。"""
import argparse
import csv
from pathlib import Path

try:
    from .evaluate import load_jsonl, validate_predictions
    from .config import GOLD_FILE, RESULTS_DIR, OUTPUT_DIR
except ImportError:
    from evaluate import load_jsonl, validate_predictions
    from config import GOLD_FILE, RESULTS_DIR, OUTPUT_DIR


def safe_name(path):
    return Path(path).stem.replace(" ", "_")


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def analyze(results_dir=RESULTS_DIR, gold_path=GOLD_FILE, output_dir=OUTPUT_DIR):
    gold = load_jsonl(gold_path)
    gold_by_id = {str(r["id"]): r for r in gold}
    results_dir, output_dir = Path(results_dir), Path(output_dir)
    pred_files = sorted(p for p in results_dir.glob("*.jsonl") if p.name != Path(gold_path).name)
    if not pred_files:
        raise FileNotFoundError(f"{results_dir} 下没有模型预测 JSONL 文件")

    models = {}
    for path in pred_files:
        records = validate_predictions(load_jsonl(path), gold)
        model_name = records[0].get("method", path.stem) if records else path.stem
        # 文件名优先作为模型唯一键，避免不同文件 method 字段意外重复
        key = path.stem
        by_id = {str(r["id"]): r for r in records}
        models[key] = {"name": model_name, "records": by_id}
        wrong_rows = []
        for rid, row in by_id.items():
            if int(row["label"]) != int(row["prediction"]):
                g = gold_by_id[rid]
                wrong_rows.append({
                    "id": g["id"], "sentence1": g.get("sentence1", ""),
                    "sentence2": g.get("sentence2", ""), "label": g["label"],
                    "prediction": row["prediction"], "method": model_name,
                    "error_type": "FP" if int(g["label"]) == 0 else "FN",
                    "latency_ms": row.get("latency_ms", "")
                })
        write_csv(output_dir / "errors" / f"{key}_errors.csv", wrong_rows,
                  ["id", "sentence1", "sentence2", "label", "prediction", "method", "error_type", "latency_ms"])

    cross_rows = []
    for g in gold:
        rid = str(g["id"])
        row = {"id": g["id"], "sentence1": g.get("sentence1", ""),
               "sentence2": g.get("sentence2", ""), "label": g["label"]}
        correct_flags = []
        for key, info in models.items():
            pred = info["records"][rid]
            correct = int(pred["prediction"]) == int(g["label"])
            row[f"{key}_prediction"] = int(pred["prediction"])
            row[f"{key}_correct"] = int(correct)
            correct_flags.append(correct)
        if len(correct_flags) >= 2:
            if all(correct_flags):
                row["error_pattern"] = "all_correct"
            elif not any(correct_flags):
                row["error_pattern"] = "all_wrong"
            else:
                row["error_pattern"] = "models_disagree_or_partial"
        else:
            row["error_pattern"] = "single_model"
        cross_rows.append(row)

    model_keys = list(models)
    fields = ["id", "sentence1", "sentence2", "label"]
    for key in model_keys:
        fields += [f"{key}_prediction", f"{key}_correct"]
    fields += ["error_pattern"]
    write_csv(output_dir / "cross_model_predictions_and_errors.csv", cross_rows, fields)

    summary = []
    for key, info in models.items():
        correct = sum(row[f"{key}_correct"] for row in cross_rows)
        wrong = len(cross_rows) - correct
        summary.append({"model_file": key, "method": info["name"], "n_samples": len(cross_rows),
                        "correct": correct, "errors": wrong, "error_rate": wrong / len(cross_rows)})
    write_csv(output_dir / "error_summary.csv", summary,
              ["model_file", "method", "n_samples", "correct", "errors", "error_rate"])
    pattern_counts = {}
    for row in cross_rows:
        pattern_counts[row["error_pattern"]] = pattern_counts.get(row["error_pattern"], 0) + 1
    return {"models": list(models), "n_samples": len(gold), "pattern_counts": pattern_counts,
            "output_dir": str(output_dir)}


def main():
    parser = argparse.ArgumentParser(description="生成各模型错例及跨模型错误对比")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--gold", default=str(GOLD_FILE))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()
    import json
    print(json.dumps(analyze(args.results_dir, args.gold, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
